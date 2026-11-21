#pragma once

#include <cuda_runtime_api.h>
#include <stdint.h>

#if defined(__cplusplus)
#define SEVENTY_EXTERN extern "C"
#else
#define SEVENTY_EXTERN
#endif
#define SEVENTY_API SEVENTY_EXTERN __attribute__((visibility("default")))

#define SEVENTY_ABI_VERSION 1

/*
 * Every function returns 0 on success. A value below 10000 is a cudaError_t from the launch. A value from
 * 10000 up is one of the codes below. seventy_error_string() gives the text for any code.
 */
enum seventy_status {
  SEVENTY_OK = 0,
  SEVENTY_ERR_INVALID_ARGUMENT = 10001,
  SEVENTY_ERR_MISALIGNED = 10002,
};

SEVENTY_API int seventy_abi_version(void);
SEVENTY_API const char *seventy_error_string(int code);

/*
 * RMS norm of each row of a contiguous FP16 matrix [rows, dim].
 * Without a residual: out = fp16(x * rsqrt(mean(x^2) + eps) * weight).
 * With a residual: s = fp16(x + residual) goes to residual_out, and out is the norm of s.
 * weight is FP32 [dim] (the 1 + w offset of zero-centered norms is already in it).
 * out may alias x, and residual_out may alias residual. dim must be even.
 */
SEVENTY_API int seventy_rms_norm(void *out, void *residual_out, const void *x, const void *residual,
                                 const float *weight, int64_t rows, int dim, float eps, cudaStream_t stream);

/*
 * Dequantize an NVFP4 matrix [n, k] to FP16.
 * packed: uint8 [n, k / 2], two E2M1 values per byte, the even column in the low nibble.
 * scales: FP8 E4M3 bytes [n, k / 16], one scale for each group of 16 values along k.
 * value = e2m1 * (scale / global_scale). k must be a multiple of 16.
 */
SEVENTY_API int seventy_nvfp4_dequantize(void *out, const uint8_t *packed, const uint8_t *scales,
                                         float global_scale, int64_t n, int64_t k, cudaStream_t stream);

/*
 * Bytes of FP32 workspace that seventy_nvfp4_gemm needs for m rows, n_padded columns and `splits` k splits.
 */
SEVENTY_API int64_t seventy_nvfp4_gemm_workspace(int m, int n_padded, int splits);

/*
 * out[m, n] (FP16, row stride n) = alpha[n] * x[m, k] (FP16, row stride k) times the NVFP4 weight [n, k],
 * repacked by seventy.nvfp4.pack_for_gemm into wpk (16 bytes per lane, column and 32-wide k chunk) and wsc (the
 * E4M3 group scales, 4 per lane and 64-wide k block). alpha holds 2^7 / global_scale for each output column.
 * 1 <= m <= 64, n_padded is n rounded up to 32, k is a multiple of 64. splits >= 1 divides k between blocks;
 * with splits > 1 the workspace must hold seventy_nvfp4_gemm_workspace() bytes. warps (2, 4 or 8) is the number of
 * 32-column tiles per block.
 */
SEVENTY_API int seventy_nvfp4_gemm(void *out, const void *x, const void *wpk, const void *wsc, const float *alpha,
                                   int m, int n, int n_padded, int k, int splits, int warps, void *workspace,
                                   int64_t workspace_bytes, cudaStream_t stream);

/*
 * The dense FP16 weight [n, k] of a weight repacked for seventy_nvfp4_gemm: e2m1 * e4m3 * alpha / 2^7, rounded once.
 */
SEVENTY_API int seventy_nvfp4_gemm_dequantize(void *out, const void *wpk, const void *wsc, const float *alpha, int n,
                                              int n_padded, int k, cudaStream_t stream);

/*
 * Gated DeltaNet for decode and verify steps, fused (see csrc/gdn.cu): convolution with the cached history, SiLU,
 * q/k L2 norms, gates, delta-rule update and output, gated RMS norm.
 * out: FP16 [T, hv * dv]. conv_state: FP16 [slots, conv_dim, taps - 1]. state: FP32 [slots, hv, dk, dv].
 * y: FP16 [T, ld], the in_proj output [q k v (conv_dim) | z (hv * dv) | b (hv) | a (hv)].
 * conv_w: FP32 [conv_dim, taps]. a_log, dt_bias: FP32 [hv]. norm_w: FP32 [dv].
 * query_start: int32 [num_seqs + 1], context_lens and state_slots: int32 [num_seqs], all on the device.
 * dk = dv = 128, 2 <= taps <= 4, hv a multiple of hk.
 * commit = 0 leaves the state and history unchanged (speculative verify). commit_lens (int32 [num_seqs], may be
 * null) limits each sequence to its first commit_lens[b] rows (the commit pass after acceptance; out may be null).
 */
SEVENTY_API int seventy_gdn_recurrent(void *out, void *conv_state, float *state, const void *y, int ld,
                                      const float *conv_w, const float *a_log, const float *dt_bias,
                                      const float *norm_w, const int *query_start, const int *context_lens,
                                      const int *state_slots, const int *commit_lens, int commit, int num_seqs,
                                      int hk, int hv, int dk, int dv, int taps, float eps, cudaStream_t stream);

/*
 * The commit pass of seventy_gdn_recurrent for `layers` layers in one launch. The arrays of layer l start at a fixed
 * stride (in elements) from those of layer 0: y at l * y_ls, conv_state at l * conv_ls, state at l * state_ls;
 * conv_w at l * conv_dim * taps, a_log and dt_bias at l * hv, norm_w at l * dv. The sequence arrays and commit_lens
 * are the same for all layers.
 */
SEVENTY_API int seventy_gdn_commit_layers(void *conv_state, float *state, const void *y, int ld, int64_t y_ls,
                                          int64_t conv_ls, int64_t state_ls, const float *conv_w, const float *a_log,
                                          const float *dt_bias, const float *norm_w, const int *query_start,
                                          const int *context_lens, const int *state_slots, const int *commit_lens,
                                          int num_seqs, int layers, int hk, int hv, int dk, int dv, int taps,
                                          float eps, cudaStream_t stream);

/*
 * Attention prep for full-attention layers (head size 256), one launch per layer step: RMS norm of each query and
 * key head (weights with the 1 + w offset in them), multimodal rotary on the first rotary_dim values, queries to
 * q_out [T, hq, 256], keys and raw values to the paged caches [blocks, block_size, hkv, 256] at slots[t].
 * kv_type 0: FP16 caches. kv_type 1: int8 caches with FP32 scales [blocks, block_size, hkv] (k_scale, v_scale):
 * scale = max |x| / 127 over the head's 256 values, value = round(x / scale).
 * qkv: FP16 [T, ld] = [q_proj (per head: query 256 | gate 256) | k_proj | v_proj]. positions: int64 [3, T]
 * (row stride pos_stride). inv_freq: FP32 [rotary_dim / 2]. section1, section2: the height and width sections of
 * the interleaved multimodal rotary.
 */
SEVENTY_API int seventy_attn_prep(void *q_out, void *k_cache, void *v_cache, float *k_scale, float *v_scale,
                                  int kv_type, const void *qkv, int ld, const float *q_norm, const float *k_norm,
                                  const int64_t *positions, int64_t pos_stride, const float *inv_freq, int section1,
                                  int section2, int rotary_dim, const int *slots, int tokens, int hq, int hkv,
                                  int head_dim, float eps, cudaStream_t stream);

SEVENTY_API int64_t seventy_decode_attention_workspace(int tokens, int hq, int splits);

/*
 * Causal attention for steps with few new tokens per sequence ((hq / hkv) * max_query_len <= 64), over the paged
 * caches (kv_type as in seventy_attn_prep), each sequence's cache split into `splits` equal ranges of blocks (so
 * the launch does not depend on context lengths and can be captured in a CUDA graph), then merged and
 * multiplied by sigmoid(gate) (the gate half of each query head in qkv). out: FP16 [T, hq * 256]. block_tables:
 * int32 [num_seqs, table_width].
 */
SEVENTY_API int seventy_decode_attention(void *out, const void *q, const void *k_cache, const void *v_cache,
                                         const float *k_scale, const float *v_scale, int kv_type, const void *qkv,
                                         int ld, const int *block_tables, int table_width,
                                         const int *query_start, const int *context_lens, int num_seqs, int tokens,
                                         int max_query_len, int block_size, int hq, int hkv, int head_dim, int splits,
                                         float scale, void *workspace, int64_t workspace_bytes, cudaStream_t stream);

/*
 * Gated DeltaNet prefill, sequential part (see csrc/gdn_chunk.cu): one block per (sequence, value head) walks the
 * sequence's chunks of 64 tokens with the FP32 state, from inputs prepared per chunk by seventy.ops.gdn_prefill.
 * out: FP16 [T, hv, 128]. state: FP32 [slots, hv, 128, 128]. w, u, gq: FP16 [hv, total_chunks, 64, 128];
 * p: FP16 [hv, total_chunks, 64, 64]; klt: FP16 [hv, total_chunks, 128, 64]; eg: FP32 [hv, total_chunks].
 * chunk_start: int32 [num_seqs], the index of each sequence's first chunk. With the workspaces hs FP16
 * [hv, total_chunks, 128, 128] and vt FP16 [hv, total_chunks, 128, 64], the state recurrence runs alone (4 blocks per
 * value head) and the outputs of all chunks follow in parallel; with null workspaces, one kernel does both.
 */
SEVENTY_API int seventy_gdn_chunk_scan(void *out, float *state, const void *w, const void *u, const void *gq,
                                       const void *p, const void *klt, const float *eg, const int *chunk_start,
                                       const int *query_start, const int *context_lens, const int *state_slots,
                                       int num_seqs, int hv, int64_t total_chunks, void *hs, void *vt,
                                       cudaStream_t stream);

/*
 * Gated DeltaNet prefill, parallel part (see csrc/gdn_prepare.cu): the per-chunk inputs of seventy_gdn_chunk_scan
 * (outputs as listed there) from the convolution output qkv FP16 [T, ld] = [q (hk * 128) | k (hk * 128) |
 * v (hv * 128) | ...], g and beta FP32 [T, hv]. Chunk c covers rows chunk_row[c] .. + chunk_len[c] (1 to 64),
 * int32 [chunks] on the device. ld even, dk = dv = 128, hv a multiple of hk.
 */
SEVENTY_API int seventy_gdn_prepare(void *w, void *u, void *gq, void *p, void *klt, float *eg, const void *qkv,
                                    int ld, const float *g, const float *beta, const int *chunk_row,
                                    const int *chunk_len, int chunks, int hk, int hv, int dk, int dv,
                                    cudaStream_t stream);

/*
 * Gated DeltaNet prefill, element-wise parts (see csrc/gdn_glue.cu). seventy_gdn_conv: out FP16 [T, channels] =
 * silu(causal depthwise convolution of the first channels columns of y [T, ld] with conv_w FP32 [channels, taps]),
 * continuing each sequence from its history in conv_state FP16 [slots, channels, taps - 1] (none if its context
 * length is 0), which it then updates. seventy_gdn_gates: g and beta FP32 [T, hv] from the a and b columns of y
 * (offsets a_off, b_off). seventy_gated_rms_norm: out = rms_norm(x) * weight * silu(z) for x FP16 [T * hv, 128]
 * and z the columns z_off + h * 128 of y-like rows (row stride z_ld). Sequence arrays as in seventy_gdn_recurrent.
 */
SEVENTY_API int seventy_gdn_conv(void *out, void *conv_state, const void *y, int ld, const float *conv_w,
                                 const int *query_start, const int *context_lens, const int *state_slots,
                                 int num_seqs, int tokens, int channels, int taps, cudaStream_t stream);
SEVENTY_API int seventy_gdn_gates(float *g, float *beta, const void *y, int ld, int a_off, int b_off,
                                  const float *a_log, const float *dt_bias, int tokens, int hv, cudaStream_t stream);
SEVENTY_API int seventy_gated_rms_norm(void *out, const void *x, const void *z, int z_ld, int z_off,
                                       const float *weight, int tokens, int hv, int dv, float eps,
                                       cudaStream_t stream);

/*
 * Prefill attention (head size 256) over the paged caches (kv_type as in seventy_attn_prep), causal, with the
 * output gate (see csrc/flash_attention.cu). tiles: int32 [num_tiles, 2] = (sequence, first new token of a
 * 64-token query tile). out: FP16 [T, hq * 256].
 */
SEVENTY_API int seventy_flash_attention(void *out, const void *q, const void *k_cache, const void *v_cache,
                                        const float *k_scale, const float *v_scale, int kv_type, const void *qkv,
                                        int ld, const int *tiles, int num_tiles, const int *block_tables,
                                        int table_width, const int *query_start, const int *context_lens,
                                        int block_size, int hq, int hkv, int head_dim, float scale,
                                        cudaStream_t stream);

/*
 * Causal attention for speculative verify steps ((hq / hkv) * max_query_len <= 64) on tensor cores, with the same
 * arguments and result as seventy_decode_attention (see csrc/flash_attention.cu). Each sequence uses
 * min(splits, ceil(keys / min_split_keys)) of the `splits` blocks.
 */
SEVENTY_API int seventy_verify_attention(void *out, const void *q, const void *k_cache, const void *v_cache,
                                         const float *k_scale, const float *v_scale, int kv_type, const void *qkv,
                                         int ld, const int *block_tables, int table_width, const int *query_start,
                                         const int *context_lens, int num_seqs, int tokens, int max_query_len,
                                         int block_size, int hq, int hkv, int head_dim, int splits,
                                         int min_split_keys, float scale, void *workspace, int64_t workspace_bytes,
                                         cudaStream_t stream);

/*
 * Prefill attention over the INT8 caches on the verify kernel (csrc/verify_int8.cu): the new tokens of each
 * sequence are cut into groups of at most max_group_tokens (with (hq / hkv) * max_group_tokens <= 48), and every
 * group runs as a verify sequence of its own. Group g owns token rows [group_start[g], group_start[g + 1]), sees
 * group_context[g] cached keys before its first token, and reads block-table row group_rows[g]. Workspace and
 * result as in seventy_verify_attention (tokens = group_start[num_groups]).
 */
SEVENTY_API int seventy_grouped_attention(void *out, const void *q, const void *k_cache, const void *v_cache,
                                          const float *k_scale, const float *v_scale, const void *qkv, int ld,
                                          const int *block_tables, int table_width, const int *group_rows,
                                          const int *group_start, const int *group_context, int num_groups,
                                          int tokens, int max_group_tokens, int block_size, int hq, int hkv,
                                          int head_dim, int splits, int min_split_keys, float scale, void *workspace,
                                          int64_t workspace_bytes, cudaStream_t stream);

/*
 * Prefill attention over the INT8 caches (head_dim 256, power-of-two block_size) on the split-D kernel
 * (csrc/flash_attention.cu). tiles: int32 [num_tiles, 4], each {sequence, first token, first head h0, g << 8 | n}:
 * n tokens of the g heads h0 .. h0 + g - 1 (one KV head), n * g <= 64; together the tiles cover every new token of
 * every head once. tokens = query_start[num_seqs]. splits > 1 also divides the keys of every tile among splits
 * blocks; the workspace then holds tokens * hq * splits * 258 floats (null and 0 for splits = 1).
 */
SEVENTY_API int seventy_prefill_attention(void *out, const void *q, const void *k_cache, const void *v_cache,
                                          const float *k_scale, const float *v_scale, const void *qkv, int ld,
                                          const int *tiles, int num_tiles, const int *block_tables, int table_width,
                                          const int *query_start, const int *context_lens, int tokens,
                                          int block_size, int hq, int hkv, int head_dim, int splits, float scale,
                                          void *workspace, int64_t workspace_bytes, cudaStream_t stream);

/*
 * The k <= 64 largest values of every row of x [rows, n] (FP32, row stride ld) and their ids, sorted by value
 * descending and then id ascending (csrc/topk.cu): values [rows, k], ids_out [rows, k] (int32). The id of element e
 * is ids[r * ids_ld + e] if ids is given (negative ids are skipped), else id_offset + e. Rows with fewer than k
 * elements are padded with -inf and id -1. Rows longer than 4096 need seventy_topk_workspace(rows, n, k) bytes of
 * workspace (else it may be null).
 */
SEVENTY_API int64_t seventy_topk_workspace(int rows, int n, int k);
SEVENTY_API int seventy_topk(float *values, int *ids_out, const float *x, int64_t ld, const int *ids, int64_t ids_ld,
                             int rows, int n, int k, int id_offset, void *workspace, int64_t workspace_bytes,
                             cudaStream_t stream);

/*
 * Speculative acceptance on the device (csrc/spec.cu): device_accept_top and cut_at_stops of seventy/engine/spec.py
 * in one kernel. pred int64 [rows, k + 1]; vals FP32 and idx int64 [rows, k + 1, top] (top = 64, descending);
 * drafts int64 [rows, k] (row stride drafts_ld); q FP32 and cand int64 [rows, k, c]; temp, top_p, min_p FP32 [rows];
 * top_k int64 [rows]; u_accept FP32 [rows, k]; u_residual FP32 [rows, k + 1]; stop int64 [rows, stops] (-1 pads);
 * limit int64 [rows]. Writes n_out int64 [rows] (accepted drafts after the cut) and new_out int64 [rows, k + 1].
 */
SEVENTY_API int seventy_spec_accept(int64_t *n_out, int64_t *new_out, const int64_t *pred, const float *vals,
                                    const int64_t *idx, const int64_t *drafts, int64_t drafts_ld, const float *q,
                                    const int64_t *cand, const float *temp, const int64_t *top_k, const float *top_p,
                                    const float *min_p, const float *u_accept, const float *u_residual,
                                    const int64_t *stop, int stops, const int64_t *limit, int rows, int k, int top,
                                    int c, cudaStream_t stream);

/*
 * The DFlash2 selector walk on the device (csrc/spec.cu): tokens int64 [rows, n] and probs FP32 [rows, n, k] from
 * anchors int64 [rows], candidate values FP32 [rows, n, k] and ids int64 [rows, n, k], the selector hidden states FP32
 * [rows, n, rank], the predecessor and successor codebooks FP16 [vocab, rank], temperatures FP32 [rows] (0 = greedy)
 * and optional Gumbel noise FP32 [rows, n, k]. k <= 32; rank a multiple of 32, at most 1024.
 */
SEVENTY_API int seventy_drafter_walk(int64_t *tokens, float *probs, const int64_t *anchors, const float *values,
                                     const int64_t *ids, const float *hid, const void *pred, const void *succ,
                                     const float *temps, const float *gumbel, int rows, int n, int k, int rank,
                                     cudaStream_t stream);

/*
 * Small-batch tensor-parallel all-reduce over peer GPU memory, fused with the residual add and RMS norm
 * (see csrc/allreduce.cu). seventy_ar_alloc allocates this rank's buffer of seventy_ar_buffer_bytes(world,
 * slot_bytes) bytes (slot_bytes a multiple of 32), prepares it, and returns its 64-byte CUDA IPC handle; seventy_ar_open maps a peer's handle; seventy_ar_context builds the
 * per-rank context from all ranks' buffer pointers (in rank order).
 * seventy_ar_norm: s = all-reduce of x [rows, dim] (FP16, FP32 sum rounded once). With weight: residual += s
 * (FP16), out = rms_norm(residual) * weight (residual may be null: out = rms_norm(s) * weight). Without weight:
 * out = s. rows * dim * 2 must fit in slot_bytes. Every rank must make the same calls in the same order.
 */
SEVENTY_API int64_t seventy_ar_buffer_bytes(int world, int64_t slot_bytes);
SEVENTY_API int seventy_ar_alloc(int world, int64_t slot_bytes, void **ptr, void *handle);
SEVENTY_API int seventy_ar_open(const void *handle, void **ptr);
SEVENTY_API int seventy_ar_context(int rank, int world, void *const *buffers, int64_t slot_bytes, void **ctx_out);
SEVENTY_API int seventy_ar_norm(void *ctx, void *out, void *residual, const void *x, const float *weight, int rows,
                                int dim, float eps, cudaStream_t stream);

/*
 * Two-shot all-reduce + residual add + RMS norm (csrc/allreduce.cu, ar_norm_push_all_kernel): rows <= 64 are split
 * into world contiguous shards; the home rank of each row reduces it and sends the sum to every rank, and every rank
 * adds its residual and norms. out and residual are complete on every rank (gather_residual is ignored). Without
 * weight (may be null), out is the plain sum. SEVENTY_AR=home selects the previous kernel, where the home rank also
 * norms and the residual is updated only there unless gather_residual is set (weight then required). Same context
 * and limits as seventy_ar_norm.
 */
SEVENTY_API int seventy_ar_norm_2shot(void *ctx_ptr, void *out, void *residual, const void *x, const float *weight,
                                      int rows, int dim, float eps, int gather_residual, cudaStream_t stream);
SEVENTY_API int seventy_ar_sum_f32(void *ctx, float *out, const float *x, int rows, int dim, cudaStream_t stream);

/*
 * FP16 weight GEMM for 1 to 64 rows (see csrc/dense_gemm.cu): out[m, n] = x[m, k] w[n, k]^T, FP32 accumulation,
 * FP32 output (out_fp32 = 1) or FP16. wpk: the weight repacked by seventy.ops.kernels.pack_dense.
 */
SEVENTY_API int64_t seventy_dense_gemm_workspace(int m, int n_padded, int splits);
SEVENTY_API int seventy_dense_gemm(void *out, int out_fp32, const void *x, const void *wpk, int m, int n, int n_padded,
                                   int k, int splits, int warps, void *workspace, int64_t workspace_bytes,
                                   const float *row_scale, cudaStream_t stream);

/*
 * The same GEMM with FP8 E4M3 weights (csrc/dense_gemm.cu, dense_gemm_fp8_kernel): wpk packed one byte per value
 * as [n_padded / 32, k / 64, 32 lanes, 64 bytes] (seventy.ops.kernels.DenseWeightFP8); col_scale FP32 [n_padded],
 * the per-column scale times 2^8. Same splits, warps, workspace and row_scale as seventy_dense_gemm.
 */
SEVENTY_API int seventy_dense_gemm_fp8(void *out, int out_fp32, const void *x, const void *wpk, const float *col_scale,
                                       int m, int n, int n_padded, int k, int splits, int warps, void *workspace,
                                       int64_t workspace_bytes, const float *row_scale, cudaStream_t stream);

/* DFlash2 drafter kernels (see csrc/drafter.cu; head size 128). */
SEVENTY_API int seventy_drafter_norm(void *out, float *residual_out, const void *x, int x_fp32,
                                     const float *residual_in, const float *weight, int rows, int dim, float eps,
                                     cudaStream_t stream);
SEVENTY_API int seventy_drafter_conv(void *out, int out_fp32, const void *x, int x_fp32, const float *base,
                                     const float *delta, int64_t delta_row_stride, int rows, int groups,
                                     int group_size, int block, cudaStream_t stream);
SEVENTY_API int seventy_drafter_qk(void *q_out, void *k_ring, void *v_ring, const float *q, int64_t q_row_stride,
                                   const float *kv, int64_t kv_row_stride, const float *q_norm, const float *k_norm,
                                   const int64_t *positions, const int64_t *slots, const float *inv_freq, int rows,
                                   int heads, int kv_heads, int head_dim, int ring, float eps, cudaStream_t stream);
SEVENTY_API int seventy_drafter_attention_splits(int block, int window);
SEVENTY_API int64_t seventy_drafter_attention_workspace(int seqs, int kv_heads, int splits);
SEVENTY_API int seventy_drafter_attention(void *out, const void *q, const void *k_ring, const void *v_ring,
                                          const int64_t *slots, const int64_t *starts, int seqs, int heads,
                                          int kv_heads, int head_dim, int block, int window, int ring, float scale,
                                          void *workspace, int64_t workspace_bytes, cudaStream_t stream);
SEVENTY_API int seventy_silu_mul_scale(void *out, float *scale, const float *gate_up, int rows, int inter,
                                       cudaStream_t stream);

/* silu(gate) * up for FP16 gate_up [rows, 2 * inter] (csrc/activation.cu). */
SEVENTY_API int seventy_silu_mul(void *out, const void *gate_up, int64_t rows, int inter, cudaStream_t stream);
