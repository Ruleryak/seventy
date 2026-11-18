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
 */
SEVENTY_API int seventy_gdn_recurrent(void *out, void *conv_state, float *state, const void *y, int ld,
                                      const float *conv_w, const float *a_log, const float *dt_bias,
                                      const float *norm_w, const int *query_start, const int *context_lens,
                                      const int *state_slots, int num_seqs, int hk, int hv, int dk, int dv, int taps,
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
 * caches (kv_type as in seventy_attn_prep), split over cache ranges of blocks_per_split blocks, then merged and
 * multiplied by sigmoid(gate) (the gate half of each query head in qkv). out: FP16 [T, hq * 256]. block_tables:
 * int32 [num_seqs, table_width].
 */
SEVENTY_API int seventy_decode_attention(void *out, const void *q, const void *k_cache, const void *v_cache,
                                         const float *k_scale, const float *v_scale, int kv_type, const void *qkv,
                                         int ld, const int *block_tables, int table_width,
                                         const int *query_start, const int *context_lens, int num_seqs, int tokens,
                                         int max_query_len, int block_size, int hq, int hkv, int head_dim, int splits,
                                         int blocks_per_split, float scale, void *workspace, int64_t workspace_bytes,
                                         cudaStream_t stream);

/*
 * Gated DeltaNet prefill, sequential part (see csrc/gdn_chunk.cu): one block per (sequence, value head) walks the
 * sequence's chunks of 64 tokens with the FP32 state, from inputs prepared per chunk by seventy.ops.gdn_prefill.
 * out: FP16 [T, hv, 128]. state: FP32 [slots, hv, 128, 128]. w, u, gq: FP16 [hv, total_chunks, 64, 128];
 * p: FP16 [hv, total_chunks, 64, 64]; klt: FP16 [hv, total_chunks, 128, 64]; eg: FP32 [hv, total_chunks].
 * chunk_start: int32 [num_seqs], the index of each sequence's first chunk.
 */
SEVENTY_API int seventy_gdn_chunk_scan(void *out, float *state, const void *w, const void *u, const void *gq,
                                       const void *p, const void *klt, const float *eg, const int *chunk_start,
                                       const int *query_start, const int *context_lens, const int *state_slots,
                                       int num_seqs, int hv, int64_t total_chunks, cudaStream_t stream);

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
