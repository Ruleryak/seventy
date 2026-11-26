#include "common.cuh"

#include <cfloat>
#include <cstdlib>
#include <cstring>

/*
 * Qwen Sparse Attention (Qwen4-Exp) for the attention layers of a step, all on the GPU (CUDA graphs).
 *
 * A sequence's keys go in blocks of `compress` (c) tokens. The indexer has `heads` query heads and one key head of
 * kIdx values. qsa_pool: for every new token that completes a block (position p with (p + 1) % c == 0), the
 * block's key = RMS norm (weight already 1 + w) of the mean of its c raw indexer keys (FP16), rounded to FP16,
 * rotated (multimodal rotary on the first rotary_dim values, cos and sin times rope_scale) at the block's first
 * position (cache positions), into
 * the pooled cache [blocks, block_size / c, kIdx] at that position's block and offset. Raw keys are not kept per
 * token: the step's own come from raw [tokens, kIdx], older ones from the sequence's ring (ring [slots, ring_len,
 * kIdx], position p at p % ring_len). After the pooling, the ring takes the step's last ring_len raw keys. With
 * ring_len >= c + the rows of a verify step, a block that a rejected draft completed is pooled again from the
 * right keys when its real last token comes.
 *
 * qsa_index_prep: from the indexer projection qk FP32 [tokens, (heads + 1) * kIdx], the normed, rotated queries (FP16
 * [tokens, heads, kIdx]) and the raw keys (FP16, into idx_cache at each token's slot, or at row t with kv_slots
 * NULL).
 *
 * seventy_qsa_select_scores: qsa_select from scores that the caller computed (row lt of scores for query row row0 + lt,
 * block j at column j; only the row's visible complete blocks are read).
 *
 * qsa_score, qsa_select and qsa_attention take the rows row0 .. row0 + tokens - 1 of the step (their q, scores,
 * sel and outputs hold only those).
 * qsa_score: for every query row whose sequence has more than `top` complete blocks visible, the score of each
 * visible complete block j: sum over heads of relu(q_h . key_j) / sqrt(kIdx), into scores [rows, ld].
 * qsa_select: the positions [rows, width = top * c + c - 1] (-1: none) that a query attends to: all its visible
 * positions while they have no more than `top` complete blocks; else the positions of its `top` best blocks (ties:
 * the lower block) and of its unfinished last block.
 * qsa_attention: attention of each query row over its selected positions (one block per split, row and KV head; the
 * group's query heads, at most 32, share the row's positions; split s takes the entries s * ceil(width / splits)
 * onward of the row's sel), with partial outputs for attention_combine in csrc/attention.cu
 * (seventy_decode_attention's workspace layout with one token per row). With a split KV cache (kv_split ranks share
 * the KV head, seventy_attn_prep_split), the rank takes the entries e with e % kv_split == kv_part (sel entry e
 * holds a position p with p % compress == e % compress, so with compress % kv_split == 0 these are the positions of
 * its tokens) and reads them at row slot / kv_split; the parts of the ranks then merge (seventy_attention_partial,
 * seventy_attention_merge, or seventy_ar_attn_merge over peer-to-peer memory).
 */

namespace seventy {
namespace {

constexpr int kIdx = 128;
constexpr int kHeadD = 256;

__device__ __forceinline__ int seq_of(const int *query_start, int num_seqs, int t) {
  int b = 0;
  while (b + 1 < num_seqs && query_start[b + 1] <= t) ++b;
  return b;
}

__global__ void __launch_bounds__(kIdx) qsa_pool_kernel(half *pooled, const half *raw, const half *ring, int ring_len,
                                                        const int *state_slots, const int *positions,
                                                        const int *block_tables, int table_width,
                                                        const int *query_start, const int *context_lens,
                                                        int num_seqs, const float *k_norm, const float *inv_freq,
                                                        float rope_scale, int s1, int s2, int rotary_dim, int c,
                                                        int block_size, float eps) {
  __shared__ float scratch[32];
  __shared__ float vals[kIdx];
  const int t = blockIdx.x, d = threadIdx.x;
  const int b = seq_of(query_start, num_seqs, t);
  const int ctx = context_lens[b];
  const int p = ctx + (t - query_start[b]);
  if ((p + 1) % c) return;
  const int j0 = p + 1 - c;
  const int *table = block_tables + static_cast<int64_t>(b) * table_width;
  const half *mine = ring + static_cast<int64_t>(state_slots[b]) * ring_len * kIdx;
  float sum = 0.0f;
  for (int i = 0; i < c; ++i) {
    const int pos = j0 + i;
    const half key = pos >= ctx ? raw[static_cast<int64_t>(query_start[b] + pos - ctx) * kIdx + d]
                                : mine[static_cast<int64_t>(pos % ring_len) * kIdx + d];
    sum += __half2float(key);
  }
  const float mean = __half2float(__float2half_rn(sum / c));
  const float ss = block_sum(mean * mean, scratch);
  vals[d] = __half2float(__float2half_rn(mean * rsqrtf(ss / kIdx + eps) * k_norm[d]));
  __syncthreads();
  const int half_rot = rotary_dim / 2;
  const int64_t slot0 = static_cast<int64_t>(table[j0 / block_size]) * block_size + j0 % block_size;
  float r = vals[d];
  if (d < rotary_dim) {
    const int i = d < half_rot ? d : d - half_rot;
    int axis = 0;
    if (i % 3 == 1 && i < 3 * s1) axis = 1;
    if (i % 3 == 2 && i < 3 * s2) axis = 2;
    const float pos = static_cast<float>(positions[slot0 * 3 + axis]);
    float sn, cs;
    sincosf(pos * inv_freq[i], &sn, &cs);
    sn *= rope_scale;
    cs *= rope_scale;
    r = d < half_rot ? vals[d] * cs - vals[d + half_rot] * sn : vals[d] * cs + vals[d - half_rot] * sn;
  }
  const int per_block = block_size / c;
  const int64_t out = (static_cast<int64_t>(table[j0 / block_size]) * per_block + (j0 % block_size) / c) * kIdx;
  pooled[out + d] = __float2half_rn(r);
}

__global__ void __launch_bounds__(kIdx) qsa_ring_kernel(half *ring, int ring_len, const half *raw,
                                                        const int *state_slots, const int *query_start,
                                                        const int *context_lens) {
  const int b = blockIdx.y, d = threadIdx.x;
  const int q0 = query_start[b], len = query_start[b + 1] - q0;
  if (len <= 0) return;
  const int ctx = context_lens[b];
  half *mine = ring + static_cast<int64_t>(state_slots[b]) * ring_len * kIdx;
  for (int i = max(0, len - ring_len) + blockIdx.x; i < len; i += gridDim.x) {
    const int pos = ctx + i;
    mine[static_cast<int64_t>(pos % ring_len) * kIdx + d] = raw[static_cast<int64_t>(q0 + i) * kIdx + d];
  }
}

__global__ void __launch_bounds__(128) qsa_score_kernel(float *scores, int ld, const half *q, int heads,
                                                        const half *pooled, const int *block_tables, int table_width,
                                                        const int *query_start, const int *context_lens,
                                                        int num_seqs, int c, int top, int block_size, int row0) {
  __shared__ float qs[8][kIdx];
  const int lt = blockIdx.x, t = row0 + lt;
  const int b = seq_of(query_start, num_seqs, t);
  const int visible = context_lens[b] + (t - query_start[b]) + 1;
  const int nb = visible / c;
  if (nb <= top) return;
  for (int e = threadIdx.x; e < heads * kIdx; e += blockDim.x)
    qs[e / kIdx][e % kIdx] = __half2float(q[static_cast<int64_t>(lt) * heads * kIdx + e]);
  __syncthreads();
  const int j = blockIdx.y * blockDim.x + threadIdx.x;
  if (j >= nb) return;
  const int pos = j * c;
  const int per_block = block_size / c;
  const int *table = block_tables + static_cast<int64_t>(b) * table_width;
  const half *key = pooled + (static_cast<int64_t>(table[pos / block_size]) * per_block + (pos % block_size) / c) * kIdx;
  float kf[kIdx];
#pragma unroll
  for (int x = 0; x < kIdx; x += 8) {
    const uint4 raw = *reinterpret_cast<const uint4 *>(key + x);
    const half2 *h = reinterpret_cast<const half2 *>(&raw);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float2 f = __half22float2(h[i]);
      kf[x + 2 * i] = f.x;
      kf[x + 2 * i + 1] = f.y;
    }
  }
  float total = 0.0f;
  for (int h = 0; h < heads; ++h) {
    float dot = 0.0f;
#pragma unroll
    for (int x = 0; x < kIdx; ++x) dot = fmaf(qs[h][x], kf[x], dot);
    total += fmaxf(dot, 0.0f);
  }
  scores[static_cast<int64_t>(lt) * ld + j] = total * rsqrtf(static_cast<float>(kIdx));
}

__device__ __forceinline__ uint32_t order_key(float v) {
  const uint32_t u = __float_as_uint(v);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__global__ void __launch_bounds__(1024) qsa_select_kernel(int *sel, int width, const float *scores, int ld,
                                                         const int *query_start, const int *context_lens,
                                                         int num_seqs, int c, int top, int row0) {
  __shared__ int hist[256];
  __shared__ uint32_t prefix_s;
  __shared__ int need_s;
  __shared__ int warp_gt[32], warp_eq[32];
  const int lt = blockIdx.x, t = row0 + lt, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int b = seq_of(query_start, num_seqs, t);
  const int visible = context_lens[b] + (t - query_start[b]) + 1;
  const int nb = visible / c;
  int *row = sel + static_cast<int64_t>(lt) * width;
  if (nb <= top) {
    for (int i = tid; i < width; i += blockDim.x) row[i] = i < visible ? i : -1;
    return;
  }
  const float *sc = scores + static_cast<int64_t>(lt) * ld;
  uint32_t prefix = 0;
  int need = top;
  for (int shift = 24; shift >= 0; shift -= 8) {
    for (int i = tid; i < 256; i += blockDim.x) hist[i] = 0;
    __syncthreads();
    const uint32_t mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
    for (int j = tid; j < nb; j += blockDim.x) {
      const uint32_t k = order_key(sc[j]);
      if ((k & mask) == (prefix & mask)) atomicAdd(&hist[(k >> shift) & 0xff], 1);
    }
    __syncthreads();
    if (tid == 0) {
      int acc = 0, digit = 255;
      for (; digit > 0; --digit) {
        if (acc + hist[digit] >= need) break;
        acc += hist[digit];
      }
      prefix_s = prefix | (static_cast<uint32_t>(digit) << shift);
      need_s = need - acc;
    }
    __syncthreads();
    prefix = prefix_s;
    need = need_s;
    __syncthreads();
  }
  const uint32_t thr = prefix;
  const int per = (nb + blockDim.x - 1) / blockDim.x;
  const int j0 = tid * per, j1 = min(nb, j0 + per);
  int gt = 0, eq = 0;
  for (int j = j0; j < j1; ++j) {
    const uint32_t k = order_key(sc[j]);
    gt += k > thr;
    eq += k == thr;
  }
  int sg = gt, se = eq;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int a = __shfl_up_sync(0xffffffffu, sg, o), e = __shfl_up_sync(0xffffffffu, se, o);
    if (lane >= o) {
      sg += a;
      se += e;
    }
  }
  if (lane == 31) {
    warp_gt[warp] = sg;
    warp_eq[warp] = se;
  }
  __syncthreads();
  int wg = 0, we = 0;
  for (int w = 0; w < warp; ++w) {
    wg += warp_gt[w];
    we += warp_eq[w];
  }
  int total_gt = 0;
  for (int w = 0; w < (blockDim.x + 31) / 32; ++w) total_gt += warp_gt[w];
  int pos_gt = wg + sg - gt, pos_eq = we + se - eq;
  const int take_eq = top - total_gt;
  for (int j = j0; j < j1; ++j) {
    const uint32_t k = order_key(sc[j]);
    int slot = -1;
    if (k > thr) {
      slot = pos_gt++;
    } else if (k == thr) {
      if (pos_eq < take_eq) slot = total_gt + pos_eq;
      ++pos_eq;
    }
    if (slot >= 0)
      for (int i = 0; i < c; ++i) row[slot * c + i] = j * c + i;
  }
  for (int i = tid; i < c - 1; i += blockDim.x) {
    const int pos = nb * c + i;
    row[top * c + i] = pos < visible ? pos : -1;
  }
}

/*
 * qsa_index_prep_kernel: one block of kIdx threads per (token, head); head `heads` is the raw key. Query heads: RMS
 * norm in FP32 times q_norm (already 1 + w), rotation of the first 2 * half_rot values (value j with value
 * j + half_rot), FP16 out. The raw key: FP16 into the indexer cache at the token's slot.
 */
__global__ void __launch_bounds__(kIdx) qsa_index_prep_kernel(half *q_out, half *idx_cache, const float *qk,
                                                              int64_t qk_ld, const float *q_norm,
                                                              const int *kv_slots, const float *cos,
                                                              const float *sin, int heads, int half_rot, float eps) {
  __shared__ float scratch[32];
  __shared__ float vals[kIdx];
  const int t = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
  const float v = qk[static_cast<int64_t>(t) * qk_ld + h * kIdx + d];
  if (h == heads) {
    idx_cache[static_cast<int64_t>(kv_slots != nullptr ? kv_slots[t] : t) * kIdx + d] = __float2half_rn(v);
    return;
  }
  const float ss = block_sum(v * v, scratch);
  const float n = v * rsqrtf(ss / kIdx + eps) * q_norm[d];
  vals[d] = n;
  __syncthreads();
  float o = n;
  if (d < 2 * half_rot) {
    const int j = d < half_rot ? d : d - half_rot;
    const float c = cos[static_cast<int64_t>(t) * half_rot + j], s = sin[static_cast<int64_t>(t) * half_rot + j];
    o = d < half_rot ? n * c - vals[d + half_rot] * s : n * c + vals[j] * s;
  }
  q_out[(static_cast<int64_t>(t) * heads + h) * kIdx + d] = __float2half_rn(o);
}

/*
 * qsa_attention_kernel: one block of kAttnThreads threads per (split, query row, KV head). A split takes a fixed
 * range of the row's sel entries (entries < 0 are skipped), in chunks of kChunk positions: the cache slots of the
 * chunk's positions are looked up once, each thread loads the next chunk's keys, values and scales into registers
 * while the block computes the current chunk, and the keys and values (dequantized to FP16) go to shared memory with a row stride of kHeadD + 2 halves, so that the 32 lanes of a
 * warp, one position each, read different banks. Scores: lane = position, warp w takes the rows w, w + 4, ...;
 * online softmax per row with warp shuffles; the outputs: each thread keeps 2 of the 256 values of every row.
 */
constexpr int kAttnThreads = 128;
constexpr int kKStride = kHeadD + 2;

template <typename T>
struct ChunkLoad;

template <>
struct ChunkLoad<int8_t> {
  static constexpr int kParts = kHeadD / 16;

  template <int kChunk>
  struct Regs {
    static constexpr int kIter = kChunk * kParts / kAttnThreads;
    uint4 kr[kIter], vr[kIter];
    float ks[kIter], vs[kIter];
  };

  template <int kChunk>
  static __device__ __forceinline__ void fetch(Regs<kChunk> &r, const int8_t *k_cache, const int8_t *v_cache,
                                               const float *k_scale, const float *v_scale, const int64_t *slots,
                                               int hkv, int kh) {
#pragma unroll
    for (int i = 0; i < Regs<kChunk>::kIter; ++i) {
      const int e = threadIdx.x + i * kAttnThreads;
      const int64_t slot = slots[e / kParts];
      r.kr[i] = r.vr[i] = make_uint4(0, 0, 0, 0);
      r.ks[i] = r.vs[i] = 0.0f;
      if (slot >= 0) {
        const int64_t vec = slot * hkv + kh;
        r.kr[i] = *reinterpret_cast<const uint4 *>(k_cache + vec * kHeadD + (e % kParts) * 16);
        r.vr[i] = *reinterpret_cast<const uint4 *>(v_cache + vec * kHeadD + (e % kParts) * 16);
        r.ks[i] = k_scale[vec];
        r.vs[i] = v_scale[vec];
      }
    }
  }

  template <int kChunk>
  static __device__ __forceinline__ void store(const Regs<kChunk> &r, half (*kd)[kKStride], half (*vd)[kKStride]) {
#pragma unroll
    for (int i = 0; i < Regs<kChunk>::kIter; ++i) {
      const int e = threadIdx.x + i * kAttnThreads;
      store16(&kd[e / kParts][(e % kParts) * 16], r.kr[i], r.ks[i]);
      store16(&vd[e / kParts][(e % kParts) * 16], r.vr[i], r.vs[i]);
    }
  }

  static __device__ __forceinline__ void store16(half *dst, const uint4 raw, float sc) {
    half2 *row = reinterpret_cast<half2 *>(dst);
    const uint32_t w[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      row[2 * i] = __floats2half2_rn(static_cast<float>(static_cast<int8_t>(w[i] & 0xff)) * sc,
                                     static_cast<float>(static_cast<int8_t>((w[i] >> 8) & 0xff)) * sc);
      row[2 * i + 1] = __floats2half2_rn(static_cast<float>(static_cast<int8_t>((w[i] >> 16) & 0xff)) * sc,
                                         static_cast<float>(static_cast<int8_t>(w[i] >> 24)) * sc);
    }
  }
};

template <>
struct ChunkLoad<half> {
  static constexpr int kParts = kHeadD / 8;

  template <int kChunk>
  struct Regs {
    static constexpr int kIter = kChunk * kParts / kAttnThreads;
    uint4 kr[kIter], vr[kIter];
  };

  template <int kChunk>
  static __device__ __forceinline__ void fetch(Regs<kChunk> &r, const half *k_cache, const half *v_cache,
                                               const float *, const float *, const int64_t *slots, int hkv,
                                               int kh) {
#pragma unroll
    for (int i = 0; i < Regs<kChunk>::kIter; ++i) {
      const int e = threadIdx.x + i * kAttnThreads;
      const int64_t slot = slots[e / kParts];
      r.kr[i] = r.vr[i] = make_uint4(0, 0, 0, 0);
      if (slot >= 0) {
        const int64_t at = (slot * hkv + kh) * kHeadD + (e % kParts) * 8;
        r.kr[i] = *reinterpret_cast<const uint4 *>(k_cache + at);
        r.vr[i] = *reinterpret_cast<const uint4 *>(v_cache + at);
      }
    }
  }

  template <int kChunk>
  static __device__ __forceinline__ void store(const Regs<kChunk> &r, half (*kd)[kKStride], half (*vd)[kKStride]) {
#pragma unroll
    for (int i = 0; i < Regs<kChunk>::kIter; ++i) {
      const int e = threadIdx.x + i * kAttnThreads;
      half2 *kw = reinterpret_cast<half2 *>(&kd[e / kParts][(e % kParts) * 8]);
      half2 *vw = reinterpret_cast<half2 *>(&vd[e / kParts][(e % kParts) * 8]);
      const half2 *k2 = reinterpret_cast<const half2 *>(&r.kr[i]);
      const half2 *v2 = reinterpret_cast<const half2 *>(&r.vr[i]);
#pragma unroll
      for (int x = 0; x < 4; ++x) {
        kw[x] = k2[x];
        vw[x] = v2[x];
      }
    }
  }
};

template <typename T, int kRows, int kChunk>
__global__ void __launch_bounds__(kAttnThreads) qsa_attention_kernel(
    float *partial_o, float *partial_ml, const half *q, const T *k_cache, const T *v_cache, const float *k_scale,
    const float *v_scale, const int *sel, int width, const int *block_tables, int table_width,
    const int *query_start, int num_seqs, int block_size, int hq, int hkv, int splits, float scale, int row0,
    int kv_split, int kv_part) {
  __shared__ __align__(16) half qs[kRows][kHeadD];
  __shared__ half ks[kChunk][kKStride];
  __shared__ half vs[kChunk][kKStride];
  __shared__ float p[kRows][kChunk + 1];
  __shared__ float row_m[kRows], row_l[kRows], row_scale[kRows];
  __shared__ int pos_list[2][kChunk];
  __shared__ int64_t slot_list[2][kChunk];
  const int split = blockIdx.x, t = blockIdx.y, kh = blockIdx.z, tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  const int rows = hq / hkv;
  const int b = seq_of(query_start, num_seqs, row0 + t);
  const int *row_sel = sel + static_cast<int64_t>(t) * width + kv_part;
  const int entries = (width - kv_part + kv_split - 1) / kv_split;
  const int per = (entries + splits - 1) / splits;
  const int first = split * per, last = min(entries, first + per);
  auto pidx = [&](int r) { return (static_cast<int64_t>(t) * hq + kh * rows + r) * splits + split; };
  for (int e = tid; e < rows * (kHeadD / 8); e += kAttnThreads) {
    const int r = e / (kHeadD / 8), part = e % (kHeadD / 8);
    *reinterpret_cast<uint4 *>(&qs[r][part * 8]) = *reinterpret_cast<const uint4 *>(
        q + (static_cast<int64_t>(t) * hq + kh * rows + r) * kHeadD + part * 8);
  }
  if (tid < kRows) {
    row_m[tid] = -FLT_MAX;
    row_l[tid] = 0.0f;
  }
  float acc[kRows][2];
#pragma unroll
  for (int r = 0; r < kRows; ++r) acc[r][0] = acc[r][1] = 0.0f;
  const int *table = block_tables + static_cast<int64_t>(b) * table_width;
  auto prepare = [&](int c, int buf) {
    if (tid < kChunk) {
      const int pos = c + tid < last ? row_sel[(c + tid) * kv_split] : -1;
      pos_list[buf][tid] = pos;
      slot_list[buf][tid] =
          pos < 0 ? -1
                  : (static_cast<int64_t>(table[pos / block_size]) * block_size + pos % block_size) / kv_split;
    }
  };
  typename ChunkLoad<T>::template Regs<kChunk> regs;
  prepare(first, 0);
  __syncthreads();
  if (first < last) ChunkLoad<T>::template fetch<kChunk>(regs, k_cache, v_cache, k_scale, v_scale, slot_list[0], hkv, kh);
  int buf = 0;
  for (int c0 = first; c0 < last; c0 += kChunk, buf ^= 1) {
    const int n = min(kChunk, last - c0);
    ChunkLoad<T>::template store<kChunk>(regs, ks, vs);
    prepare(c0 + kChunk, buf ^ 1);
    __syncthreads();
    if (c0 + kChunk < last)
      ChunkLoad<T>::template fetch<kChunk>(regs, k_cache, v_cache, k_scale, v_scale, slot_list[buf ^ 1], hkv, kh);
    const int *pos_now = pos_list[buf];
    for (int j = lane; j < kChunk; j += 32) {
      float dot[kRows / 4 > 0 ? kRows / 4 : 1];
#pragma unroll
      for (int i = 0; i < (kRows / 4 > 0 ? kRows / 4 : 1); ++i) dot[i] = 0.0f;
      const half2 *kr = reinterpret_cast<const half2 *>(&ks[j][0]);
#pragma unroll 4
      for (int x = 0; x < kHeadD / 2; ++x) {
        const float2 k2 = __half22float2(kr[x]);
#pragma unroll
        for (int i = 0; i < (kRows / 4 > 0 ? kRows / 4 : 1); ++i) {
          const int r = warp + 4 * i;
          if (r < kRows) {
            const float2 a = __half22float2(reinterpret_cast<const half2 *>(&qs[r][0])[x]);
            dot[i] = fmaf(a.x, k2.x, fmaf(a.y, k2.y, dot[i]));
          }
        }
      }
      const bool ok = pos_now[j] >= 0;
#pragma unroll
      for (int i = 0; i < (kRows / 4 > 0 ? kRows / 4 : 1); ++i) {
        const int r = warp + 4 * i;
        if (r < kRows) p[r][j] = ok ? dot[i] * scale : -FLT_MAX;
      }
    }
    __syncthreads();
    for (int r = warp; r < rows; r += 4) {
      float m = -FLT_MAX;
      for (int j = lane; j < kChunk; j += 32) m = fmaxf(m, p[r][j]);
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
      const float m_old = row_m[r];
      const float m_new = fmaxf(m_old, m);
      float l = 0.0f;
      for (int j = lane; j < kChunk; j += 32) {
        const float s = p[r][j];
        const float e = s == -FLT_MAX ? 0.0f : expf(s - m_new);
        p[r][j] = e;
        l += e;
      }
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) l += __shfl_xor_sync(0xffffffffu, l, o);
      if (lane == 0) {
        const float corr = m_old == -FLT_MAX ? 0.0f : expf(m_old - m_new);
        row_scale[r] = corr;
        row_l[r] = row_l[r] * corr + l;
        row_m[r] = m_new;
      }
    }
    __syncthreads();
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
      if (r < rows) {
        const float c = row_scale[r];
        float a0 = acc[r][0] * c, a1 = acc[r][1] * c;
        for (int j = 0; j < n; ++j) {
          const float2 v2 = __half22float2(reinterpret_cast<const half2 *>(&vs[j][0])[tid]);
          const float pj = p[r][j];
          a0 = fmaf(pj, v2.x, a0);
          a1 = fmaf(pj, v2.y, a1);
        }
        acc[r][0] = a0;
        acc[r][1] = a1;
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int r = 0; r < kRows; ++r) {
    if (r < rows) {
      float2 out;
      out.x = acc[r][0];
      out.y = acc[r][1];
      reinterpret_cast<float2 *>(partial_o + pidx(r) * kHeadD)[tid] = out;
    }
  }
  if (tid < rows) {
    partial_ml[pidx(tid) * 2] = row_m[tid];
    partial_ml[pidx(tid) * 2 + 1] = row_l[tid];
  }
}

/*
 * qsa_attention_mma_kernel: qsa_attention_kernel on the tensor cores, for INT8 caches and up to 16 query heads per KV
 * head (kMT m-tiles of 8). One block of 4 warps per (split, query row, KV head), chunks of 32 positions.
 * - The chunk's keys stay INT8 in shared memory (rows of 260 bytes) and its values INT8 transposed (vt[dim][pos],
 *   rows of 36 bytes); the key scale multiplies the scores after the products and the value scale the
 *   probabilities before them. INT8 values are exact in FP16.
 * - Scores: mma.m8n8k4 (fragment layout in csrc/nvfp4_gemm.cu), quad-pair p and index i take position 4 i + p, so
 *   the lanes of a half-warp read consecutive positions (no bank conflicts); warp w adds dims 64 w .. 64 w + 63 and
 *   the four partial scores meet in shared memory. Then the online softmax per row (one warp per row).
 * - Output: warp w keeps dims 64 w .. 64 w + 63 (two tiles of 32; quad-pair p, index i: dim 4 i + p of a tile) of
 *   every row in its accumulators, scaled by each row's correction per chunk; the probabilities go to the tensor
 *   cores in FP16.
 * - Each thread loads the next chunk's keys, values and scales into registers while the block computes the current
 *   one. The values load one position per lane, so that their transposed stores do not conflict.
 */
constexpr int kMmaChunk = 32;
constexpr int kMmaKStride = 260;
constexpr int kMmaVStride = 36;
constexpr int kMmaQStride = 260;
constexpr int kMmaPStride = 36;

__device__ __forceinline__ void i8x4_to_half2x2(uint32_t w, uint32_t &lo, uint32_t &hi) {
  const uint32_t u = w ^ 0x80808080u;
  uint32_t a = __byte_perm(u, 0x64646464u, 0x4140);
  uint32_t b = __byte_perm(u, 0x64646464u, 0x4342);
  const __half2 bias = __float2half2_rn(1152.0f);
  __half2 ha = __hsub2(*reinterpret_cast<__half2 *>(&a), bias);
  __half2 hb = __hsub2(*reinterpret_cast<__half2 *>(&b), bias);
  lo = *reinterpret_cast<uint32_t *>(&ha);
  hi = *reinterpret_cast<uint32_t *>(&hb);
}

template <int kMT>
__global__ void __launch_bounds__(kAttnThreads) qsa_attention_mma_kernel(
    float *partial_o, float *partial_ml, const half *q, const int8_t *k_cache, const int8_t *v_cache,
    const float *k_scale, const float *v_scale, const int *sel, int width, const int *block_tables, int table_width,
    const int *query_start, int num_seqs, int block_size, int hq, int hkv, int splits, float scale, int row0,
    int kv_split, int kv_part) {
  constexpr int kR = kMT * 8;
  __shared__ __align__(16) half qs[kR][kMmaQStride];
  __shared__ __align__(16) int8_t ks[kMmaChunk][kMmaKStride];
  __shared__ __align__(16) int8_t vt[kHeadD][kMmaVStride];
  __shared__ float spart[4][kR][kMmaChunk + 1];
  __shared__ __align__(16) half p16[kR][kMmaPStride];
  __shared__ float kscl[kMmaChunk], vscl[kMmaChunk];
  __shared__ float row_m[kR], row_l[kR], row_scale[kR];
  __shared__ int pos_list[2][kMmaChunk];
  __shared__ int64_t slot_list[2][kMmaChunk];
  const int split = blockIdx.x, t = blockIdx.y, kh = blockIdx.z, tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  const int qp = (lane >> 2) & 3, idx = (lane & 3) + ((lane >> 4) << 2);
  const int rows = hq / hkv;
  const int b = seq_of(query_start, num_seqs, row0 + t);
  const int *row_sel = sel + static_cast<int64_t>(t) * width + kv_part;
  const int entries = (width - kv_part + kv_split - 1) / kv_split;
  const int per = (entries + splits - 1) / splits;
  const int first = split * per, last = min(entries, first + per);
  auto pidx = [&](int r) { return (static_cast<int64_t>(t) * hq + kh * rows + r) * splits + split; };
  for (int e = tid; e < kR * (kHeadD / 8); e += kAttnThreads) {
    const int r = e / (kHeadD / 8), part = e % (kHeadD / 8);
    uint4 v = make_uint4(0, 0, 0, 0);
    if (r < rows)
      v = *reinterpret_cast<const uint4 *>(q + (static_cast<int64_t>(t) * hq + kh * rows + r) * kHeadD + part * 8);
    uint32_t *dst = reinterpret_cast<uint32_t *>(&qs[r][part * 8]);
    dst[0] = v.x;
    dst[1] = v.y;
    dst[2] = v.z;
    dst[3] = v.w;
  }
  if (tid < kR) {
    row_m[tid] = -FLT_MAX;
    row_l[tid] = 0.0f;
  }
  const int *table = block_tables + static_cast<int64_t>(b) * table_width;
  auto prepare = [&](int c, int buf) {
    if (tid < kMmaChunk) {
      const int pos = c + tid < last ? row_sel[(c + tid) * kv_split] : -1;
      pos_list[buf][tid] = pos;
      slot_list[buf][tid] =
          pos < 0 ? -1
                  : (static_cast<int64_t>(table[pos / block_size]) * block_size + pos % block_size) / kv_split;
    }
  };
  constexpr int kIter = kMmaChunk * 16 / kAttnThreads;
  uint4 kr[kIter], vr[kIter];
  float ksr = 0.0f, vsr = 0.0f;
  auto fetch = [&](int buf) {
#pragma unroll
    for (int i = 0; i < kIter; ++i) {
      const int e = tid + i * kAttnThreads;
      const int64_t ks_slot = slot_list[buf][e / 16];
      kr[i] = ks_slot >= 0 ? *reinterpret_cast<const uint4 *>(k_cache + (ks_slot * hkv + kh) * kHeadD + (e % 16) * 16)
                           : make_uint4(0, 0, 0, 0);
      const int64_t vs_slot = slot_list[buf][e % kMmaChunk];
      vr[i] = vs_slot >= 0
                  ? *reinterpret_cast<const uint4 *>(v_cache + (vs_slot * hkv + kh) * kHeadD + (e / kMmaChunk) * 16)
                  : make_uint4(0, 0, 0, 0);
    }
    if (tid < kMmaChunk) {
      const int64_t slot = slot_list[buf][tid];
      ksr = slot >= 0 ? k_scale[slot * hkv + kh] : 0.0f;
      vsr = slot >= 0 ? v_scale[slot * hkv + kh] : 0.0f;
    }
  };
  float acc[kMT][2][8];
#pragma unroll
  for (int m = 0; m < kMT; ++m)
#pragma unroll
    for (int nt = 0; nt < 2; ++nt)
#pragma unroll
      for (int j = 0; j < 8; ++j) acc[m][nt][j] = 0.0f;
  prepare(first, 0);
  __syncthreads();
  if (first < last) fetch(0);
  int buf = 0;
  for (int c0 = first; c0 < last; c0 += kMmaChunk, buf ^= 1) {
#pragma unroll
    for (int i = 0; i < kIter; ++i) {
      const int e = tid + i * kAttnThreads;
      uint32_t *kd = reinterpret_cast<uint32_t *>(&ks[e / 16][(e % 16) * 16]);
      kd[0] = kr[i].x;
      kd[1] = kr[i].y;
      kd[2] = kr[i].z;
      kd[3] = kr[i].w;
      const int pos = e % kMmaChunk, d0 = (e / kMmaChunk) * 16;
      const uint32_t w[4] = {vr[i].x, vr[i].y, vr[i].z, vr[i].w};
#pragma unroll
      for (int q4 = 0; q4 < 16; ++q4) vt[d0 + q4][pos] = static_cast<int8_t>((w[q4 >> 2] >> (8 * (q4 & 3))) & 0xff);
    }
    if (tid < kMmaChunk) {
      kscl[tid] = ksr;
      vscl[tid] = vsr;
    }
    prepare(c0 + kMmaChunk, buf ^ 1);
    __syncthreads();
    if (c0 + kMmaChunk < last) fetch(buf ^ 1);
    {
      float sc[kMT][8];
#pragma unroll
      for (int m = 0; m < kMT; ++m)
#pragma unroll
        for (int j = 0; j < 8; ++j) sc[m][j] = 0.0f;
      const int pos = 4 * idx + qp;
#pragma unroll 4
      for (int s4 = 0; s4 < 16; ++s4) {
        const int d0 = warp * 64 + s4 * 4;
        uint32_t b0, b1;
        i8x4_to_half2x2(*reinterpret_cast<const uint32_t *>(&ks[pos][d0]), b0, b1);
#pragma unroll
        for (int m = 0; m < kMT; ++m) {
          const uint2 a = *reinterpret_cast<const uint2 *>(&qs[m * 8 + idx][d0]);
          mma_m8n8k4(sc[m], a.x, a.y, b0, b1);
        }
      }
#pragma unroll
      for (int m = 0; m < kMT; ++m)
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const int r = m * 8 + (idx & 1) + 2 * ((j >> 1) & 1) + 4 * (idx >> 2);
          const int col = (j & 1) + 2 * ((idx >> 1) & 1) + 4 * (j >> 2);
          spart[warp][r][4 * col + qp] = sc[m][j];
        }
    }
    __syncthreads();
    const int *pos_now = pos_list[buf];
    for (int r = warp; r < rows; r += 4) {
      const float raw = spart[0][r][lane] + spart[1][r][lane] + spart[2][r][lane] + spart[3][r][lane];
      const bool ok = pos_now[lane] >= 0;
      const float sv = ok ? raw * kscl[lane] * scale : -FLT_MAX;
      float m = sv;
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
      const float m_old = row_m[r];
      const float m_new = fmaxf(m_old, m);
      const float e = ok ? expf(sv - m_new) : 0.0f;
      float l = e;
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) l += __shfl_xor_sync(0xffffffffu, l, o);
      p16[r][lane] = __float2half_rn(e * vscl[lane]);
      if (lane == 0) {
        const float corr = m_old == -FLT_MAX ? 0.0f : expf(m_old - m_new);
        row_scale[r] = corr;
        row_l[r] = row_l[r] * corr + l;
        row_m[r] = m_new;
      }
    }
    for (int r = rows + warp; r < kR; r += 4) {
      p16[r][lane] = __float2half_rn(0.0f);
      if (lane == 0) row_scale[r] = 0.0f;
    }
    __syncthreads();
#pragma unroll
    for (int m = 0; m < kMT; ++m)
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float cr = row_scale[m * 8 + (idx & 1) + 2 * ((j >> 1) & 1) + 4 * (idx >> 2)];
        acc[m][0][j] *= cr;
        acc[m][1][j] *= cr;
      }
#pragma unroll
    for (int s4 = 0; s4 < kMmaChunk / 4; ++s4) {
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        const int dim = warp * 64 + nt * 32 + 4 * idx + qp;
        uint32_t b0, b1;
        i8x4_to_half2x2(*reinterpret_cast<const uint32_t *>(&vt[dim][s4 * 4]), b0, b1);
#pragma unroll
        for (int m = 0; m < kMT; ++m) {
          const uint2 a = *reinterpret_cast<const uint2 *>(&p16[m * 8 + idx][s4 * 4]);
          mma_m8n8k4(acc[m][nt], a.x, a.y, b0, b1);
        }
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int m = 0; m < kMT; ++m)
#pragma unroll
    for (int nt = 0; nt < 2; ++nt)
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int r = m * 8 + (idx & 1) + 2 * ((j >> 1) & 1) + 4 * (idx >> 2);
        const int col = (j & 1) + 2 * ((idx >> 1) & 1) + 4 * (j >> 2);
        if (r < rows) partial_o[pidx(r) * kHeadD + warp * 64 + nt * 32 + 4 * col + qp] = acc[m][nt][j];
      }
  if (tid < rows) {
    partial_ml[pidx(tid) * 2] = row_m[tid];
    partial_ml[pidx(tid) * 2 + 1] = row_l[tid];
  }
}

}  // namespace
}  // namespace seventy

SEVENTY_API int seventy_qsa_pool(void *pooled, const void *raw, void *ring, int ring_len, const int *state_slots,
                                 const int *positions, const int *block_tables, int table_width,
                                 const int *query_start, const int *context_lens, int num_seqs, int tokens,
                                 const float *k_norm, const float *inv_freq, float rope_scale, int s1, int s2,
                                 int rotary_dim, int compress, int block_size, float eps, cudaStream_t stream) {
  using namespace seventy;
  if (pooled == nullptr || raw == nullptr || ring == nullptr || positions == nullptr || state_slots == nullptr ||
      tokens < 0 || num_seqs < 1 || compress < 1 || block_size % compress || rotary_dim > kIdx ||
      ring_len < compress)
    return SEVENTY_ERR_INVALID_ARGUMENT;
  if (tokens == 0) return SEVENTY_OK;
  qsa_pool_kernel<<<tokens, kIdx, 0, stream>>>(static_cast<half *>(pooled), static_cast<const half *>(raw),
                                               static_cast<const half *>(ring), ring_len, state_slots, positions,
                                               block_tables, table_width, query_start, context_lens, num_seqs,
                                               k_norm, inv_freq, rope_scale, s1, s2, rotary_dim, compress, block_size,
                                               eps);
  qsa_ring_kernel<<<dim3(8, num_seqs), kIdx, 0, stream>>>(static_cast<half *>(ring), ring_len,
                                                          static_cast<const half *>(raw), state_slots, query_start,
                                                          context_lens);
  return launch_status();
}

SEVENTY_API int seventy_qsa_index_prep(void *q_out, void *idx_cache, const float *qk, int64_t qk_ld,
                                       const float *q_norm, const int *kv_slots, const float *cos, const float *sin,
                                       int tokens, int heads, int dim, int half_rot, float eps,
                                       cudaStream_t stream) {
  using namespace seventy;
  if (q_out == nullptr || idx_cache == nullptr || qk == nullptr || q_norm == nullptr ||
      tokens < 0 || heads < 1 || dim != kIdx || half_rot < 0 || 2 * half_rot > kIdx || qk_ld < (heads + 1) * kIdx ||
      (half_rot > 0 && (cos == nullptr || sin == nullptr)))
    return SEVENTY_ERR_INVALID_ARGUMENT;
  if (tokens == 0) return SEVENTY_OK;
  qsa_index_prep_kernel<<<dim3(tokens, heads + 1), kIdx, 0, stream>>>(
      static_cast<half *>(q_out), static_cast<half *>(idx_cache), qk, qk_ld, q_norm, kv_slots, cos, sin, heads,
      half_rot, eps);
  return launch_status();
}

SEVENTY_API int seventy_qsa_select_scores(int *sel, int width, const float *scores, int ld, const int *query_start,
                                          const int *context_lens, int num_seqs, int row0, int tokens, int compress,
                                          int top, cudaStream_t stream) {
  using namespace seventy;
  if (sel == nullptr || scores == nullptr || tokens < 0 || width != top * compress + compress - 1 || num_seqs < 1 ||
      ld < 1)
    return SEVENTY_ERR_INVALID_ARGUMENT;
  if (tokens == 0) return SEVENTY_OK;
  qsa_select_kernel<<<tokens, 1024, 0, stream>>>(sel, width, scores, ld, query_start, context_lens, num_seqs,
                                                 compress, top, row0);
  return launch_status();
}

SEVENTY_API int seventy_qsa_select(int *sel, int width, float *scores, int ld, const void *q, int heads,
                                   const void *pooled, const int *block_tables, int table_width,
                                   const int *query_start, const int *context_lens, int num_seqs, int row0,
                                   int tokens, int compress, int top, int block_size, cudaStream_t stream) {
  using namespace seventy;
  if (sel == nullptr || scores == nullptr || q == nullptr || pooled == nullptr || tokens < 0 || heads < 1 ||
      heads > 8 || width != top * compress + compress - 1 || num_seqs < 1)
    return SEVENTY_ERR_INVALID_ARGUMENT;
  if (tokens == 0) return SEVENTY_OK;
  qsa_score_kernel<<<dim3(tokens, (ld + 127) / 128), 128, 0, stream>>>(
      scores, ld, static_cast<const half *>(q), heads, static_cast<const half *>(pooled), block_tables, table_width,
      query_start, context_lens, num_seqs, compress, top, block_size, row0);
  qsa_select_kernel<<<tokens, 1024, 0, stream>>>(sel, width, scores, ld, query_start, context_lens, num_seqs,
                                                 compress, top, row0);
  return launch_status();
}

SEVENTY_API int seventy_qsa_attention(float *partial_o, float *partial_ml, const void *q, const void *k_cache,
                                      const void *v_cache, const float *k_scale, const float *v_scale,
                                      const int *sel, int width, const int *block_tables, int table_width,
                                      const int *query_start, int num_seqs, int row0, int tokens, int block_size,
                                      int hq, int hkv, int splits, float scale, int kv_split, int kv_part,
                                      cudaStream_t stream) {
  using namespace seventy;
  if (partial_o == nullptr || partial_ml == nullptr || q == nullptr || sel == nullptr || tokens < 0 || hkv < 1 ||
      hq % hkv || hq / hkv > 32 || splits < 1 || kv_split < 1 || kv_part < 0 || kv_part >= kv_split ||
      block_size % kv_split)
    return SEVENTY_ERR_INVALID_ARGUMENT;
  if (tokens == 0) return SEVENTY_OK;
  const dim3 grid(splits, tokens, hkv);
  const int rows = hq / hkv;
#define SEVENTY_QSA_ATTN(T, R, C, K, V, KS, VS)                                                                    \
  qsa_attention_kernel<T, R, C><<<grid, kAttnThreads, 0, stream>>>(                                                \
      partial_o, partial_ml, static_cast<const half *>(q), static_cast<const T *>(K), static_cast<const T *>(V), KS, \
      VS, sel, width, block_tables, table_width, query_start, num_seqs, block_size, hq, hkv, splits, scale, row0,   \
      kv_split, kv_part)
  static const bool mma = getenv("SEVENTY_QSA_MMA") == nullptr || strcmp(getenv("SEVENTY_QSA_MMA"), "0") != 0;
  if (k_scale != nullptr && mma && rows <= 16) {
    if (rows <= 8)
      qsa_attention_mma_kernel<1><<<grid, kAttnThreads, 0, stream>>>(
          partial_o, partial_ml, static_cast<const half *>(q), static_cast<const int8_t *>(k_cache),
          static_cast<const int8_t *>(v_cache), k_scale, v_scale, sel, width, block_tables, table_width, query_start,
          num_seqs, block_size, hq, hkv, splits, scale, row0, kv_split, kv_part);
    else
      qsa_attention_mma_kernel<2><<<grid, kAttnThreads, 0, stream>>>(
          partial_o, partial_ml, static_cast<const half *>(q), static_cast<const int8_t *>(k_cache),
          static_cast<const int8_t *>(v_cache), k_scale, v_scale, sel, width, block_tables, table_width, query_start,
          num_seqs, block_size, hq, hkv, splits, scale, row0, kv_split, kv_part);
    return launch_status();
  }
  if (k_scale != nullptr) {
    if (rows <= 8)
      SEVENTY_QSA_ATTN(int8_t, 8, 32, k_cache, v_cache, k_scale, v_scale);
    else if (rows <= 16)
      SEVENTY_QSA_ATTN(int8_t, 16, 32, k_cache, v_cache, k_scale, v_scale);
    else
      SEVENTY_QSA_ATTN(int8_t, 32, 16, k_cache, v_cache, k_scale, v_scale);
  } else {
    if (rows <= 8)
      SEVENTY_QSA_ATTN(half, 8, 32, k_cache, v_cache, nullptr, nullptr);
    else if (rows <= 16)
      SEVENTY_QSA_ATTN(half, 16, 32, k_cache, v_cache, nullptr, nullptr);
    else
      SEVENTY_QSA_ATTN(half, 32, 16, k_cache, v_cache, nullptr, nullptr);
  }
#undef SEVENTY_QSA_ATTN
  return launch_status();
}
