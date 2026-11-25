#include "common.cuh"

#include <cfloat>

/*
 * Qwen Sparse Attention (Qwen4-Exp) for the attention layers of a step, all on the GPU (CUDA graphs).
 *
 * A sequence's keys go in blocks of `compress` (c) tokens. The indexer has `heads` query heads and one key head of
 * kIdx values. qsa_pool: for every new token that completes a block (position p with (p + 1) % c == 0), the
 * block's key = RMS norm (weight already 1 + w) of the mean of its c raw indexer keys (cache idx_k, FP16), rounded
 * to FP16, rotated (multimodal rotary on the first rotary_dim values) at the block's first position (cache
 * positions), into the pooled cache [blocks, block_size / c, kIdx] at that position's block and offset.
 *
 * qsa_score, qsa_select and qsa_attention take the rows row0 .. row0 + tokens - 1 of the step (their q, scores,
 * sel and outputs hold only those).
 * qsa_score: for every query row whose sequence has more than `top` complete blocks visible, the score of each
 * visible complete block j: sum over heads of relu(q_h . key_j) / sqrt(kIdx), into scores [rows, ld].
 * qsa_select: the positions [rows, width = top * c + c - 1] (-1: none) that a query attends to: all its visible
 * positions while they have no more than `top` complete blocks; else the positions of its `top` best blocks (ties:
 * the lower block) and of its unfinished last block.
 * qsa_attention: causal attention of each query row over its selected positions (one block per split, row and KV
 * head; the group's query heads share the row's positions), with partial outputs for attention_combine in
 * csrc/attention.cu (seventy_decode_attention's workspace layout with one token per row).
 */

namespace seventy {
namespace {

constexpr int kIdx = 128;
constexpr int kHeadD = 256;
constexpr int kSelChunk = 16;
constexpr int kAttnRows = 64;

__device__ __forceinline__ int seq_of(const int *query_start, int num_seqs, int t) {
  int b = 0;
  while (b + 1 < num_seqs && query_start[b + 1] <= t) ++b;
  return b;
}

__global__ void __launch_bounds__(kIdx) qsa_pool_kernel(half *pooled, const half *idx_keys, const int *positions,
                                                        const int *block_tables, int table_width,
                                                        const int *query_start, const int *context_lens,
                                                        int num_seqs, const float *k_norm, const float *inv_freq,
                                                        int s1, int s2, int rotary_dim, int c, int block_size,
                                                        float eps) {
  __shared__ float scratch[32];
  __shared__ float vals[kIdx];
  const int t = blockIdx.x, d = threadIdx.x;
  const int b = seq_of(query_start, num_seqs, t);
  const int p = context_lens[b] + (t - query_start[b]);
  if ((p + 1) % c) return;
  const int j0 = p + 1 - c;
  const int *table = block_tables + static_cast<int64_t>(b) * table_width;
  float sum = 0.0f;
  for (int i = 0; i < c; ++i) {
    const int pos = j0 + i;
    const int64_t slot = static_cast<int64_t>(table[pos / block_size]) * block_size + pos % block_size;
    sum += __half2float(idx_keys[slot * kIdx + d]);
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
    r = d < half_rot ? vals[d] * cs - vals[d + half_rot] * sn : vals[d] * cs + vals[d - half_rot] * sn;
  }
  const int per_block = block_size / c;
  const int64_t out = (static_cast<int64_t>(table[j0 / block_size]) * per_block + (j0 % block_size) / c) * kIdx;
  pooled[out + d] = __float2half_rn(r);
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

template <typename T>
__device__ __forceinline__ void load_sel(half (*dst)[kHeadD + 8], const T *cache, const float *scales,
                                         const int *table, const int *pos_list, int n, int block_size, int hkv,
                                         int kh);

template <>
__device__ __forceinline__ void load_sel<int8_t>(half (*dst)[kHeadD + 8], const int8_t *cache, const float *scales,
                                                 const int *table, const int *pos_list, int n, int block_size,
                                                 int hkv, int kh) {
  for (int e = threadIdx.x; e < n * (kHeadD / 16); e += blockDim.x) {
    const int j = e / (kHeadD / 16);
    const int part = e % (kHeadD / 16);
    const int pos = pos_list[j];
    const int64_t slot = static_cast<int64_t>(table[pos / block_size]) * block_size + pos % block_size;
    const int64_t vec = slot * hkv + kh;
    const float s = scales[vec];
    const uint4 raw = *reinterpret_cast<const uint4 *>(cache + vec * kHeadD + part * 16);
    const uint32_t w[4] = {raw.x, raw.y, raw.z, raw.w};
    uint32_t h[8];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const half2 lo = __floats2half2_rn(static_cast<float>(static_cast<int8_t>(w[i] & 0xff)) * s,
                                         static_cast<float>(static_cast<int8_t>((w[i] >> 8) & 0xff)) * s);
      const half2 hi = __floats2half2_rn(static_cast<float>(static_cast<int8_t>((w[i] >> 16) & 0xff)) * s,
                                         static_cast<float>(static_cast<int8_t>(w[i] >> 24)) * s);
      h[2 * i] = *reinterpret_cast<const uint32_t *>(&lo);
      h[2 * i + 1] = *reinterpret_cast<const uint32_t *>(&hi);
    }
    *reinterpret_cast<uint4 *>(&dst[j][part * 16]) = make_uint4(h[0], h[1], h[2], h[3]);
    *reinterpret_cast<uint4 *>(&dst[j][part * 16 + 8]) = make_uint4(h[4], h[5], h[6], h[7]);
  }
}

template <>
__device__ __forceinline__ void load_sel<half>(half (*dst)[kHeadD + 8], const half *cache, const float *,
                                               const int *table, const int *pos_list, int n, int block_size, int hkv,
                                               int kh) {
  for (int e = threadIdx.x; e < n * (kHeadD / 8); e += blockDim.x) {
    const int j = e / (kHeadD / 8);
    const int part = e % (kHeadD / 8);
    const int pos = pos_list[j];
    const int64_t slot = static_cast<int64_t>(table[pos / block_size]) * block_size + pos % block_size;
    *reinterpret_cast<uint4 *>(&dst[j][part * 8]) =
        *reinterpret_cast<const uint4 *>(cache + (slot * hkv + kh) * kHeadD + part * 8);
  }
}

template <typename T>
__global__ void __launch_bounds__(kHeadD) qsa_attention_kernel(
    float *partial_o, float *partial_ml, const half *q, const T *k_cache, const T *v_cache, const float *k_scale,
    const float *v_scale, const int *sel, int width, const int *block_tables, int table_width,
    const int *query_start, int num_seqs, int block_size, int hq, int hkv, int splits, float scale, int row0) {
  __shared__ half qs[kAttnRows][kHeadD];
  __shared__ half kvs[kSelChunk][kHeadD + 8];
  __shared__ float p[kAttnRows][kSelChunk];
  __shared__ float row_m[kAttnRows], row_l[kAttnRows], row_scale[kAttnRows];
  __shared__ int count_s;
  const int split = blockIdx.x, t = blockIdx.y, kh = blockIdx.z, tid = threadIdx.x;
  const int group = hq / hkv;
  const int rows = group;
  const int b = seq_of(query_start, num_seqs, row0 + t);
  const int *row_sel = sel + static_cast<int64_t>(t) * width;
  if (tid == 0) {
    int n = 0;
    while (n < width && row_sel[n] >= 0) ++n;
    for (int i = n; i < width; ++i)
      if (row_sel[i] >= 0) n = i + 1;
    count_s = n;
  }
  __syncthreads();
  const int total = count_s;
  const int per = (total + splits - 1) / splits;
  const int first = split * per, last = min(total, first + per);
  auto pidx = [&](int r) { return (static_cast<int64_t>(t) * hq + kh * group + r) * splits + split; };
  if (rows > kAttnRows) return;
  if (first >= last) {
    for (int r = tid; r < rows; r += kHeadD) {
      partial_ml[pidx(r) * 2] = -FLT_MAX;
      partial_ml[pidx(r) * 2 + 1] = 0.0f;
    }
    return;
  }
  for (int e = tid; e < rows * kHeadD; e += kHeadD) {
    const int r = e / kHeadD;
    qs[r][e % kHeadD] = q[(static_cast<int64_t>(t) * hq + kh * group + r) * kHeadD + e % kHeadD];
  }
  for (int r = tid; r < rows; r += kHeadD) {
    row_m[r] = -FLT_MAX;
    row_l[r] = 0.0f;
  }
  float acc[kAttnRows];
#pragma unroll
  for (int r = 0; r < kAttnRows; ++r) acc[r] = 0.0f;
  const int *table = block_tables + static_cast<int64_t>(b) * table_width;
  __syncthreads();
  for (int c0 = first; c0 < last; c0 += kSelChunk) {
    const int n = min(kSelChunk, last - c0);
    __shared__ int pos_list[kSelChunk];
    __shared__ bool ok[kSelChunk];
    if (tid < kSelChunk) {
      const int v = tid < n ? row_sel[c0 + tid] : -1;
      ok[tid] = v >= 0;
      pos_list[tid] = v >= 0 ? v : 0;
    }
    __syncthreads();
    load_sel<T>(kvs, k_cache, k_scale, table, pos_list, n, block_size, hkv, kh);
    __syncthreads();
    for (int e = tid; e < rows * kSelChunk; e += kHeadD) {
      const int r = e / kSelChunk, j = e % kSelChunk;
      float s = -FLT_MAX;
      if (j < n && ok[j]) {
        float dot = 0.0f;
#pragma unroll 8
        for (int x = 0; x < kHeadD; x += 2) {
          const float2 a = __half22float2(*reinterpret_cast<const half2 *>(&qs[r][x]));
          const float2 k2 = __half22float2(*reinterpret_cast<const half2 *>(&kvs[j][x]));
          dot += a.x * k2.x + a.y * k2.y;
        }
        s = dot * scale;
      }
      p[r][j] = s;
    }
    __syncthreads();
    for (int r = tid; r < rows; r += kHeadD) {
      float m = row_m[r];
      for (int j = 0; j < n; ++j) m = fmaxf(m, p[r][j]);
      const float corr = (row_m[r] == -FLT_MAX) ? 0.0f : expf(row_m[r] - m);
      float l = row_l[r] * corr;
      for (int j = 0; j < n; ++j) {
        const float e = (p[r][j] == -FLT_MAX) ? 0.0f : expf(p[r][j] - m);
        p[r][j] = e;
        l += e;
      }
      row_m[r] = m;
      row_l[r] = l;
      row_scale[r] = corr;
    }
    load_sel<T>(kvs, v_cache, v_scale, table, pos_list, n, block_size, hkv, kh);
    __syncthreads();
#pragma unroll
    for (int r = 0; r < kAttnRows; ++r) {
      if (r < rows) {
        float a = acc[r] * row_scale[r];
        for (int j = 0; j < n; ++j) a += p[r][j] * __half2float(kvs[j][tid]);
        acc[r] = a;
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int r = 0; r < kAttnRows; ++r)
    if (r < rows) partial_o[pidx(r) * kHeadD + tid] = acc[r];
  for (int r = tid; r < rows; r += kHeadD) {
    partial_ml[pidx(r) * 2] = row_m[r];
    partial_ml[pidx(r) * 2 + 1] = row_l[r];
  }
}

}  // namespace
}  // namespace seventy

SEVENTY_API int seventy_qsa_pool(void *pooled, const void *idx_keys, const int *positions, const int *block_tables,
                                 int table_width, const int *query_start, const int *context_lens, int num_seqs,
                                 int tokens, const float *k_norm, const float *inv_freq, int s1, int s2,
                                 int rotary_dim, int compress, int block_size, float eps, cudaStream_t stream) {
  using namespace seventy;
  if (pooled == nullptr || idx_keys == nullptr || positions == nullptr || tokens < 0 || num_seqs < 1 ||
      compress < 1 || block_size % compress || rotary_dim > kIdx)
    return SEVENTY_ERR_INVALID_ARGUMENT;
  if (tokens == 0) return SEVENTY_OK;
  qsa_pool_kernel<<<tokens, kIdx, 0, stream>>>(static_cast<half *>(pooled), static_cast<const half *>(idx_keys),
                                               positions, block_tables, table_width, query_start, context_lens,
                                               num_seqs, k_norm, inv_freq, s1, s2, rotary_dim, compress, block_size,
                                               eps);
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
                                      int hq, int hkv, int splits, float scale, cudaStream_t stream) {
  using namespace seventy;
  if (partial_o == nullptr || partial_ml == nullptr || q == nullptr || sel == nullptr || tokens < 0 || hkv < 1 ||
      hq % hkv || hq / hkv > kAttnRows || splits < 1)
    return SEVENTY_ERR_INVALID_ARGUMENT;
  if (tokens == 0) return SEVENTY_OK;
  const dim3 grid(splits, tokens, hkv);
  if (k_scale != nullptr) {
    qsa_attention_kernel<int8_t><<<grid, kHeadD, 0, stream>>>(
        partial_o, partial_ml, static_cast<const half *>(q), static_cast<const int8_t *>(k_cache),
        static_cast<const int8_t *>(v_cache), k_scale, v_scale, sel, width, block_tables, table_width, query_start,
        num_seqs, block_size, hq, hkv, splits, scale, row0);
  } else {
    qsa_attention_kernel<half><<<grid, kHeadD, 0, stream>>>(
        partial_o, partial_ml, static_cast<const half *>(q), static_cast<const half *>(k_cache),
        static_cast<const half *>(v_cache), nullptr, nullptr, sel, width, block_tables, table_width, query_start,
        num_seqs, block_size, hq, hkv, splits, scale, row0);
  }
  return launch_status();
}
