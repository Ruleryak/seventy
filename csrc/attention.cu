#include <cfloat>

#include "common.cuh"

/*
 * Full-attention layers, head size 256.
 *
 * attn_prep: one block per (token, head). Query heads: RMS norm of the query (weight with the 1 + w offset already
 * in it), rounded to FP16, then multimodal rotary on the first rotary_dim values, to q_out [T, Hq, 256]. KV heads:
 * the same for the key, written with the raw value to the paged caches at the token's slot.
 * The projection row is [q_proj (per head: query 256 | gate 256) | k_proj | v_proj].
 *
 * decode_attention: causal attention of a sequence's few new tokens over its cached tokens, which include them.
 * One block per (split, sequence, KV head) computes all query heads of the KV head's group for all new tokens of
 * the sequence (rows = group * tokens), over one range of cache blocks, with an online softmax in FP32. Partial
 * outputs, row maxima and sums go to a workspace; attention_combine merges the splits, multiplies by
 * sigmoid(gate) and writes FP16 [T, Hq * 256].
 */

namespace seventy {
namespace {

constexpr int kHead = 256;
constexpr int kChunk = 16;
constexpr int kMaxRows = 64;

__global__ void __launch_bounds__(kHead) attn_prep_kernel(half *q_out, half *k_cache, half *v_cache, const half *qkv,
                                                          int ld, const float *q_norm, const float *k_norm,
                                                          const int64_t *positions, int64_t pos_stride,
                                                          const float *inv_freq, int s1, int s2, int rotary_dim,
                                                          const int *slots, int hq, int hkv, float eps) {
  __shared__ float scratch[32];
  __shared__ float vals[kHead];
  const int t = blockIdx.x;
  const int head = blockIdx.y;
  const int d = threadIdx.x;
  const bool is_q = head < hq;
  const int kv = head - hq;
  const half *row = qkv + static_cast<int64_t>(t) * ld;
  const float x = __half2float(is_q ? row[head * 2 * kHead + d] : row[2 * hq * kHead + kv * kHead + d]);
  const float ss = block_sum(x * x, scratch);
  const float w = is_q ? q_norm[d] : k_norm[d];
  vals[d] = __half2float(__float2half_rn(x * rsqrtf(ss / kHead + eps) * w));
  __syncthreads();
  const int half_rot = rotary_dim / 2;
  float r = vals[d];
  if (d < rotary_dim) {
    const int i = d < half_rot ? d : d - half_rot;
    int axis = 0;
    if (i % 3 == 1 && i < 3 * s1) axis = 1;
    if (i % 3 == 2 && i < 3 * s2) axis = 2;
    const float pos = static_cast<float>(positions[axis * pos_stride + t]);
    float sn, cs;
    sincosf(pos * inv_freq[i], &sn, &cs);
    r = d < half_rot ? vals[d] * cs - vals[d + half_rot] * sn : vals[d] * cs + vals[d - half_rot] * sn;
  }
  const half out = __float2half_rn(r);
  if (is_q) {
    q_out[(static_cast<int64_t>(t) * hq + head) * kHead + d] = out;
  } else {
    const int64_t base = (static_cast<int64_t>(slots[t]) * hkv + kv) * kHead + d;
    k_cache[base] = out;
    v_cache[base] = row[2 * hq * kHead + hkv * kHead + kv * kHead + d];
  }
}

__global__ void __launch_bounds__(kHead) decode_attention_kernel(
    float *partial_o, float *partial_ml, const half *q, const half *k_cache, const half *v_cache,
    const int *block_tables, int table_width, const int *query_start, const int *context_lens, int block_size,
    int hq, int hkv, int blocks_per_split, int max_splits, float scale) {
  __shared__ half qs[kMaxRows][kHead];
  __shared__ half kvs[kChunk][kHead + 8];
  __shared__ float p[kMaxRows][kChunk];
  __shared__ float row_m[kMaxRows], row_l[kMaxRows], row_scale[kMaxRows];
  const int split = blockIdx.x;
  const int b = blockIdx.y;
  const int kh = blockIdx.z;
  const int group = hq / hkv;
  const int t0 = query_start[b];
  const int tokens = query_start[b + 1] - t0;
  const int rows = group * tokens;
  const int total = context_lens[b] + tokens;
  const int first = split * blocks_per_split * block_size;
  const int last = min(total, first + blocks_per_split * block_size);
  const int tid = threadIdx.x;
  if (tokens <= 0 || rows > kMaxRows) return;
  auto pidx = [&](int r) {
    return (static_cast<int64_t>(t0 + r / group) * hq + kh * group + r % group) * max_splits + split;
  };
  if (first >= last) {
    for (int r = tid; r < rows; r += kHead) {
      const int64_t ml = pidx(r) * 2;
      partial_ml[ml] = -FLT_MAX;
      partial_ml[ml + 1] = 0.0f;
    }
    return;
  }
  for (int e = tid; e < rows * kHead; e += kHead) {
    const int r = e / kHead;
    const int tok = r / group;
    const int head = kh * group + r % group;
    qs[r][e % kHead] = q[(static_cast<int64_t>(t0 + tok) * hq + head) * kHead + e % kHead];
  }
  for (int r = tid; r < rows; r += kHead) {
    row_m[r] = -FLT_MAX;
    row_l[r] = 0.0f;
  }
  float acc[kMaxRows];
#pragma unroll
  for (int r = 0; r < kMaxRows; ++r) acc[r] = 0.0f;
  const int *table = block_tables + static_cast<int64_t>(b) * table_width;
  __syncthreads();

  for (int c0 = first; c0 < last; c0 += kChunk) {
    const int n = min(kChunk, last - c0);
    for (int e = tid; e < n * (kHead / 8); e += kHead) {
      const int j = e / (kHead / 8);
      const int part = e % (kHead / 8);
      const int pos = c0 + j;
      const int64_t slot = static_cast<int64_t>(table[pos / block_size]) * block_size + pos % block_size;
      *reinterpret_cast<uint4 *>(&kvs[j][part * 8]) =
          *reinterpret_cast<const uint4 *>(k_cache + (slot * hkv + kh) * kHead + part * 8);
    }
    __syncthreads();
    for (int e = tid; e < rows * kChunk; e += kHead) {
      const int r = e / kChunk;
      const int j = e % kChunk;
      float s = -FLT_MAX;
      const int qpos = context_lens[b] + r / group;
      if (j < n && c0 + j <= qpos) {
        float dot = 0.0f;
#pragma unroll 8
        for (int x = 0; x < kHead; x += 2) {
          const float2 a = __half22float2(*reinterpret_cast<const half2 *>(&qs[r][x]));
          const float2 k2 = __half22float2(*reinterpret_cast<const half2 *>(&kvs[j][x]));
          dot += a.x * k2.x + a.y * k2.y;
        }
        s = dot * scale;
      }
      p[r][j] = s;
    }
    __syncthreads();
    for (int r = tid; r < rows; r += kHead) {
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
    for (int e = tid; e < n * (kHead / 8); e += kHead) {
      const int j = e / (kHead / 8);
      const int part = e % (kHead / 8);
      const int pos = c0 + j;
      const int64_t slot = static_cast<int64_t>(table[pos / block_size]) * block_size + pos % block_size;
      *reinterpret_cast<uint4 *>(&kvs[j][part * 8]) =
          *reinterpret_cast<const uint4 *>(v_cache + (slot * hkv + kh) * kHead + part * 8);
    }
    __syncthreads();
#pragma unroll
    for (int r = 0; r < kMaxRows; ++r) {
      if (r < rows) {
        float a = acc[r] * row_scale[r];
        for (int j = 0; j < n; ++j) a += p[r][j] * __half2float(kvs[j][tid]);
        acc[r] = a;
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int r = 0; r < kMaxRows; ++r) {
    if (r < rows) partial_o[pidx(r) * kHead + tid] = acc[r];
  }
  for (int r = tid; r < rows; r += kHead) {
    const int64_t ml = pidx(r) * 2;
    partial_ml[ml] = row_m[r];
    partial_ml[ml + 1] = row_l[r];
  }
}

__global__ void __launch_bounds__(kHead) attention_combine_kernel(half *out, const float *partial_o,
                                                                  const float *partial_ml, const half *qkv, int ld,
                                                                  int hq, int splits) {
  const int t = blockIdx.x;
  const int head = blockIdx.y;
  const int d = threadIdx.x;
  const int64_t base = (static_cast<int64_t>(t) * hq + head) * splits;
  float m = -FLT_MAX;
  for (int s = 0; s < splits; ++s) m = fmaxf(m, partial_ml[(base + s) * 2]);
  float l = 0.0f, o = 0.0f;
  for (int s = 0; s < splits; ++s) {
    const float ms = partial_ml[(base + s) * 2];
    if (ms == -FLT_MAX) continue;
    const float w = expf(ms - m);
    l += partial_ml[(base + s) * 2 + 1] * w;
    o += partial_o[(base + s) * kHead + d] * w;
  }
  const float a = __half2float(__float2half_rn(o / l));
  const float g = __half2float(qkv[static_cast<int64_t>(t) * ld + head * 2 * kHead + kHead + d]);
  out[(static_cast<int64_t>(t) * hq + head) * kHead + d] = __float2half_rn(a / (1.0f + expf(-g)));
}

}  // namespace
}  // namespace seventy

SEVENTY_API int seventy_attn_prep(void *q_out, void *k_cache, void *v_cache, const void *qkv, int ld,
                                  const float *q_norm, const float *k_norm, const int64_t *positions,
                                  int64_t pos_stride, const float *inv_freq, int section1, int section2,
                                  int rotary_dim, const int *slots, int tokens, int hq, int hkv, int head_dim,
                                  float eps, cudaStream_t stream) {
  using namespace seventy;
  if (q_out == nullptr || k_cache == nullptr || v_cache == nullptr || qkv == nullptr || q_norm == nullptr ||
      k_norm == nullptr || positions == nullptr || inv_freq == nullptr || slots == nullptr)
    return SEVENTY_ERR_INVALID_ARGUMENT;
  if (head_dim != kHead || rotary_dim <= 0 || rotary_dim > kHead || rotary_dim % 2 || hq <= 0 || hkv <= 0 ||
      tokens < 0)
    return SEVENTY_ERR_INVALID_ARGUMENT;
  if (tokens == 0) return SEVENTY_OK;
  attn_prep_kernel<<<dim3(tokens, hq + hkv), kHead, 0, stream>>>(
      static_cast<half *>(q_out), static_cast<half *>(k_cache), static_cast<half *>(v_cache),
      static_cast<const half *>(qkv), ld, q_norm, k_norm, positions, pos_stride, inv_freq, section1, section2,
      rotary_dim, slots, hq, hkv, eps);
  return launch_status();
}

SEVENTY_API int64_t seventy_decode_attention_workspace(int tokens, int hq, int splits) {
  return static_cast<int64_t>(tokens) * hq * splits * (seventy::kHead + 2) * static_cast<int64_t>(sizeof(float));
}

SEVENTY_API int seventy_decode_attention(void *out, const void *q, const void *k_cache, const void *v_cache,
                                         const void *qkv, int ld, const int *block_tables, int table_width,
                                         const int *query_start, const int *context_lens, int num_seqs, int tokens,
                                         int max_query_len, int block_size, int hq, int hkv, int head_dim, int splits,
                                         int blocks_per_split, float scale, void *workspace, int64_t workspace_bytes,
                                         cudaStream_t stream) {
  using namespace seventy;
  if (out == nullptr || q == nullptr || k_cache == nullptr || v_cache == nullptr || qkv == nullptr ||
      block_tables == nullptr || query_start == nullptr || context_lens == nullptr || workspace == nullptr)
    return SEVENTY_ERR_INVALID_ARGUMENT;
  if (head_dim != kHead || hq % hkv || (hq / hkv) * max_query_len > kMaxRows || splits < 1 || blocks_per_split < 1 ||
      block_size < 1)
    return SEVENTY_ERR_INVALID_ARGUMENT;
  if (workspace_bytes < seventy_decode_attention_workspace(tokens, hq, splits)) return SEVENTY_ERR_INVALID_ARGUMENT;
  if (num_seqs == 0 || tokens == 0) return SEVENTY_OK;
  auto *po = static_cast<float *>(workspace);
  auto *pml = po + static_cast<int64_t>(tokens) * hq * splits * kHead;
  decode_attention_kernel<<<dim3(splits, num_seqs, hkv), kHead, 0, stream>>>(
      po, pml, static_cast<const half *>(q), static_cast<const half *>(k_cache), static_cast<const half *>(v_cache),
      block_tables, table_width, query_start, context_lens, block_size, hq, hkv, blocks_per_split, splits, scale);
  attention_combine_kernel<<<dim3(tokens, hq), kHead, 0, stream>>>(static_cast<half *>(out), po, pml,
                                                                   static_cast<const half *>(qkv), ld, hq, splits);
  return launch_status();
}
