#include <mma.h>

#include "common.cuh"

/*
 * Kernels of the DFlash2 drafter (seventy/model/dflash2.py has the reference of each one):
 *
 *   drafter_norm     RMS norm with an FP32 residual stream: s = x + residual; residual = s; out = norm(s) * w.
 *   drafter_conv     the two-tap grouped convolution inside each block of `block` rows:
 *                    out[t, c] = (base[0, c] + d[t, 0, c / gs]) x[t, c] + (base[1, c] + d[t, 1, c / gs]) x[t - 1, c],
 *                    the second term only when t is not the first row of its block.
 *   drafter_qk       per head: RMS norm (rounded to FP16), rotary (neox halves over the whole head, FP32 math), for
 *                    query heads (to q_out) and key heads (into the K ring), and V into the V ring, at ring index
 *                    position % ring.
 *   drafter_attention the block attention: each block row (token t of sequence b, at position start + t) sees the
 *                    keys at positions [max(0, start + t - window + 1), start + block) of its ring, so a block of
 *                    queries reads at most window + block - 1 keys. Splits of 128 keys run on the tensor cores
 *                    (WMMA 16 x 16 x 16, FP16 in, FP32 accumulation) with an exact softmax per split; a combine
 *                    kernel merges the splits by their log-sum-exp.
 *   silu_mul_scale   p = silu(gate) * up in FP32, rows scaled into the FP16 range (scale = max(1, max|p| / 16384)).
 */

namespace seventy {
namespace {

using namespace nvcuda;

constexpr int kD = 128;
constexpr int kRows = 32;
constexpr int kKeys = 128;
constexpr int kLd = kD + 8;
constexpr int kLdS = kKeys + 4;

__global__ void drafter_norm_kernel(half *out, float *res_out, const void *x, int x_fp32, const float *res_in,
                                    const float *weight, int dim, float eps) {
  __shared__ float scratch[32];
  const int64_t row = blockIdx.x;
  float sumsq = 0.0f;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    float v = x_fp32 ? static_cast<const float *>(x)[row * dim + i]
                     : __half2float(static_cast<const half *>(x)[row * dim + i]);
    if (res_in != nullptr) v += res_in[row * dim + i];
    if (res_out != nullptr) res_out[row * dim + i] = v;
    sumsq += v * v;
  }
  const float inv = rsqrtf(block_sum(sumsq, scratch) / dim + eps);
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    float v = res_out != nullptr ? res_out[row * dim + i]
                                 : (x_fp32 ? static_cast<const float *>(x)[row * dim + i]
                                           : __half2float(static_cast<const half *>(x)[row * dim + i]));
    if (res_out == nullptr && res_in != nullptr) v += res_in[row * dim + i];
    out[row * dim + i] = __float2half_rn(v * inv * weight[i]);
  }
}

__global__ void drafter_conv_kernel(void *out, int out_fp32, const void *x, int x_fp32, const float *base,
                                    const float *delta, int64_t delta_row_stride, int groups, int group_size,
                                    int channels, int block) {
  const int64_t t = blockIdx.y;
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= channels) return;
  const int g = c / group_size;
  auto load = [&](int64_t r) {
    return x_fp32 ? static_cast<const float *>(x)[r * channels + c]
                  : __half2float(static_cast<const half *>(x)[r * channels + c]);
  };
  const float *d = delta + t * delta_row_stride;
  float v = (base[c] + d[g]) * load(t);
  if (t % block) v += (base[channels + c] + d[groups + g]) * load(t - 1);
  if (out_fp32) static_cast<float *>(out)[t * channels + c] = v;
  else static_cast<half *>(out)[t * channels + c] = __float2half_rn(v);
}

__global__ void drafter_qk_kernel(half *q_out, half *k_ring, half *v_ring, const float *q, int64_t q_row_stride,
                                  const float *kv, int64_t kv_row_stride, const float *q_norm, const float *k_norm,
                                  const int64_t *positions, const int64_t *slots, const float *inv_freq, int heads,
                                  int kv_heads, int ring, float eps) {
  const int64_t row = blockIdx.x;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int64_t pos = positions[row];
  const int64_t base = (slots[row] * ring + pos % ring) * kv_heads * kD;
  const int total = (q_out != nullptr ? heads : 0) + 2 * kv_heads;
  for (int h = warp; h < total; h += blockDim.x >> 5) {
    const bool is_q = q_out != nullptr && h < heads;
    const int kh = h - (q_out != nullptr ? heads : 0);
    float x[4];
    const float *src = is_q ? q + row * q_row_stride + h * kD : kv + row * kv_row_stride + kh * kD;
    if (!is_q && kh >= kv_heads) {
      half *dst = v_ring + base + (kh - kv_heads) * kD;
#pragma unroll
      for (int i = 0; i < 4; ++i) dst[lane * 4 + i] = __float2half_rn(src[lane * 4 + i]);
      continue;
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) x[i] = src[lane * 4 + i];
    const float ss = warp_sum(x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]);
    const float inv = rsqrtf(ss / kD + eps);
    const float *w = is_q ? q_norm : k_norm;
#pragma unroll
    for (int i = 0; i < 4; ++i) x[i] = __half2float(__float2half_rn(x[i] * inv * w[lane * 4 + i]));
    float o[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float partner = __shfl_xor_sync(0xffffffffu, x[i], 16);
      const int f = (lane & 15) * 4 + i;
      float sn, cs;
      sincosf(static_cast<float>(pos) * inv_freq[f], &sn, &cs);
      o[i] = lane < 16 ? x[i] * cs - partner * sn : x[i] * cs + partner * sn;
    }
    half *dst = is_q ? q_out + (row * heads + h) * kD : k_ring + base + kh * kD;
#pragma unroll
    for (int i = 0; i < 4; ++i) dst[lane * 4 + i] = __float2half_rn(o[i]);
  }
}

__global__ void __launch_bounds__(128) drafter_attention_kernel(float *ws_o, float *ws_lse, const half *q,
                                                                const half *k_ring, const half *v_ring,
                                                                const int64_t *slots, const int64_t *starts,
                                                                int heads, int kv_heads, int block, int window,
                                                                int ring, int splits, float scale) {
  extern __shared__ __align__(16) unsigned char smem[];
  half *qs = reinterpret_cast<half *>(smem);
  half *ks = qs + kRows * kLd;
  half *vs = ks + kKeys * kLd;
  float *ss = reinterpret_cast<float *>(vs + kKeys * kLd);
  float *row_l = ss + kRows * kLdS;
  half *ps = ks;
  float *os = ss;

  const int s = blockIdx.x, g = blockIdx.y, b = blockIdx.z;
  const int rep = heads / kv_heads;
  const int rows = block * rep;
  const int64_t start = starts[b];
  const int64_t end = start + block;
  const int64_t k0 = start - (window - 1) > 0 ? start - (window - 1) : 0;
  const int64_t kb = k0 + static_cast<int64_t>(s) * kKeys;
  const int64_t ke = kb + kKeys < end ? kb + kKeys : end;
  const int64_t out_base = ((static_cast<int64_t>(b) * kv_heads + g) * splits + s) * kRows;
  if (kb >= end) {
    for (int r = threadIdx.x; r < kRows; r += blockDim.x) ws_lse[out_base + r] = -INFINITY;
    return;
  }
  const int64_t slot = slots[b];

  for (int e = threadIdx.x; e < kRows * (kD / 8); e += blockDim.x) {
    const int r = e / (kD / 8), c = (e % (kD / 8)) * 8;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (r < rows) {
      const int t = r / rep, h = g * rep + r % rep;
      v = *reinterpret_cast<const uint4 *>(q + ((static_cast<int64_t>(b) * block + t) * heads + h) * kD + c);
    }
    *reinterpret_cast<uint4 *>(qs + r * kLd + c) = v;
  }
  for (int e = threadIdx.x; e < kKeys * (kD / 8); e += blockDim.x) {
    const int j = e / (kD / 8), c = (e % (kD / 8)) * 8;
    uint4 kv4 = make_uint4(0, 0, 0, 0), vv4 = make_uint4(0, 0, 0, 0);
    const int64_t pos = kb + j;
    if (pos < ke) {
      const int64_t off = ((slot * ring + pos % ring) * kv_heads + g) * kD + c;
      kv4 = *reinterpret_cast<const uint4 *>(k_ring + off);
      vv4 = *reinterpret_cast<const uint4 *>(v_ring + off);
    }
    *reinterpret_cast<uint4 *>(ks + j * kLd + c) = kv4;
    *reinterpret_cast<uint4 *>(vs + j * kLd + c) = vv4;
  }
  __syncthreads();

  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  {
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
#pragma unroll
    for (int rt = 0; rt < 2; ++rt)
#pragma unroll
      for (int ct = 0; ct < 2; ++ct) wmma::fill_fragment(acc[rt][ct], 0.0f);
#pragma unroll
    for (int k = 0; k < kD / 16; ++k) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a[2];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> bf[2];
#pragma unroll
      for (int rt = 0; rt < 2; ++rt) wmma::load_matrix_sync(a[rt], qs + rt * 16 * kLd + k * 16, kLd);
#pragma unroll
      for (int ct = 0; ct < 2; ++ct) wmma::load_matrix_sync(bf[ct], ks + (warp * 32 + ct * 16) * kLd + k * 16, kLd);
#pragma unroll
      for (int rt = 0; rt < 2; ++rt)
#pragma unroll
        for (int ct = 0; ct < 2; ++ct) wmma::mma_sync(acc[rt][ct], a[rt], bf[ct], acc[rt][ct]);
    }
#pragma unroll
    for (int rt = 0; rt < 2; ++rt)
#pragma unroll
      for (int ct = 0; ct < 2; ++ct)
        wmma::store_matrix_sync(ss + rt * 16 * kLdS + warp * 32 + ct * 16, acc[rt][ct], kLdS, wmma::mem_row_major);
  }
  __syncthreads();

  for (int r = warp; r < kRows; r += 4) {
    const int t = r / rep;
    const int64_t lo = start + t - (window - 1);
    float v[4];
    float m = -INFINITY;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int j = lane + 32 * i;
      const int64_t pos = kb + j;
      const bool ok = r < rows && pos < ke && pos >= lo;
      v[i] = ok ? ss[r * kLdS + j] * scale : -INFINITY;
      m = fmaxf(m, v[i]);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
    float l = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float p = m == -INFINITY ? 0.0f : __expf(v[i] - m);
      v[i] = p;
      l += p;
    }
    l = warp_sum(l);
#pragma unroll
    for (int i = 0; i < 4; ++i) ps[r * kLd + lane + 32 * i] = __float2half_rn(v[i]);
    if (lane == 0) {
      row_l[r] = l;
      ws_lse[out_base + r] = l > 0.0f ? m + __logf(l) : -INFINITY;
    }
  }
  __syncthreads();

  {
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
#pragma unroll
    for (int rt = 0; rt < 2; ++rt)
#pragma unroll
      for (int ct = 0; ct < 2; ++ct) wmma::fill_fragment(acc[rt][ct], 0.0f);
#pragma unroll
    for (int k = 0; k < kKeys / 16; ++k) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a[2];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf[2];
#pragma unroll
      for (int rt = 0; rt < 2; ++rt) wmma::load_matrix_sync(a[rt], ps + rt * 16 * kLd + k * 16, kLd);
#pragma unroll
      for (int ct = 0; ct < 2; ++ct) wmma::load_matrix_sync(bf[ct], vs + k * 16 * kLd + warp * 32 + ct * 16, kLd);
#pragma unroll
      for (int rt = 0; rt < 2; ++rt)
#pragma unroll
        for (int ct = 0; ct < 2; ++ct) wmma::mma_sync(acc[rt][ct], a[rt], bf[ct], acc[rt][ct]);
    }
    __syncthreads();
#pragma unroll
    for (int rt = 0; rt < 2; ++rt)
#pragma unroll
      for (int ct = 0; ct < 2; ++ct)
        wmma::store_matrix_sync(os + rt * 16 * kLdS + warp * 32 + ct * 16, acc[rt][ct], kLdS, wmma::mem_row_major);
  }
  __syncthreads();
  for (int e = threadIdx.x; e < kRows * kD; e += blockDim.x) {
    const int r = e / kD, c = e % kD;
    const float l = row_l[r];
    ws_o[(out_base + r) * kD + c] = l > 0.0f ? os[r * kLdS + c] / l : 0.0f;
  }
}

__global__ void drafter_combine_kernel(half *out, const float *ws_o, const float *ws_lse, int heads, int kv_heads,
                                       int block, int splits) {
  const int g = blockIdx.y, b = blockIdx.z;
  const int rep = heads / kv_heads;
  const int r = blockIdx.x;
  const int c = threadIdx.x;
  const int64_t base = (static_cast<int64_t>(b) * kv_heads + g) * splits;
  float m = -INFINITY;
  for (int s = 0; s < splits; ++s) m = fmaxf(m, ws_lse[(base + s) * kRows + r]);
  float acc = 0.0f, wsum = 0.0f;
  if (m != -INFINITY) {
    for (int s = 0; s < splits; ++s) {
      const float lse = ws_lse[(base + s) * kRows + r];
      if (lse == -INFINITY) continue;
      const float w = __expf(lse - m);
      wsum += w;
      acc += w * ws_o[((base + s) * kRows + r) * kD + c];
    }
  }
  const int t = r / rep, h = g * rep + r % rep;
  out[((static_cast<int64_t>(b) * block + t) * heads + h) * kD + c] = __float2half_rn(wsum > 0.0f ? acc / wsum : 0.0f);
}

__global__ void silu_mul_scale_kernel(half *out, float *scale_out, const float *gu, int inter) {
  __shared__ float red[32];
  const int64_t row = blockIdx.x;
  const float *g = gu + row * 2 * inter;
  const float *u = g + inter;
  float amax = 0.0f;
  for (int i = threadIdx.x; i < inter; i += blockDim.x) {
    const float x = g[i];
    amax = fmaxf(amax, fabsf(x / (1.0f + __expf(-x)) * u[i]));
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = amax;
  __syncthreads();
  if (threadIdx.x < 32) {
    float v = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.0f;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    if (threadIdx.x == 0) red[0] = v;
  }
  __syncthreads();
  const float sc = fmaxf(1.0f, red[0] / 16384.0f);
  if (threadIdx.x == 0) scale_out[row] = sc;
  for (int i = threadIdx.x; i < inter; i += blockDim.x) {
    const float x = g[i];
    out[row * inter + i] = __float2half_rn(x / (1.0f + __expf(-x)) * u[i] / sc);
  }
}

constexpr int kAttnSmem = (kRows * kLd + 2 * kKeys * kLd) * 2 + (kRows * kLdS + kRows) * 4;

}  // namespace
}  // namespace seventy

SEVENTY_API int seventy_drafter_norm(void *out, float *residual_out, const void *x, int x_fp32,
                                     const float *residual_in, const float *weight, int rows, int dim, float eps,
                                     cudaStream_t stream) {
  using namespace seventy;
  if (out == nullptr || x == nullptr || weight == nullptr || rows < 0 || dim <= 0) return SEVENTY_ERR_INVALID_ARGUMENT;
  if (rows == 0) return SEVENTY_OK;
  drafter_norm_kernel<<<rows, 512, 0, stream>>>(static_cast<half *>(out), residual_out, x, x_fp32, residual_in,
                                                weight, dim, eps);
  return launch_status();
}

SEVENTY_API int seventy_drafter_conv(void *out, int out_fp32, const void *x, int x_fp32, const float *base,
                                     const float *delta, int64_t delta_row_stride, int rows, int groups,
                                     int group_size, int block, cudaStream_t stream) {
  using namespace seventy;
  if (out == nullptr || x == nullptr || base == nullptr || delta == nullptr || rows < 0 || groups <= 0 ||
      group_size <= 0 || block <= 0)
    return SEVENTY_ERR_INVALID_ARGUMENT;
  if (rows == 0) return SEVENTY_OK;
  const int channels = groups * group_size;
  drafter_conv_kernel<<<dim3((channels + 255) / 256, rows), 256, 0, stream>>>(
      out, out_fp32, x, x_fp32, base, delta, delta_row_stride, groups, group_size, channels, block);
  return launch_status();
}

SEVENTY_API int seventy_drafter_qk(void *q_out, void *k_ring, void *v_ring, const float *q, int64_t q_row_stride,
                                   const float *kv, int64_t kv_row_stride, const float *q_norm, const float *k_norm,
                                   const int64_t *positions, const int64_t *slots, const float *inv_freq, int rows,
                                   int heads, int kv_heads, int head_dim, int ring, float eps, cudaStream_t stream) {
  using namespace seventy;
  if (k_ring == nullptr || v_ring == nullptr || kv == nullptr || k_norm == nullptr || positions == nullptr ||
      slots == nullptr || inv_freq == nullptr || head_dim != kD || rows < 0 || kv_heads <= 0 || ring <= 0 ||
      (q_out != nullptr && (q == nullptr || q_norm == nullptr || heads <= 0)))
    return SEVENTY_ERR_INVALID_ARGUMENT;
  if (rows == 0) return SEVENTY_OK;
  drafter_qk_kernel<<<rows, 256, 0, stream>>>(static_cast<half *>(q_out), static_cast<half *>(k_ring),
                                              static_cast<half *>(v_ring), q, q_row_stride, kv, kv_row_stride, q_norm,
                                              k_norm, positions, slots, inv_freq, heads, kv_heads, ring, eps);
  return launch_status();
}

SEVENTY_API int seventy_drafter_attention_splits(int block, int window) {
  using namespace seventy;
  return (window + block - 1 + kKeys - 1) / kKeys;
}

SEVENTY_API int64_t seventy_drafter_attention_workspace(int seqs, int kv_heads, int splits) {
  using namespace seventy;
  return static_cast<int64_t>(seqs) * kv_heads * splits * kRows * (kD + 1) * static_cast<int64_t>(sizeof(float));
}

SEVENTY_API int seventy_drafter_attention(void *out, const void *q, const void *k_ring, const void *v_ring,
                                          const int64_t *slots, const int64_t *starts, int seqs, int heads,
                                          int kv_heads, int head_dim, int block, int window, int ring, float scale,
                                          void *workspace, int64_t workspace_bytes, cudaStream_t stream) {
  using namespace seventy;
  if (out == nullptr || q == nullptr || k_ring == nullptr || v_ring == nullptr || slots == nullptr ||
      starts == nullptr || head_dim != kD || seqs < 0 || kv_heads <= 0 || heads % kv_heads ||
      block * (heads / kv_heads) > kRows || window + block > ring)
    return SEVENTY_ERR_INVALID_ARGUMENT;
  if (seqs == 0) return SEVENTY_OK;
  const int splits = seventy_drafter_attention_splits(block, window);
  if (workspace == nullptr || workspace_bytes < seventy_drafter_attention_workspace(seqs, kv_heads, splits))
    return SEVENTY_ERR_INVALID_ARGUMENT;
  if (!aligned(q, 16) || !aligned(k_ring, 16) || !aligned(v_ring, 16)) return SEVENTY_ERR_MISALIGNED;
  static bool configured = false;
  if (!configured) {
    cudaFuncSetAttribute(drafter_attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kAttnSmem);
    configured = true;
  }
  auto *ws_o = static_cast<float *>(workspace);
  float *ws_lse = ws_o + static_cast<int64_t>(seqs) * kv_heads * splits * kRows * kD;
  drafter_attention_kernel<<<dim3(splits, kv_heads, seqs), 128, kAttnSmem, stream>>>(
      ws_o, ws_lse, static_cast<const half *>(q), static_cast<const half *>(k_ring), static_cast<const half *>(v_ring),
      slots, starts, heads, kv_heads, block, window, ring, splits, scale);
  const int rows = block * (heads / kv_heads);
  drafter_combine_kernel<<<dim3(rows, kv_heads, seqs), kD, 0, stream>>>(static_cast<half *>(out), ws_o, ws_lse,
                                                                       heads, kv_heads, block, splits);
  return launch_status();
}

SEVENTY_API int seventy_silu_mul_scale(void *out, float *scale, const float *gate_up, int rows, int inter,
                                       cudaStream_t stream) {
  using namespace seventy;
  if (out == nullptr || scale == nullptr || gate_up == nullptr || rows < 0 || inter <= 0)
    return SEVENTY_ERR_INVALID_ARGUMENT;
  if (rows == 0) return SEVENTY_OK;
  silu_mul_scale_kernel<<<rows, 512, 0, stream>>>(static_cast<half *>(out), scale, gate_up, inter);
  return launch_status();
}
