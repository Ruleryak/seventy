#include <mma.h>

#include "common.cuh"

/*
 * Kernels of the DFlash2 drafter (seventy/model/dflash2.py has the reference of each one):
 *
 *   drafter_norm     RMS norm with an FP32 residual stream: s = x + residual; residual = s; out = norm(s) * w.
 *                    Rows of up to 8192 values (a multiple of 4) stay in registers: 16-byte loads and stores, one
 *                    pass over x and the residual.
 *   drafter_conv     the two-tap grouped convolution inside each block of `block` rows:
 *                    out[t, c] = (base[0, c] + d[t, 0, c / gs]) x[t, c] + (base[1, c] + d[t, 1, c / gs]) x[t - 1, c],
 *                    the second term only when t is not the first row of its block.
 *   drafter_qk       per head: RMS norm (rounded to FP16), rotary (neox halves over the whole head, FP32 math), for
 *                    query heads (to q_out) and key heads (into the K ring), and V into the V ring, at ring index
 *                    position % ring.
 *   drafter_attention the block attention: each block row (token t of sequence b, at position start + t) sees the
 *                    keys at positions [max(0, start + t - window + 1), start + block) of its ring, so a block of
 *                    queries reads at most window + block - 1 keys. Splits of 64 keys run on the tensor cores
 *                    (WMMA 16 x 16 x 16, FP16 in, FP32 accumulation), one block of 4 warps each, with an exact
 *                    softmax per split. A block issues all its loads before the first store to shared memory, and
 *                    its 43.5 KB of shared memory (the scores take the place of q, P the place of K) let two
 *                    blocks share an SM. Each split writes its unnormalized output, row maximum and row sum; a
 *                    combine kernel merges the splits. (A version with several tiles per block and an online
 *                    softmax needs 74 KB of shared memory, one block of 4 warps per SM, and was no faster.)
 *   silu_mul_scale   p = silu(gate) * up in FP32, rows scaled into the FP16 range (scale = max(1, max|p| / 16384)).
 */

namespace seventy {
namespace {

using namespace nvcuda;

constexpr int kD = 128;
constexpr int kRows = 32;
constexpr int kKeys = 64;
constexpr int kLd = kD + 8;
constexpr int kLdS = kKeys + 4;
constexpr int kLdP = kKeys + 8;
constexpr int kAttnThreads = 128;

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

template <int kPer>
__global__ void __launch_bounds__(256) drafter_norm_vec_kernel(half *out, float *res_out, const void *x, int x_fp32,
                                                               const float *res_in, const float *weight, int dim,
                                                               float eps) {
  __shared__ float scratch[32];
  const int64_t row = blockIdx.x;
  const int n4 = dim / 4;
  float4 v[kPer];
  float sumsq = 0.0f;
#pragma unroll
  for (int i = 0; i < kPer; ++i) {
    const int c = threadIdx.x + i * 256;
    v[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (c < n4) {
      if (x_fp32) {
        v[i] = reinterpret_cast<const float4 *>(x)[row * n4 + c];
      } else {
        const uint2 hv = reinterpret_cast<const uint2 *>(x)[row * n4 + c];
        const float2 a = __half22float2(*reinterpret_cast<const half2 *>(&hv.x));
        const float2 b = __half22float2(*reinterpret_cast<const half2 *>(&hv.y));
        v[i] = make_float4(a.x, a.y, b.x, b.y);
      }
      if (res_in != nullptr) {
        const float4 r = reinterpret_cast<const float4 *>(res_in)[row * n4 + c];
        v[i] = make_float4(v[i].x + r.x, v[i].y + r.y, v[i].z + r.z, v[i].w + r.w);
      }
      if (res_out != nullptr) reinterpret_cast<float4 *>(res_out)[row * n4 + c] = v[i];
      sumsq += v[i].x * v[i].x + v[i].y * v[i].y + v[i].z * v[i].z + v[i].w * v[i].w;
    }
  }
  const float inv = rsqrtf(block_sum(sumsq, scratch) / dim + eps);
#pragma unroll
  for (int i = 0; i < kPer; ++i) {
    const int c = threadIdx.x + i * 256;
    if (c >= n4) continue;
    const float4 w = reinterpret_cast<const float4 *>(weight)[c];
    const half2 a = __floats2half2_rn(v[i].x * inv * w.x, v[i].y * inv * w.y);
    const half2 b = __floats2half2_rn(v[i].z * inv * w.z, v[i].w * inv * w.w);
    uint2 o;
    o.x = *reinterpret_cast<const uint32_t *>(&a);
    o.y = *reinterpret_cast<const uint32_t *>(&b);
    reinterpret_cast<uint2 *>(out)[row * n4 + c] = o;
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

__global__ void __launch_bounds__(kAttnThreads) drafter_attention_kernel(float *ws_o, float *ws_ml, const half *q,
                                                                         const half *k_ring, const half *v_ring,
                                                                         const int64_t *slots, const int64_t *starts,
                                                                         int heads, int kv_heads, int block,
                                                                         int window, int ring, int splits,
                                                                         float scale) {
  __shared__ __align__(32) half qs[kRows * kLd];
  __shared__ __align__(32) half ks[kKeys * kLd];
  __shared__ __align__(32) half vs[kKeys * kLd];
  float *ss = reinterpret_cast<float *>(qs);
  half *ps = ks;

  const int s = blockIdx.x, g = blockIdx.y, b = blockIdx.z;
  const int rep = heads / kv_heads;
  const int rows = block * rep;
  const int64_t start = starts[b];
  const int64_t end = start + block;
  const int64_t k0 = start - (window - 1) > 0 ? start - (window - 1) : 0;
  const int64_t kb = k0 + static_cast<int64_t>(s) * kKeys;
  const int n = end - kb < kKeys ? static_cast<int>(end - kb) : kKeys;
  const int64_t entries = static_cast<int64_t>(gridDim.z) * gridDim.y * gridDim.x * kRows;
  const int64_t out_base = ((static_cast<int64_t>(b) * kv_heads + g) * splits + s) * kRows;
  float *ws_m = ws_ml, *ws_l = ws_ml + entries;
  const int tid = threadIdx.x;
  if (n <= 0) {
    if (tid < kRows) {
      ws_m[out_base + tid] = -INFINITY;
      ws_l[out_base + tid] = 0.0f;
    }
    return;
  }
  const int64_t slot = slots[b];
  const int base_mod = static_cast<int>(kb % ring);

  constexpr int kQLoads = kRows * (kD / 8) / kAttnThreads;
  constexpr int kKLoads = kKeys * (kD / 8) / kAttnThreads;
  uint4 qv[kQLoads], kv[kKLoads], vv[kKLoads];
#pragma unroll
  for (int i = 0; i < kQLoads; ++i) {
    const int e = tid + i * kAttnThreads;
    const int r = e >> 4, c = (e & 15) * 8;
    qv[i] = make_uint4(0, 0, 0, 0);
    if (r < rows) {
      const int t = r / rep, h = g * rep + r % rep;
      qv[i] = *reinterpret_cast<const uint4 *>(q + ((static_cast<int64_t>(b) * block + t) * heads + h) * kD + c);
    }
  }
#pragma unroll
  for (int i = 0; i < kKLoads; ++i) {
    const int e = tid + i * kAttnThreads;
    const int j = e >> 4, c = (e & 15) * 8;
    kv[i] = vv[i] = make_uint4(0, 0, 0, 0);
    if (j < n) {
      const int idx = base_mod + j < ring ? base_mod + j : base_mod + j - ring;
      const int64_t off = ((slot * ring + idx) * kv_heads + g) * kD + c;
      kv[i] = *reinterpret_cast<const uint4 *>(k_ring + off);
      vv[i] = *reinterpret_cast<const uint4 *>(v_ring + off);
    }
  }
#pragma unroll
  for (int i = 0; i < kQLoads; ++i) {
    const int e = tid + i * kAttnThreads;
    *reinterpret_cast<uint4 *>(qs + (e >> 4) * kLd + (e & 15) * 8) = qv[i];
  }
#pragma unroll
  for (int i = 0; i < kKLoads; ++i) {
    const int e = tid + i * kAttnThreads;
    *reinterpret_cast<uint4 *>(ks + (e >> 4) * kLd + (e & 15) * 8) = kv[i];
    *reinterpret_cast<uint4 *>(vs + (e >> 4) * kLd + (e & 15) * 8) = vv[i];
  }
  __syncthreads();

  const int warp = tid >> 5, lane = tid & 31;
  {
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2];
#pragma unroll
    for (int rt = 0; rt < 2; ++rt) wmma::fill_fragment(acc[rt], 0.0f);
#pragma unroll
    for (int k = 0; k < kD / 16; ++k) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a[2];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> bf;
#pragma unroll
      for (int rt = 0; rt < 2; ++rt) wmma::load_matrix_sync(a[rt], qs + rt * 16 * kLd + k * 16, kLd);
      wmma::load_matrix_sync(bf, ks + warp * 16 * kLd + k * 16, kLd);
#pragma unroll
      for (int rt = 0; rt < 2; ++rt) wmma::mma_sync(acc[rt], a[rt], bf, acc[rt]);
    }
    __syncthreads();
#pragma unroll
    for (int rt = 0; rt < 2; ++rt)
      wmma::store_matrix_sync(ss + rt * 16 * kLdS + warp * 16, acc[rt], kLdS, wmma::mem_row_major);
  }
  __syncthreads();

  for (int r = warp; r < kRows; r += kAttnThreads / 32) {
    const int t = r / rep;
    const int64_t lo = start + t - (window - 1);
    float v[2];
    float m = -INFINITY;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int j = lane + 32 * i;
      const bool ok = r < rows && j < n && kb + j >= lo;
      v[i] = ok ? ss[r * kLdS + j] * scale : -INFINITY;
      m = fmaxf(m, v[i]);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
    float l = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const float p = m == -INFINITY ? 0.0f : __expf(v[i] - m);
      v[i] = p;
      l += p;
    }
    l = warp_sum(l);
#pragma unroll
    for (int i = 0; i < 2; ++i) ps[r * kLdP + lane + 32 * i] = __float2half_rn(v[i]);
    if (lane == 0) {
      ws_m[out_base + r] = m;
      ws_l[out_base + r] = l;
    }
  }
  __syncthreads();

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
    for (int rt = 0; rt < 2; ++rt) wmma::load_matrix_sync(a[rt], ps + rt * 16 * kLdP + k * 16, kLdP);
#pragma unroll
    for (int ct = 0; ct < 2; ++ct) wmma::load_matrix_sync(bf[ct], vs + k * 16 * kLd + warp * 32 + ct * 16, kLd);
#pragma unroll
    for (int rt = 0; rt < 2; ++rt)
#pragma unroll
      for (int ct = 0; ct < 2; ++ct) wmma::mma_sync(acc[rt][ct], a[rt], bf[ct], acc[rt][ct]);
  }
#pragma unroll
  for (int rt = 0; rt < 2; ++rt)
#pragma unroll
    for (int ct = 0; ct < 2; ++ct)
      wmma::store_matrix_sync(ws_o + (out_base + rt * 16) * kD + warp * 32 + ct * 16, acc[rt][ct], kD,
                              wmma::mem_row_major);
}

__global__ void drafter_combine_kernel(half *out, const float *ws_o, const float *ws_ml, int heads, int kv_heads,
                                       int block, int splits, int64_t entries) {
  const int g = blockIdx.y, b = blockIdx.z;
  const int rep = heads / kv_heads;
  const int r = blockIdx.x;
  const int c = threadIdx.x;
  const int64_t base = (static_cast<int64_t>(b) * kv_heads + g) * splits;
  const float *ws_m = ws_ml, *ws_l = ws_ml + entries;
  float mx = -INFINITY;
#pragma unroll 8
  for (int s = 0; s < splits; ++s) mx = fmaxf(mx, ws_m[(base + s) * kRows + r]);
  float acc = 0.0f, wsum = 0.0f;
  if (mx != -INFINITY) {
#pragma unroll 8
    for (int s = 0; s < splits; ++s) {
      const int64_t i = (base + s) * kRows + r;
      const float m = ws_m[i];
      if (m != -INFINITY) {
        const float w = __expf(m - mx);
        wsum += w * ws_l[i];
        acc += w * ws_o[i * kD + c];
      }
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

}  // namespace
}  // namespace seventy

SEVENTY_API int seventy_drafter_norm(void *out, float *residual_out, const void *x, int x_fp32,
                                     const float *residual_in, const float *weight, int rows, int dim, float eps,
                                     cudaStream_t stream) {
  using namespace seventy;
  if (out == nullptr || x == nullptr || weight == nullptr || rows < 0 || dim <= 0) return SEVENTY_ERR_INVALID_ARGUMENT;
  if (rows == 0) return SEVENTY_OK;
  const int per = (dim / 4 + 255) / 256;
  const bool vec = dim % 4 == 0 && per <= 8 && aligned(out, 8) && aligned(x, 16) && aligned(weight, 16) &&
                   (residual_in == nullptr || aligned(residual_in, 16)) &&
                   (residual_out == nullptr || aligned(residual_out, 16));
  if (!vec) {
    drafter_norm_kernel<<<rows, 512, 0, stream>>>(static_cast<half *>(out), residual_out, x, x_fp32, residual_in,
                                                  weight, dim, eps);
    return launch_status();
  }
  auto *o = static_cast<half *>(out);
  switch (per) {
    case 1: drafter_norm_vec_kernel<1><<<rows, 256, 0, stream>>>(o, residual_out, x, x_fp32, residual_in, weight, dim, eps); break;
    case 2: drafter_norm_vec_kernel<2><<<rows, 256, 0, stream>>>(o, residual_out, x, x_fp32, residual_in, weight, dim, eps); break;
    case 3: drafter_norm_vec_kernel<3><<<rows, 256, 0, stream>>>(o, residual_out, x, x_fp32, residual_in, weight, dim, eps); break;
    case 4: drafter_norm_vec_kernel<4><<<rows, 256, 0, stream>>>(o, residual_out, x, x_fp32, residual_in, weight, dim, eps); break;
    case 5: drafter_norm_vec_kernel<5><<<rows, 256, 0, stream>>>(o, residual_out, x, x_fp32, residual_in, weight, dim, eps); break;
    case 6: drafter_norm_vec_kernel<6><<<rows, 256, 0, stream>>>(o, residual_out, x, x_fp32, residual_in, weight, dim, eps); break;
    case 7: drafter_norm_vec_kernel<7><<<rows, 256, 0, stream>>>(o, residual_out, x, x_fp32, residual_in, weight, dim, eps); break;
    default: drafter_norm_vec_kernel<8><<<rows, 256, 0, stream>>>(o, residual_out, x, x_fp32, residual_in, weight, dim, eps); break;
  }
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

SEVENTY_API int seventy_drafter_attention_splits(int seqs, int kv_heads, int block, int window) {
  using namespace seventy;
  return (window + block - 1 + kKeys - 1) / kKeys;
}

SEVENTY_API int64_t seventy_drafter_attention_workspace(int seqs, int kv_heads, int splits) {
  using namespace seventy;
  return static_cast<int64_t>(seqs) * kv_heads * splits * kRows * (kD + 2) * static_cast<int64_t>(sizeof(float));
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
  const int splits = seventy_drafter_attention_splits(seqs, kv_heads, block, window);
  if (workspace == nullptr || workspace_bytes < seventy_drafter_attention_workspace(seqs, kv_heads, splits))
    return SEVENTY_ERR_INVALID_ARGUMENT;
  if (!aligned(q, 16) || !aligned(k_ring, 16) || !aligned(v_ring, 16)) return SEVENTY_ERR_MISALIGNED;
  auto *ws_o = static_cast<float *>(workspace);
  const int64_t entries = static_cast<int64_t>(seqs) * kv_heads * splits * kRows;
  float *ws_ml = ws_o + entries * kD;
  drafter_attention_kernel<<<dim3(splits, kv_heads, seqs), kAttnThreads, 0, stream>>>(
      ws_o, ws_ml, static_cast<const half *>(q), static_cast<const half *>(k_ring), static_cast<const half *>(v_ring),
      slots, starts, heads, kv_heads, block, window, ring, splits, scale);
  const int rows = block * (heads / kv_heads);
  drafter_combine_kernel<<<dim3(rows, kv_heads, seqs), kD, 0, stream>>>(static_cast<half *>(out), ws_o, ws_ml, heads,
                                                                       kv_heads, block, splits, entries);
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
