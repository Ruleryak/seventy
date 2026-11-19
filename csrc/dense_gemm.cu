#include "common.cuh"

/*
 * FP16 weight GEMM for small batches (1 to 64 rows): out[m, n] = sum_k x[m, k] * w[n, k], FP32 accumulation, FP32 or
 * FP16 output, optionally times a per-row scale (row_scale[m], for inputs scaled into the FP16 range). Same
 * structure as csrc/nvfp4_gemm.cu: a warp owns 32 output columns (mma.m8n8k4 B fragments, layout
 * there), the block stages 64-wide k steps of the activations in shared memory, and k splits go to an FP32
 * workspace that a reduce kernel adds in split order.
 *
 * Weights are repacked at load (seventy.ops.kernels.pack_dense): [n_padded / 32, k / 64, 32 lanes, 64 halves]; a
 * lane's 128 bytes are the 64 k values of its column, so a warp reads 4 KB of contiguous weights per step.
 */

namespace seventy {
namespace {

constexpr int kStepK = 64;
constexpr int kRowStride = 9;

__device__ __forceinline__ void mma_8x8x4(float (&c)[8], uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7])
      : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

template <int MT, int W>
__global__ void __launch_bounds__(W * 32) dense_gemm_kernel(void *out, int out_fp32, float *ws, const half *x,
                                                             const uint4 *wpk, int m, int n, int n_tiles, int k,
                                                             int steps_per_split, const float *row_scale) {
  __shared__ uint4 smem[2][MT * 8 * kRowStride];
  constexpr int kRows = MT * 8;
  constexpr int kLoads = (kRows * 8 + W * 32 - 1) / (W * 32);
  const int lane = threadIdx.x & 31;
  const int tile = blockIdx.x * W + (threadIdx.x >> 5);
  const bool active = tile < n_tiles;
  const int ksteps = k / kStepK;
  const int s0 = blockIdx.y * steps_per_split;
  const int s1 = min(ksteps, s0 + steps_per_split);
  const int steps = s1 - s0;
  const int quad = (lane >> 2) & 3;
  const int idx = (lane & 3) + ((lane >> 4) << 2);
  const uint4 *wp = wpk + (static_cast<int64_t>(active ? tile : 0) * ksteps + s0) * 32 * 8 + lane * 8;

  float acc[MT][8];
#pragma unroll
  for (int t = 0; t < MT; ++t)
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[t][j] = 0.0f;

  auto load_a = [&](uint4 (&r)[kLoads], int kbase) {
#pragma unroll
    for (int l = 0; l < kLoads; ++l) {
      const int e = threadIdx.x + l * W * 32;
      const int row = e >> 3;
      r[l] = make_uint4(0, 0, 0, 0);
      if (row < m && e < kRows * 8)
        r[l] = __ldg(reinterpret_cast<const uint4 *>(x + static_cast<int64_t>(row) * k + kbase + (e & 7) * 8));
    }
  };
  auto store_a = [&](uint4 *stage, const uint4 (&r)[kLoads]) {
#pragma unroll
    for (int l = 0; l < kLoads; ++l) {
      const int e = threadIdx.x + l * W * 32;
      if (e < kRows * 8) stage[(e >> 3) * kRowStride + (e & 7)] = r[l];
    }
  };

  uint4 ar[kLoads];
  load_a(ar, s0 * kStepK);
  store_a(smem[0], ar);
  __syncthreads();
  uint4 wb[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) wb[i] = active ? __ldg(wp + i) : make_uint4(0, 0, 0, 0);

  for (int s = 0; s < steps; ++s) {
    const bool more = s + 1 < steps;
    if (more) load_a(ar, (s0 + s + 1) * kStepK);
    uint4 cur[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) cur[i] = wb[i];
    if (active && more) {
#pragma unroll
      for (int i = 0; i < 8; ++i) wb[i] = __ldg(wp + static_cast<int64_t>(s + 1) * 32 * 8 + i);
    }
    if (active) {
      const uint4 *stage = smem[s & 1];
#pragma unroll
      for (int i = 0; i < 8; ++i) {
#pragma unroll
        for (int t = 0; t < MT; ++t) {
          const uint4 a = stage[(t * 8 + idx) * kRowStride + i];
          mma_8x8x4(acc[t], a.x, a.y, cur[i].x, cur[i].y);
          mma_8x8x4(acc[t], a.z, a.w, cur[i].z, cur[i].w);
        }
      }
    }
    if (more) store_a(smem[(s + 1) & 1], ar);
    __syncthreads();
  }
  if (!active) return;
  const int n_padded = n_tiles * 32;
#pragma unroll
  for (int t = 0; t < MT; ++t)
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int row = t * 8 + (idx & 1) + 2 * ((j >> 1) & 1) + 4 * (idx >> 2);
      const int col = tile * 32 + quad * 8 + (j & 1) + 2 * ((idx >> 1) & 1) + 4 * (j >> 2);
      if (row >= m) continue;
      if (gridDim.y == 1) {
        if (col < n) {
          const float v = row_scale != nullptr ? acc[t][j] * row_scale[row] : acc[t][j];
          if (out_fp32) static_cast<float *>(out)[static_cast<int64_t>(row) * n + col] = v;
          else static_cast<half *>(out)[static_cast<int64_t>(row) * n + col] = __float2half_rn(v);
        }
      } else {
        ws[(static_cast<int64_t>(blockIdx.y) * m + row) * n_padded + col] = acc[t][j];
      }
    }
}

/*
 * dense_gemm_fp8_kernel: the same GEMM with FP8 E4M3 weights and a per-column scale (lossy weights: used for the
 * speculative drafter only). Weights are packed like the FP16 ones but one byte per value, [n_padded / 32, k / 64,
 * 32 lanes, 64 bytes]. E4M3 to FP16 without a table: with the byte in the high byte of a half, (h >> 1) keeps the
 * exponent and mantissa (bits 13..7) and the sign stays in bit 15, which is the E4M3 value times 2^-8 exactly for
 * every finite value (subnormals included); col_scale holds the channel scale times 2^8.
 */
__device__ __forceinline__ uint32_t e4m3x2_to_half2(uint32_t word, uint32_t sel) {
  const uint32_t v = __byte_perm(word, 0u, sel);
  return ((v >> 1) & 0x3F803F80u) | (v & 0x80008000u);
}

template <int MT, int W>
__global__ void __launch_bounds__(W * 32) dense_gemm_fp8_kernel(void *out, int out_fp32, float *ws, const half *x,
                                                                 const uint4 *wpk, const float *col_scale, int m,
                                                                 int n, int n_tiles, int k, int steps_per_split,
                                                                 const float *row_scale) {
  __shared__ uint4 smem[2][MT * 8 * kRowStride];
  constexpr int kRows = MT * 8;
  constexpr int kLoads = (kRows * 8 + W * 32 - 1) / (W * 32);
  const int lane = threadIdx.x & 31;
  const int tile = blockIdx.x * W + (threadIdx.x >> 5);
  const bool active = tile < n_tiles;
  const int ksteps = k / kStepK;
  const int s0 = blockIdx.y * steps_per_split;
  const int s1 = min(ksteps, s0 + steps_per_split);
  const int steps = s1 - s0;
  const int quad = (lane >> 2) & 3;
  const int idx = (lane & 3) + ((lane >> 4) << 2);
  const uint4 *wp = wpk + (static_cast<int64_t>(active ? tile : 0) * ksteps + s0) * 32 * 4 + lane * 4;

  float acc[MT][8];
#pragma unroll
  for (int t = 0; t < MT; ++t)
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[t][j] = 0.0f;

  auto load_a = [&](uint4 (&r)[kLoads], int kbase) {
#pragma unroll
    for (int l = 0; l < kLoads; ++l) {
      const int e = threadIdx.x + l * W * 32;
      const int row = e >> 3;
      r[l] = make_uint4(0, 0, 0, 0);
      if (row < m && e < kRows * 8)
        r[l] = __ldg(reinterpret_cast<const uint4 *>(x + static_cast<int64_t>(row) * k + kbase + (e & 7) * 8));
    }
  };
  auto store_a = [&](uint4 *stage, const uint4 (&r)[kLoads]) {
#pragma unroll
    for (int l = 0; l < kLoads; ++l) {
      const int e = threadIdx.x + l * W * 32;
      if (e < kRows * 8) stage[(e >> 3) * kRowStride + (e & 7)] = r[l];
    }
  };

  uint4 ar[kLoads];
  load_a(ar, s0 * kStepK);
  store_a(smem[0], ar);
  __syncthreads();
  uint4 wb[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) wb[i] = active ? __ldg(wp + i) : make_uint4(0, 0, 0, 0);

  for (int s = 0; s < steps; ++s) {
    const bool more = s + 1 < steps;
    if (more) load_a(ar, (s0 + s + 1) * kStepK);
    uint4 cur[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) cur[i] = wb[i];
    if (active && more) {
#pragma unroll
      for (int i = 0; i < 4; ++i) wb[i] = __ldg(wp + static_cast<int64_t>(s + 1) * 32 * 4 + i);
    }
    if (active) {
      const uint4 *stage = smem[s & 1];
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const uint32_t lo = (i & 1) ? cur[i >> 1].z : cur[i >> 1].x;
        const uint32_t hi = (i & 1) ? cur[i >> 1].w : cur[i >> 1].y;
        const uint32_t b0 = e4m3x2_to_half2(lo, 0x1404u), b1 = e4m3x2_to_half2(lo, 0x3424u);
        const uint32_t b2 = e4m3x2_to_half2(hi, 0x1404u), b3 = e4m3x2_to_half2(hi, 0x3424u);
#pragma unroll
        for (int t = 0; t < MT; ++t) {
          const uint4 a = stage[(t * 8 + idx) * kRowStride + i];
          mma_8x8x4(acc[t], a.x, a.y, b0, b1);
          mma_8x8x4(acc[t], a.z, a.w, b2, b3);
        }
      }
    }
    if (more) store_a(smem[(s + 1) & 1], ar);
    __syncthreads();
  }
  if (!active) return;
  const int n_padded = n_tiles * 32;
#pragma unroll
  for (int t = 0; t < MT; ++t)
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int row = t * 8 + (idx & 1) + 2 * ((j >> 1) & 1) + 4 * (idx >> 2);
      const int col = tile * 32 + quad * 8 + (j & 1) + 2 * ((idx >> 1) & 1) + 4 * (j >> 2);
      if (row >= m) continue;
      if (gridDim.y == 1) {
        if (col < n) {
          float v = acc[t][j] * col_scale[col];
          if (row_scale != nullptr) v *= row_scale[row];
          if (out_fp32) static_cast<float *>(out)[static_cast<int64_t>(row) * n + col] = v;
          else static_cast<half *>(out)[static_cast<int64_t>(row) * n + col] = __float2half_rn(v);
        }
      } else {
        ws[(static_cast<int64_t>(blockIdx.y) * m + row) * n_padded + col] = acc[t][j];
      }
    }
}

__global__ void dense_reduce_scaled_kernel(void *out, int out_fp32, const float *ws, const float *col_scale, int m,
                                           int n, int n_padded, int splits, const float *row_scale) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= static_cast<int64_t>(m) * n) return;
  const int row = static_cast<int>(idx / n);
  const int col = static_cast<int>(idx % n);
  float s = 0.0f;
  for (int p = 0; p < splits; ++p) s += ws[(static_cast<int64_t>(p) * m + row) * n_padded + col];
  s *= col_scale[col];
  if (row_scale != nullptr) s *= row_scale[row];
  if (out_fp32) static_cast<float *>(out)[idx] = s;
  else static_cast<half *>(out)[idx] = __float2half_rn(s);
}

template <int MT>
int launch_fp8(int warps, void *out, int out_fp32, float *ws, const half *x, const uint4 *wpk, const float *cs, int m,
               int n, int n_tiles, int k, int splits, const float *row_scale, cudaStream_t stream) {
  const int ksteps = k / kStepK;
  const int per = (ksteps + splits - 1) / splits;
  const int used = (ksteps + per - 1) / per;
  if (warps >= 8) {
    dense_gemm_fp8_kernel<MT, 8><<<dim3((n_tiles + 7) / 8, used), 256, 0, stream>>>(out, out_fp32, ws, x, wpk, cs, m,
                                                                                      n, n_tiles, k, per, row_scale);
  } else {
    dense_gemm_fp8_kernel<MT, 4><<<dim3((n_tiles + 3) / 4, used), 128, 0, stream>>>(out, out_fp32, ws, x, wpk, cs, m,
                                                                                      n, n_tiles, k, per, row_scale);
  }
  return used;
}

__global__ void dense_reduce_kernel(void *out, int out_fp32, const float *ws, int m, int n, int n_padded, int splits,
                                    const float *row_scale) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= static_cast<int64_t>(m) * n) return;
  const int row = static_cast<int>(idx / n);
  const int col = static_cast<int>(idx % n);
  float s = 0.0f;
  for (int p = 0; p < splits; ++p) s += ws[(static_cast<int64_t>(p) * m + row) * n_padded + col];
  if (row_scale != nullptr) s *= row_scale[row];
  if (out_fp32) static_cast<float *>(out)[idx] = s;
  else static_cast<half *>(out)[idx] = __float2half_rn(s);
}

template <int MT>
int launch(int warps, void *out, int out_fp32, float *ws, const half *x, const uint4 *wpk, int m, int n, int n_tiles,
           int k, int splits, const float *row_scale, cudaStream_t stream) {
  const int ksteps = k / kStepK;
  const int per = (ksteps + splits - 1) / splits;
  const int used = (ksteps + per - 1) / per;
  if (warps >= 8) {
    dense_gemm_kernel<MT, 8><<<dim3((n_tiles + 7) / 8, used), 256, 0, stream>>>(out, out_fp32, ws, x, wpk, m, n,
                                                                                  n_tiles, k, per, row_scale);
  } else {
    dense_gemm_kernel<MT, 4><<<dim3((n_tiles + 3) / 4, used), 128, 0, stream>>>(out, out_fp32, ws, x, wpk, m, n,
                                                                                  n_tiles, k, per, row_scale);
  }
  return used;
}

}  // namespace
}  // namespace seventy

SEVENTY_API int64_t seventy_dense_gemm_workspace(int m, int n_padded, int splits) {
  return splits > 1 ? static_cast<int64_t>(splits) * m * n_padded * static_cast<int64_t>(sizeof(float)) : 0;
}

SEVENTY_API int seventy_dense_gemm(void *out, int out_fp32, const void *x, const void *wpk, int m, int n, int n_padded,
                                   int k, int splits, int warps, void *workspace, int64_t workspace_bytes,
                                   const float *row_scale, cudaStream_t stream) {
  using namespace seventy;
  if (out == nullptr || x == nullptr || wpk == nullptr || m <= 0 || m > 64 || n <= 0 || n_padded < n ||
      n_padded % 32 || k <= 0 || k % kStepK || splits < 1)
    return SEVENTY_ERR_INVALID_ARGUMENT;
  const int ksteps = k / kStepK;
  const int per = (ksteps + splits - 1) / splits;
  const int used = (ksteps + per - 1) / per;
  if (used > 1 && (workspace == nullptr || workspace_bytes < seventy_dense_gemm_workspace(m, n_padded, used)))
    return SEVENTY_ERR_INVALID_ARGUMENT;
  if (!aligned(x, 16) || !aligned(wpk, 16)) return SEVENTY_ERR_MISALIGNED;
  const int n_tiles = n_padded / 32;
  auto *ws = static_cast<float *>(workspace);
  const auto *xh = static_cast<const half *>(x);
  const auto *wp = static_cast<const uint4 *>(wpk);
  int got = 0;
  switch ((m + 7) / 8) {
    case 1: got = launch<1>(warps, out, out_fp32, ws, xh, wp, m, n, n_tiles, k, used, row_scale, stream); break;
    case 2: got = launch<2>(warps, out, out_fp32, ws, xh, wp, m, n, n_tiles, k, used, row_scale, stream); break;
    case 3: got = launch<3>(warps, out, out_fp32, ws, xh, wp, m, n, n_tiles, k, used, row_scale, stream); break;
    case 4: got = launch<4>(warps, out, out_fp32, ws, xh, wp, m, n, n_tiles, k, used, row_scale, stream); break;
    case 5: got = launch<5>(warps, out, out_fp32, ws, xh, wp, m, n, n_tiles, k, used, row_scale, stream); break;
    case 6: got = launch<6>(warps, out, out_fp32, ws, xh, wp, m, n, n_tiles, k, used, row_scale, stream); break;
    case 7: got = launch<7>(warps, out, out_fp32, ws, xh, wp, m, n, n_tiles, k, used, row_scale, stream); break;
    default: got = launch<8>(warps, out, out_fp32, ws, xh, wp, m, n, n_tiles, k, used, row_scale, stream); break;
  }
  if (got > 1) {
    const int64_t total = static_cast<int64_t>(m) * n;
    dense_reduce_kernel<<<static_cast<unsigned>((total + 255) / 256), 256, 0, stream>>>(out, out_fp32, ws, m, n,
                                                                                       n_padded, got, row_scale);
  }
  return launch_status();
}

SEVENTY_API int seventy_dense_gemm_fp8(void *out, int out_fp32, const void *x, const void *wpk, const float *col_scale,
                                       int m, int n, int n_padded, int k, int splits, int warps, void *workspace,
                                       int64_t workspace_bytes, const float *row_scale, cudaStream_t stream) {
  using namespace seventy;
  if (out == nullptr || x == nullptr || wpk == nullptr || col_scale == nullptr || m <= 0 || m > 64 || n <= 0 ||
      n_padded < n || n_padded % 32 || k <= 0 || k % kStepK || splits < 1)
    return SEVENTY_ERR_INVALID_ARGUMENT;
  const int ksteps = k / kStepK;
  const int per = (ksteps + splits - 1) / splits;
  const int used = (ksteps + per - 1) / per;
  if (used > 1 && (workspace == nullptr || workspace_bytes < seventy_dense_gemm_workspace(m, n_padded, used)))
    return SEVENTY_ERR_INVALID_ARGUMENT;
  if (!aligned(x, 16) || !aligned(wpk, 16)) return SEVENTY_ERR_MISALIGNED;
  const int n_tiles = n_padded / 32;
  auto *ws = static_cast<float *>(workspace);
  const auto *xh = static_cast<const half *>(x);
  const auto *wp = static_cast<const uint4 *>(wpk);
  int got = 0;
  switch ((m + 7) / 8) {
    case 1: got = launch_fp8<1>(warps, out, out_fp32, ws, xh, wp, col_scale, m, n, n_tiles, k, used, row_scale, stream); break;
    case 2: got = launch_fp8<2>(warps, out, out_fp32, ws, xh, wp, col_scale, m, n, n_tiles, k, used, row_scale, stream); break;
    case 3: got = launch_fp8<3>(warps, out, out_fp32, ws, xh, wp, col_scale, m, n, n_tiles, k, used, row_scale, stream); break;
    case 4: got = launch_fp8<4>(warps, out, out_fp32, ws, xh, wp, col_scale, m, n, n_tiles, k, used, row_scale, stream); break;
    case 5: got = launch_fp8<5>(warps, out, out_fp32, ws, xh, wp, col_scale, m, n, n_tiles, k, used, row_scale, stream); break;
    case 6: got = launch_fp8<6>(warps, out, out_fp32, ws, xh, wp, col_scale, m, n, n_tiles, k, used, row_scale, stream); break;
    case 7: got = launch_fp8<7>(warps, out, out_fp32, ws, xh, wp, col_scale, m, n, n_tiles, k, used, row_scale, stream); break;
    default: got = launch_fp8<8>(warps, out, out_fp32, ws, xh, wp, col_scale, m, n, n_tiles, k, used, row_scale, stream); break;
  }
  if (got > 1) {
    const int64_t total = static_cast<int64_t>(m) * n;
    dense_reduce_scaled_kernel<<<static_cast<unsigned>((total + 255) / 256), 256, 0, stream>>>(
        out, out_fp32, ws, col_scale, m, n, n_padded, got, row_scale);
  }
  return launch_status();
}
