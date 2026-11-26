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
 *
 * With one split, dense_gemm_wk_kernel runs instead (the structure of nvfp4_gemm_wk_kernel, csrc/nvfp4_gemm.cu):
 * one block of W warps per 32-column tile, the warps split k into contiguous ranges of 64-steps, each warp stages its
 * own activations and prefetches the next step's weights and activations into registers, and the warps add their
 * partial tiles in shared memory at the end: no workspace, no reduce kernel and no block barrier in the main loop.
 *
 * FP8 weights use dense_gemm_fp8_wk_kernel, the same structure with k also split over gridDim.y blocks for narrow
 * weights: each block writes its partial tile to the workspace, and the last block of a tile (a counter per tile,
 * which that block sets back to zero) adds the partials in split order. The sum is thus the same whichever block
 * ends last, and no reduce kernel runs.
 */

namespace seventy {
namespace {

constexpr int kStepK = 64;
constexpr int kRowStride = 9;

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
          mma_m8n8k4(acc[t], a.x, a.y, cur[i].x, cur[i].y);
          mma_m8n8k4(acc[t], a.z, a.w, cur[i].z, cur[i].w);
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
 * every finite value (subnormals included); col_scale holds the channel scale times 2^8. In three instructions: a
 * PRMT that puts byte b in the low byte of a half and copies of its sign bit in the high byte (selector 0x9180 for
 * bytes 0 and 1, 0xB3A2 for bytes 2 and 3), a shift left by 7 and a mask that keeps the sign and bits 13..7.
 */
__device__ __forceinline__ uint32_t e4m3x2_to_half2(uint32_t word, uint32_t sel) {
  uint32_t v;
  asm("prmt.b32 %0, %1, 0, %2;" : "=r"(v) : "r"(word), "r"(sel));
  return (v << 7) & 0xBF80BF80u;
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
        const uint32_t b0 = e4m3x2_to_half2(lo, 0x9180u), b1 = e4m3x2_to_half2(lo, 0xB3A2u);
        const uint32_t b2 = e4m3x2_to_half2(hi, 0x9180u), b3 = e4m3x2_to_half2(hi, 0xB3A2u);
#pragma unroll
        for (int t = 0; t < MT; ++t) {
          const uint4 a = stage[(t * 8 + idx) * kRowStride + i];
          mma_m8n8k4(acc[t], a.x, a.y, b0, b1);
          mma_m8n8k4(acc[t], a.z, a.w, b2, b3);
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

template <int MT, int W>
__global__ void __launch_bounds__(W * 32) dense_gemm_wk_kernel(void *out, int out_fp32, const half *x,
                                                               const uint4 *wpk, int m, int n, int k,
                                                               const float *row_scale) {
  constexpr int kRows = MT * 8;
  constexpr int kLoads = kRows * 8 / 32;
  extern __shared__ uint4 dwk_smem[];
  auto stage = reinterpret_cast<uint4 (*)[kRows * kRowStride]>(dwk_smem);
  float *red = reinterpret_cast<float *>(dwk_smem);
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int tile = blockIdx.x;
  const int ksteps = k / kStepK;
  const int s0 = warp * ksteps / W;
  const int s1 = (warp + 1) * ksteps / W;
  const int steps = s1 - s0;
  const int quad = (lane >> 2) & 3;
  const int idx = (lane & 3) + ((lane >> 4) << 2);
  const uint4 *wp = wpk + (static_cast<int64_t>(tile) * ksteps + s0) * 32 * 8 + lane * 8;
  const uint4 *xbase = reinterpret_cast<const uint4 *>(x) + s0 * 8 + (lane & 7);
  int xrow[kLoads];
#pragma unroll
  for (int l = 0; l < kLoads; ++l) xrow[l] = min((lane >> 3) + 4 * l, m - 1) * (k / 8);
  const int dst0 = (lane >> 3) * kRowStride + (lane & 7);

  float acc[MT][8];
#pragma unroll
  for (int t = 0; t < MT; ++t)
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[t][j] = 0.0f;

  uint4 *mine = stage[warp];
  uint4 ar[kLoads];
  uint4 wb[8];
  if (steps > 0) {
#pragma unroll
    for (int l = 0; l < kLoads; ++l) mine[dst0 + 4 * kRowStride * l] = __ldg(xbase + xrow[l]);
#pragma unroll
    for (int i = 0; i < 8; ++i) wb[i] = __ldg(wp + i);
  }
  __syncwarp();
  for (int s = 0; s < steps; ++s) {
    const bool more = s + 1 < steps;
    uint4 cur[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) cur[i] = wb[i];
    if (more) {
#pragma unroll
      for (int l = 0; l < kLoads; ++l) ar[l] = __ldg(xbase + xrow[l] + (s + 1) * 8);
#pragma unroll
      for (int i = 0; i < 8; ++i) wb[i] = __ldg(wp + static_cast<int64_t>(s + 1) * 32 * 8 + i);
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
#pragma unroll
      for (int t = 0; t < MT; ++t) {
        const uint4 a = mine[(t * 8 + idx) * kRowStride + i];
        mma_m8n8k4(acc[t], a.x, a.y, cur[i].x, cur[i].y);
        mma_m8n8k4(acc[t], a.z, a.w, cur[i].z, cur[i].w);
      }
    }
    __syncwarp();
    if (more) {
#pragma unroll
      for (int l = 0; l < kLoads; ++l) mine[dst0 + 4 * kRowStride * l] = ar[l];
    }
    __syncwarp();
  }

  __syncthreads();
  if (warp > 0) {
#pragma unroll
    for (int t = 0; t < MT; ++t)
#pragma unroll
      for (int j = 0; j < 8; ++j) red[((warp - 1) * kRows * 8 + t * 64 + j * 8) * 4 + lane] = acc[t][j];
  }
  __syncthreads();
  if (warp > 0) return;
#pragma unroll
  for (int t = 0; t < MT; ++t)
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      float v = acc[t][j];
#pragma unroll
      for (int w = 0; w < W - 1; ++w) v += red[(w * kRows * 8 + t * 64 + j * 8) * 4 + lane];
      const int row = t * 8 + (idx & 1) + 2 * ((j >> 1) & 1) + 4 * (idx >> 2);
      const int col = tile * 32 + quad * 8 + (j & 1) + 2 * ((idx >> 1) & 1) + 4 * (j >> 2);
      if (row >= m || col >= n) continue;
      if (row_scale != nullptr) v *= row_scale[row];
      if (out_fp32) static_cast<float *>(out)[static_cast<int64_t>(row) * n + col] = v;
      else static_cast<half *>(out)[static_cast<int64_t>(row) * n + col] = __float2half_rn(v);
    }
}

template <int MT, int W>
void launch_wk(void *out, int out_fp32, const half *x, const uint4 *wpk, int m, int n, int n_tiles, int k,
               const float *row_scale, cudaStream_t stream) {
  constexpr int kBytes = W * MT * 8 * kRowStride * static_cast<int>(sizeof(uint4));
  static_assert(kBytes >= (W - 1) * MT * 8 * 32 * static_cast<int>(sizeof(float)), "reduction must fit the staging area");
  if constexpr (kBytes > 48 * 1024) {
    static bool configured = false;
    if (!configured) {
      cudaFuncSetAttribute(dense_gemm_wk_kernel<MT, W>, cudaFuncAttributeMaxDynamicSharedMemorySize, kBytes);
      configured = true;
    }
  }
  dense_gemm_wk_kernel<MT, W><<<n_tiles, W * 32, kBytes, stream>>>(out, out_fp32, x, wpk, m, n, k, row_scale);
}

template <int MT, int W>
__global__ void __launch_bounds__(W * 32) dense_gemm_fp8_wk_kernel(void *out, int out_fp32, const half *x,
                                                                   const uint4 *wpk, const float *col_scale, int m,
                                                                   int n, int k, const float *row_scale, float *ws,
                                                                   int *counters) {
  constexpr int kRows = MT * 8;
  constexpr int kLoads = kRows * 8 / 32;
  extern __shared__ uint4 fwk_smem[];
  auto stage = reinterpret_cast<uint4 (*)[kRows * kRowStride]>(fwk_smem);
  float *red = reinterpret_cast<float *>(fwk_smem);
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int tile = blockIdx.x;
  const int ksteps = k / kStepK;
  const int splits = gridDim.y;
  const int q0 = blockIdx.y * ksteps / splits;
  const int q1 = (blockIdx.y + 1) * ksteps / splits;
  const int s0 = q0 + warp * (q1 - q0) / W;
  const int s1 = q0 + (warp + 1) * (q1 - q0) / W;
  const int steps = s1 - s0;
  const int quad = (lane >> 2) & 3;
  const int idx = (lane & 3) + ((lane >> 4) << 2);
  const uint4 *wp = wpk + (static_cast<int64_t>(tile) * ksteps + s0) * 32 * 4 + lane * 4;
  const uint4 *xbase = reinterpret_cast<const uint4 *>(x) + s0 * 8 + (lane & 7);
  int xrow[kLoads];
#pragma unroll
  for (int l = 0; l < kLoads; ++l) xrow[l] = min((lane >> 3) + 4 * l, m - 1) * (k / 8);
  const int dst0 = (lane >> 3) * kRowStride + (lane & 7);

  float acc[MT][8];
#pragma unroll
  for (int t = 0; t < MT; ++t)
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[t][j] = 0.0f;

  uint4 *mine = stage[warp];
  uint4 ar[kLoads];
  uint4 wb[4];
  if (steps > 0) {
#pragma unroll
    for (int l = 0; l < kLoads; ++l) mine[dst0 + 4 * kRowStride * l] = __ldg(xbase + xrow[l]);
#pragma unroll
    for (int i = 0; i < 4; ++i) wb[i] = __ldg(wp + i);
  }
  __syncwarp();
  for (int s = 0; s < steps; ++s) {
    const bool more = s + 1 < steps;
    uint4 cur[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) cur[i] = wb[i];
    if (more) {
#pragma unroll
      for (int l = 0; l < kLoads; ++l) ar[l] = __ldg(xbase + xrow[l] + (s + 1) * 8);
#pragma unroll
      for (int i = 0; i < 4; ++i) wb[i] = __ldg(wp + static_cast<int64_t>(s + 1) * 32 * 4 + i);
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const uint32_t lo = (i & 1) ? cur[i >> 1].z : cur[i >> 1].x;
      const uint32_t hi = (i & 1) ? cur[i >> 1].w : cur[i >> 1].y;
      const uint32_t b0 = e4m3x2_to_half2(lo, 0x9180u), b1 = e4m3x2_to_half2(lo, 0xB3A2u);
      const uint32_t b2 = e4m3x2_to_half2(hi, 0x9180u), b3 = e4m3x2_to_half2(hi, 0xB3A2u);
#pragma unroll
      for (int t = 0; t < MT; ++t) {
        const uint4 a = mine[(t * 8 + idx) * kRowStride + i];
        mma_m8n8k4(acc[t], a.x, a.y, b0, b1);
        mma_m8n8k4(acc[t], a.z, a.w, b2, b3);
      }
    }
    __syncwarp();
    if (more) {
#pragma unroll
      for (int l = 0; l < kLoads; ++l) mine[dst0 + 4 * kRowStride * l] = ar[l];
    }
    __syncwarp();
  }

  __syncthreads();
  if (warp > 0) {
#pragma unroll
    for (int t = 0; t < MT; ++t)
#pragma unroll
      for (int j = 0; j < 8; ++j) red[(warp - 1) * kRows * 32 + t * 256 + j * 32 + lane] = acc[t][j];
  }
  __syncthreads();
  if (warp > 0) return;
#pragma unroll
  for (int t = 0; t < MT; ++t)
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
      for (int w = 0; w < W - 1; ++w) acc[t][j] += red[w * kRows * 32 + t * 256 + j * 32 + lane];
  if (splits > 1) {
    float *part = ws + (static_cast<int64_t>(blockIdx.y) * gridDim.x + tile) * (kRows * 32);
#pragma unroll
    for (int t = 0; t < MT; ++t)
#pragma unroll
      for (int j = 0; j < 8; ++j) part[t * 256 + j * 32 + lane] = acc[t][j];
    __threadfence();
    __syncwarp();
    int last = 0;
    if (lane == 0) last = atomicAdd(counters + tile, 1) == splits - 1;
    if (!__shfl_sync(0xffffffffu, last, 0)) return;
    __threadfence();
#pragma unroll
    for (int t = 0; t < MT; ++t)
#pragma unroll
      for (int j = 0; j < 8; ++j) acc[t][j] = 0.0f;
    for (int p = 0; p < splits; ++p) {
      const float *src = ws + (static_cast<int64_t>(p) * gridDim.x + tile) * (kRows * 32);
#pragma unroll
      for (int t = 0; t < MT; ++t)
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[t][j] += __ldcg(src + t * 256 + j * 32 + lane);
    }
    if (lane == 0) counters[tile] = 0;
  }
#pragma unroll
  for (int t = 0; t < MT; ++t)
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int row = t * 8 + (idx & 1) + 2 * ((j >> 1) & 1) + 4 * (idx >> 2);
      const int col = tile * 32 + quad * 8 + (j & 1) + 2 * ((idx >> 1) & 1) + 4 * (j >> 2);
      if (row >= m || col >= n) continue;
      float v = acc[t][j] * col_scale[col];
      if (row_scale != nullptr) v *= row_scale[row];
      if (out_fp32) static_cast<float *>(out)[static_cast<int64_t>(row) * n + col] = v;
      else static_cast<half *>(out)[static_cast<int64_t>(row) * n + col] = __float2half_rn(v);
    }
}

template <int MT, int W>
void launch_fp8_wk(void *out, int out_fp32, const half *x, const uint4 *wpk, const float *cs, int m, int n,
                   int n_tiles, int k, int splits, const float *row_scale, float *ws, int *counters,
                   cudaStream_t stream) {
  constexpr int kBytes = W * MT * 8 * kRowStride * static_cast<int>(sizeof(uint4));
  static_assert(kBytes >= (W - 1) * MT * 8 * 32 * static_cast<int>(sizeof(float)), "reduction must fit the staging area");
  if constexpr (kBytes > 48 * 1024) {
    static bool configured = false;
    if (!configured) {
      cudaFuncSetAttribute(dense_gemm_fp8_wk_kernel<MT, W>, cudaFuncAttributeMaxDynamicSharedMemorySize, kBytes);
      configured = true;
    }
  }
  dense_gemm_fp8_wk_kernel<MT, W><<<dim3(n_tiles, splits), W * 32, kBytes, stream>>>(out, out_fp32, x, wpk, cs, m, n, k,
                                                                                     row_scale, ws, counters);
}

template <int MT>
void launch_fp8_wk_warps(int warps, void *out, int out_fp32, const half *x, const uint4 *wpk, const float *cs, int m,
                         int n, int n_tiles, int k, int splits, const float *row_scale, float *ws, int *counters,
                         cudaStream_t stream) {
  if (warps >= 8) launch_fp8_wk<MT, 8>(out, out_fp32, x, wpk, cs, m, n, n_tiles, k, splits, row_scale, ws, counters, stream);
  else if (warps >= 4) launch_fp8_wk<MT, 4>(out, out_fp32, x, wpk, cs, m, n, n_tiles, k, splits, row_scale, ws, counters, stream);
  else launch_fp8_wk<MT, 2>(out, out_fp32, x, wpk, cs, m, n, n_tiles, k, splits, row_scale, ws, counters, stream);
}

template <int MT>
int launch(int warps, void *out, int out_fp32, float *ws, const half *x, const uint4 *wpk, int m, int n, int n_tiles,
           int k, int splits, const float *row_scale, cudaStream_t stream) {
  if (splits == 1 && k / kStepK >= warps) {
    if (warps >= 8) launch_wk<MT, 8>(out, out_fp32, x, wpk, m, n, n_tiles, k, row_scale, stream);
    else if (warps >= 4) launch_wk<MT, 4>(out, out_fp32, x, wpk, m, n, n_tiles, k, row_scale, stream);
    else launch_wk<MT, 2>(out, out_fp32, x, wpk, m, n, n_tiles, k, row_scale, stream);
    return 1;
  }
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

SEVENTY_API int64_t seventy_dense_gemm_fp8_wk_workspace(int m, int n_padded, int splits) {
  return splits > 1 ? static_cast<int64_t>(splits) * (n_padded / 32) * ((m + 7) / 8) * 256 *
                          static_cast<int64_t>(sizeof(float))
                    : 0;
}

SEVENTY_API int seventy_dense_gemm_fp8_wk(void *out, int out_fp32, const void *x, const void *wpk,
                                          const float *col_scale, int m, int n, int n_padded, int k, int splits,
                                          int warps, void *workspace, int64_t workspace_bytes, int *counters,
                                          const float *row_scale, cudaStream_t stream) {
  using namespace seventy;
  if (out == nullptr || x == nullptr || wpk == nullptr || col_scale == nullptr || m <= 0 || m > 64 || n <= 0 ||
      n_padded < n || n_padded % 32 || k <= 0 || k % kStepK || splits < 1 || splits > 65535 || warps < 1)
    return SEVENTY_ERR_INVALID_ARGUMENT;
  if (splits > 1 && (workspace == nullptr || counters == nullptr ||
                     workspace_bytes < seventy_dense_gemm_fp8_wk_workspace(m, n_padded, splits)))
    return SEVENTY_ERR_INVALID_ARGUMENT;
  if (!aligned(x, 16) || !aligned(wpk, 16)) return SEVENTY_ERR_MISALIGNED;
  const int n_tiles = n_padded / 32;
  auto *ws = static_cast<float *>(workspace);
  const auto *xh = static_cast<const half *>(x);
  const auto *wp = static_cast<const uint4 *>(wpk);
  switch ((m + 7) / 8) {
    case 1: launch_fp8_wk_warps<1>(warps, out, out_fp32, xh, wp, col_scale, m, n, n_tiles, k, splits, row_scale, ws, counters, stream); break;
    case 2: launch_fp8_wk_warps<2>(warps, out, out_fp32, xh, wp, col_scale, m, n, n_tiles, k, splits, row_scale, ws, counters, stream); break;
    case 3: launch_fp8_wk_warps<3>(warps, out, out_fp32, xh, wp, col_scale, m, n, n_tiles, k, splits, row_scale, ws, counters, stream); break;
    case 4: launch_fp8_wk_warps<4>(warps, out, out_fp32, xh, wp, col_scale, m, n, n_tiles, k, splits, row_scale, ws, counters, stream); break;
    case 5: launch_fp8_wk_warps<5>(warps, out, out_fp32, xh, wp, col_scale, m, n, n_tiles, k, splits, row_scale, ws, counters, stream); break;
    case 6: launch_fp8_wk_warps<6>(warps, out, out_fp32, xh, wp, col_scale, m, n, n_tiles, k, splits, row_scale, ws, counters, stream); break;
    case 7: launch_fp8_wk_warps<7>(warps, out, out_fp32, xh, wp, col_scale, m, n, n_tiles, k, splits, row_scale, ws, counters, stream); break;
    default: launch_fp8_wk_warps<8>(warps, out, out_fp32, xh, wp, col_scale, m, n, n_tiles, k, splits, row_scale, ws, counters, stream); break;
  }
  return launch_status();
}
