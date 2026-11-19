#include "common.cuh"

#include <cstdlib>

/*
 * NVFP4 weight GEMM for small batches: out[m, n] = alpha[n] * sum_k x[m, k] * q[n, k], with q = e2m1 * e4m3 the
 * unscaled NVFP4 weight. Volta tensor cores (mma.m8n8k4, FP16 inputs, FP32 accumulation).
 *
 * Fragment layout of mma.m8n8k4.row.col on sm70 (probed): quad-pair p = (lane >> 2) & 3, index
 * i = (lane & 3) + 4 * (lane >= 16). The thread holds row i of A (k = 0..3), column i of B (k = 0..3), and the
 * 8 accumulators c[j] at row (i & 1) + 2 * ((j >> 1) & 1) + 4 * (i >> 2), column (j & 1) + 2 * ((i >> 1) & 1)
 * + 4 * (j >> 2).
 *
 * Two kernels. nvfp4_gemm_wk (see its comment) gives each block one 32-column tile and splits k over the block's
 * warps; it runs for all sizes except wide and long ones above 40 rows (more than 200 column tiles and more than
 * 72 steps of 64 in k, such as the MLP gate_up projection), where nvfp4_gemm_kernel is as fast or faster (measured
 * on V100). Below 8 steps of k nvfp4_gemm_kernel runs too; it is described here. SEVENTY_GEMM_WK=0 selects
 * nvfp4_gemm_kernel for all sizes.
 *
 * Work split: a warp owns a tile of 32 output columns (quad-pair p: columns 8 p .. 8 p + 7) and all batch rows
 * (8 per m-tile). A block of W warps owns W adjacent tiles. blockIdx.y takes a range of k, in steps of 64.
 *
 * Weights: repacked at load (seventy.nvfp4.pack_for_gemm). Per step a lane reads two 16-byte words (the 64 codes
 * of its column) and one 4-byte word of the 4 group scales; a warp reads 1 KB of contiguous codes. Each warp keeps
 * kDepth steps of weights in flight in registers.
 *
 * Activations: each step's 64 k of all batch rows go to shared memory once per block (double buffered, row stride
 * 144 bytes so the 8 rows of a fragment load hit different banks), and every warp of the block reads them there.
 *
 * Codes become FP16 without a table: with the 4-bit code (sign, e1, e0, m) in bits 15..12 of a half,
 * (sign << 15) | (e1 e0 m << 9) is e2m1 * 2^-14 exactly. Each word holds 8 codes so that shifts by 0, 4, 8, 12
 * bring the pairs (k0, k1), (k2, k3), (k4, k5), (k6, k7) into bits 15..12 and 31..28. The E4M3 group scale
 * becomes FP16 as (byte << 7) * 2^15 = e4m3 * 2^7. Each product e2m1 * 2^-14 * e4m3 * 2^7 is exact in FP16, so
 * the tensor cores see q * 2^-7 without rounding; alpha (from pack_for_gemm) holds 2^7 / global_scale.
 *
 * Tensor-core latency: each m-tile has its own accumulators, and with fewer than 4 m-tiles the k steps rotate
 * over C = 4 / MT accumulator sets (added up at the end), so at least 4 independent chains are in flight.
 *
 * With one k split the block writes FP16 results. With more, it writes FP32 partial sums to a workspace and the
 * reduce kernel adds them in split order, so results do not depend on timing.
 */

namespace seventy {
namespace {

constexpr int kStepK = 64;
constexpr int kRowStride = 9;
constexpr int kDepth = 3;

__device__ __forceinline__ uint32_t code_pair_to_half2(uint32_t x) {
  return (x & 0x80008000u) | ((x >> 3) & 0x0E000E00u);
}

__device__ __forceinline__ uint32_t hmul2_bits(uint32_t a, __half2 s) {
  __half2 h = *reinterpret_cast<__half2 *>(&a);
  h = __hmul2(h, s);
  return *reinterpret_cast<uint32_t *>(&h);
}

__device__ __forceinline__ void mma_8x8x4(float (&c)[8], uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7])
      : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

template <int MT, int W>
struct Stage {
  static constexpr int kRows = MT * 8;
  static constexpr int kUint4 = kRows * kRowStride;
  static constexpr int kLoads = (kRows * 8 + W * 32 - 1) / (W * 32);
};

template <int MT, int W>
__device__ __forceinline__ void load_activations(uint4 (&r)[Stage<MT, W>::kLoads], const half *x, int m, int k,
                                                 int kbase) {
#pragma unroll
  for (int l = 0; l < Stage<MT, W>::kLoads; ++l) {
    const int idx = threadIdx.x + l * W * 32;
    const int row = idx >> 3;
    const int part = idx & 7;
    r[l] = make_uint4(0, 0, 0, 0);
    if (row < m && idx < Stage<MT, W>::kRows * 8)
      r[l] = __ldg(reinterpret_cast<const uint4 *>(x + static_cast<int64_t>(row) * k + kbase + part * 8));
  }
}

template <int MT, int W>
__device__ __forceinline__ void store_activations(uint4 *stage, const uint4 (&r)[Stage<MT, W>::kLoads]) {
#pragma unroll
  for (int l = 0; l < Stage<MT, W>::kLoads; ++l) {
    const int idx = threadIdx.x + l * W * 32;
    if (idx < Stage<MT, W>::kRows * 8) stage[(idx >> 3) * kRowStride + (idx & 7)] = r[l];
  }
}

template <int MT>
__device__ __forceinline__ void step_mma(float (&acc)[MT][8], const uint4 *stage, uint4 w0, uint4 w1, uint32_t sw,
                                         int idx) {
  uint32_t sbits = ((sw & 0xffu) << 7) | (((sw >> 8) & 0xffu) << 23);
  const __half2 two15 = __float2half2_rn(32768.0f);
  const __half2 s01 = __hmul2(*reinterpret_cast<__half2 *>(&sbits), two15);
  sbits = (((sw >> 16) & 0xffu) << 7) | ((sw >> 24) << 23);
  const __half2 s23 = __hmul2(*reinterpret_cast<__half2 *>(&sbits), two15);
  const uint32_t words[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
  const __half2 scales[4] = {__low2half2(s01), __high2half2(s01), __low2half2(s23), __high2half2(s23)};
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const __half2 s = scales[j >> 1];
    const uint32_t v = words[j];
    const uint32_t b0 = hmul2_bits(code_pair_to_half2(v), s);
    const uint32_t b1 = hmul2_bits(code_pair_to_half2(v << 4), s);
    const uint32_t b2 = hmul2_bits(code_pair_to_half2(v << 8), s);
    const uint32_t b3 = hmul2_bits(code_pair_to_half2(v << 12), s);
#pragma unroll
    for (int t = 0; t < MT; ++t) {
      const uint4 a = stage[(t * 8 + idx) * kRowStride + j];
      mma_8x8x4(acc[t], a.x, a.y, b0, b1);
      mma_8x8x4(acc[t], a.z, a.w, b2, b3);
    }
  }
}

template <int MT, int W>
__global__ void __launch_bounds__(W * 32) nvfp4_gemm_kernel(half *out, float *ws, const half *x, const uint4 *wpk,
                                                             const uint32_t *wsc, const float *alpha, int m, int n,
                                                             int n_tiles, int k, int steps_per_split) {
  constexpr int C = (MT >= 4) ? 1 : 4 / MT;
  __shared__ uint4 smem[2][Stage<MT, W>::kUint4];
  const int lane = threadIdx.x & 31;
  const int tile = blockIdx.x * W + (threadIdx.x >> 5);
  const bool active = tile < n_tiles;
  const int ksteps = k / kStepK;
  const int s0 = blockIdx.y * steps_per_split;
  const int s1 = min(ksteps, s0 + steps_per_split);
  const int steps = s1 - s0;
  const int quad = (lane >> 2) & 3;
  const int idx = (lane & 3) + ((lane >> 4) << 2);
  const int safe_tile = active ? tile : 0;
  const uint4 *wp = wpk + (static_cast<int64_t>(safe_tile) * (k / 32) + 2 * s0) * 32 + lane;
  const uint32_t *sp = wsc + (static_cast<int64_t>(safe_tile) * ksteps + s0) * 32 + lane;

  float acc[C][MT][8];
#pragma unroll
  for (int c = 0; c < C; ++c)
#pragma unroll
    for (int t = 0; t < MT; ++t)
#pragma unroll
      for (int j = 0; j < 8; ++j) acc[c][t][j] = 0.0f;

  uint4 wb[kDepth][2];
  uint32_t sb[kDepth];
#pragma unroll
  for (int d = 0; d < kDepth; ++d) {
    const int s = min(d, steps - 1);
    wb[d][0] = __ldg(wp + static_cast<int64_t>(2 * s) * 32);
    wb[d][1] = __ldg(wp + static_cast<int64_t>(2 * s + 1) * 32);
    sb[d] = __ldg(sp + static_cast<int64_t>(s) * 32);
  }

  uint4 ar[Stage<MT, W>::kLoads];
  load_activations<MT, W>(ar, x, m, k, s0 * kStepK);
  store_activations<MT, W>(smem[0], ar);
  __syncthreads();

  for (int base = 0; base < steps; base += kDepth) {
#pragma unroll
    for (int d = 0; d < kDepth; ++d) {
      const int s = base + d;
      if (s < steps) {
        const bool more = s + 1 < steps;
        if (more) load_activations<MT, W>(ar, x, m, k, (s0 + s + 1) * kStepK);
        if (active) step_mma<MT>(acc[d % C], smem[s & 1], wb[d][0], wb[d][1], sb[d], idx);
        const int nxt = min(s + kDepth, steps - 1);
        wb[d][0] = __ldg(wp + static_cast<int64_t>(2 * nxt) * 32);
        wb[d][1] = __ldg(wp + static_cast<int64_t>(2 * nxt + 1) * 32);
        sb[d] = __ldg(sp + static_cast<int64_t>(nxt) * 32);
        if (more) store_activations<MT, W>(smem[(s + 1) & 1], ar);
        __syncthreads();
      }
    }
  }
  if (!active) return;
#pragma unroll
  for (int c = 1; c < C; ++c)
#pragma unroll
    for (int t = 0; t < MT; ++t)
#pragma unroll
      for (int j = 0; j < 8; ++j) acc[0][t][j] += acc[c][t][j];
  const int n_padded = n_tiles * 32;
#pragma unroll
  for (int t = 0; t < MT; ++t) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int row = t * 8 + (idx & 1) + 2 * ((j >> 1) & 1) + 4 * (idx >> 2);
      const int col = tile * 32 + quad * 8 + (j & 1) + 2 * ((idx >> 1) & 1) + 4 * (j >> 2);
      if (row >= m) continue;
      if (out != nullptr) {
        if (col < n) out[static_cast<int64_t>(row) * n + col] = __float2half_rn(acc[0][t][j] * alpha[col]);
      } else {
        ws[(static_cast<int64_t>(blockIdx.y) * m + row) * n_padded + col] = acc[0][t][j];
      }
    }
  }
}

/*
 * nvfp4_gemm_wk (at least 8 k steps; selection in seventy_nvfp4_gemm): one block of W warps (8 up to 16 rows, 4
 * from 17 rows: two such blocks share an SM and each warp gets twice the k range) per 32-column tile; the warps split k (contiguous ranges of
 * 64-steps) and add their partial tiles in shared memory at the end, so there is no workspace and no reduce
 * kernel, and no block barrier in the main loop. Each warp stages its own activations: the next step's 64 k of
 * all rows are loaded into registers (coalesced, rows past m clamped to row m - 1, their outputs not written)
 * while the current step computes from the warp's shared-memory buffer (step_mma, as nvfp4_gemm_kernel), then
 * stored there after a warp barrier. On sm70 all of a warp's outstanding global loads wait together, so the
 * prefetch is covered by one whole step of tensor-core work. Weights and scales are prefetched one step ahead.
 */
template <int MT, int kW>
__global__ void __launch_bounds__(kW * 32, (MT >= 4 ? 8 / kW : 16 / kW)) nvfp4_gemm_wk_kernel(
    half *out, const half *x, const uint4 *wpk, const uint32_t *wsc, const float *alpha, int m, int n, int k) {
  constexpr int kRows = MT * 8;
  constexpr int kLoads = kRows * 8 / 32;
  extern __shared__ uint4 wk_smem[];
  auto stage = reinterpret_cast<uint4 (*)[kRows * kRowStride]>(wk_smem);
  float *red = reinterpret_cast<float *>(wk_smem);
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int tile = blockIdx.x;
  const int ksteps = k / kStepK;
  const int s0 = warp * ksteps / kW;
  const int s1 = (warp + 1) * ksteps / kW;
  const int steps = s1 - s0;
  const int quad = (lane >> 2) & 3;
  const int idx = (lane & 3) + ((lane >> 4) << 2);
  const uint4 *wp = wpk + (static_cast<int64_t>(tile) * (k / 32) + 2 * s0) * 32 + lane;
  const uint32_t *sp = wsc + (static_cast<int64_t>(tile) * ksteps + s0) * 32 + lane;
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
  uint4 w0 = make_uint4(0, 0, 0, 0), w1 = w0;
  uint32_t sw = 0;
  if (steps > 0) {
#pragma unroll
    for (int l = 0; l < kLoads; ++l) mine[dst0 + 4 * kRowStride * l] = __ldg(xbase + xrow[l]);
    w0 = __ldg(wp);
    w1 = __ldg(wp + 32);
    sw = __ldg(sp);
  }
  __syncwarp();
  for (int i = 0; i < steps; ++i) {
    const bool more = i + 1 < steps;
    uint4 nw0 = w0, nw1 = w1;
    uint32_t nsw = sw;
    if (more) {
#pragma unroll
      for (int l = 0; l < kLoads; ++l) ar[l] = __ldg(xbase + xrow[l] + (i + 1) * 8);
      nw0 = __ldg(wp + static_cast<int64_t>(2 * (i + 1)) * 32);
      nw1 = __ldg(wp + static_cast<int64_t>(2 * (i + 1) + 1) * 32);
      nsw = __ldg(sp + static_cast<int64_t>(i + 1) * 32);
    }
    step_mma<MT>(acc, mine, w0, w1, sw, idx);
    __syncwarp();
    if (more) {
#pragma unroll
      for (int l = 0; l < kLoads; ++l) mine[dst0 + 4 * kRowStride * l] = ar[l];
    }
    __syncwarp();
    w0 = nw0;
    w1 = nw1;
    sw = nsw;
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
      for (int w = 0; w < kW - 1; ++w) v += red[(w * kRows * 8 + t * 64 + j * 8) * 4 + lane];
      const int row = t * 8 + (idx & 1) + 2 * ((j >> 1) & 1) + 4 * (idx >> 2);
      const int col = tile * 32 + quad * 8 + (j & 1) + 2 * ((idx >> 1) & 1) + 4 * (j >> 2);
      if (row < m && col < n) out[static_cast<int64_t>(row) * n + col] = __float2half_rn(v * alpha[col]);
    }
}

template <int MT, int W>
void launch_wk_w(half *out, const half *x, const uint4 *wpk, const uint32_t *wsc, const float *alpha, int m, int n,
                 int k, int n_tiles, cudaStream_t stream) {
  constexpr int kBytes = W * MT * 8 * kRowStride * static_cast<int>(sizeof(uint4));
  static_assert(kBytes >= (W - 1) * MT * 8 * 32 * static_cast<int>(sizeof(float)), "reduction must fit the staging area");
  if constexpr (kBytes > 48 * 1024) {
    static bool configured = false;
    if (!configured) {
      cudaFuncSetAttribute(nvfp4_gemm_wk_kernel<MT, W>, cudaFuncAttributeMaxDynamicSharedMemorySize, kBytes);
      configured = true;
    }
  }
  nvfp4_gemm_wk_kernel<MT, W><<<n_tiles, W * 32, kBytes, stream>>>(out, x, wpk, wsc, alpha, m, n, k);
}

template <int MT>
void launch_wk(half *out, const half *x, const uint4 *wpk, const uint32_t *wsc, const float *alpha, int m, int n,
               int k, int n_tiles, cudaStream_t stream) {
  if constexpr (MT >= 3) launch_wk_w<MT, 4>(out, x, wpk, wsc, alpha, m, n, k, n_tiles, stream);
  else launch_wk_w<MT, 8>(out, x, wpk, wsc, alpha, m, n, k, n_tiles, stream);
}

__global__ void nvfp4_gemm_reduce_kernel(half *out, const float *ws, const float *alpha, int m, int n, int n_padded,
                                         int splits) {
  const int quads = n_padded / 4;
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= static_cast<int64_t>(m) * quads) return;
  const int row = static_cast<int>(idx / quads);
  const int col = static_cast<int>(idx % quads) * 4;
  if (col >= n) return;
  float4 s = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  for (int p = 0; p < splits; ++p) {
    const float4 v = *reinterpret_cast<const float4 *>(ws + (static_cast<int64_t>(p) * m + row) * n_padded + col);
    s.x += v.x;
    s.y += v.y;
    s.z += v.z;
    s.w += v.w;
  }
  const float r[4] = {s.x, s.y, s.z, s.w};
  half *o = out + static_cast<int64_t>(row) * n + col;
#pragma unroll
  for (int e = 0; e < 4; ++e)
    if (col + e < n) o[e] = __float2half_rn(r[e] * alpha[col + e]);
}

template <int MT, int W>
int launch(half *out, float *ws, const half *x, const uint4 *wpk, const uint32_t *wsc, const float *alpha, int m,
           int n, int n_tiles, int k, int splits, cudaStream_t stream) {
  const int ksteps = k / kStepK;
  const int per = (ksteps + splits - 1) / splits;
  const dim3 grid((n_tiles + W - 1) / W, (ksteps + per - 1) / per);
  nvfp4_gemm_kernel<MT, W><<<grid, W * 32, 0, stream>>>(grid.y == 1 ? out : nullptr, ws, x, wpk, wsc, alpha, m, n,
                                                        n_tiles, k, per);
  return static_cast<int>(grid.y);
}

template <int MT>
int launch_mt(int warps, half *out, float *ws, const half *x, const uint4 *wpk, const uint32_t *wsc,
              const float *alpha, int m, int n, int n_tiles, int k, int splits, cudaStream_t stream) {
  if (warps >= 8) return launch<MT, 8>(out, ws, x, wpk, wsc, alpha, m, n, n_tiles, k, splits, stream);
  if (warps >= 4) return launch<MT, 4>(out, ws, x, wpk, wsc, alpha, m, n, n_tiles, k, splits, stream);
  return launch<MT, 2>(out, ws, x, wpk, wsc, alpha, m, n, n_tiles, k, splits, stream);
}

__global__ void nvfp4_gemm_dequantize_kernel(half *out, const uint4 *wpk, const uint32_t *wsc, const float *alpha,
                                             int n, int k, int64_t words) {
  const int64_t g = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (g >= words) return;
  const int chunks = k / 32;
  const int lane = static_cast<int>(g & 31);
  const int64_t tc = g >> 5;
  const int chunk = static_cast<int>(tc % chunks);
  const int tile = static_cast<int>(tc / chunks);
  const int col = tile * 32 + 8 * ((lane >> 2) & 3) + (lane & 3) + 4 * (lane >> 4);
  if (col >= n) return;
  const uint4 w = wpk[g];
  const uint32_t sw = wsc[(static_cast<int64_t>(tile) * (k / 64) + (chunk >> 1)) * 32 + lane] >> ((chunk & 1) * 16);
  const float a = alpha[col] * (1.0f / 128.0f);
  const float s0 = e4m3_to_float(sw & 0xffu) * a;
  const float s1 = e4m3_to_float((sw >> 8) & 0xffu) * a;
  const uint32_t words4[4] = {w.x, w.y, w.z, w.w};
  const int pos[8] = {3, 7, 2, 6, 1, 5, 0, 4};
  uint32_t outw[16];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float s = (j < 2) ? s0 : s1;
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      const float lo = e2m1_to_float((words4[j] >> (4 * pos[2 * p])) & 0xfu) * s;
      const float hi = e2m1_to_float((words4[j] >> (4 * pos[2 * p + 1])) & 0xfu) * s;
      const half2 h = __floats2half2_rn(lo, hi);
      outw[j * 4 + p] = *reinterpret_cast<const uint32_t *>(&h);
    }
  }
  uint4 *dst = reinterpret_cast<uint4 *>(out + static_cast<int64_t>(col) * k + chunk * 32);
#pragma unroll
  for (int q = 0; q < 4; ++q) dst[q] = make_uint4(outw[4 * q], outw[4 * q + 1], outw[4 * q + 2], outw[4 * q + 3]);
}

}  // namespace
}  // namespace seventy

SEVENTY_API int seventy_nvfp4_gemm_dequantize(void *out, const void *wpk, const void *wsc, const float *alpha, int n,
                                              int n_padded, int k, cudaStream_t stream) {
  using namespace seventy;
  if (out == nullptr || wpk == nullptr || wsc == nullptr || alpha == nullptr || n <= 0 || n_padded < n ||
      n_padded % 32 || k <= 0 || k % kStepK)
    return SEVENTY_ERR_INVALID_ARGUMENT;
  if (!aligned(out, 16) || !aligned(wpk, 16) || !aligned(wsc, 4)) return SEVENTY_ERR_MISALIGNED;
  const int64_t words = static_cast<int64_t>(n_padded / 32) * (k / 32) * 32;
  nvfp4_gemm_dequantize_kernel<<<static_cast<unsigned>((words + 255) / 256), 256, 0, stream>>>(
      static_cast<half *>(out), static_cast<const uint4 *>(wpk), static_cast<const uint32_t *>(wsc), alpha, n, k,
      words);
  return launch_status();
}

SEVENTY_API int64_t seventy_nvfp4_gemm_workspace(int m, int n_padded, int splits) {
  return splits > 1 ? static_cast<int64_t>(splits) * m * n_padded * static_cast<int64_t>(sizeof(float)) : 0;
}

SEVENTY_API int seventy_nvfp4_gemm(void *out, const void *x, const void *wpk, const void *wsc, const float *alpha,
                                   int m, int n, int n_padded, int k, int splits, int warps, void *workspace,
                                   int64_t workspace_bytes, cudaStream_t stream) {
  using namespace seventy;
  if (out == nullptr || x == nullptr || wpk == nullptr || wsc == nullptr || alpha == nullptr)
    return SEVENTY_ERR_INVALID_ARGUMENT;
  if (m <= 0 || m > 64 || n <= 0 || n_padded < n || n_padded % 32 || k <= 0 || k % kStepK || splits < 1)
    return SEVENTY_ERR_INVALID_ARGUMENT;
  const int ksteps = k / kStepK;
  const int per = (ksteps + splits - 1) / splits;
  const int used = (ksteps + per - 1) / per;
  if (used > 1 && (workspace == nullptr || workspace_bytes < seventy_nvfp4_gemm_workspace(m, n_padded, used)))
    return SEVENTY_ERR_INVALID_ARGUMENT;
  if (!aligned(x, 16) || !aligned(wpk, 16) || !aligned(wsc, 4) || !aligned(out, 2) || !aligned(workspace, 16))
    return SEVENTY_ERR_MISALIGNED;
  const int n_tiles = n_padded / 32;
  auto *o = static_cast<half *>(out);
  auto *ws = static_cast<float *>(workspace);
  const auto *xh = static_cast<const half *>(x);
  const auto *wp = static_cast<const uint4 *>(wpk);
  const auto *sp = static_cast<const uint32_t *>(wsc);
  static const bool wk_off = getenv("SEVENTY_GEMM_WK") != nullptr && atoi(getenv("SEVENTY_GEMM_WK")) == 0;
  const bool use_wk = m <= 40 || ksteps <= 72 || n_tiles < 200;
  if (!wk_off && use_wk && ksteps >= 8) {
    if (m > 56)
      launch_wk<8>(o, xh, wp, sp, alpha, m, n, k, n_tiles, stream);
    else if (m > 48)
      launch_wk<7>(o, xh, wp, sp, alpha, m, n, k, n_tiles, stream);
    else if (m > 40)
      launch_wk<6>(o, xh, wp, sp, alpha, m, n, k, n_tiles, stream);
    else if (m > 32)
      launch_wk<5>(o, xh, wp, sp, alpha, m, n, k, n_tiles, stream);
    else if (m > 24)
      launch_wk<4>(o, xh, wp, sp, alpha, m, n, k, n_tiles, stream);
    else if (m > 16)
      launch_wk<3>(o, xh, wp, sp, alpha, m, n, k, n_tiles, stream);
    else if (m > 8)
      launch_wk<2>(o, xh, wp, sp, alpha, m, n, k, n_tiles, stream);
    else
      launch_wk<1>(o, xh, wp, sp, alpha, m, n, k, n_tiles, stream);
    return launch_status();
  }
  switch ((m + 7) / 8) {
    case 1: launch_mt<1>(warps, o, ws, xh, wp, sp, alpha, m, n, n_tiles, k, used, stream); break;
    case 2: launch_mt<2>(warps, o, ws, xh, wp, sp, alpha, m, n, n_tiles, k, used, stream); break;
    case 3: launch_mt<3>(warps, o, ws, xh, wp, sp, alpha, m, n, n_tiles, k, used, stream); break;
    case 4: launch_mt<4>(warps, o, ws, xh, wp, sp, alpha, m, n, n_tiles, k, used, stream); break;
    case 5: launch_mt<5>(warps, o, ws, xh, wp, sp, alpha, m, n, n_tiles, k, used, stream); break;
    case 6: launch_mt<6>(warps, o, ws, xh, wp, sp, alpha, m, n, n_tiles, k, used, stream); break;
    case 7: launch_mt<7>(warps, o, ws, xh, wp, sp, alpha, m, n, n_tiles, k, used, stream); break;
    default: launch_mt<8>(warps, o, ws, xh, wp, sp, alpha, m, n, n_tiles, k, used, stream); break;
  }
  if (used > 1) {
    const int64_t total = static_cast<int64_t>(m) * (n_padded / 4);
    nvfp4_gemm_reduce_kernel<<<static_cast<unsigned>((total + 255) / 256), 256, 0, stream>>>(o, ws, alpha, m, n,
                                                                                            n_padded, used);
  }
  return launch_status();
}
