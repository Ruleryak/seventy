#include "common.cuh"

#include <cstdlib>

/*
 * Mixture-of-experts kernels for small batches (decode and speculative verify), all inside CUDA graphs: the plan
 * and the counts stay on the GPU, and each grid has a fixed size from the largest plan.
 *
 * A step has P = tokens x k (token, expert) pairs; pair p = token * k + j is the token's j-th pick. A rank keeps
 * the experts [e0, e1) (local expert e - e0).
 *
 * seventy_moe_plan (one block): the pairs of local experts in order of (local expert, pair), and slots of at most
 * `per` pairs of one expert: slot s covers order[start[s] .. start[s] + count[s]) of expert slot_expert[s]. The
 * order is stable, so results do not depend on timing.
 *
 * seventy_moe_gemm: for each slot, the rows of its pairs times the weight of its expert. Row r of x for pair p is
 * x[p / row_div] (row_div k: x holds the tokens, for gate_up; row_div 1: x holds the pairs, for down). One block
 * takes 64 output columns of one slot; each warp takes 8 columns at once: each lane reads the same group of 16
 * weights (one NVFP4 group, 8 bytes, or 32 bytes of FP16) of all 8 columns per step, so 8 loads are in flight,
 * and adds them up in FP32 for every row of the slot; the warp then sums its lanes. NVFP4 weights: packed [E, n, k / 2], scales [E, n, k / 16] (E4M3 bytes) and
 * alpha [E, n] (1 / global scale of the row's module); W = e2m1 * e4m3 * alpha. FP16 weights: [E, n, k], alpha
 * ignored. Output: FP16 out[p, n], or with pair_weight FP32 out[p, n] * pair_weight[p].
 */

namespace seventy {
namespace {

constexpr int kMoeMaxExperts = 1024;
constexpr int kMoeMaxRows = 8;
constexpr int kMoeWarpCols = 2;
constexpr int kMoeThreads = 256;

__device__ __forceinline__ uint32_t code_pair_to_half2(uint32_t x) {
  return (x & 0x80008000u) | ((x >> 3) & 0x0E000E00u);
}

__global__ void moe_plan_kernel(const int32_t *ids, int pairs, int e0, int e1, int per, int max_slots,
                                int32_t *order, int32_t *slot_expert, int32_t *slot_start, int32_t *slot_count,
                                int32_t *num_slots) {
  __shared__ int counts[kMoeMaxExperts];
  __shared__ int offsets[kMoeMaxExperts];
  __shared__ int slot_base[kMoeMaxExperts];
  const int experts = e1 - e0;
  for (int e = threadIdx.x; e < experts; e += blockDim.x) counts[e] = 0;
  __syncthreads();
  for (int p = threadIdx.x; p < pairs; p += blockDim.x) {
    const int e = ids[p];
    if (e >= e0 && e < e1) atomicAdd(&counts[e - e0], 1);
  }
  __syncthreads();
  for (int e = threadIdx.x; e < experts; e += blockDim.x) {
    int off = 0, slots = 0;
    for (int f = 0; f < e; ++f) {
      off += counts[f];
      slots += (counts[f] + per - 1) / per;
    }
    offsets[e] = off;
    slot_base[e] = slots;
  }
  __syncthreads();
  for (int e = threadIdx.x; e < experts; e += blockDim.x) {
    const int c = counts[e];
    for (int j = 0; j * per < c; ++j) {
      const int s = slot_base[e] + j;
      if (s < max_slots) {
        slot_expert[s] = e;
        slot_start[s] = offsets[e] + j * per;
        slot_count[s] = min(per, c - j * per);
      }
    }
    if (e == experts - 1) *num_slots = min(max_slots, slot_base[e] + (c + per - 1) / per);
  }
  for (int p = threadIdx.x; p < pairs; p += blockDim.x) {
    const int e = ids[p];
    if (e < e0 || e >= e1) continue;
    int rank = 0;
    for (int q = 0; q < p; ++q) rank += ids[q] == e;
    order[offsets[e - e0] + rank] = p;
  }
  if (experts == 0 && threadIdx.x == 0) *num_slots = 0;
}

template <bool kNvfp4, int kRows, int kCols>
__global__ void __launch_bounds__(kMoeThreads) moe_gemm_kernel(
    void *out, const half *x, int row_div, const uint8_t *weight, const uint8_t *scales, const float *alpha, int n,
    int k, const int32_t *order, const int32_t *slot_expert, const int32_t *slot_start, const int32_t *slot_count,
    const int32_t *num_slots, const float *pair_weight) {
  extern __shared__ __align__(16) half xs[];
  __shared__ int pair_of[kRows];
  const int s = blockIdx.y;
  if (s >= *num_slots) return;
  const int e = slot_expert[s];
  const int start = slot_start[s];
  const int count = slot_count[s];
  if (threadIdx.x < kRows) pair_of[threadIdx.x] = threadIdx.x < count ? order[start + threadIdx.x] : 0;
  __syncthreads();
  const int words = k / 8;
  for (int i = threadIdx.x; i < count * words; i += blockDim.x) {
    const int r = i / words, w = i % words;
    reinterpret_cast<uint4 *>(xs)[r * words + w] =
        reinterpret_cast<const uint4 *>(x + static_cast<int64_t>(pair_of[r] / row_div) * k)[w];
  }
  __syncthreads();
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int groups = k / 16;
  const int col0 = blockIdx.x * (kMoeThreads / 32) * kCols + warp * kCols;
  float acc[kRows][kCols];
#pragma unroll
  for (int r = 0; r < kRows; ++r)
#pragma unroll
    for (int c = 0; c < kCols; ++c) acc[r][c] = 0.0f;
  const int64_t row0 = static_cast<int64_t>(e) * n + col0;
  for (int g = lane; g < groups; g += 32) {
    if (kNvfp4) {
      uint2 codes[kCols];
      float sc[kCols];
#pragma unroll
      for (int c = 0; c < kCols; ++c) {
        const bool ok = col0 + c < n;
        codes[c] = ok ? reinterpret_cast<const uint2 *>(weight + (row0 + c) * (k / 2))[g] : make_uint2(0, 0);
        sc[c] = ok ? e4m3_to_float(scales[(row0 + c) * groups + g]) * 16384.0f : 0.0f;
      }
#pragma unroll
      for (int r = 0; r < kRows; ++r) {
        if (r >= count) break;
        const half *xr = xs + r * k + g * 16;
        float xv[16];
#pragma unroll
        for (int q = 0; q < 16; ++q) xv[q] = __half2float(xr[q]);
#pragma unroll
        for (int c = 0; c < kCols; ++c) {
          float sum = 0.0f;
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const uint32_t word = h ? codes[c].y : codes[c].x;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
              const uint32_t bits = code_pair_to_half2(word << (12 - 4 * j));
              const float2 f = __half22float2(*reinterpret_cast<const half2 *>(&bits));
              sum = fmaf(xv[8 * h + j], f.x, sum);
              sum = fmaf(xv[8 * h + j + 4], f.y, sum);
            }
          }
          acc[r][c] = fmaf(sum, sc[c], acc[r][c]);
        }
      }
    } else {
      uint4 a[kCols], b[kCols];
#pragma unroll
      for (int c = 0; c < kCols; ++c) {
        a[c] = b[c] = make_uint4(0, 0, 0, 0);
        if (col0 + c < n) {
          const uint4 *src =
              reinterpret_cast<const uint4 *>(reinterpret_cast<const half *>(weight) + (row0 + c) * k) + 2 * g;
          a[c] = src[0];
          b[c] = src[1];
        }
      }
#pragma unroll
      for (int r = 0; r < kRows; ++r) {
        if (r >= count) break;
        const half2 *xr = reinterpret_cast<const half2 *>(xs + r * k + g * 16);
#pragma unroll
        for (int c = 0; c < kCols; ++c) {
          const half2 *ha = reinterpret_cast<const half2 *>(&a[c]);
          const half2 *hb = reinterpret_cast<const half2 *>(&b[c]);
          float sum = 0.0f;
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const float2 xa = __half22float2(xr[q]), xb = __half22float2(xr[4 + q]);
            const float2 fa = __half22float2(ha[q]), fb = __half22float2(hb[q]);
            sum = fmaf(xa.x, fa.x, fmaf(xa.y, fa.y, sum));
            sum = fmaf(xb.x, fb.x, fmaf(xb.y, fb.y, sum));
          }
          acc[r][c] += sum;
        }
      }
    }
  }
#pragma unroll
  for (int r = 0; r < kRows; ++r) {
    if (r >= count) break;
    const int p = pair_of[r];
#pragma unroll
    for (int c = 0; c < kCols; ++c) {
      const float v = warp_sum(acc[r][c]);
      const int col = col0 + c;
      if (lane == c && col < n) {
        const float a = kNvfp4 ? alpha[row0 + c] : 1.0f;
        if (pair_weight != nullptr) {
          static_cast<float *>(out)[static_cast<int64_t>(p) * n + col] = v * a * pair_weight[p];
        } else {
          static_cast<half *>(out)[static_cast<int64_t>(p) * n + col] = __float2half_rn(v * a);
        }
      }
    }
  }
}

}  // namespace
}  // namespace seventy

SEVENTY_API int seventy_moe_plan(const int32_t *ids, int pairs, int e0, int e1, int per, int max_slots,
                                 int32_t *order, int32_t *slot_expert, int32_t *slot_start, int32_t *slot_count,
                                 int32_t *num_slots, cudaStream_t stream) {
  using namespace seventy;
  if (ids == nullptr || order == nullptr || slot_expert == nullptr || slot_start == nullptr || slot_count == nullptr ||
      num_slots == nullptr || pairs < 0 || e1 < e0 || e1 - e0 > kMoeMaxExperts || per < 1 || per > kMoeMaxRows ||
      max_slots < 1)
    return SEVENTY_ERR_INVALID_ARGUMENT;
  moe_plan_kernel<<<1, 1024, 0, stream>>>(ids, pairs, e0, e1, per, max_slots, order, slot_expert, slot_start,
                                          slot_count, num_slots);
  return launch_status();
}

SEVENTY_API int seventy_moe_gemm(void *out, const void *x, int row_div, const void *weight, const void *scales,
                                 const float *alpha, int nvfp4, int n, int k, const int32_t *order,
                                 const int32_t *slot_expert, const int32_t *slot_start, const int32_t *slot_count,
                                 const int32_t *num_slots, int max_slots, int rows, const float *pair_weight,
                                 cudaStream_t stream) {
  using namespace seventy;
  if (out == nullptr || x == nullptr || weight == nullptr || order == nullptr || n <= 0 || k <= 0 || k % 16 ||
      row_div < 1 || max_slots < 1 || (nvfp4 && (scales == nullptr || alpha == nullptr)))
    return SEVENTY_ERR_INVALID_ARGUMENT;
  if (!aligned(x, 16) || !aligned(weight, 16)) return SEVENTY_ERR_MISALIGNED;
  const size_t smem = static_cast<size_t>(rows) * k * sizeof(half);
  if (smem > 48 * 1024) return SEVENTY_ERR_INVALID_ARGUMENT;
  const char *env = getenv("SEVENTY_MOE_COLS");
  const int cols = env ? atoi(env) : kMoeWarpCols;
  const int per_block = (kMoeThreads / 32) * cols;
  const dim3 grid((n + per_block - 1) / per_block, max_slots);
  const auto *xh = static_cast<const half *>(x);
  const auto *w = static_cast<const uint8_t *>(weight);
  const auto *sc = static_cast<const uint8_t *>(scales);
#define SEVENTY_MOE_LAUNCH(NV, R, C)                                                                              \
  moe_gemm_kernel<NV, R, C><<<grid, kMoeThreads, smem, stream>>>(out, xh, row_div, w, sc, alpha, n, k, order,      \
                                                                   slot_expert, slot_start, slot_count, num_slots, \
                                                                   pair_weight)
  if (rows != 2 && rows != 4 && rows != 8) return SEVENTY_ERR_INVALID_ARGUMENT;
  if (cols != 2 && cols != 4 && cols != 8) return SEVENTY_ERR_INVALID_ARGUMENT;
#define SEVENTY_MOE_COLS(NV, R)                 \
  if (cols == 2) SEVENTY_MOE_LAUNCH(NV, R, 2); \
  else if (cols == 4) SEVENTY_MOE_LAUNCH(NV, R, 4); \
  else SEVENTY_MOE_LAUNCH(NV, R, 8)
  if (nvfp4) {
    if (rows == 2) { SEVENTY_MOE_COLS(true, 2); } else if (rows == 4) { SEVENTY_MOE_COLS(true, 4); } else { SEVENTY_MOE_COLS(true, 8); }
  } else {
    if (rows == 2) { SEVENTY_MOE_COLS(false, 2); } else if (rows == 4) { SEVENTY_MOE_COLS(false, 4); } else { SEVENTY_MOE_COLS(false, 8); }
  }
#undef SEVENTY_MOE_COLS
#undef SEVENTY_MOE_LAUNCH
  return launch_status();
}
