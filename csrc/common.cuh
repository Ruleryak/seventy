#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "seventy.h"

namespace seventy {

inline int launch_status() { return static_cast<int>(cudaGetLastError()); }

inline bool aligned(const void *p, uintptr_t bytes) { return (reinterpret_cast<uintptr_t>(p) % bytes) == 0; }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_xor_sync(0xffffffffu, v, offset);
  return v;
}

/*
 * One Volta tensor-core product (mma.m8n8k4.row.col, FP16 inputs, FP32 accumulators): each quad-pair of the warp adds
 * its 8 x 4 A (a0, a1) times its 4 x 8 B (b0, b1) into its 8 x 8 accumulators c (layout in csrc/nvfp4_gemm.cu).
 */
__device__ __forceinline__ void mma_m8n8k4(float (&c)[8], uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7])
      : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

/*
 * Sum of v over the block. Every thread gets the result. scratch needs 32 floats of shared memory.
 */
__device__ __forceinline__ float block_sum(float v, float *scratch) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warps = (blockDim.x + 31) >> 5;
  v = warp_sum(v);
  __syncthreads();
  if (lane == 0) scratch[warp] = v;
  __syncthreads();
  v = (lane < warps) ? scratch[lane] : 0.0f;
  return warp_sum(v);
}

/*
 * FP8 E4M3 (the "fn" variant: no infinities, 0x7f and 0xff are NaN) to FP32. Exact.
 */
__device__ __forceinline__ float e4m3_to_float(uint8_t b) {
  const uint32_t sign = (b & 0x80u) << 24;
  const uint32_t exp = (b >> 3) & 0xfu;
  const uint32_t man = b & 0x7u;
  float mag;
  if (exp == 0) {
    mag = static_cast<float>(man) * 0.001953125f;
  } else if (exp == 0xf && man == 0x7) {
    mag = __int_as_float(0x7fc00000);
  } else {
    mag = __int_as_float(static_cast<int>(((exp + 120u) << 23) | (man << 20)));
  }
  return __int_as_float(__float_as_int(mag) | static_cast<int>(sign));
}

/*
 * E2M1 (4-bit float: sign, 2 exponent bits, 1 mantissa bit) to FP32. Exact.
 */
__device__ __forceinline__ float e2m1_to_float(uint32_t nibble) {
  const float mags[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  const float m = mags[nibble & 0x7u];
  return (nibble & 0x8u) ? -m : m;
}

}  // namespace seventy
