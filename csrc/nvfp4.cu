#include "common.cuh"

namespace seventy {
namespace {

/*
 * One thread per group of 16 values. The groups of a row are contiguous, so group g of the whole matrix reads
 * packed bytes [8 g, 8 g + 8) and scale byte g, and writes output values [16 g, 16 g + 16).
 */
__global__ void nvfp4_dequantize_kernel(uint4 *out, const uint2 *packed, const uint8_t *scales, float global_scale,
                                        int64_t groups) {
  const int64_t g = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (g >= groups) return;
  const float scale = e4m3_to_float(scales[g]) / global_scale;
  const uint2 p = packed[g];
  const uint32_t words[2] = {p.x, p.y};
  uint32_t halves[8];
#pragma unroll
  for (int w = 0; w < 2; ++w) {
#pragma unroll
    for (int b = 0; b < 4; ++b) {
      const uint32_t byte = (words[w] >> (8 * b)) & 0xffu;
      const half2 h = __floats2half2_rn(e2m1_to_float(byte & 0xfu) * scale, e2m1_to_float(byte >> 4) * scale);
      halves[w * 4 + b] = *reinterpret_cast<const uint32_t *>(&h);
    }
  }
  out[2 * g] = make_uint4(halves[0], halves[1], halves[2], halves[3]);
  out[2 * g + 1] = make_uint4(halves[4], halves[5], halves[6], halves[7]);
}

}  // namespace
}  // namespace seventy

SEVENTY_API int seventy_nvfp4_dequantize(void *out, const uint8_t *packed, const uint8_t *scales,
                                         float global_scale, int64_t n, int64_t k, cudaStream_t stream) {
  using namespace seventy;
  if (out == nullptr || packed == nullptr || scales == nullptr || n < 0 || k <= 0 || (k % 16) != 0)
    return SEVENTY_ERR_INVALID_ARGUMENT;
  if (!aligned(out, 16) || !aligned(packed, 8)) return SEVENTY_ERR_MISALIGNED;
  const int64_t groups = n * (k / 16);
  if (groups == 0) return SEVENTY_OK;
  const int threads = 256;
  const int64_t blocks = (groups + threads - 1) / threads;
  nvfp4_dequantize_kernel<<<static_cast<unsigned>(blocks), threads, 0, stream>>>(
      static_cast<uint4 *>(out), reinterpret_cast<const uint2 *>(packed), scales, global_scale, groups);
  return launch_status();
}
