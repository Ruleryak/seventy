#include "common.cuh"

/*
 * silu_mul: out[r, i] = silu(gate[r, i]) * up[r, i] for FP16 gate_up [rows, 2 * inter] = [gate | up], in FP32 with
 * one rounding to FP16 (as the reference).
 */

namespace seventy {
namespace {

__global__ void silu_mul_kernel(half2 *out, const half2 *gate_up, int inter2, int64_t total) {
  const int64_t e = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (e >= total) return;
  const int64_t row = e / inter2;
  const int i = static_cast<int>(e % inter2);
  const float2 g = __half22float2(gate_up[row * 2 * inter2 + i]);
  const float2 u = __half22float2(gate_up[row * 2 * inter2 + inter2 + i]);
  out[e] = __floats2half2_rn(g.x / (1.0f + expf(-g.x)) * u.x, g.y / (1.0f + expf(-g.y)) * u.y);
}

}  // namespace
}  // namespace seventy

SEVENTY_API int seventy_silu_mul(void *out, const void *gate_up, int64_t rows, int inter, cudaStream_t stream) {
  using namespace seventy;
  if (out == nullptr || gate_up == nullptr || rows < 0 || inter <= 0 || inter % 2) return SEVENTY_ERR_INVALID_ARGUMENT;
  if (!aligned(out, 4) || !aligned(gate_up, 4)) return SEVENTY_ERR_MISALIGNED;
  const int64_t total = rows * (inter / 2);
  if (total == 0) return SEVENTY_OK;
  silu_mul_kernel<<<static_cast<unsigned>((total + 255) / 256), 256, 0, stream>>>(
      static_cast<half2 *>(out), static_cast<const half2 *>(gate_up), inter / 2, total);
  return launch_status();
}
