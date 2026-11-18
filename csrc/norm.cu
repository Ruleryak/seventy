#include "common.cuh"

namespace seventy {
namespace {

/*
 * One block per row. Pass 1 forms s (x, or x + residual rounded to FP16) and its sum of squares. Pass 2 reads s
 * back and writes the normalized row. Each thread touches the same elements in both passes, so in-place use
 * (out == x, residual_out == residual) is safe without extra synchronization.
 */
__global__ void rms_norm_kernel(half2 *out, half2 *residual_out, const half2 *x, const half2 *residual,
                                const float2 *weight, int pairs, float inv_dim, float eps) {
  __shared__ float scratch[32];
  const int64_t row = blockIdx.x;
  const half2 *xr = x + row * pairs;
  half2 *outr = out + row * pairs;
  float sumsq = 0.0f;
  if (residual != nullptr) {
    const half2 *rr = residual + row * pairs;
    half2 *ror = residual_out + row * pairs;
    for (int i = threadIdx.x; i < pairs; i += blockDim.x) {
      const float2 a = __half22float2(xr[i]);
      const float2 b = __half22float2(rr[i]);
      const half2 s = __floats2half2_rn(a.x + b.x, a.y + b.y);
      ror[i] = s;
      const float2 f = __half22float2(s);
      sumsq += f.x * f.x + f.y * f.y;
    }
  } else {
    for (int i = threadIdx.x; i < pairs; i += blockDim.x) {
      const float2 f = __half22float2(xr[i]);
      sumsq += f.x * f.x + f.y * f.y;
    }
  }
  const float total = block_sum(sumsq, scratch);
  const float inv_rms = 1.0f / sqrtf(total * inv_dim + eps);
  const half2 *src = (residual != nullptr) ? residual_out + row * pairs : xr;
  for (int i = threadIdx.x; i < pairs; i += blockDim.x) {
    const float2 f = __half22float2(src[i]);
    const float2 w = weight[i];
    outr[i] = __floats2half2_rn(f.x * inv_rms * w.x, f.y * inv_rms * w.y);
  }
}

}  // namespace
}  // namespace seventy

SEVENTY_API int seventy_rms_norm(void *out, void *residual_out, const void *x, const void *residual,
                                 const float *weight, int64_t rows, int dim, float eps, cudaStream_t stream) {
  using namespace seventy;
  if (out == nullptr || x == nullptr || weight == nullptr || rows < 0 || dim <= 0 || (dim & 1) != 0)
    return SEVENTY_ERR_INVALID_ARGUMENT;
  if ((residual == nullptr) != (residual_out == nullptr)) return SEVENTY_ERR_INVALID_ARGUMENT;
  if (!aligned(out, 4) || !aligned(x, 4) || !aligned(weight, 8) ||
      (residual != nullptr && (!aligned(residual, 4) || !aligned(residual_out, 4))))
    return SEVENTY_ERR_MISALIGNED;
  if (rows == 0) return SEVENTY_OK;
  const int pairs = dim / 2;
  int threads = ((pairs + 31) / 32) * 32;
  if (threads > 256) threads = 256;
  rms_norm_kernel<<<static_cast<unsigned>(rows), threads, 0, stream>>>(
      static_cast<half2 *>(out), static_cast<half2 *>(residual_out), static_cast<const half2 *>(x),
      static_cast<const half2 *>(residual), reinterpret_cast<const float2 *>(weight), pairs,
      1.0f / static_cast<float>(dim), eps);
  return launch_status();
}
