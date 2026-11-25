#include "common.cuh"

/*
 * Hyper-connections (Qwen4-Exp): the residual is hc streams of `hidden` values per token, FP32 [rows, hc * hidden].
 * Around the two products of a hyper-connection (seventy.model.qwen4_exp.HyperConnection, through
 * seventy_dense_gemm), these kernels do the rest:
 *
 * seventy_hc_norm: n = rms_norm of each stream (group of `hidden` values) times weight (already 1 + w), FP16 out.
 * seventy_hc_act: from d = [down(n) | inject(n)] FP32 [rows, rank + hc]: s = silu(d[:rank] / hc) as FP16 [rows,
 *   rank], and the injection weights 2 sigmoid(d[rank:] / hc) as FP32 [rows, hc] (with hc_inject 0: none).
 * seventy_hc_combine: the block input x[t, j] = mean over streams i of sigmoid(u[t, i * hidden + j]) *
 *   n[t, i * hidden + j], FP16 [rows, hidden], from u = up(s) FP32 [rows, hc * hidden].
 * seventy_hc_inject: h[t, i * hidden + j] += y[t, j] * w[t, i] for a block output y FP16 [rows, hidden].
 */

namespace seventy {
namespace {

__global__ void __launch_bounds__(256) hc_norm_kernel(half *out, const float *h, const float *weight, int hc,
                                                     int hidden, float eps) {
  __shared__ float scratch[32];
  const int row = blockIdx.x / hc, g = blockIdx.x % hc;
  const float *src = h + (static_cast<int64_t>(row) * hc + g) * hidden;
  float ss = 0.0f;
  for (int j = threadIdx.x; j < hidden; j += blockDim.x) ss += src[j] * src[j];
  ss = block_sum(ss, scratch);
  const float inv = rsqrtf(ss / hidden + eps);
  half *dst = out + (static_cast<int64_t>(row) * hc + g) * hidden;
  const float *w = weight + static_cast<int64_t>(g) * hidden;
  for (int j = threadIdx.x; j < hidden; j += blockDim.x) dst[j] = __float2half_rn(src[j] * inv * w[j]);
}

__global__ void hc_act_kernel(half *s, float *inject, const float *d, int rows, int rank, int hc, int ld,
                              int with_inject) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int width = rank + (with_inject ? hc : 0);
  if (i >= rows * width) return;
  const int t = i / width, c = i % width;
  const float v = d[static_cast<int64_t>(t) * ld + c] / hc;
  if (c < rank) {
    s[static_cast<int64_t>(t) * rank + c] = __float2half_rn(v / (1.0f + expf(-v)));
  } else {
    inject[static_cast<int64_t>(t) * hc + (c - rank)] = 2.0f / (1.0f + expf(-v));
  }
}

__global__ void hc_combine_kernel(half *x, const float *u, const half *n, int rows, int hc, int hidden) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * hidden) return;
  const int t = i / hidden, j = i % hidden;
  float acc = 0.0f;
  for (int g = 0; g < hc; ++g) {
    const int64_t at = (static_cast<int64_t>(t) * hc + g) * hidden + j;
    acc += __half2float(n[at]) / (1.0f + expf(-u[at]));
  }
  x[i] = __float2half_rn(acc / hc);
}

__global__ void hc_inject_kernel(float *h, const half *y, const float *w, int rows, int hc, int hidden) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<int64_t>(rows) * hc * hidden) return;
  const int j = static_cast<int>(i % hidden);
  const int64_t tg = i / hidden;
  const int t = static_cast<int>(tg / hc), g = static_cast<int>(tg % hc);
  h[i] += __half2float(y[static_cast<int64_t>(t) * hidden + j]) * w[static_cast<int64_t>(t) * hc + g];
}

}  // namespace
}  // namespace seventy

SEVENTY_API int seventy_hc_norm(void *out, const float *h, const float *weight, int rows, int hc, int hidden,
                                float eps, cudaStream_t stream) {
  using namespace seventy;
  if (out == nullptr || h == nullptr || weight == nullptr || rows < 0 || hc < 1 || hidden < 1)
    return SEVENTY_ERR_INVALID_ARGUMENT;
  if (rows == 0) return SEVENTY_OK;
  hc_norm_kernel<<<rows * hc, 256, 0, stream>>>(static_cast<half *>(out), h, weight, hc, hidden, eps);
  return launch_status();
}

SEVENTY_API int seventy_hc_act(void *s, float *inject, const float *d, int rows, int rank, int hc, int ld,
                               cudaStream_t stream) {
  using namespace seventy;
  if (s == nullptr || d == nullptr || rows < 0 || rank < 1 || hc < 1 || ld < rank) return SEVENTY_ERR_INVALID_ARGUMENT;
  const int width = rank + (inject != nullptr ? hc : 0);
  if (rows == 0) return SEVENTY_OK;
  hc_act_kernel<<<(rows * width + 255) / 256, 256, 0, stream>>>(static_cast<half *>(s), inject, d, rows, rank, hc,
                                                                 ld, inject != nullptr);
  return launch_status();
}

SEVENTY_API int seventy_hc_combine(void *x, const float *u, const void *n, int rows, int hc, int hidden,
                                   cudaStream_t stream) {
  using namespace seventy;
  if (x == nullptr || u == nullptr || n == nullptr || rows < 0 || hc < 1 || hidden < 1)
    return SEVENTY_ERR_INVALID_ARGUMENT;
  if (rows == 0) return SEVENTY_OK;
  hc_combine_kernel<<<(rows * hidden + 255) / 256, 256, 0, stream>>>(static_cast<half *>(x), u,
                                                                      static_cast<const half *>(n), rows, hc, hidden);
  return launch_status();
}

SEVENTY_API int seventy_hc_inject(float *h, const void *y, const float *w, int rows, int hc, int hidden,
                                  cudaStream_t stream) {
  using namespace seventy;
  if (h == nullptr || y == nullptr || w == nullptr || rows < 0 || hc < 1 || hidden < 1)
    return SEVENTY_ERR_INVALID_ARGUMENT;
  const int64_t total = static_cast<int64_t>(rows) * hc * hidden;
  if (total == 0) return SEVENTY_OK;
  hc_inject_kernel<<<static_cast<unsigned>((total + 255) / 256), 256, 0, stream>>>(h, static_cast<const half *>(y), w,
                                                                                    rows, hc, hidden);
  return launch_status();
}
