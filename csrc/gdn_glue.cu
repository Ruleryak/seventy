#include "common.cuh"

/*
 * The element-wise parts of Gated DeltaNet prefill around the chunked kernels (csrc/gdn_prepare.cu,
 * csrc/gdn_chunk.cu), each in one pass. They compute what seventy.ops.reference computes, in FP32, rounding where
 * the reference rounds.
 *
 * gdn_conv_kernel: the depthwise causal convolution over time with SiLU. Thread c of a block handles channel c of
 * a range of tokens: it keeps the last taps inputs in registers, so each input is read once, and starts a sequence
 * from its cached history (conv_state [slots, channels, taps - 1], oldest first; zero for a sequence without
 * context). gdn_conv_state_kernel then writes the new history of every sequence (the last taps - 1 inputs of
 * history and new tokens together); it runs after the convolution, which reads the old history.
 *
 * gdn_gates_kernel: g = -exp(a_log) * softplus(a + dt_bias) (softplus(x) = x above 20, as in PyTorch) and
 * beta = sigmoid(b), FP32 [T, hv], from the a and b columns of the input projection.
 *
 * gated_rms_norm_kernel: out = rms_norm(x) * weight * silu(z) per head row of 128 values, one warp per row; z is read
 * in place from the input projection (row t, columns z_off + h * 128 ...).
 */

namespace seventy {
namespace {

constexpr int kMaxTaps = 4;
constexpr int kConvThreads = 256;
constexpr int kConvTokens = 32;

__device__ __forceinline__ float silu(float x) { return x / (1.0f + expf(-x)); }

__device__ __forceinline__ int sequence_of(const int *query_start, int num_seqs, int t) {
  int lo = 0, hi = num_seqs - 1;
  while (lo < hi) {
    const int mid = (lo + hi + 1) / 2;
    if (query_start[mid] <= t) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

template <int kTaps>
__global__ void __launch_bounds__(kConvThreads) gdn_conv_kernel(half *out, const half *conv_state, const half *y,
                                                                int ld, const float *conv_w, const int *query_start,
                                                                const int *context_lens, const int *state_slots,
                                                                int num_seqs, int tokens, int channels) {
  const int c = blockIdx.y * kConvThreads + threadIdx.x;
  if (c >= channels) return;
  const int t0 = blockIdx.x * kConvTokens;
  const int t1 = min(tokens, t0 + kConvTokens);
  float w[kTaps];
#pragma unroll
  for (int j = 0; j < kTaps; ++j) w[j] = conv_w[c * kTaps + j];
  int b = sequence_of(query_start, num_seqs, t0);
  int seq_end = query_start[b + 1];
  float win[kTaps - 1];
  auto start_window = [&](int t) {
    const int s = query_start[b];
    const bool fresh = context_lens[b] == 0;
    const half *hist = conv_state + (static_cast<int64_t>(state_slots[b]) * channels + c) * (kTaps - 1);
#pragma unroll
    for (int j = 0; j < kTaps - 1; ++j) {
      const int p = t - (kTaps - 1) + j;
      win[j] = p >= s ? __half2float(y[static_cast<int64_t>(p) * ld + c])
                      : (fresh ? 0.0f : __half2float(hist[kTaps - 1 + (p - s)]));
    }
  };
  start_window(t0);
  for (int t = t0; t < t1; ++t) {
    if (t == seq_end) {
      while (t == seq_end) {
        ++b;
        seq_end = query_start[b + 1];
      }
      start_window(t);
    }
    const float x = __half2float(y[static_cast<int64_t>(t) * ld + c]);
    float acc = 0.0f;
#pragma unroll
    for (int j = 0; j < kTaps - 1; ++j) acc += w[j] * win[j];
    acc += w[kTaps - 1] * x;
    out[static_cast<int64_t>(t) * channels + c] = __float2half_rn(silu(acc));
#pragma unroll
    for (int j = 0; j < kTaps - 2; ++j) win[j] = win[j + 1];
    win[kTaps - 2] = x;
  }
}

template <int kTaps>
__global__ void __launch_bounds__(kConvThreads) gdn_conv_state_kernel(half *conv_state, const half *y, int ld,
                                                                      const int *query_start,
                                                                      const int *context_lens,
                                                                      const int *state_slots, int channels) {
  const int c = blockIdx.x * kConvThreads + threadIdx.x;
  const int b = blockIdx.y;
  if (c >= channels) return;
  const int s = query_start[b], n = query_start[b + 1] - s;
  if (n <= 0) return;
  const bool fresh = context_lens[b] == 0;
  half *hist = conv_state + (static_cast<int64_t>(state_slots[b]) * channels + c) * (kTaps - 1);
  half next[kTaps - 1];
#pragma unroll
  for (int j = 0; j < kTaps - 1; ++j) {
    const int p = n - (kTaps - 1) + j;
    next[j] = p >= 0 ? y[static_cast<int64_t>(s + p) * ld + c]
                     : (fresh ? __float2half(0.0f) : hist[kTaps - 1 + p]);
  }
#pragma unroll
  for (int j = 0; j < kTaps - 1; ++j) hist[j] = next[j];
}

__global__ void gdn_gates_kernel(float *g, float *beta, const half *y, int ld, int a_off, int b_off,
                                 const float *a_log, const float *dt_bias, int tokens, int hv) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<int64_t>(tokens) * hv) return;
  const int t = static_cast<int>(i / hv), h = static_cast<int>(i % hv);
  const float a = __half2float(y[static_cast<int64_t>(t) * ld + a_off + h]) + dt_bias[h];
  const float sp = a > 20.0f ? a : log1pf(expf(a));
  g[i] = -expf(a_log[h]) * sp;
  beta[i] = 1.0f / (1.0f + expf(-__half2float(y[static_cast<int64_t>(t) * ld + b_off + h])));
}

__global__ void __launch_bounds__(256) gated_rms_norm_kernel(half *out, const half *x, const half *z, int z_ld,
                                                             int z_off, const float *weight, int rows, int hv,
                                                             float eps) {
  const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (row >= rows) return;
  const int t = row / hv, h = row % hv;
  const uint2 xv = *reinterpret_cast<const uint2 *>(x + static_cast<int64_t>(row) * 128 + lane * 4);
  const uint2 zv = *reinterpret_cast<const uint2 *>(z + static_cast<int64_t>(t) * z_ld + z_off + h * 128 + lane * 4);
  const half *xh = reinterpret_cast<const half *>(&xv);
  const half *zh = reinterpret_cast<const half *>(&zv);
  float f[4], ss = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    f[i] = __half2float(xh[i]);
    ss += f[i] * f[i];
  }
  ss = warp_sum(ss);
  const float inv = rsqrtf(ss / 128.0f + eps);
  uint2 ov;
  half *oh = reinterpret_cast<half *>(&ov);
#pragma unroll
  for (int i = 0; i < 4; ++i) oh[i] = __float2half_rn(f[i] * inv * weight[lane * 4 + i] * silu(__half2float(zh[i])));
  *reinterpret_cast<uint2 *>(out + static_cast<int64_t>(row) * 128 + lane * 4) = ov;
}

template <int kTaps>
int launch_conv(half *out, half *conv_state, const half *y, int ld, const float *conv_w, const int *query_start,
                const int *context_lens, const int *state_slots, int num_seqs, int tokens, int channels,
                cudaStream_t stream) {
  const dim3 grid((tokens + kConvTokens - 1) / kConvTokens, (channels + kConvThreads - 1) / kConvThreads);
  gdn_conv_kernel<kTaps><<<grid, kConvThreads, 0, stream>>>(out, conv_state, y, ld, conv_w, query_start,
                                                            context_lens, state_slots, num_seqs, tokens, channels);
  gdn_conv_state_kernel<kTaps><<<dim3((channels + kConvThreads - 1) / kConvThreads, num_seqs), kConvThreads, 0,
                                  stream>>>(conv_state, y, ld, query_start, context_lens, state_slots, channels);
  return launch_status();
}

}  // namespace
}  // namespace seventy

SEVENTY_API int seventy_gdn_conv(void *out, void *conv_state, const void *y, int ld, const float *conv_w,
                                 const int *query_start, const int *context_lens, const int *state_slots,
                                 int num_seqs, int tokens, int channels, int taps, cudaStream_t stream) {
  using namespace seventy;
  if (out == nullptr || conv_state == nullptr || y == nullptr || conv_w == nullptr || query_start == nullptr ||
      context_lens == nullptr || state_slots == nullptr || num_seqs <= 0 || tokens < 0 || channels <= 0 ||
      ld < channels)
    return SEVENTY_ERR_INVALID_ARGUMENT;
  if (tokens == 0) return SEVENTY_OK;
  auto *o = static_cast<half *>(out);
  auto *cs = static_cast<half *>(conv_state);
  const auto *yy = static_cast<const half *>(y);
  switch (taps) {
    case 2: return launch_conv<2>(o, cs, yy, ld, conv_w, query_start, context_lens, state_slots, num_seqs, tokens,
                                  channels, stream);
    case 3: return launch_conv<3>(o, cs, yy, ld, conv_w, query_start, context_lens, state_slots, num_seqs, tokens,
                                  channels, stream);
    case 4: return launch_conv<4>(o, cs, yy, ld, conv_w, query_start, context_lens, state_slots, num_seqs, tokens,
                                  channels, stream);
    default: return SEVENTY_ERR_INVALID_ARGUMENT;
  }
}

SEVENTY_API int seventy_gdn_gates(float *g, float *beta, const void *y, int ld, int a_off, int b_off,
                                  const float *a_log, const float *dt_bias, int tokens, int hv, cudaStream_t stream) {
  using namespace seventy;
  if (g == nullptr || beta == nullptr || y == nullptr || a_log == nullptr || dt_bias == nullptr || tokens < 0 ||
      hv <= 0)
    return SEVENTY_ERR_INVALID_ARGUMENT;
  if (tokens == 0) return SEVENTY_OK;
  const int64_t n = static_cast<int64_t>(tokens) * hv;
  gdn_gates_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(
      g, beta, static_cast<const half *>(y), ld, a_off, b_off, a_log, dt_bias, tokens, hv);
  return launch_status();
}

SEVENTY_API int seventy_gated_rms_norm(void *out, const void *x, const void *z, int z_ld, int z_off,
                                       const float *weight, int tokens, int hv, int dv, float eps,
                                       cudaStream_t stream) {
  using namespace seventy;
  if (out == nullptr || x == nullptr || z == nullptr || weight == nullptr || tokens < 0 || hv <= 0 || dv != 128 ||
      z_ld % 4 || z_off % 4)
    return SEVENTY_ERR_INVALID_ARGUMENT;
  if (!aligned(out, 8) || !aligned(x, 8) || !aligned(z, 8)) return SEVENTY_ERR_MISALIGNED;
  const int rows = tokens * hv;
  if (rows == 0) return SEVENTY_OK;
  gated_rms_norm_kernel<<<(rows + 7) / 8, 256, 0, stream>>>(static_cast<half *>(out), static_cast<const half *>(x),
                                                            static_cast<const half *>(z), z_ld, z_off, weight, rows,
                                                            hv, eps);
  return launch_status();
}
