#include "common.cuh"

/*
 * Gated DeltaNet for decode and verify steps (a few new tokens per sequence), fused: causal convolution with the
 * cached inputs, SiLU, L2 norms of q and k, the decay and write gates, the delta-rule state update and output,
 * and the gated RMS norm. Prefill (long token runs) uses the chunked form instead.
 *
 * One block of 256 threads per (sequence, value head). Thread t owns column v = t % 128 and rows
 * i = 64 * (t / 128) .. + 64 of the head's FP32 state S [Dk = 128, Dv = 128], in registers, for all of the
 * sequence's new tokens; the state is read once and written once.
 *
 * Per token, with a = exp(g): kv = a * S^T k, delta = beta * (v - kv), S = a * S + k delta^T, o = S^T q,
 * out = rms_norm(o) * w * silu(z). q and k are L2-normalized and q is scaled by 1 / sqrt(Dk). The convolution
 * output, o and the result are rounded to FP16 where the reference rounds them.
 *
 * Inputs come straight from the in_proj output y [T, ld]: q, k, v (the convolution channels: key heads, key
 * heads, value heads), then z [Hv * Dv], b [Hv], a [Hv]. Value head h reads key head h / (Hv / Hk). The q and k
 * convolution history is written back by the first value head of each key head only.
 */

namespace seventy {
namespace {

constexpr int kD = 128;
constexpr int kThreads = 256;
constexpr int kRows = 64;
constexpr int kTaps = 4;

__device__ __forceinline__ float round_half(float x) { return __half2float(__float2half_rn(x)); }

__device__ __forceinline__ float softplus(float x) { return x > 20.0f ? x : log1pf(expf(x)); }

__device__ __forceinline__ float block_sum256(float v, float *scratch) { return block_sum(v, scratch); }

__global__ void __launch_bounds__(kThreads) gdn_recurrent_kernel(
    half *out, half *conv_state, float *state, const half *y, int ld, const float *conv_w, const float *a_log,
    const float *dt_bias, const float *norm_w, const int *query_start, const int *context_lens,
    const int *state_slots, int hk, int hv, int taps, float eps) {
  __shared__ float qs[kD], ks[kD], vs[kD];
  __shared__ float pair[2][kD];
  __shared__ float scratch[32];
  __shared__ float gate[2];
  const int b = blockIdx.x / hv;
  const int h = blockIdx.x % hv;
  const int t0 = query_start[b];
  const int tokens = query_start[b + 1] - t0;
  if (tokens <= 0) return;
  const bool fresh = context_lens[b] == 0;
  const int slot = state_slots[b];
  const int rep = hv / hk;
  const int kh = h / rep;
  const int key_size = hk * kD;
  const int conv_dim = 2 * key_size + hv * kD;
  const int tid = threadIdx.x;
  const int v = tid & (kD - 1);
  const int half_idx = tid >> 7;

  int ch[2];
  ch[0] = (tid < kD) ? kh * kD + tid : key_size + kh * kD + (tid - kD);
  ch[1] = (tid < kD) ? 2 * key_size + h * kD + tid : -1;
  float hist[2][kTaps - 1];
  float w[2][kTaps];
#pragma unroll
  for (int c = 0; c < 2; ++c) {
#pragma unroll
    for (int j = 0; j < kTaps; ++j) w[c][j] = 0.0f;
#pragma unroll
    for (int j = 0; j < kTaps - 1; ++j) hist[c][j] = 0.0f;
    if (ch[c] < 0) continue;
    for (int j = 0; j < taps; ++j) w[c][kTaps - taps + j] = conv_w[ch[c] * taps + j];
    if (!fresh)
      for (int j = 0; j < taps - 1; ++j)
        hist[c][kTaps - taps + j] =
            __half2float(conv_state[(static_cast<int64_t>(slot) * conv_dim + ch[c]) * (taps - 1) + j]);
  }

  float S[kRows];
  float *sp = state + ((static_cast<int64_t>(slot) * hv + h) * kD) * kD;
#pragma unroll
  for (int r = 0; r < kRows; ++r) S[r] = fresh ? 0.0f : sp[(half_idx * kRows + r) * kD + v];

  const float decay_scale = -expf(a_log[h]);
  const float bias = dt_bias[h];
  const float nw = norm_w[v];
  const float qscale = rsqrtf(static_cast<float>(kD));

  for (int t = 0; t < tokens; ++t) {
    const half *row = y + static_cast<int64_t>(t0 + t) * ld;
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      if (ch[c] < 0) continue;
      const float x = __half2float(row[ch[c]]);
      float acc = w[c][kTaps - 1] * x;
#pragma unroll
      for (int j = 0; j < kTaps - 1; ++j) acc += w[c][j] * hist[c][j];
#pragma unroll
      for (int j = 0; j < kTaps - 2; ++j) hist[c][j] = hist[c][j + 1];
      hist[c][kTaps - 2] = x;
      const float y_c = round_half(acc / (1.0f + expf(-acc)));
      if (c == 1) vs[tid] = y_c;
      else if (tid < kD) qs[tid] = y_c;
      else ks[tid - kD] = y_c;
    }
    if (tid == 0) {
      const float bv = __half2float(row[conv_dim + hv * kD + h]);
      const float av = __half2float(row[conv_dim + hv * kD + hv + h]);
      gate[0] = expf(decay_scale * softplus(av + bias));
      gate[1] = 1.0f / (1.0f + expf(-bv));
    }
    __syncthreads();
    const float qq = (tid < kD) ? qs[tid] * qs[tid] : ks[tid - kD] * ks[tid - kD];
    const float sq_q = block_sum256(tid < kD ? qq : 0.0f, scratch);
    const float sq_k = block_sum256(tid < kD ? 0.0f : qq, scratch);
    const float qn = qscale * rsqrtf(sq_q + 1e-6f);
    const float kn = rsqrtf(sq_k + 1e-6f);
    const float decay = gate[0];
    const float beta = gate[1];

    float part = 0.0f;
#pragma unroll
    for (int r = 0; r < kRows; ++r) part += S[r] * ks[half_idx * kRows + r];
    pair[half_idx][v] = part;
    __syncthreads();
    const float kv = decay * kn * (pair[0][v] + pair[1][v]);
    const float delta = beta * (vs[v] - kv);
    float opart = 0.0f;
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
      const int i = half_idx * kRows + r;
      S[r] = decay * S[r] + kn * ks[i] * delta;
      opart += S[r] * qs[i];
    }
    __syncthreads();
    pair[half_idx][v] = opart;
    __syncthreads();
    const float o = round_half(qn * (pair[0][v] + pair[1][v]));
    const float sq_o = block_sum256(half_idx == 0 ? o * o : 0.0f, scratch);
    if (half_idx == 0) {
      const float z = __half2float(row[conv_dim + h * kD + v]);
      const float r = o * rsqrtf(sq_o / kD + eps) * nw * (z / (1.0f + expf(-z)));
      out[static_cast<int64_t>(t0 + t) * hv * kD + h * kD + v] = __float2half_rn(r);
    }
    __syncthreads();
  }

#pragma unroll
  for (int r = 0; r < kRows; ++r) sp[(half_idx * kRows + r) * kD + v] = S[r];
#pragma unroll
  for (int c = 0; c < 2; ++c) {
    if (ch[c] < 0) continue;
    if (c == 0 && h % rep != 0) continue;
    for (int j = 0; j < taps - 1; ++j)
      conv_state[(static_cast<int64_t>(slot) * conv_dim + ch[c]) * (taps - 1) + j] =
          __float2half_rn(hist[c][kTaps - taps + j]);
  }
}

}  // namespace
}  // namespace seventy

SEVENTY_API int seventy_gdn_recurrent(void *out, void *conv_state, float *state, const void *y, int ld,
                                      const float *conv_w, const float *a_log, const float *dt_bias,
                                      const float *norm_w, const int *query_start, const int *context_lens,
                                      const int *state_slots, int num_seqs, int hk, int hv, int dk, int dv, int taps,
                                      float eps, cudaStream_t stream) {
  using namespace seventy;
  if (out == nullptr || conv_state == nullptr || state == nullptr || y == nullptr || conv_w == nullptr ||
      a_log == nullptr || dt_bias == nullptr || norm_w == nullptr || query_start == nullptr ||
      context_lens == nullptr || state_slots == nullptr)
    return SEVENTY_ERR_INVALID_ARGUMENT;
  if (dk != kD || dv != kD || hk <= 0 || hv <= 0 || hv % hk || taps < 2 || taps > kTaps || num_seqs < 0)
    return SEVENTY_ERR_INVALID_ARGUMENT;
  if (num_seqs == 0) return SEVENTY_OK;
  gdn_recurrent_kernel<<<static_cast<unsigned>(num_seqs * hv), kThreads, 0, stream>>>(
      static_cast<half *>(out), static_cast<half *>(conv_state), state, static_cast<const half *>(y), ld, conv_w,
      a_log, dt_bias, norm_w, query_start, context_lens, state_slots, hk, hv, taps, eps);
  return launch_status();
}
