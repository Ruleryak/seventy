#include "common.cuh"

/*
 * Gated DeltaNet for decode and verify steps (a few new tokens per sequence), fused: causal convolution with the
 * cached inputs, SiLU, L2 norms of q and k, the decay and write gates, the delta-rule state update and output,
 * and the gated RMS norm. Prefill (long token runs) uses the chunked form instead.
 *
 * Per token, with a = exp(g) and L2-normalized q (also scaled by 1 / sqrt(Dk)) and k: S = a * S,
 * delta = beta * (v - S^T k), S = S + k delta^T, o = S^T q, out = rms_norm(o) * w * silu(z). The convolution
 * output, o and the result are rounded to FP16 where the reference rounds them.
 *
 * One block of 512 threads per (sequence, value head). The columns of S [Dk = 128, Dv = 128] are independent in
 * the recurrence: column v belongs to the four lanes 4 (v % 8) .. + 3 of warp v / 8, and each of the four holds
 * 32 of its rows in registers (float4 groups of rows, interleaved, so that the lanes read different shared memory
 * banks). The state is read once and written once.
 *
 * The tokens run in chunks of kChunk. Per chunk, everything that does not depend on the state runs first, for
 * all tokens of the chunk in parallel: the inputs are staged in shared memory, then the convolution (lane 4c: the
 * q channel v of the key head, 4c + 1: the k channel, 4c + 2: the value channel), the L2 norms, k . q and the
 * gates (one warp per token). The token loop then has no barrier: one pass over the state gives S^T k and S^T q
 * after the decay, the four lanes add their parts with shuffles, and o = S'^T q = S^T q + delta * (k . q) needs
 * no second pass over q. The gated RMS norm runs after the loop for the whole chunk. Only the token loop is
 * unrolled over the registers: the code that runs once per chunk stays compact (instruction cache).
 *
 * Inputs come straight from the in_proj output y [T, ld]: q, k, v (the convolution channels: key heads, key
 * heads, value heads), then z [Hv * Dv], b [Hv], a [Hv]. Value head h reads key head h / (Hv / Hk). The q and k
 * convolution history is written back by the first value head of each key head only. The number of
 * convolution taps (2 to 4; 4 in Qwen3.5) is a template parameter, so that the taps stay in registers.
 *
 * Speculative decoding runs the kernel twice. The verify pass (commit = 0) computes the outputs of all drafted
 * tokens and leaves the state and convolution history untouched. After acceptance, the commit pass (commit = 1,
 * out may be null) replays only the first commit_lens[b] rows of each sequence and writes the state back.
 */

namespace seventy {
namespace {

constexpr int kD = 128;
constexpr int kThreads = 512;
constexpr int kLanes = 4;
constexpr int kRows = kD / kLanes;
constexpr int kChunk = 16;
constexpr int kWarps = kThreads / 32;
constexpr int kCols = 32 / kLanes;

__device__ __forceinline__ float round_half(float x) { return __half2float(__float2half_rn(x)); }

__device__ __forceinline__ float softplus(float x) { return x > 20.0f ? x : log1pf(expf(x)); }

template <int kTaps>
__global__ void __launch_bounds__(kThreads) gdn_recurrent_kernel(
    half *out, half *conv_state, float *state, const half *y, int ld, const float *conv_w, const float *a_log,
    const float *dt_bias, const float *norm_w, const int *query_start, const int *context_lens,
    const int *state_slots, const int *commit_lens, int commit, int hk, int hv, float eps) {
  __shared__ __align__(16) half raw[kChunk][4 * kD];
  __shared__ __align__(16) float qs[kChunk][kD];
  __shared__ __align__(16) float ks[kChunk][kD];
  __shared__ float vs[kChunk][kD];
  __shared__ float ab[kChunk][2];
  __shared__ float kq_s[kChunk], decay_s[kChunk], beta_s[kChunk], rinv_s[kChunk];
  const int b = blockIdx.x / hv;
  const int h = blockIdx.x % hv;
  const int t0 = query_start[b];
  const int tokens = commit_lens != nullptr ? commit_lens[b] : query_start[b + 1] - t0;
  if (tokens <= 0) return;
  const bool fresh = context_lens[b] == 0;
  const int slot = state_slots[b];
  const int rep = hv / hk;
  const int kh = h / rep;
  const int key_size = hk * kD;
  const int conv_dim = 2 * key_size + hv * kD;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int quarter = lane & (kLanes - 1);
  const int v = warp * kCols + lane / kLanes;
  const bool outputs = out != nullptr;

  const int ch = quarter == 0 ? kh * kD + v : quarter == 1 ? key_size + kh * kD + v
                 : quarter == 2 ? 2 * key_size + h * kD + v : -1;
  const bool writes_conv = ch >= 0 && (quarter == 2 || h % rep == 0);
  float w[kTaps], hist[kTaps - 1];
  half *conv_slot = conv_state + (static_cast<int64_t>(slot) * conv_dim + (ch < 0 ? 0 : ch)) * (kTaps - 1);
#pragma unroll
  for (int j = 0; j < kTaps; ++j) w[j] = ch < 0 ? 0.0f : conv_w[(ch < 0 ? 0 : ch) * kTaps + j];
#pragma unroll
  for (int j = 0; j < kTaps - 1; ++j) hist[j] = (ch < 0 || fresh) ? 0.0f : __half2float(conv_slot[j]);
  const int raw_ch = quarter * kD + v;

  float S[kRows];
  float *sp = state + ((static_cast<int64_t>(slot) * hv + h) * kD) * kD;
#pragma unroll
  for (int g = 0; g < kRows / 4; ++g)
#pragma unroll
    for (int j = 0; j < 4; ++j) S[4 * g + j] = fresh ? 0.0f : sp[(16 * g + 4 * quarter + j) * kD + v];

  const float decay_scale = -expf(a_log[h]);
  const float bias = dt_bias[h];
  const float qscale = rsqrtf(static_cast<float>(kD));
  const int ov = tid & (kD - 1);
  const float nw = norm_w[ov];
  const int seg = tid / kD;
  const int src = seg == 0 ? kh * kD + ov : seg == 1 ? key_size + kh * kD + ov
                  : seg == 2 ? 2 * key_size + h * kD + ov : conv_dim + h * kD + ov;
  const bool staged = outputs || seg < 3;

  for (int c0 = 0; c0 < tokens; c0 += kChunk) {
    const int n = min(kChunk, tokens - c0);
    const half *rows = y + static_cast<int64_t>(t0 + c0) * ld;
    half xr[kChunk];
#pragma unroll
    for (int t = 0; t < kChunk; ++t)
      xr[t] = (t < n && staged) ? rows[static_cast<int64_t>(t) * ld + src] : __float2half(0.0f);
    if (tid < 2 * n) {
      const int t = tid >> 1;
      ab[t][tid & 1] = __half2float(rows[static_cast<int64_t>(t) * ld + conv_dim + hv * kD + (tid & 1) * hv + h]);
    }
#pragma unroll
    for (int t = 0; t < kChunk; ++t) raw[t][tid] = xr[t];
    __syncthreads();

    if (ch >= 0) {
      float *dst = quarter == 0 ? &qs[0][v] : quarter == 1 ? &ks[0][v] : &vs[0][v];
#pragma unroll 1
      for (int t = 0; t < n; ++t) {
        const float x = __half2float(raw[t][raw_ch]);
        float acc = w[kTaps - 1] * x;
#pragma unroll
        for (int j = 0; j < kTaps - 1; ++j) acc += w[j] * hist[j];
#pragma unroll
        for (int j = 0; j < kTaps - 2; ++j) hist[j] = hist[j + 1];
        hist[kTaps - 2] = x;
        dst[t * kD] = round_half(acc / (1.0f + expf(-acc)));
      }
    }
    if (tid < n) {
      decay_s[tid] = expf(decay_scale * softplus(ab[tid][1] + bias));
      beta_s[tid] = 1.0f / (1.0f + expf(-ab[tid][0]));
    }
    __syncthreads();
    if (warp < n) {
      const int t = warp;
      float4 q4 = reinterpret_cast<const float4 *>(qs[t])[lane];
      float4 k4 = reinterpret_cast<const float4 *>(ks[t])[lane];
      const float qn = qscale * rsqrtf(warp_sum(q4.x * q4.x + q4.y * q4.y + q4.z * q4.z + q4.w * q4.w) + 1e-6f);
      const float kn = rsqrtf(warp_sum(k4.x * k4.x + k4.y * k4.y + k4.z * k4.z + k4.w * k4.w) + 1e-6f);
      q4 = make_float4(q4.x * qn, q4.y * qn, q4.z * qn, q4.w * qn);
      k4 = make_float4(k4.x * kn, k4.y * kn, k4.z * kn, k4.w * kn);
      const float kq = warp_sum(q4.x * k4.x + q4.y * k4.y + q4.z * k4.z + q4.w * k4.w);
      reinterpret_cast<float4 *>(qs[t])[lane] = q4;
      reinterpret_cast<float4 *>(ks[t])[lane] = k4;
      if (lane == 0) kq_s[t] = kq;
    }
    __syncthreads();

    float *os = reinterpret_cast<float *>(&raw[0][0]);
#pragma unroll 1
    for (int t = 0; t < n; ++t) {
      const float4 *k4 = reinterpret_cast<const float4 *>(ks[t]) + quarter;
      const float4 *q4 = reinterpret_cast<const float4 *>(qs[t]) + quarter;
      const float decay = decay_s[t];
      float sk[2] = {0.0f, 0.0f};
      float sq[2] = {0.0f, 0.0f};
#pragma unroll
      for (int g = 0; g < kRows / 4; ++g) {
        const float4 kk = k4[4 * g];
        S[4 * g] *= decay;
        S[4 * g + 1] *= decay;
        S[4 * g + 2] *= decay;
        S[4 * g + 3] *= decay;
        sk[g & 1] += S[4 * g] * kk.x + S[4 * g + 1] * kk.y + S[4 * g + 2] * kk.z + S[4 * g + 3] * kk.w;
        if (outputs) {
          const float4 qq = q4[4 * g];
          sq[g & 1] += S[4 * g] * qq.x + S[4 * g + 1] * qq.y + S[4 * g + 2] * qq.z + S[4 * g + 3] * qq.w;
        }
      }
      float dot_k = sk[0] + sk[1];
      dot_k += __shfl_xor_sync(0xffffffffu, dot_k, 1);
      dot_k += __shfl_xor_sync(0xffffffffu, dot_k, 2);
      const float delta = beta_s[t] * (vs[t][v] - dot_k);
#pragma unroll
      for (int g = 0; g < kRows / 4; ++g) {
        const float4 kk = k4[4 * g];
        S[4 * g] += kk.x * delta;
        S[4 * g + 1] += kk.y * delta;
        S[4 * g + 2] += kk.z * delta;
        S[4 * g + 3] += kk.w * delta;
      }
      if (outputs) {
        float dot_q = sq[0] + sq[1];
        dot_q += __shfl_xor_sync(0xffffffffu, dot_q, 1);
        dot_q += __shfl_xor_sync(0xffffffffu, dot_q, 2);
        if (quarter == 0) os[t * 2 * kD + v] = round_half(dot_q + delta * kq_s[t]);
      }
    }

    if (outputs) {
      __syncthreads();
      if (warp < n) {
        const float4 o4 = reinterpret_cast<const float4 *>(os + warp * 2 * kD)[lane];
        const float s2 = warp_sum(o4.x * o4.x + o4.y * o4.y + o4.z * o4.z + o4.w * o4.w);
        if (lane == 0) rinv_s[warp] = rsqrtf(s2 / kD + eps);
      }
      __syncthreads();
#pragma unroll 1
      for (int t = tid / kD; t < n; t += kThreads / kD) {
        const float z = __half2float(raw[t][3 * kD + ov]);
        const float r = os[t * 2 * kD + ov] * rinv_s[t] * nw * (z / (1.0f + expf(-z)));
        out[static_cast<int64_t>(t0 + c0 + t) * hv * kD + h * kD + ov] = __float2half_rn(r);
      }
    }
    __syncthreads();
  }

  if (!commit) return;
#pragma unroll
  for (int g = 0; g < kRows / 4; ++g)
#pragma unroll
    for (int j = 0; j < 4; ++j) sp[(16 * g + 4 * quarter + j) * kD + v] = S[4 * g + j];
  if (writes_conv)
#pragma unroll
    for (int j = 0; j < kTaps - 1; ++j) conv_slot[j] = __float2half_rn(hist[j]);
}

}  // namespace
}  // namespace seventy

SEVENTY_API int seventy_gdn_recurrent(void *out, void *conv_state, float *state, const void *y, int ld,
                                      const float *conv_w, const float *a_log, const float *dt_bias,
                                      const float *norm_w, const int *query_start, const int *context_lens,
                                      const int *state_slots, const int *commit_lens, int commit, int num_seqs,
                                      int hk, int hv, int dk, int dv, int taps, float eps, cudaStream_t stream) {
  using namespace seventy;
  if ((out == nullptr && commit_lens == nullptr) || conv_state == nullptr || state == nullptr || y == nullptr || conv_w == nullptr ||
      a_log == nullptr || dt_bias == nullptr || norm_w == nullptr || query_start == nullptr ||
      context_lens == nullptr || state_slots == nullptr)
    return SEVENTY_ERR_INVALID_ARGUMENT;
  if (dk != kD || dv != kD || hk <= 0 || hv <= 0 || hv % hk || taps < 2 || taps > 4 || num_seqs < 0)
    return SEVENTY_ERR_INVALID_ARGUMENT;
  if (num_seqs == 0) return SEVENTY_OK;
  const auto kernel = taps == 2 ? gdn_recurrent_kernel<2> : taps == 3 ? gdn_recurrent_kernel<3> : gdn_recurrent_kernel<4>;
  kernel<<<static_cast<unsigned>(num_seqs * hv), kThreads, 0, stream>>>(
      static_cast<half *>(out), static_cast<half *>(conv_state), state, static_cast<const half *>(y), ld, conv_w,
      a_log, dt_bias, norm_w, query_start, context_lens, state_slots, commit_lens, commit, hk, hv, eps);
  return launch_status();
}
