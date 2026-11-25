#include "common.cuh"

/*
 * Per-layer n-gram embeddings (PLE, Qwen4-Exp) on the GPU, for seventy.model.qwen4_exp.PLE.
 *
 * seventy_ple_ids: the table row of each head for each new token. Token p of a sequence and its predecessors: the
 * token at p - s counts as the end-of-sequence token (eos) when any token at p - s .. p - 1 is eos or p - s < 0.
 * The sequence's tokens before the step come from its state (state[slot]: its last n - 1 tokens, oldest first;
 * none when context_lens[b] is 0). For n-gram size g = 2 .. n: mix = XOR over positions s < g of
 * (token(p - s) * mult[s]) in 64-bit two's complement, and head h of that size takes row
 * offsets[h] + (mix mod sizes[h]) (the floor modulo: 0 .. sizes[h] - 1).
 *
 * seventy_ple_gather: NVFP4 rows (packed 4-bit values and E4M3 scales of 16, one global scale) of heads h0 .. h1 - 1
 * from tables in host memory that the GPU reads directly (pinned, mapped), at rows minus row_base; FP16 out
 * [rows, (h1 - h0) * dim].
 *
 * seventy_ple_conv: out = x + silu(depthwise causal convolution of xn over time with `taps` taps at dilation `dil`)
 * per sequence, continuing from its state (state[slot]: the last (taps - 1) * dil inputs xn, oldest first; zeros
 * when context_lens[b] is 0); x and xn FP32 [T, channels], out FP32.
 *
 * seventy_ple_state: after the step, the token state (last n - 1 tokens) and the convolution state of each
 * sequence with new tokens.
 */

namespace seventy {
namespace {

__device__ __forceinline__ int ple_seq(const int *query_start, int num_seqs, int t) {
  int b = 0;
  while (b + 1 < num_seqs && query_start[b + 1] <= t) ++b;
  return b;
}

__device__ __forceinline__ int64_t token_at(const int64_t *tokens, const int64_t *state, int n1, int t, int i,
                                            int ctx, int back, int64_t eos) {
  const int j = i - back;
  if (j >= 0) return tokens[t - back];
  const int k = -j;
  if (k > ctx || k > n1) return eos;
  return state[n1 - k];
}

__global__ void ple_ids_kernel(int64_t *ids, const int64_t *tokens, const int64_t *state, const int *slots,
                               const int *query_start, const int *context_lens, int num_seqs, int tokens_n, int n,
                               int heads_per, const int64_t *mult, const int64_t *sizes, const int64_t *offsets,
                               int64_t eos) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= tokens_n) return;
  const int b = ple_seq(query_start, num_seqs, t);
  const int i = t - query_start[b];
  const int ctx = context_lens[b];
  const int n1 = n - 1;
  const int64_t *st = state + static_cast<int64_t>(slots[b]) * n1;
  int64_t shifted[8];
  shifted[0] = tokens[t];
  bool bad = false;
  for (int s = 1; s < n; ++s) {
    const int64_t prev = token_at(tokens, st, n1, t, i, ctx, s, eos);
    bad = bad || prev == eos || (ctx + i - s) < 0;
    shifted[s] = bad ? eos : prev;
  }
  for (int g = 2; g <= n; ++g) {
    uint64_t mixed = static_cast<uint64_t>(shifted[0]) * static_cast<uint64_t>(mult[0]);
    for (int s = 1; s < g; ++s) mixed ^= static_cast<uint64_t>(shifted[s]) * static_cast<uint64_t>(mult[s]);
    const int64_t m = static_cast<int64_t>(mixed);
    for (int h = 0; h < heads_per; ++h) {
      const int head = (g - 2) * heads_per + h;
      int64_t r = m % sizes[head];
      if (r < 0) r += sizes[head];
      ids[static_cast<int64_t>(t) * (n1 * heads_per) + head] = r + offsets[head];
    }
  }
}

__global__ void ple_gather_kernel(half *out, const int64_t *ids, int heads_total, int h0, int h1,
                                  const uint8_t *packed, const uint8_t *scales, int64_t row_base, int dim,
                                  float global_scale) {
  const int t = blockIdx.x, j = blockIdx.y;
  const int h = h0 + j;
  const int64_t row = ids[static_cast<int64_t>(t) * heads_total + h] - row_base;
  const uint8_t *pk = packed + row * (dim / 2);
  const uint8_t *sc = scales + row * (dim / 16);
  for (int x = threadIdx.x; x < dim; x += blockDim.x) {
    const uint8_t byte = pk[x / 2];
    const uint32_t code = (x & 1) ? (byte >> 4) : (byte & 0xf);
    const float v = e2m1_to_float(code) * e4m3_to_float(sc[x / 16]) * global_scale;
    out[(static_cast<int64_t>(t) * (h1 - h0) + j) * dim + x] = __float2half_rn(v);
  }
}

__global__ void ple_conv_kernel(float *out, const float *x, const float *xn, const float *state, const float *w,
                                const int *slots, const int *query_start, const int *context_lens, int num_seqs,
                                int tokens_n, int channels, int taps, int dil) {
  const int64_t e = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (e >= static_cast<int64_t>(tokens_n) * channels) return;
  const int t = static_cast<int>(e / channels), ch = static_cast<int>(e % channels);
  const int b = ple_seq(query_start, num_seqs, t);
  const int i = t - query_start[b];
  const int ctx = context_lens[b];
  const int hist = (taps - 1) * dil;
  const float *st = state + static_cast<int64_t>(slots[b]) * hist * channels;
  float acc = 0.0f;
  for (int k = 0; k < taps; ++k) {
    const int lag = (taps - 1 - k) * dil;
    const int j = i - lag;
    float v = 0.0f;
    if (j >= 0) {
      v = xn[static_cast<int64_t>(t - lag) * channels + ch];
    } else if (ctx > 0 && -j <= hist) {
      v = st[static_cast<int64_t>(hist + j) * channels + ch];
    }
    acc = fmaf(w[ch * taps + k], v, acc);
  }
  out[e] = x[e] + acc / (1.0f + expf(-acc));
}

__global__ void ple_state_kernel(int64_t *tok_state, float *conv_state, const int64_t *tokens, const float *xn,
                                 const int *slots, const int *query_start, const int *context_lens, int n1,
                                 int channels, int hist, int64_t eos) {
  const int b = blockIdx.y;
  const int q0 = query_start[b], q1 = query_start[b + 1];
  const int len = q1 - q0;
  if (len <= 0) return;
  const int ctx = context_lens[b];
  int64_t *ts = tok_state + static_cast<int64_t>(slots[b]) * n1;
  float *cs = conv_state + static_cast<int64_t>(slots[b]) * hist * channels;
  const int ch = blockIdx.x * blockDim.x + threadIdx.x;
  if (ch < channels) {
    float keep[32];
    for (int r = 0; r < hist; ++r) {
      const int j = len - hist + r;
      if (j >= 0) {
        keep[r] = xn[static_cast<int64_t>(q0 + j) * channels + ch];
      } else {
        const int src = hist + j;
        keep[r] = (ctx > 0 && src >= 0) ? cs[static_cast<int64_t>(src) * channels + ch] : 0.0f;
      }
    }
    for (int r = 0; r < hist; ++r) cs[static_cast<int64_t>(r) * channels + ch] = keep[r];
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    int64_t keep[8];
    for (int r = 0; r < n1; ++r) {
      const int j = len - n1 + r;
      if (j >= 0) {
        keep[r] = tokens[q0 + j];
      } else {
        const int src = n1 + j;
        keep[r] = (ctx > 0 && src >= 0) ? ts[src] : eos;
      }
    }
    for (int r = 0; r < n1; ++r) ts[r] = keep[r];
  }
}

}  // namespace
}  // namespace seventy

SEVENTY_API int seventy_ple_ids(int64_t *ids, const int64_t *tokens, const int64_t *state, const int *slots,
                                const int *query_start, const int *context_lens, int num_seqs, int tokens_n, int n,
                                int heads_per, const int64_t *mult, const int64_t *sizes, const int64_t *offsets,
                                int64_t eos, cudaStream_t stream) {
  using namespace seventy;
  if (ids == nullptr || tokens == nullptr || state == nullptr || n < 2 || n > 8 || heads_per < 1 || num_seqs < 1)
    return SEVENTY_ERR_INVALID_ARGUMENT;
  if (tokens_n == 0) return SEVENTY_OK;
  ple_ids_kernel<<<(tokens_n + 127) / 128, 128, 0, stream>>>(ids, tokens, state, slots, query_start, context_lens,
                                                              num_seqs, tokens_n, n, heads_per, mult, sizes, offsets,
                                                              eos);
  return launch_status();
}

SEVENTY_API int seventy_ple_gather(void *out, const int64_t *ids, int heads_total, int h0, int h1, int tokens_n,
                                   const void *packed, const void *scales, int64_t row_base, int dim,
                                   float global_scale, cudaStream_t stream) {
  using namespace seventy;
  if (out == nullptr || ids == nullptr || packed == nullptr || scales == nullptr || h1 <= h0 || dim % 16)
    return SEVENTY_ERR_INVALID_ARGUMENT;
  if (tokens_n == 0) return SEVENTY_OK;
  ple_gather_kernel<<<dim3(tokens_n, h1 - h0), 160, 0, stream>>>(
      static_cast<half *>(out), ids, heads_total, h0, h1, static_cast<const uint8_t *>(packed),
      static_cast<const uint8_t *>(scales), row_base, dim, global_scale);
  return launch_status();
}

SEVENTY_API int seventy_ple_conv(float *out, const float *x, const float *xn, const float *state, const float *w,
                                 const int *slots, const int *query_start, const int *context_lens, int num_seqs,
                                 int tokens_n, int channels, int taps, int dil, cudaStream_t stream) {
  using namespace seventy;
  if (out == nullptr || x == nullptr || xn == nullptr || state == nullptr || w == nullptr || taps < 1 || dil < 1 ||
      (taps - 1) * dil > 32)
    return SEVENTY_ERR_INVALID_ARGUMENT;
  const int64_t total = static_cast<int64_t>(tokens_n) * channels;
  if (total == 0) return SEVENTY_OK;
  ple_conv_kernel<<<static_cast<unsigned>((total + 255) / 256), 256, 0, stream>>>(
      out, x, xn, state, w, slots, query_start, context_lens, num_seqs, tokens_n, channels, taps, dil);
  return launch_status();
}

SEVENTY_API int seventy_ple_state(int64_t *tok_state, float *conv_state, const int64_t *tokens, const float *xn,
                                  const int *slots, const int *query_start, const int *context_lens, int num_seqs,
                                  int n1, int channels, int hist, int64_t eos, cudaStream_t stream) {
  using namespace seventy;
  if (tok_state == nullptr || conv_state == nullptr || n1 < 1 || n1 > 8 || hist > 32 || num_seqs < 1)
    return SEVENTY_ERR_INVALID_ARGUMENT;
  ple_state_kernel<<<dim3((channels + 255) / 256, num_seqs), 256, 0, stream>>>(
      tok_state, conv_state, tokens, xn, slots, query_start, context_lens, n1, channels, hist, eos);
  return launch_status();
}
