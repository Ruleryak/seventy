#include "common.cuh"

/*
 * Gated DeltaNet prefill: the parallel part of the chunked delta rule (the algebra is in
 * seventy/ops/gdn_prefill.py), fused. One block of 256 threads per (chunk of 64 tokens, value head), all in
 * FP32 from the FP16 convolution outputs:
 *
 *   1. q, k (key head h / (Hv / Hk)) and v rows go to shared memory; per row the L2 inverse norms (q also
 *      scaled by 1 / sqrt(Dk)), G = cumsum(g) over the chunk, and beta. Padded rows (past the sequence) are zero.
 *   2. Gamma Q = exp(G) Q, (K Lambda)^T = (K exp(G_last - G))^T and exp(G_last) are written.
 *   3. One pass over Dk gives q_t . k_j and k_t . k_j: each thread owns 16 (t, j) pairs, t = ti + 16 a and
 *      j = tj + 16 b, so the lower-triangular work is balanced and the k rows of a warp hit distinct banks (odd
 *      row pitch in words). P[t, j] = exp(G_t - G_j) q_t . k_j (j <= t) is written; A[t, j] =
 *      beta_t exp(G_t - G_j) k_t . k_j (j < t) stays in shared memory.
 *   4. T = (I + A)^-1 by forward substitution, four threads per column.
 *   5. U' = T (beta V) and W = T (beta exp(G) K): a thread per output column (128 for U', 128 for W) keeps its
 *      right-hand-side column in registers and walks the rows of T.
 *
 * Outputs (FP16 unless noted) as gdn_chunk_scan reads them: w, u, gq [Hv, chunks, 64, 128], p [Hv, chunks, 64,
 * 64], klt [Hv, chunks, 128, 64], eg FP32 [Hv, chunks].
 */

namespace seventy {
namespace {

constexpr int kL = 64;
constexpr int kD = 128;
constexpr int kThreads = 256;
constexpr int kPitch = kD + 2;
constexpr int kLdA = kL + 1;
constexpr int kSmem = 3 * kL * kPitch * static_cast<int>(sizeof(half)) +
                      (2 * kL * kLdA + 6 * kL) * static_cast<int>(sizeof(float));

__global__ void __launch_bounds__(kThreads) gdn_prepare_kernel(
    half *w_out, half *u_out, half *gq_out, half *p_out, half *klt_out, float *eg_out, const half *qkv, int ld,
    const float *g, const float *beta, const int *chunk_row, const int *chunk_len, int chunks, int hk, int hv) {
  extern __shared__ __align__(16) unsigned char smem[];
  half *qs = reinterpret_cast<half *>(smem);
  half *ks = qs + kL * kPitch;
  half *vs = ks + kL * kPitch;
  float *A = reinterpret_cast<float *>(vs + kL * kPitch);
  float *T = A + kL * kLdA;
  float *G = T + kL * kLdA;
  float *bt = G + kL;
  float *qn = bt + kL;
  float *kn = qn + kL;
  float *rw = kn + kL;
  float *eG = rw + kL;

  const int c = blockIdx.x;
  const int h = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int row0 = chunk_row[c];
  const int len = chunk_len[c];
  const int kh = h / (hv / hk);
  const int key_size = hk * kD;
  const int64_t slab = static_cast<int64_t>(h) * chunks + c;

  for (int e = tid; e < kL * (kD / 2); e += kThreads) {
    const int r = e / (kD / 2);
    const int d2 = e % (kD / 2);
    half2 q2 = __float2half2_rn(0.0f), k2 = q2, v2 = q2;
    if (r < len) {
      const half *row = qkv + static_cast<int64_t>(row0 + r) * ld;
      q2 = reinterpret_cast<const half2 *>(row + kh * kD)[d2];
      k2 = reinterpret_cast<const half2 *>(row + key_size + kh * kD)[d2];
      v2 = reinterpret_cast<const half2 *>(row + 2 * key_size + h * kD)[d2];
    }
    reinterpret_cast<half2 *>(qs + r * kPitch)[d2] = q2;
    reinterpret_cast<half2 *>(ks + r * kPitch)[d2] = k2;
    reinterpret_cast<half2 *>(vs + r * kPitch)[d2] = v2;
  }
  if (warp == 0) {
    float g0 = 0.0f, g1 = 0.0f;
    const int r0 = 2 * lane, r1 = 2 * lane + 1;
    if (r0 < len) g0 = g[static_cast<int64_t>(row0 + r0) * hv + h];
    if (r1 < len) g1 = g[static_cast<int64_t>(row0 + r1) * hv + h];
    float s = g0 + g1;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      const float o = __shfl_up_sync(0xffffffffu, s, off);
      if (lane >= off) s += o;
    }
    G[r0] = s - g1;
    G[r1] = s;
  } else if (warp == 1) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int r = lane + 32 * i;
      bt[r] = r < len ? beta[static_cast<int64_t>(row0 + r) * hv + h] : 0.0f;
    }
  }
  __syncthreads();

  for (int r = warp; r < kL; r += kThreads / 32) {
    float sq = 0.0f, sk = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const float2 q2 = __half22float2(reinterpret_cast<const half2 *>(qs + r * kPitch)[lane + 32 * i]);
      const float2 k2 = __half22float2(reinterpret_cast<const half2 *>(ks + r * kPitch)[lane + 32 * i]);
      sq += q2.x * q2.x + q2.y * q2.y;
      sk += k2.x * k2.x + k2.y * k2.y;
    }
    sq = warp_sum(sq);
    sk = warp_sum(sk);
    if (lane == 0) {
      qn[r] = rsqrtf(sq + 1e-6f) * rsqrtf(static_cast<float>(kD));
      kn[r] = rsqrtf(sk + 1e-6f);
    }
  }
  __syncthreads();
  const float last = G[kL - 1];
  if (tid < kL) {
    eG[tid] = expf(G[tid]);
    rw[tid] = bt[tid] * expf(G[tid]) * kn[tid];
  }
  if (tid == 0) eg_out[slab] = expf(last);
  __syncthreads();

  for (int e = tid; e < kL * kD; e += kThreads) {
    const int t = e / kD;
    const int d = e % kD;
    gq_out[slab * kL * kD + e] = __float2half_rn(eG[t] * (qn[t] * __half2float(qs[t * kPitch + d])));
  }
  for (int e = tid; e < kD * kL; e += kThreads) {
    const int d = e / kL;
    const int t = e % kL;
    klt_out[slab * kD * kL + e] =
        __float2half_rn(kn[t] * __half2float(ks[t * kPitch + d]) * expf(last - G[t]));
  }

  {
    const int ti = tid / 16, tj = tid % 16;
    float qk[4][4], kk[4][4];
#pragma unroll
    for (int a = 0; a < 4; ++a)
#pragma unroll
      for (int b = 0; b < 4; ++b) qk[a][b] = kk[a][b] = 0.0f;
#pragma unroll 2
    for (int d2 = 0; d2 < kD / 2; ++d2) {
      float2 qa[4], ka[4], kb[4];
#pragma unroll
      for (int a = 0; a < 4; ++a) {
        qa[a] = __half22float2(reinterpret_cast<const half2 *>(qs + (ti + 16 * a) * kPitch)[d2]);
        ka[a] = __half22float2(reinterpret_cast<const half2 *>(ks + (ti + 16 * a) * kPitch)[d2]);
        kb[a] = __half22float2(reinterpret_cast<const half2 *>(ks + (tj + 16 * a) * kPitch)[d2]);
      }
#pragma unroll
      for (int a = 0; a < 4; ++a)
#pragma unroll
        for (int b = 0; b < 4; ++b) {
          qk[a][b] += qa[a].x * kb[b].x + qa[a].y * kb[b].y;
          kk[a][b] += ka[a].x * kb[b].x + ka[a].y * kb[b].y;
        }
    }
#pragma unroll
    for (int a = 0; a < 4; ++a)
#pragma unroll
      for (int b = 0; b < 4; ++b) {
        const int t = ti + 16 * a, j = tj + 16 * b;
        const float decay = j <= t ? expf(G[t] - G[j]) : 0.0f;
        p_out[slab * kL * kL + t * kL + j] = __float2half_rn(j <= t ? decay * (qn[t] * kn[j] * qk[a][b]) : 0.0f);
        A[t * kLdA + j] = j < t ? bt[t] * decay * (kn[t] * kn[j] * kk[a][b]) : 0.0f;
      }
  }
  __syncthreads();

  {
    const int j = tid / 4, part = tid % 4;
    for (int t = 0; t < kL; ++t) {
      float s = 0.0f;
      for (int i = j + part; i < t; i += 4) s += A[t * kLdA + i] * T[i * kLdA + j];
      s += __shfl_xor_sync(0xffffffffu, s, 1);
      s += __shfl_xor_sync(0xffffffffu, s, 2);
      if (part == 0) T[t * kLdA + j] = t < j ? 0.0f : (t == j ? 1.0f : -s);
      __syncwarp();
    }
  }
  __syncthreads();

  {
    const bool is_u = tid < kD;
    const int d = tid & (kD - 1);
    const half *src = is_u ? vs : ks;
    float r[kL];
#pragma unroll
    for (int jj = 0; jj < kL; ++jj) r[jj] = (is_u ? bt[jj] : rw[jj]) * __half2float(src[jj * kPitch + d]);
    half *dst = (is_u ? u_out : w_out) + slab * kL * kD + d;
#pragma unroll 1
    for (int t = 0; t < kL; ++t) {
      const float *trow = T + t * kLdA;
      float acc0 = 0.0f, acc1 = 0.0f;
#pragma unroll
      for (int jj = 0; jj < kL; jj += 2) {
        acc0 += trow[jj] * r[jj];
        acc1 += trow[jj + 1] * r[jj + 1];
      }
      dst[t * kD] = __float2half_rn(acc0 + acc1);
    }
  }
}

}  // namespace
}  // namespace seventy

SEVENTY_API int seventy_gdn_prepare(void *w, void *u, void *gq, void *p, void *klt, float *eg, const void *qkv,
                                    int ld, const float *g, const float *beta, const int *chunk_row,
                                    const int *chunk_len, int chunks, int hk, int hv, int dk, int dv,
                                    cudaStream_t stream) {
  using namespace seventy;
  if (w == nullptr || u == nullptr || gq == nullptr || p == nullptr || klt == nullptr || eg == nullptr ||
      qkv == nullptr || g == nullptr || beta == nullptr || chunk_row == nullptr || chunk_len == nullptr)
    return SEVENTY_ERR_INVALID_ARGUMENT;
  if (dk != kD || dv != kD || hk <= 0 || hv <= 0 || hv % hk || chunks < 0 || ld % 2) return SEVENTY_ERR_INVALID_ARGUMENT;
  if (!aligned(qkv, 4)) return SEVENTY_ERR_MISALIGNED;
  if (chunks == 0) return SEVENTY_OK;
  static bool configured = false;
  if (!configured) {
    const cudaError_t e = cudaFuncSetAttribute(gdn_prepare_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem);
    if (e != cudaSuccess) return static_cast<int>(e);
    configured = true;
  }
  gdn_prepare_kernel<<<dim3(chunks, hv), kThreads, kSmem, stream>>>(
      static_cast<half *>(w), static_cast<half *>(u), static_cast<half *>(gq), static_cast<half *>(p),
      static_cast<half *>(klt), eg, static_cast<const half *>(qkv), ld, g, beta, chunk_row, chunk_len, chunks, hk, hv);
  return launch_status();
}
