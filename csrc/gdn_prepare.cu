#include "common.cuh"

/*
 * Gated DeltaNet prefill: the parallel part of the chunked delta rule (the algebra is in
 * seventy/ops/gdn_prefill.py), fused. One block of 256 threads per (chunk of 64 tokens, value head), all in
 * FP32 from the FP16 convolution outputs:
 *
 *   1. q, k (key head h / (Hv / Hk)) and v rows go to shared memory; per row the L2 inverse norms (q also
 *      scaled by 1 / sqrt(Dk)), G = cumsum(g) over the chunk, and beta. Padded rows (past the sequence) are zero.
 *   2. Gamma Q = exp(G) Q, (K Lambda)^T = (K exp(G_last - G))^T and exp(G_last) are written.
 *   3. q_t . k_j and k_t . k_j on the tensor cores (mma.m8n8k4, FP16 inputs, FP32 sums; the products of FP16
 *      values are exact in FP32): warp w takes rows t = 8 w .. 8 w + 7 of both 64 x 64 products.
 *      P[t, j] = exp(G_t - G_j) q_t . k_j (j <= t) is written; A[t, j] = beta_t exp(G_t - G_j) k_t . k_j (j < t)
 *      stays in shared memory. The q, k and v rows have a pitch of 132 halves (8-byte aligned fragment loads;
 *      the 8 rows of a load hit distinct banks).
 *   4. U' = (I + A)^-1 (beta V) and W = (I + A)^-1 (beta exp(G) K) by forward substitution on the right-hand
 *      sides, X[t] = R[t] - sum_{j < t} A[t, j] X[j]: a thread per output column (128 for U', 128 for W) keeps its
 *      column of X in registers and reads the rows of A with 16-byte broadcast loads (rows padded to a multiple of
 *      4 floats; A[t, j] = 0 for j >= t).
 *
 * Outputs (FP16 unless noted) as gdn_chunk_scan reads them: w, u, gq [Hv, chunks, 64, 128], p [Hv, chunks, 64,
 * 64], klt [Hv, chunks, 128, 64], eg FP32 [Hv, chunks].
 */

namespace seventy {
namespace {

constexpr int kL = 64;
constexpr int kD = 128;
constexpr int kThreads = 256;
constexpr int kPitch = kD + 4;
constexpr int kLdA = kL + 4;
constexpr int kSmem = 3 * kL * kPitch * static_cast<int>(sizeof(half)) +
                      (kL * kLdA + 6 * kL) * static_cast<int>(sizeof(float));

__device__ __forceinline__ void mma_8x8x4(float (&c)[8], uint2 a, uint2 b) {
  asm volatile(
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7])
      : "r"(a.x), "r"(a.y), "r"(b.x), "r"(b.y));
}

__global__ void __launch_bounds__(kThreads, 1) gdn_prepare_kernel(
    half *w_out, half *u_out, half *gq_out, half *p_out, half *klt_out, float *eg_out, const half *qkv, int ld,
    const float *g, const float *beta, const int *chunk_row, const int *chunk_len, int chunks, int hk, int hv) {
  extern __shared__ __align__(16) unsigned char smem[];
  half *qs = reinterpret_cast<half *>(smem);
  half *ks = qs + kL * kPitch;
  half *vs = ks + kL * kPitch;
  float *A = reinterpret_cast<float *>(vs + kL * kPitch);
  float *G = A + kL * kLdA;
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

#pragma unroll
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
    const int quad = (lane >> 2) & 3;
    const int idx = (lane & 3) + ((lane >> 4) << 2);
    float qk[2][8], kk[2][8];
#pragma unroll
    for (int n = 0; n < 2; ++n)
#pragma unroll
      for (int jj = 0; jj < 8; ++jj) qk[n][jj] = kk[n][jj] = 0.0f;
#pragma unroll 4
    for (int k = 0; k < kD; k += 4) {
      const uint2 aq = *reinterpret_cast<const uint2 *>(qs + (8 * warp + idx) * kPitch + k);
      const uint2 ak = *reinterpret_cast<const uint2 *>(ks + (8 * warp + idx) * kPitch + k);
#pragma unroll
      for (int n = 0; n < 2; ++n) {
        const uint2 bk = *reinterpret_cast<const uint2 *>(ks + (32 * n + 8 * quad + idx) * kPitch + k);
        mma_8x8x4(qk[n], aq, bk);
        mma_8x8x4(kk[n], ak, bk);
      }
    }
#pragma unroll
    for (int n = 0; n < 2; ++n)
#pragma unroll
      for (int jj = 0; jj < 8; ++jj) {
        const int t = 8 * warp + (idx & 1) + 2 * ((jj >> 1) & 1) + 4 * (idx >> 2);
        const int j = 32 * n + 8 * quad + (jj & 1) + 2 * ((idx >> 1) & 1) + 4 * (jj >> 2);
        const float decay = j <= t ? expf(G[t] - G[j]) : 0.0f;
        p_out[slab * kL * kL + t * kL + j] = __float2half_rn(j <= t ? decay * (qn[t] * kn[j] * qk[n][jj]) : 0.0f);
        A[t * kLdA + j] = j < t ? bt[t] * decay * (kn[t] * kn[j] * kk[n][jj]) : 0.0f;
      }
  }
  __syncthreads();

  {
    const bool is_u = tid < kD;
    const int d = tid & (kD - 1);
    const half *src = is_u ? vs : ks;
    float x[kL];
#pragma unroll
    for (int t = 0; t < kL; ++t) x[t] = 0.0f;
#pragma unroll
    for (int t = 0; t < kL; ++t) {
      float acc[4] = {(is_u ? bt[t] : rw[t]) * __half2float(src[t * kPitch + d]), 0.0f, 0.0f, 0.0f};
      const float4 *arow = reinterpret_cast<const float4 *>(A + t * kLdA);
#pragma unroll
      for (int q = 0; q < kL / 4; ++q) {
        if (4 * q < t) {
          const float4 a = arow[q];
          acc[0] = fmaf(-a.x, x[4 * q], acc[0]);
          acc[1] = fmaf(-a.y, x[4 * q + 1], acc[1]);
          acc[2] = fmaf(-a.z, x[4 * q + 2], acc[2]);
          acc[3] = fmaf(-a.w, x[4 * q + 3], acc[3]);
        }
      }
      x[t] = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    }
    half *dst = (is_u ? u_out : w_out) + slab * kL * kD + d;
#pragma unroll
    for (int t = 0; t < kL; ++t) dst[t * kD] = __float2half_rn(x[t]);
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
