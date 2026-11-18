#include "common.cuh"

/*
 * Gated DeltaNet prefill: the sequential part of the chunked delta rule (see seventy/ops/gdn_prefill.py for the
 * parallel part and the algebra). One block of 8 warps per (sequence, value head) walks the sequence's chunks of
 * 64 tokens in order. With the state S [Dk = 128, Dv = 128] at the start of chunk c:
 *
 *   U = U' - W S                      (64 x 128)
 *   O = (Gamma Q) S + P U             (64 x 128, written as FP16)
 *   S = exp(G_last) S + (K Lambda)^T U
 *
 * S stays in FP32 registers in mma accumulator layout (warp w: rows 16 w .. 16 w + 15, all 128 columns); an FP16
 * copy of S^T in shared memory feeds the tensor cores. Inputs per chunk (FP16, padded rows are zero): W, U',
 * Gamma Q [64, 128], P [64, 64], (K Lambda)^T [128, 64], and exp(G_last) in FP32.
 *
 * Shared-memory rows are padded (132 or 68 halves) so the fragment loads of a warp hit distinct bank pairs.
 */

namespace seventy {
namespace {

constexpr int kL = 64;
constexpr int kD = 128;
constexpr int kWarps = 8;
constexpr int kLdD = kD + 4;
constexpr int kLdL = kL + 4;

__device__ __forceinline__ void mma_8x8x4(float (&c)[8], uint2 a, uint2 b) {
  asm volatile(
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7])
      : "r"(a.x), "r"(a.y), "r"(b.x), "r"(b.y));
}

/*
 * acc[t][n] += A[r0 + 8 t .. + 8, 0..K) * B[0..K, c0 + 32 n .. + 32), with A row-major (lda) and B given as B^T
 * row-major (ldb), both FP16 in shared memory.
 */
template <int MT, int NT>
__device__ __forceinline__ void warp_mma(float (&acc)[MT][NT][8], const half *A, int lda, int r0, const half *BT,
                                         int ldb, int c0, int K) {
  const int lane = threadIdx.x & 31;
  const int quad = (lane >> 2) & 3;
  const int idx = (lane & 3) + ((lane >> 4) << 2);
#pragma unroll 4
  for (int k = 0; k < K; k += 4) {
    uint2 a[MT], b[NT];
#pragma unroll
    for (int t = 0; t < MT; ++t) a[t] = *reinterpret_cast<const uint2 *>(A + (r0 + 8 * t + idx) * lda + k);
#pragma unroll
    for (int n = 0; n < NT; ++n) b[n] = *reinterpret_cast<const uint2 *>(BT + (c0 + 32 * n + 8 * quad + idx) * ldb + k);
#pragma unroll
    for (int t = 0; t < MT; ++t)
#pragma unroll
      for (int n = 0; n < NT; ++n) mma_8x8x4(acc[t][n], a[t], b[n]);
  }
}

__device__ __forceinline__ int frag_row(int j) {
  const int idx = ((threadIdx.x & 31) & 3) + (((threadIdx.x & 31) >> 4) << 2);
  return (idx & 1) + 2 * ((j >> 1) & 1) + 4 * (idx >> 2);
}

__device__ __forceinline__ int frag_col(int j) {
  const int lane = threadIdx.x & 31;
  const int idx = (lane & 3) + ((lane >> 4) << 2);
  return 8 * ((lane >> 2) & 3) + (j & 1) + 2 * ((idx >> 1) & 1) + 4 * (j >> 2);
}

__device__ __forceinline__ void load_tile(half *dst, int ld, const half *src, int rows, int cols) {
  const int per_row = cols / 8;
  for (int e = threadIdx.x; e < rows * per_row; e += kWarps * 32) {
    const int r = e / per_row;
    const int c = (e % per_row) * 8;
    const uint4 v = *reinterpret_cast<const uint4 *>(src + static_cast<int64_t>(r) * cols + c);
    *reinterpret_cast<uint2 *>(dst + r * ld + c) = make_uint2(v.x, v.y);
    *reinterpret_cast<uint2 *>(dst + r * ld + c + 4) = make_uint2(v.z, v.w);
  }
}

__global__ void __launch_bounds__(kWarps * 32, 1) gdn_chunk_scan_kernel(
    half *out, float *state, const half *w_g, const half *u_g, const half *gq_g, const half *p_g, const half *klt_g,
    const float *eg_g, const int *chunk_start, const int *query_start, const int *context_lens,
    const int *state_slots, int hv, int64_t total_chunks) {
  extern __shared__ half smem[];
  half *wq = smem;
  half *st = wq + kL * kLdD;
  half *ut = st + kD * kLdD;
  half *pk = ut + kD * kLdL;
  const int b = blockIdx.x;
  const int h = blockIdx.y;
  const int warp = threadIdx.x >> 5;
  const int t0 = query_start[b];
  const int tokens = query_start[b + 1] - t0;
  const int nchunks = (tokens + kL - 1) / kL;
  const int slot = state_slots[b];
  const bool fresh = context_lens[b] == 0;
  float *sp = state + (static_cast<int64_t>(slot) * hv + h) * kD * kD;

  float S[2][4][8];
#pragma unroll
  for (int t = 0; t < 2; ++t)
#pragma unroll
    for (int n = 0; n < 4; ++n)
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int row = 16 * warp + 8 * t + frag_row(j);
        const int col = 32 * n + frag_col(j);
        S[t][n][j] = fresh ? 0.0f : sp[row * kD + col];
      }

  for (int c = 0; c < nchunks; ++c) {
    const int64_t ci = static_cast<int64_t>(h) * total_chunks + chunk_start[b] + c;
#pragma unroll
    for (int t = 0; t < 2; ++t)
#pragma unroll
      for (int n = 0; n < 4; ++n)
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const int row = 16 * warp + 8 * t + frag_row(j);
          const int col = 32 * n + frag_col(j);
          st[col * kLdD + row] = __float2half_rn(S[t][n][j]);
        }
    load_tile(wq, kLdD, w_g + ci * kL * kD, kL, kD);
    __syncthreads();

    float U[1][4][8];
#pragma unroll
    for (int n = 0; n < 4; ++n)
#pragma unroll
      for (int j = 0; j < 8; ++j) U[0][n][j] = 0.0f;
    warp_mma<1, 4>(U, wq, kLdD, 8 * warp, st, kLdD, 0, kD);
    const half *ug = u_g + ci * kL * kD;
#pragma unroll
    for (int n = 0; n < 4; ++n)
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int row = 8 * warp + frag_row(j);
        const int col = 32 * n + frag_col(j);
        const float u = __half2float(ug[row * kD + col]) - U[0][n][j];
        U[0][n][j] = u;
        ut[col * kLdL + row] = __float2half_rn(u);
      }
    __syncthreads();
    load_tile(wq, kLdD, gq_g + ci * kL * kD, kL, kD);
    load_tile(pk, kLdL, p_g + ci * kL * kL, kL, kL);
    __syncthreads();

    float O[1][4][8];
#pragma unroll
    for (int n = 0; n < 4; ++n)
#pragma unroll
      for (int j = 0; j < 8; ++j) O[0][n][j] = 0.0f;
    warp_mma<1, 4>(O, wq, kLdD, 8 * warp, st, kLdD, 0, kD);
    warp_mma<1, 4>(O, pk, kLdL, 8 * warp, ut, kLdL, 0, kL);
    const int valid = min(kL, tokens - c * kL);
#pragma unroll
    for (int n = 0; n < 4; ++n)
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int row = 8 * warp + frag_row(j);
        const int col = 32 * n + frag_col(j);
        if (row < valid) out[(static_cast<int64_t>(t0 + c * kL + row) * hv + h) * kD + col] = __float2half_rn(O[0][n][j]);
      }
    __syncthreads();
    load_tile(pk, kLdL, klt_g + ci * kD * kL, kD, kL);
    __syncthreads();

    const float decay = eg_g[ci];
#pragma unroll
    for (int t = 0; t < 2; ++t)
#pragma unroll
      for (int n = 0; n < 4; ++n)
#pragma unroll
        for (int j = 0; j < 8; ++j) S[t][n][j] *= decay;
    warp_mma<2, 4>(S, pk, kLdL, 16 * warp, ut, kLdL, 0, kL);
    __syncthreads();
  }

#pragma unroll
  for (int t = 0; t < 2; ++t)
#pragma unroll
    for (int n = 0; n < 4; ++n)
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int row = 16 * warp + 8 * t + frag_row(j);
        const int col = 32 * n + frag_col(j);
        sp[row * kD + col] = S[t][n][j];
      }
}

constexpr int kSmemBytes = (kL * kLdD + kD * kLdD + kD * kLdL + kD * kLdL) * static_cast<int>(sizeof(half));

}  // namespace
}  // namespace seventy

SEVENTY_API int seventy_gdn_chunk_scan(void *out, float *state, const void *w, const void *u, const void *gq,
                                       const void *p, const void *klt, const float *eg, const int *chunk_start,
                                       const int *query_start, const int *context_lens, const int *state_slots,
                                       int num_seqs, int hv, int64_t total_chunks, cudaStream_t stream) {
  using namespace seventy;
  if (out == nullptr || state == nullptr || w == nullptr || u == nullptr || gq == nullptr || p == nullptr ||
      klt == nullptr || eg == nullptr || chunk_start == nullptr || query_start == nullptr ||
      context_lens == nullptr || state_slots == nullptr || hv <= 0 || num_seqs < 0)
    return SEVENTY_ERR_INVALID_ARGUMENT;
  if (num_seqs == 0) return SEVENTY_OK;
  const cudaError_t e =
      cudaFuncSetAttribute(gdn_chunk_scan_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes);
  if (e != cudaSuccess) return static_cast<int>(e);
  gdn_chunk_scan_kernel<<<dim3(num_seqs, hv), kWarps * 32, kSmemBytes, stream>>>(
      static_cast<half *>(out), state, static_cast<const half *>(w), static_cast<const half *>(u),
      static_cast<const half *>(gq), static_cast<const half *>(p), static_cast<const half *>(klt), eg, chunk_start,
      query_start, context_lens, state_slots, hv, total_chunks);
  return launch_status();
}
