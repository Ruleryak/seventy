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
 *
 * With a workspace, two kernels run instead, as flash-linear-attention's chunked gated delta rule splits the work
 * (FLA, MIT license): only the state recurrence is sequential.
 *   gdn_state_scan_kernel: one block per (sequence, value head, 32 state columns); the columns of S are
 *     independent in U = U' - W S and S = exp(G_last) S + (K Lambda)^T U, so 4 blocks share a head. Per chunk it
 *     writes S at the chunk start (FP16, as S^T [Dv, Dk]) and U (FP16, as U^T [Dv, 64]) to the workspace. The next
 *     chunk's inputs load into registers while the current one computes.
 *   gdn_chunk_out_kernel: one block per (chunk, value head), all in parallel: O = (Gamma Q) S + P U.
 * The FP16 roundings are those of gdn_chunk_scan_kernel (S and U feed the tensor cores in FP16 there too).
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

constexpr int kCols = 32;
constexpr int kSplits = kD / kCols;

/*
 * Per chunk, the block's inputs: W [64, 128], (K Lambda)^T [128, 64] and the block's 32 columns of U' [64, 32]
 * (36 KB), 9 uint4 per thread of 256.
 */
struct ScanLoads {
  uint4 v[9];
};

__device__ __forceinline__ void scan_load(ScanLoads &r, const half *w, const half *klt, const half *u, int col0) {
#pragma unroll
  for (int i = 0; i < 4; ++i) r.v[i] = reinterpret_cast<const uint4 *>(w)[threadIdx.x + i * 256];
#pragma unroll
  for (int i = 0; i < 4; ++i) r.v[4 + i] = reinterpret_cast<const uint4 *>(klt)[threadIdx.x + i * 256];
  const int row = threadIdx.x >> 2, part = threadIdx.x & 3;
  r.v[8] = *reinterpret_cast<const uint4 *>(u + row * kD + col0 + part * 8);
}

__device__ __forceinline__ void scan_store(const ScanLoads &r, half *ws, half *ks, half *us) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int e = threadIdx.x + i * 256;
    const int row = e / (kD / 8), c = (e % (kD / 8)) * 8;
    *reinterpret_cast<uint2 *>(ws + row * kLdD + c) = make_uint2(r.v[i].x, r.v[i].y);
    *reinterpret_cast<uint2 *>(ws + row * kLdD + c + 4) = make_uint2(r.v[i].z, r.v[i].w);
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int e = threadIdx.x + i * 256;
    const int row = e / (kL / 8), c = (e % (kL / 8)) * 8;
    *reinterpret_cast<uint2 *>(ks + row * kLdL + c) = make_uint2(r.v[4 + i].x, r.v[4 + i].y);
    *reinterpret_cast<uint2 *>(ks + row * kLdL + c + 4) = make_uint2(r.v[4 + i].z, r.v[4 + i].w);
  }
  const int row = threadIdx.x >> 2, part = threadIdx.x & 3;
  const half *h = reinterpret_cast<const half *>(&r.v[8]);
#pragma unroll
  for (int j = 0; j < 8; ++j) us[row * (kCols + 4) + part * 8 + j] = h[j];
}

constexpr int kScanSmem =
    (kL * kLdD + kD * kLdL + kL * (kCols + 4) + kCols * kLdD + kCols * kLdL) * static_cast<int>(sizeof(half));

__global__ void __launch_bounds__(kWarps * 32, 1) gdn_state_scan_kernel(
    float *state, half *hs, half *vt, const half *w_g, const half *u_g, const half *klt_g, const float *eg_g,
    const int *chunk_start, const int *query_start, const int *context_lens, const int *state_slots, int hv,
    int64_t total_chunks) {
  extern __shared__ half smem[];
  half *ws = smem;
  half *ks = ws + kL * kLdD;
  half *us = ks + kD * kLdL;
  half *st = us + kL * (kCols + 4);
  half *ut = st + kCols * kLdD;
  const int b = blockIdx.x;
  const int h = blockIdx.y / kSplits;
  const int col0 = (blockIdx.y % kSplits) * kCols;
  const int warp = threadIdx.x >> 5;
  const int tokens = query_start[b + 1] - query_start[b];
  const int nchunks = (tokens + kL - 1) / kL;
  if (nchunks == 0) return;
  const bool fresh = context_lens[b] == 0;
  float *sp = state + (static_cast<int64_t>(state_slots[b]) * hv + h) * kD * kD;

  float S[2][1][8];
#pragma unroll
  for (int t = 0; t < 2; ++t)
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int row = 16 * warp + 8 * t + frag_row(j);
      S[t][0][j] = fresh ? 0.0f : sp[row * kD + col0 + frag_col(j)];
    }

  const int64_t base = static_cast<int64_t>(h) * total_chunks + chunk_start[b];
  ScanLoads ld;
  scan_load(ld, w_g + base * kL * kD, klt_g + base * kD * kL, u_g + base * kL * kD, col0);
  for (int c = 0; c < nchunks; ++c) {
    const int64_t ci = base + c;
    scan_store(ld, ws, ks, us);
#pragma unroll
    for (int t = 0; t < 2; ++t)
#pragma unroll
      for (int j = 0; j < 8; ++j)
        st[frag_col(j) * kLdD + 16 * warp + 8 * t + frag_row(j)] = __float2half_rn(S[t][0][j]);
    __syncthreads();
    if (c + 1 < nchunks)
      scan_load(ld, w_g + (ci + 1) * kL * kD, klt_g + (ci + 1) * kD * kL, u_g + (ci + 1) * kL * kD, col0);
    {
      half *dst = hs + ci * kD * kD + static_cast<int64_t>(col0) * kD;
      for (int e = threadIdx.x; e < kCols * (kD / 8); e += kWarps * 32) {
        const int r = e / (kD / 8), k8 = (e % (kD / 8)) * 8;
        const uint2 a = *reinterpret_cast<const uint2 *>(st + r * kLdD + k8);
        const uint2 bb = *reinterpret_cast<const uint2 *>(st + r * kLdD + k8 + 4);
        *reinterpret_cast<uint4 *>(dst + r * kD + k8) = make_uint4(a.x, a.y, bb.x, bb.y);
      }
    }
    float U[1][1][8];
#pragma unroll
    for (int j = 0; j < 8; ++j) U[0][0][j] = 0.0f;
    warp_mma<1, 1>(U, ws, kLdD, 8 * warp, st, kLdD, 0, kD);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int row = 8 * warp + frag_row(j);
      const int col = frag_col(j);
      ut[col * kLdL + row] = __float2half_rn(__half2float(us[row * (kCols + 4) + col]) - U[0][0][j]);
    }
    __syncthreads();
    {
      half *dst = vt + ci * kD * kL + static_cast<int64_t>(col0) * kL;
      for (int e = threadIdx.x; e < kCols * (kL / 8); e += kWarps * 32) {
        const int r = e / (kL / 8), t8 = (e % (kL / 8)) * 8;
        const uint2 a = *reinterpret_cast<const uint2 *>(ut + r * kLdL + t8);
        const uint2 bb = *reinterpret_cast<const uint2 *>(ut + r * kLdL + t8 + 4);
        *reinterpret_cast<uint4 *>(dst + r * kL + t8) = make_uint4(a.x, a.y, bb.x, bb.y);
      }
    }
    const float decay = eg_g[ci];
#pragma unroll
    for (int t = 0; t < 2; ++t)
#pragma unroll
      for (int j = 0; j < 8; ++j) S[t][0][j] *= decay;
    warp_mma<2, 1>(S, ks, kLdL, 16 * warp, ut, kLdL, 0, kL);
    __syncthreads();
  }

#pragma unroll
  for (int t = 0; t < 2; ++t)
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int row = 16 * warp + 8 * t + frag_row(j);
      sp[row * kD + col0 + frag_col(j)] = S[t][0][j];
    }
}

constexpr int kOutSmem = (kL * kLdD + kD * kLdD + kL * kLdL + kD * kLdL) * static_cast<int>(sizeof(half));

__global__ void __launch_bounds__(kWarps * 32) gdn_chunk_out_kernel(half *out, const half *hs, const half *vt,
                                                                    const half *gq_g, const half *p_g,
                                                                    const int *chunk_start, const int *query_start,
                                                                    int num_seqs, int hv, int64_t total_chunks) {
  extern __shared__ half smem[];
  half *gq = smem;
  half *ht = gq + kL * kLdD;
  half *pk = ht + kD * kLdD;
  half *vs = pk + kL * kLdL;
  const int g = blockIdx.x;
  const int h = blockIdx.y;
  const int warp = threadIdx.x >> 5;
  int lo = 0, hi = num_seqs - 1;
  while (lo < hi) {
    const int mid = (lo + hi + 1) / 2;
    if (chunk_start[mid] <= g) lo = mid;
    else hi = mid - 1;
  }
  const int b = lo;
  const int t0 = query_start[b] + (g - chunk_start[b]) * kL;
  const int valid = min(kL, query_start[b + 1] - t0);
  if (valid <= 0) return;
  const int64_t ci = static_cast<int64_t>(h) * total_chunks + g;
  load_tile(gq, kLdD, gq_g + ci * kL * kD, kL, kD);
  load_tile(ht, kLdD, hs + ci * kD * kD, kD, kD);
  load_tile(pk, kLdL, p_g + ci * kL * kL, kL, kL);
  load_tile(vs, kLdL, vt + ci * kD * kL, kD, kL);
  __syncthreads();
  float O[1][4][8];
#pragma unroll
  for (int n = 0; n < 4; ++n)
#pragma unroll
    for (int j = 0; j < 8; ++j) O[0][n][j] = 0.0f;
  warp_mma<1, 4>(O, gq, kLdD, 8 * warp, ht, kLdD, 0, kD);
  warp_mma<1, 4>(O, pk, kLdL, 8 * warp, vs, kLdL, 0, kL);
#pragma unroll
  for (int n = 0; n < 4; ++n)
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int row = 8 * warp + frag_row(j);
      const int col = 32 * n + frag_col(j);
      if (row < valid) out[(static_cast<int64_t>(t0 + row) * hv + h) * kD + col] = __float2half_rn(O[0][n][j]);
    }
}

}  // namespace
}  // namespace seventy

SEVENTY_API int seventy_gdn_chunk_scan(void *out, float *state, const void *w, const void *u, const void *gq,
                                       const void *p, const void *klt, const float *eg, const int *chunk_start,
                                       const int *query_start, const int *context_lens, const int *state_slots,
                                       int num_seqs, int hv, int64_t total_chunks, void *hs, void *vt,
                                       cudaStream_t stream) {
  using namespace seventy;
  if (out == nullptr || state == nullptr || w == nullptr || u == nullptr || gq == nullptr || p == nullptr ||
      klt == nullptr || eg == nullptr || chunk_start == nullptr || query_start == nullptr ||
      context_lens == nullptr || state_slots == nullptr || hv <= 0 || num_seqs < 0)
    return SEVENTY_ERR_INVALID_ARGUMENT;
  if (num_seqs == 0) return SEVENTY_OK;
  if (hs != nullptr && vt != nullptr) {
    cudaError_t e = cudaFuncSetAttribute(gdn_state_scan_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kScanSmem);
    if (e == cudaSuccess)
      e = cudaFuncSetAttribute(gdn_chunk_out_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kOutSmem);
    if (e != cudaSuccess) return static_cast<int>(e);
    gdn_state_scan_kernel<<<dim3(num_seqs, hv * kSplits), kWarps * 32, kScanSmem, stream>>>(
        state, static_cast<half *>(hs), static_cast<half *>(vt), static_cast<const half *>(w),
        static_cast<const half *>(u), static_cast<const half *>(klt), eg, chunk_start, query_start, context_lens,
        state_slots, hv, total_chunks);
    gdn_chunk_out_kernel<<<dim3(static_cast<unsigned>(total_chunks), hv), kWarps * 32, kOutSmem, stream>>>(
        static_cast<half *>(out), static_cast<const half *>(hs), static_cast<const half *>(vt),
        static_cast<const half *>(gq), static_cast<const half *>(p), chunk_start, query_start, num_seqs, hv,
        total_chunks);
    return launch_status();
  }
  const cudaError_t e =
      cudaFuncSetAttribute(gdn_chunk_scan_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes);
  if (e != cudaSuccess) return static_cast<int>(e);
  gdn_chunk_scan_kernel<<<dim3(num_seqs, hv), kWarps * 32, kSmemBytes, stream>>>(
      static_cast<half *>(out), state, static_cast<const half *>(w), static_cast<const half *>(u),
      static_cast<const half *>(gq), static_cast<const half *>(p), static_cast<const half *>(klt), eg, chunk_start,
      query_start, context_lens, state_slots, hv, total_chunks);
  return launch_status();
}
