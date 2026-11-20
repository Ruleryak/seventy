#include <cfloat>
#include <mma.h>

#include "common.cuh"
#include "verify_int8.cuh"

/*
 * verify_int8: speculative verify attention over the INT8 KV cache (head size 256) for up to 48 query rows per
 * sequence and KV head, on Volta WMMA tiles. It writes the same partial outputs as verify_attention_wmma
 * (csrc/flash_attention.cu), which verify_combine_kernel merges and gates.
 *
 * One block of 16 warps per (split, sequence, KV head), 1 block per SM. Rows: r = token * group + head, as in
 * verify_attention_wmma. Each split takes a range of whole 64-key tiles. Per tile:
 *   - K and V (64 keys x 256 bytes each) and their scales are in registers from the previous iteration. K goes to
 *     shared memory as exact FP16 integers; V stays INT8 and is converted while its B fragments load, so each V
 *     value is converted once per block.
 *   - 12 warps compute S = Q K^T, one 16 x 16 tile each (3 row tiles x 4 key tiles), and do the softmax on the
 *     accumulator registers. Each row tile's 4 warps reduce the row maxima and sums through shared memory with a
 *     named barrier of 128 threads. The K scale multiplies each score column in FP32. The V scale times 256
 *     multiplies each P column before P is rounded to FP16; the output is divided by 256 at the end.
 *   - 16 warps compute O += P V; warp w owns output columns 16 w .. 16 w + 15 for all row tiles.
 *   - The next tile's K, V and scales load into registers with predicated loads (a C++ "if (ok) load" compiles to
 *     load + select, and the select waits for the data). The block-table entries for the tile after it load one
 *     iteration earlier, so no load waits for a table lookup.
 * Precision: FP16 Q, K, V and P operands with FP32 accumulation and FP32 softmax statistics.
 *
 * Shared-memory layout: Q and P rows are XOR-swizzled (byte-offset bit 6 flips on rows with bit 3 set): an 8-lane
 * phase of a row_major A fragment load reads rows {0-3, 8-11} or {4-7, 12-15}, which a 16-byte-aligned stride maps
 * to the same banks without the swizzle.
 *
 * WMMA fragment layouts on sm70 (probed):
 *   accumulator 16 x 16 FP32: lane l holds rows r, r + 2 and columns c, c + 1, c + 4, c + 5, with
 *     r = (l & 1) + 8 ((l >> 2) & 1) + 4 ((l >> 4) & 1), c = 2 ((l >> 1) & 1) + 8 ((l >> 3) & 1);
 *     element i: row + 2 ((i >> 1) & 1), column + (i & 1) + 4 ((i >> 2) & 1).
 *   matrix_a row_major 16 x 16: lane l holds row (l & 3) + 8 ((l >> 2) & 1) + 4 ((l >> 4) & 1), columns 0..15 in
 *     order.
 *   matrix_b row_major 16 x 16: lane l holds B[q + 4 j][n0 .. n0 + 3] in x[4 j .. 4 j + 3], j = 0..3, with
 *     q = l & 3, n0 = 4 ((l >> 4) & 1) + 8 ((l >> 3) & 1).
 */

namespace seventy {
namespace {

using namespace nvcuda;

constexpr int kD = 256;
constexpr int kRows = 48;
constexpr int kMTiles = kRows / 16;
constexpr int kBN = 64;
constexpr int kNTiles = kBN / 16;
constexpr int kQKWarps = kMTiles * kNTiles;
constexpr int kWarps = 16;
constexpr int kThreads = kWarps * 32;

constexpr int kQStride = kD + 8;
constexpr int kKStride = kD + 8;
constexpr int kVStride = kD + 16;
constexpr int kPStride = kBN + 8;

struct Smem {
  alignas(16) half q[kRows * kQStride];
  alignas(16) half k[kBN * kKStride];
  alignas(16) uint8_t v[kBN * kVStride];
  alignas(16) half p[kRows * kPStride];
  float red[kMTiles][kNTiles][16];
  float row_max[kRows];
  float row_sum[kRows];
  float row_alpha[kRows];
  float ks[kBN];
  float vs[kBN];
};

/*
 * Bytes 0, 1 (lo) or 2, 3 (hi) of four int8 values as two exact FP16 integers: b ^ 0x80 is b + 128 as a uint8;
 * under the FP16 exponent byte 0x64 it reads 1024 + b + 128, and 1152 is subtracted.
 */
__device__ __forceinline__ uint32_t i8x2_lo(uint32_t w) {
  const uint32_t x = __byte_perm(w ^ 0x80808080u, 0x64646464u, 0x4140);
  const __half2 h = __hsub2(*reinterpret_cast<const __half2 *>(&x), __float2half2_rn(1152.f));
  return *reinterpret_cast<const uint32_t *>(&h);
}

__device__ __forceinline__ uint32_t i8x2_hi(uint32_t w) {
  const uint32_t x = __byte_perm(w ^ 0x80808080u, 0x64646464u, 0x4342);
  const __half2 h = __hsub2(*reinterpret_cast<const __half2 *>(&x), __float2half2_rn(1152.f));
  return *reinterpret_cast<const uint32_t *>(&h);
}

__device__ __forceinline__ int swz(int row, int byte_off) { return byte_off ^ (((row >> 3) & 1) << 6); }

__device__ __forceinline__ void load_a_swz(wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> &f,
                                           const half *tile, int stride_halves, int col0, int lane) {
  const int r = (lane & 3) + ((lane >> 2) & 1) * 8 + ((lane >> 4) & 1) * 4;
  const char *p = reinterpret_cast<const char *>(tile) + r * stride_halves * 2 + swz(r, col0 * 2);
  uint4 *x = reinterpret_cast<uint4 *>(&f.x[0]);
  x[0] = *reinterpret_cast<const uint4 *>(p);
  x[1] = *reinterpret_cast<const uint4 *>(p + 16);
}

__device__ __forceinline__ void ldg_cg_pred(uint4 &r, const void *p, bool pred) {
  asm volatile("{\n\t.reg .pred q;\n\tsetp.ne.b32 q, %5, 0;\n\t@q ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];\n\t}"
               : "+r"(r.x), "+r"(r.y), "+r"(r.z), "+r"(r.w)
               : "l"(p), "r"(static_cast<int>(pred)));
}

__device__ __forceinline__ void ldg_f32_pred(float &r, const void *p, bool pred) {
  asm volatile("{\n\t.reg .pred q;\n\tsetp.ne.b32 q, %2, 0;\n\t@q ld.global.cg.f32 %0, [%1];\n\t}"
               : "+f"(r)
               : "l"(p), "r"(static_cast<int>(pred)));
}

__device__ __forceinline__ void bar_group(int id) { asm volatile("bar.sync %0, 128;" ::"r"(id) : "memory"); }

__global__ void __launch_bounds__(kThreads, 1)
    verify_int8_kernel(float *__restrict__ partial_o, float *__restrict__ partial_ml, const half *__restrict__ q,
                       const int8_t *__restrict__ k_cache, const int8_t *__restrict__ v_cache,
                       const float *__restrict__ k_scale, const float *__restrict__ v_scale,
                       const int *__restrict__ block_tables, int table_width, const int *__restrict__ query_start,
                       const int *__restrict__ context_lens, int block_shift, int hq, int hkv, int splits,
                       int min_split_keys, float qk_scale) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  Smem &s = *reinterpret_cast<Smem *>(smem_raw);
  const int split = blockIdx.x, b = blockIdx.y, kh = blockIdx.z;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int group = hq / hkv;
  const int t0 = query_start[b];
  const int tokens = query_start[b + 1] - t0;
  const int rows = group * tokens;
  if (tokens <= 0 || rows > kRows) return;
  const int ctx = context_lens[b];
  const int total = ctx + tokens;
  const int total_tiles = (total + kBN - 1) / kBN;
  const int used = max(1, min(min(splits, total_tiles), (total + min_split_keys - 1) / min_split_keys));
  auto pidx = [&](int r) {
    return (static_cast<int64_t>(t0 + r / group) * hq + kh * group + r % group) * splits + split;
  };
  if (split >= used) {
    for (int r = tid; r < rows; r += kThreads) {
      partial_ml[pidx(r) * 2] = -FLT_MAX;
      partial_ml[pidx(r) * 2 + 1] = 0.0f;
    }
    return;
  }
  const int base_tiles = total_tiles / used;
  const int extra = total_tiles - base_tiles * used;
  const int tile0 = split * base_tiles + min(split, extra);
  const int ntiles = base_tiles + (split < extra ? 1 : 0);
  const int row_tiles = (rows + 15) / 16;

  for (int idx = tid; idx < kRows * (kD / 8); idx += kThreads) {
    const int r = idx / (kD / 8), c = idx % (kD / 8);
    uint4 val = make_uint4(0, 0, 0, 0);
    if (r < rows)
      val = __ldg(reinterpret_cast<const uint4 *>(
                      q + (static_cast<int64_t>(t0 + r / group) * hq + kh * group + r % group) * kD) +
                  c);
    *reinterpret_cast<uint4 *>(reinterpret_cast<char *>(s.q + r * kQStride) + swz(r, c * 16)) = val;
  }
  if (tid < kRows) {
    s.row_max[tid] = -INFINITY;
    s.row_sum[tid] = 0.f;
  }

  const int *table = block_tables + static_cast<int64_t>(b) * table_width;
  const int last_block = (total - 1) >> block_shift;
  const int bmask = (1 << block_shift) - 1;
  const int my_tok = tid >> 4, chunk = tid & 15;
  int blk_next[2], blk_scale_next = 0;
  auto lookup = [&](int t) {
    const int tok0 = t * kBN;
#pragma unroll
    for (int j = 0; j < 2; ++j) blk_next[j] = __ldg(table + min((tok0 + my_tok + 32 * j) >> block_shift, last_block));
    if (tid < kBN) blk_scale_next = __ldg(table + min((tok0 + tid) >> block_shift, last_block));
  };

  uint4 pk[2], pv[2];
  float psk = 0.f, psv = 0.f;
  auto load_tile = [&](int t) {
    const int tok0 = t * kBN;
    int blk[2];
    blk[0] = blk_next[0];
    blk[1] = blk_next[1];
    const int blk_scale = blk_scale_next;
    if (t + 1 < tile0 + ntiles) lookup(t + 1);
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int pos = tok0 + my_tok + 32 * j;
      const bool ok = pos < total;
      const int64_t vec = ((static_cast<int64_t>(blk[j]) << block_shift) + (pos & bmask)) * hkv + kh;
      pk[j] = make_uint4(0, 0, 0, 0);
      pv[j] = make_uint4(0, 0, 0, 0);
      ldg_cg_pred(pk[j], k_cache + vec * kD + chunk * 16, ok);
      ldg_cg_pred(pv[j], v_cache + vec * kD + chunk * 16, ok);
    }
    if (tid < kBN) {
      const int pos = tok0 + tid;
      const bool ok = pos < total;
      const int64_t vec = ((static_cast<int64_t>(blk_scale) << block_shift) + (pos & bmask)) * hkv + kh;
      psk = 0.f;
      psv = 0.f;
      ldg_f32_pred(psk, k_scale + vec, ok);
      ldg_f32_pred(psv, v_scale + vec, ok);
    }
  };
  auto store_tile = [&]() {
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int tok = my_tok + 32 * j;
      uint4 lo, hi;
      lo.x = i8x2_lo(pk[j].x);
      lo.y = i8x2_hi(pk[j].x);
      lo.z = i8x2_lo(pk[j].y);
      lo.w = i8x2_hi(pk[j].y);
      hi.x = i8x2_lo(pk[j].z);
      hi.y = i8x2_hi(pk[j].z);
      hi.z = i8x2_lo(pk[j].w);
      hi.w = i8x2_hi(pk[j].w);
      uint4 *kd = reinterpret_cast<uint4 *>(s.k + tok * kKStride + chunk * 16);
      const bool swap = chunk & 4;
      kd[swap ? 1 : 0] = swap ? hi : lo;
      kd[swap ? 0 : 1] = swap ? lo : hi;
      *reinterpret_cast<uint4 *>(s.v + tok * kVStride + chunk * 16) = pv[j];
    }
    if (tid < kBN) {
      s.ks[tid] = psk;
      s.vs[tid] = psv * 256.f;
    }
  };

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> o[kMTiles];
#pragma unroll
  for (int i = 0; i < kMTiles; ++i) wmma::fill_fragment(o[i], 0.f);

  const int ar = (lane & 1) + ((lane >> 2) & 1) * 8 + ((lane >> 4) & 1) * 4;
  const int ac = ((lane >> 1) & 1) * 2 + ((lane >> 3) & 1) * 8;
  const bool qk_warp = warp < kQKWarps && warp / kNTiles < row_tiles;
  const int mt = warp < kQKWarps ? warp / kNTiles : 0;
  const int wn = warp < kQKWarps ? warp % kNTiles : 0;
  const int grow0 = mt * 16 + ar;
  const int grow1 = grow0 + 2;
  const int len0 = grow0 < rows ? ctx + grow0 / group + 1 : 0;
  const int len1 = grow1 < rows ? ctx + grow1 / group + 1 : 0;
  const int vq = lane & 3;
  const int vn0 = ((lane >> 4) & 1) * 4 + ((lane >> 3) & 1) * 8;

  lookup(tile0);
  __syncthreads();
  load_tile(tile0);

  for (int it = 0; it < ntiles; ++it) {
    const int t = tile0 + it;
    store_tile();
    __syncthreads();
    if (it + 1 < ntiles) load_tile(t + 1);
    if (qk_warp) {
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> sc;
      wmma::fill_fragment(sc, 0.f);
      {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> qa;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> kb;
        const half *qbase = s.q + mt * 16 * kQStride;
        const half *kbase = s.k + wn * 16 * kKStride;
#pragma unroll 4
        for (int kk = 0; kk < kD; kk += 16) {
          load_a_swz(qa, qbase, kQStride, kk, lane);
          wmma::load_matrix_sync(kb, kbase + kk, kKStride);
          wmma::mma_sync(sc, qa, kb, sc);
        }
      }
      const int tc = wn * 16 + ac;
      const float ksc[4] = {s.ks[tc], s.ks[tc + 1], s.ks[tc + 4], s.ks[tc + 5]};
      const float vsc[4] = {s.vs[tc], s.vs[tc + 1], s.vs[tc + 4], s.vs[tc + 5]};
      const int col_base = t * kBN + tc;
      float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int col = col_base + (i & 1) + ((i >> 2) & 1) * 4;
        const bool second = (i >> 1) & 1;
        float val = sc.x[i] * qk_scale * ksc[(i & 1) + ((i >> 2) & 1) * 2];
        val = col < (second ? len1 : len0) ? val : -INFINITY;
        sc.x[i] = val;
        if (second)
          mx1 = fmaxf(mx1, val);
        else
          mx0 = fmaxf(mx0, val);
      }
      mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
      mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 8));
      mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
      mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 8));
      if ((lane & 0xA) == 0) {
        s.red[mt][wn][ar] = mx0;
        s.red[mt][wn][ar + 2] = mx1;
      }
      bar_group(1 + mt);
      float tmax0 = s.red[mt][0][ar], tmax1 = s.red[mt][0][ar + 2];
#pragma unroll
      for (int n = 1; n < kNTiles; ++n) {
        tmax0 = fmaxf(tmax0, s.red[mt][n][ar]);
        tmax1 = fmaxf(tmax1, s.red[mt][n][ar + 2]);
      }
      const float om0 = s.row_max[grow0], om1 = s.row_max[grow1];
      const float nm0 = fmaxf(om0, tmax0), nm1 = fmaxf(om1, tmax1);
      const float sub0 = nm0 == -INFINITY ? 0.f : nm0;
      const float sub1 = nm1 == -INFINITY ? 0.f : nm1;
      float sum0 = 0.f, sum1 = 0.f;
      char *prow0 = reinterpret_cast<char *>(s.p + grow0 * kPStride);
      char *prow1 = reinterpret_cast<char *>(s.p + grow1 * kPStride);
#pragma unroll
      for (int i = 0; i < 8; i += 2) {
        const bool second = (i >> 1) & 1;
        const float sb = second ? sub1 : sub0;
        const float p0 = __expf(sc.x[i] - sb);
        const float p1 = __expf(sc.x[i + 1] - sb);
        const int vi = ((i >> 2) & 1) * 2;
        const int cl = ac + ((i >> 2) & 1) * 4;
        *reinterpret_cast<__half2 *>((second ? prow1 : prow0) + swz(second ? grow1 : grow0, (wn * 16 + cl) * 2)) =
            __floats2half2_rn(p0 * vsc[vi], p1 * vsc[vi + 1]);
        if (second)
          sum1 += p0 + p1;
        else
          sum0 += p0 + p1;
      }
      sum0 += __shfl_xor_sync(0xffffffffu, sum0, 2);
      sum0 += __shfl_xor_sync(0xffffffffu, sum0, 8);
      sum1 += __shfl_xor_sync(0xffffffffu, sum1, 2);
      sum1 += __shfl_xor_sync(0xffffffffu, sum1, 8);
      bar_group(1 + mt);
      if ((lane & 0xA) == 0) {
        s.red[mt][wn][ar] = sum0;
        s.red[mt][wn][ar + 2] = sum1;
      }
      bar_group(1 + mt);
      if (wn == 0 && (lane & 0xA) == 0) {
        const float a0 = om0 == -INFINITY ? 0.f : __expf(om0 - nm0);
        const float a1 = om1 == -INFINITY ? 0.f : __expf(om1 - nm1);
        float ts0 = 0.f, ts1 = 0.f;
#pragma unroll
        for (int n = 0; n < kNTiles; ++n) {
          ts0 += s.red[mt][n][ar];
          ts1 += s.red[mt][n][ar + 2];
        }
        s.row_sum[grow0] = s.row_sum[grow0] * a0 + ts0;
        s.row_sum[grow1] = s.row_sum[grow1] * a1 + ts1;
        s.row_max[grow0] = nm0;
        s.row_max[grow1] = nm1;
        s.row_alpha[grow0] = a0;
        s.row_alpha[grow1] = a1;
      }
    }
    __syncthreads();

#pragma unroll
    for (int m = 0; m < kMTiles; ++m) {
      if (m >= row_tiles) break;
      const float a0 = s.row_alpha[m * 16 + ar];
      const float a1 = s.row_alpha[m * 16 + ar + 2];
#pragma unroll
      for (int i = 0; i < 8; ++i) o[m].x[i] *= ((i >> 1) & 1) ? a1 : a0;
    }
    {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> pa;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> vb;
      const uint8_t *vcol = s.v + vq * kVStride + warp * 16 + vn0;
#pragma unroll
      for (int kt = 0; kt < kNTiles; ++kt) {
        uint32_t *vx = reinterpret_cast<uint32_t *>(&vb.x[0]);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const uint32_t w = *reinterpret_cast<const uint32_t *>(vcol + (kt * 16 + 4 * j) * kVStride);
          vx[2 * j] = i8x2_lo(w);
          vx[2 * j + 1] = i8x2_hi(w);
        }
#pragma unroll
        for (int m = 0; m < kMTiles; ++m) {
          if (m >= row_tiles) break;
          load_a_swz(pa, s.p + m * 16 * kPStride, kPStride, kt * 16, lane);
          wmma::mma_sync(o[m], pa, vb, o[m]);
        }
      }
    }
    __syncthreads();
  }

  constexpr float kInv256 = 1.f / 256.f;
#pragma unroll
  for (int m = 0; m < kMTiles; ++m) {
#pragma unroll
    for (int i = 0; i < 8; i += 2) {
      const int row = m * 16 + ar + ((i >> 1) & 1) * 2;
      const int col = warp * 16 + ac + ((i >> 2) & 1) * 4;
      if (row < rows)
        *reinterpret_cast<float2 *>(partial_o + pidx(row) * kD + col) =
            make_float2(o[m].x[i] * kInv256, o[m].x[i + 1] * kInv256);
    }
  }
  if (tid < rows) {
    const float sm = s.row_sum[tid];
    partial_ml[pidx(tid) * 2] = sm > 0.f ? s.row_max[tid] : -FLT_MAX;
    partial_ml[pidx(tid) * 2 + 1] = sm;
  }
}

}  // namespace

bool verify_int8_supported(int hq, int hkv, int max_query_len, int block_size) {
  return hkv > 0 && hq % hkv == 0 && (hq / hkv) * max_query_len <= kRows && block_size > 0 &&
         (block_size & (block_size - 1)) == 0;
}

int launch_verify_int8(float *partial_o, float *partial_ml, const half *q, const int8_t *k_cache,
                       const int8_t *v_cache, const float *k_scale, const float *v_scale, const int *block_tables,
                       int table_width, const int *query_start, const int *context_lens, int num_seqs, int block_size,
                       int hq, int hkv, int splits, int min_split_keys, float scale, cudaStream_t stream) {
  cudaError_t e = cudaFuncSetAttribute(verify_int8_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(Smem));
  if (e == cudaSuccess) e = cudaFuncSetAttribute(verify_int8_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  if (e != cudaSuccess) return static_cast<int>(e);
  int shift = 0;
  while ((1 << shift) < block_size) ++shift;
  verify_int8_kernel<<<dim3(splits, num_seqs, hkv), kThreads, sizeof(Smem), stream>>>(
      partial_o, partial_ml, q, k_cache, v_cache, k_scale, v_scale, block_tables, table_width, query_start,
      context_lens, shift, hq, hkv, splits, max(1, min_split_keys), scale);
  return launch_status();
}

}  // namespace seventy
