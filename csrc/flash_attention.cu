#include <cfloat>
#include <mma.h>

#include "common.cuh"
#include "verify_int8.cuh"

#include <cstdlib>
#include <cstring>

/*
 * Attention over the paged KV cache (head size 256) on Volta tensor cores, in tiles of 64 keys with an online
 * softmax in FP32.
 *
 * flash_attention_wmma (prefill): one block per (query tile of 64 tokens of one sequence, query head), keys up to the
 * causal limit of the tile's last row; the epilogue divides by the row sums and multiplies by sigmoid(gate).
 * tiles: int32 [num_tiles, 2] = (sequence, first new token of the tile within the sequence).
 *
 * verify_attention_wmma / verify_attention_kernel (speculative verify, a few new tokens per sequence): one block per
 * (split, sequence, KV head); the block's query rows are the group's query heads times the sequence's new tokens
 * (row r: token r / group, head r % group), so every K/V tile load serves the whole group. Each sequence's cache is
 * split into `splits` ranges on the device; blocks write unnormalized partial outputs with their row maxima and
 * sums, which verify_combine_kernel merges and gates. The WMMA kernel takes up to 48 rows; the m8n8k4 kernel
 * (fragment layout: see csrc/nvfp4_gemm.cu) takes larger blocks. For the INT8 cache with up to 48 rows,
 * seventy_verify_attention uses verify_int8 (csrc/verify_int8.cu) instead; SEVENTY_VERIFY_ATTN=wmma selects
 * verify_attention_wmma.
 */

namespace seventy {
namespace {

constexpr int kHead = 256;
constexpr int kTile = 64;
constexpr int kWarps = 8;
constexpr int kLdQ = kHead + 4;
constexpr int kLdV = kTile + 4;

__device__ __forceinline__ void mma_8x8x4(float (&c)[8], uint2 a, uint2 b) {
  asm volatile(
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7])
      : "r"(a.x), "r"(a.y), "r"(b.x), "r"(b.y));
}

__device__ __forceinline__ int lane_idx() {
  const int lane = threadIdx.x & 31;
  return (lane & 3) + ((lane >> 4) << 2);
}

__device__ __forceinline__ int frag_row(int j) {
  const int idx = lane_idx();
  return (idx & 1) + 2 * ((j >> 1) & 1) + 4 * (idx >> 2);
}

__device__ __forceinline__ int frag_col(int j) {
  const int idx = lane_idx();
  return 8 * (((threadIdx.x & 31) >> 2) & 3) + (j & 1) + 2 * ((idx >> 1) & 1) + 4 * (j >> 2);
}

__device__ __forceinline__ float row_reduce_max(float v) {
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 4));
  return fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 8));
}

__device__ __forceinline__ float row_reduce_sum(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 2);
  v += __shfl_xor_sync(0xffffffffu, v, 4);
  return v + __shfl_xor_sync(0xffffffffu, v, 8);
}

__device__ __forceinline__ void int8x4_to_half(uint32_t w, float s, half *dst) {
  dst[0] = __float2half_rn(static_cast<float>(static_cast<int8_t>(w & 0xff)) * s);
  dst[1] = __float2half_rn(static_cast<float>(static_cast<int8_t>((w >> 8) & 0xff)) * s);
  dst[2] = __float2half_rn(static_cast<float>(static_cast<int8_t>((w >> 16) & 0xff)) * s);
  dst[3] = __float2half_rn(static_cast<float>(static_cast<int8_t>(w >> 24)) * s);
}


constexpr int kLdVR = kHead + 32;

__device__ __forceinline__ void mma_8x8x4_rr(float (&c)[8], uint2 a, uint2 b) {
  asm volatile(
      "mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7])
      : "r"(a.x), "r"(a.y), "r"(b.x), "r"(b.y));
}

/*
 * Four int8 values (one 32-bit word) as four exact FP16 integers: the bytes, offset by 128, go into the low byte of
 * 1024.0 (0x6400) and 1152 is subtracted.
 */
__device__ __forceinline__ uint2 int8x4_to_half_exact(uint32_t w) {
  const uint32_t u = w ^ 0x80808080u;
  const uint32_t lo = __byte_perm(u, 0x64646464u, 0x5140);
  const uint32_t hi = __byte_perm(u, 0x64646464u, 0x7362);
  const __half2 bias = __float2half2_rn(1152.0f);
  __half2 a = __hsub2(*reinterpret_cast<const __half2 *>(&lo), bias);
  __half2 b = __hsub2(*reinterpret_cast<const __half2 *>(&hi), bias);
  return make_uint2(*reinterpret_cast<uint32_t *>(&a), *reinterpret_cast<uint32_t *>(&b));
}

/*
 * Stage a tile of 64 cache vectors row-major into dst (row stride ld) as FP16, and their per-vector scales into
 * scales_out (int8 caches; 1 for FP16). scale_values: multiply the int8 values by their scales while staging
 * (values); otherwise keep exact integers (keys, whose scales are applied to the scores).
 */
template <typename T>
__device__ __forceinline__ void stage_rows(half *dst, int ld, float *scales_out, const T *cache, const float *scales,
                                           const int *table, int k0, int n, int block_size, int hkv, int kh,
                                           bool scale_values) {
  constexpr int kPer = sizeof(T) == 1 ? 16 : 8;
  for (int e = threadIdx.x; e < kTile * (kHead / kPer); e += kWarps * 32) {
    const int j = e / (kHead / kPer);
    const int c = (e % (kHead / kPer)) * kPer;
    uint4 raw = make_uint4(0, 0, 0, 0);
    float s = 0.0f;
    if (j < n) {
      const int pos = k0 + j;
      const int64_t vec = (static_cast<int64_t>(table[pos / block_size]) * block_size + pos % block_size) * hkv + kh;
      raw = *reinterpret_cast<const uint4 *>(cache + vec * kHead + c);
      s = (sizeof(T) == 1) ? scales[vec] : 1.0f;
    }
    if (c == 0) scales_out[j] = s;
    if constexpr (sizeof(T) == 2) {
      *reinterpret_cast<uint2 *>(dst + j * ld + c) = make_uint2(raw.x, raw.y);
      *reinterpret_cast<uint2 *>(dst + j * ld + c + 4) = make_uint2(raw.z, raw.w);
    } else {
      const uint32_t w[4] = {raw.x, raw.y, raw.z, raw.w};
      const __half2 sc = __float2half2_rn(s);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        uint2 h = int8x4_to_half_exact(w[i]);
        if (scale_values) {
          __half2 a = __hmul2(*reinterpret_cast<__half2 *>(&h.x), sc);
          __half2 b = __hmul2(*reinterpret_cast<__half2 *>(&h.y), sc);
          h = make_uint2(*reinterpret_cast<uint32_t *>(&a), *reinterpret_cast<uint32_t *>(&b));
        }
        *reinterpret_cast<uint2 *>(dst + j * ld + c + 4 * i) = h;
      }
    }
  }
}

/*
 * Register-staged tile loads, so the next tile's global loads are in flight during the current tile's math.
 * A thread owns kChunks 16-byte chunks of the 64 x 256 tile.
 */
template <typename T>
struct TileRaw {
  static constexpr int kPer = 16 / sizeof(T);
  static constexpr int kChunks = kTile * kHead / kPer / (kWarps * 32);
  uint4 v[kChunks];
  float s[kChunks];
};

template <typename T>
__device__ __forceinline__ void load_raw(TileRaw<T> &raw, const T *cache, const float *scales, const int *table, int k0,
                                         int n, int block_size, int hkv, int kh) {
  constexpr int kPer = TileRaw<T>::kPer;
#pragma unroll
  for (int i = 0; i < TileRaw<T>::kChunks; ++i) {
    const int e = threadIdx.x + i * kWarps * 32;
    const int j = e / (kHead / kPer);
    const int c = (e % (kHead / kPer)) * kPer;
    raw.v[i] = make_uint4(0, 0, 0, 0);
    raw.s[i] = 0.0f;
    if (j < n) {
      const int pos = k0 + j;
      const int64_t vec = (static_cast<int64_t>(table[pos / block_size]) * block_size + pos % block_size) * hkv + kh;
      raw.v[i] = *reinterpret_cast<const uint4 *>(cache + vec * kHead + c);
      if (sizeof(T) == 1 && c == 0) raw.s[i] = scales[vec];
      if (sizeof(T) == 2) raw.s[i] = 1.0f;
    }
  }
}

template <typename T>
__device__ __forceinline__ void store_raw(half *dst, int ld, float *scales_out, const TileRaw<T> &raw) {
  constexpr int kPer = TileRaw<T>::kPer;
#pragma unroll
  for (int i = 0; i < TileRaw<T>::kChunks; ++i) {
    const int e = threadIdx.x + i * kWarps * 32;
    const int j = e / (kHead / kPer);
    const int c = (e % (kHead / kPer)) * kPer;
    if (c == 0) scales_out[j] = raw.s[i];
    if constexpr (sizeof(T) == 2) {
      *reinterpret_cast<uint2 *>(dst + j * ld + c) = make_uint2(raw.v[i].x, raw.v[i].y);
      *reinterpret_cast<uint2 *>(dst + j * ld + c + 4) = make_uint2(raw.v[i].z, raw.v[i].w);
    } else {
      const uint32_t w[4] = {raw.v[i].x, raw.v[i].y, raw.v[i].z, raw.v[i].w};
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        uint2 h = int8x4_to_half_exact(w[q]);
        *reinterpret_cast<uint2 *>(dst + j * ld + c + 4 * q) = h;
      }
    }
  }
}

template <typename T>
__global__ void __launch_bounds__(kWarps * 32, 1) verify_attention_kernel(
    float *partial_o, float *partial_ml, const half *q, const T *k_cache, const T *v_cache, const float *k_scale,
    const float *v_scale, const int *block_tables, int table_width, const int *query_start, const int *context_lens,
    int block_size, int hq, int hkv, int splits, int min_split_keys, float scale) {
  extern __shared__ half smem[];
  half *qs = smem;
  half *kv = qs + kTile * kLdQ;
  half *ps = kv + kTile * kLdVR;
  float *kscale = reinterpret_cast<float *>(ps + kTile * kLdV);
  float *vscale = kscale + kTile;
  const int split = blockIdx.x;
  const int b = blockIdx.y;
  const int kh = blockIdx.z;
  const int group = hq / hkv;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int quad = (lane >> 2) & 3;
  const int idx = lane_idx();
  const int t0 = query_start[b];
  const int tokens = query_start[b + 1] - t0;
  const int rows = group * tokens;
  if (tokens <= 0 || rows > kTile) return;
  const int ctx = context_lens[b];
  const int total = ctx + tokens;
  const int seq_blocks = (total + block_size - 1) / block_size;
  const int used = max(1, min(splits, (total + min_split_keys - 1) / min_split_keys));
  const int per = (seq_blocks + used - 1) / used;
  const int first = split < used ? split * per * block_size : total;
  const int last = min(total, first + per * block_size);
  const int *table = block_tables + static_cast<int64_t>(b) * table_width;
  auto pidx = [&](int r) {
    return (static_cast<int64_t>(t0 + r / group) * hq + kh * group + r % group) * splits + split;
  };

  if (first >= last) {
    for (int r = threadIdx.x; r < rows; r += kWarps * 32) {
      partial_ml[pidx(r) * 2] = -FLT_MAX;
      partial_ml[pidx(r) * 2 + 1] = 0.0f;
    }
    return;
  }
  for (int e = threadIdx.x; e < kTile * (kHead / 8); e += kWarps * 32) {
    const int r = e / (kHead / 8);
    const int c = (e % (kHead / 8)) * 8;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (r < rows)
      v = *reinterpret_cast<const uint4 *>(q + (static_cast<int64_t>(t0 + r / group) * hq + kh * group + r % group) *
                                                   kHead + c);
    *reinterpret_cast<uint2 *>(qs + r * kLdQ + c) = make_uint2(v.x, v.y);
    *reinterpret_cast<uint2 *>(qs + r * kLdQ + c + 4) = make_uint2(v.z, v.w);
  }

  float o[8][8];
#pragma unroll
  for (int n = 0; n < 8; ++n)
#pragma unroll
    for (int j = 0; j < 8; ++j) o[n][j] = 0.0f;
  float m_row[2] = {-FLT_MAX, -FLT_MAX};
  float l_row[2] = {0.0f, 0.0f};
  const int r_lo = 8 * warp + frag_row(0);
  const int r_hi = 8 * warp + frag_row(2);

  float unit = 1.0f;
  TileRaw<T> rk, rv;
  load_raw<T>(rk, k_cache, k_scale, table, first, min(kTile, last - first), block_size, hkv, kh);
  load_raw<T>(rv, v_cache, v_scale, table, first, min(kTile, last - first), block_size, hkv, kh);
  for (int k0 = first; k0 < last; k0 += kTile) {
    const int n = min(kTile, last - k0);
    const int k1 = k0 + kTile;
    const int n1 = min(kTile, last - k1);
    __syncthreads();
    store_raw<T>(kv, kLdQ, kscale, rk);
    __syncthreads();
    if (k1 < last) load_raw<T>(rk, k_cache, k_scale, table, k1, n1, block_size, hkv, kh);
#pragma unroll
    for (int i = 0; i < TileRaw<T>::kChunks; ++i) {
      const int e = threadIdx.x + i * kWarps * 32;
      if ((e % (kHead / TileRaw<T>::kPer)) == 0) vscale[e / (kHead / TileRaw<T>::kPer)] = rv.s[i];
    }
    float s[2][8], s2[2][8];
#pragma unroll
    for (int g = 0; g < 2; ++g)
#pragma unroll
      for (int j = 0; j < 8; ++j) s[g][j] = s2[g][j] = 0.0f;
    const half *qrow = qs + (8 * warp + idx) * kLdQ;
    const half *krow0 = kv + (8 * quad + idx) * kLdQ;
    const half *krow1 = krow0 + 32 * kLdQ;
#pragma unroll 8
    for (int k = 0; k < kHead; k += 8) {
      const uint2 a0 = *reinterpret_cast<const uint2 *>(qrow + k);
      const uint2 a1 = *reinterpret_cast<const uint2 *>(qrow + k + 4);
      const uint2 b00 = *reinterpret_cast<const uint2 *>(krow0 + k);
      const uint2 b10 = *reinterpret_cast<const uint2 *>(krow1 + k);
      const uint2 b01 = *reinterpret_cast<const uint2 *>(krow0 + k + 4);
      const uint2 b11 = *reinterpret_cast<const uint2 *>(krow1 + k + 4);
      mma_8x8x4(s[0], a0, b00);
      mma_8x8x4(s[1], a0, b10);
      mma_8x8x4(s2[0], a1, b01);
      mma_8x8x4(s2[1], a1, b11);
    }
#pragma unroll
    for (int g = 0; g < 2; ++g)
#pragma unroll
      for (int j = 0; j < 8; ++j) s[g][j] += s2[g][j];
    float mx[2] = {-FLT_MAX, -FLT_MAX};
#pragma unroll
    for (int g = 0; g < 2; ++g)
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int row = 8 * warp + frag_row(j);
        const int col = 32 * g + frag_col(j);
        const int key = k0 + col;
        const bool ok = row < rows && key < last && key <= ctx + row / group;
        s[g][j] = ok ? s[g][j] * kscale[col] * scale : -FLT_MAX;
        mx[(j >> 1) & 1] = fmaxf(mx[(j >> 1) & 1], s[g][j]);
      }
    float corr[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const float m_new = fmaxf(m_row[h], row_reduce_max(mx[h]));
      corr[h] = (m_row[h] == -FLT_MAX) ? 0.0f : expf(m_row[h] - m_new);
      m_row[h] = m_new;
    }
    float tile_unit = 0.0f;
    for (int j = 0; j < kTile; ++j) tile_unit = fmaxf(tile_unit, vscale[j]);
    tile_unit = tile_unit > 0.0f ? tile_unit : 1.0f;
    const float rescale = unit / tile_unit;
    unit = tile_unit;
    float sum[2] = {0.0f, 0.0f};
#pragma unroll
    for (int g = 0; g < 2; ++g)
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int h = (j >> 1) & 1;
        const int col = 32 * g + frag_col(j);
        const float p = (s[g][j] == -FLT_MAX) ? 0.0f : expf(s[g][j] - m_row[h]);
        sum[h] += __half2float(__float2half_rn(p));
        ps[(8 * warp + frag_row(j)) * kLdV + col] = __float2half_rn(p * (vscale[col] / tile_unit));
      }
#pragma unroll
    for (int h = 0; h < 2; ++h) l_row[h] = l_row[h] * corr[h] + row_reduce_sum(sum[h]);
#pragma unroll
    for (int nn = 0; nn < 8; ++nn)
#pragma unroll
      for (int j = 0; j < 8; ++j) o[nn][j] *= corr[(j >> 1) & 1] * rescale;
    __syncthreads();
    store_raw<T>(kv, kLdVR, vscale, rv);
    __syncthreads();
    if (k1 < last) load_raw<T>(rv, v_cache, v_scale, table, k1, n1, block_size, hkv, kh);
    const int vk = lane & 3;
    const int vn = 8 * quad + 4 * (lane >> 4);
#pragma unroll 4
    for (int k = 0; k < kTile; k += 4) {
      const uint2 a = *reinterpret_cast<const uint2 *>(ps + (8 * warp + idx) * kLdV + k);
#pragma unroll
      for (int nn = 0; nn < 8; ++nn) {
        const uint2 bv = *reinterpret_cast<const uint2 *>(kv + (k + vk) * kLdVR + 32 * nn + vn);
        mma_8x8x4_rr(o[nn], a, bv);
      }
    }
  }

#pragma unroll
  for (int nn = 0; nn < 8; ++nn)
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int h = (j >> 1) & 1;
      const int row = h ? r_hi : r_lo;
      if (row >= rows) continue;
      partial_o[pidx(row) * kHead + 32 * nn + frag_col(j)] = o[nn][j] * unit;
    }
  if ((lane & 0xe) == 0) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int row = h ? r_hi : r_lo;
      if (row < rows) {
        partial_ml[pidx(row) * 2] = m_row[h];
        partial_ml[pidx(row) * 2 + 1] = l_row[h];
      }
    }
  }
}

/*
 * verify_attention_wmma: the speculative verify attention on WMMA 16 x 16 x 16 tiles, for blocks of up to 48 rows
 * (6 query heads per KV head x 8 new tokens). Same work split, numerics and outputs as verify_attention_kernel
 * (partial O times the running unit, row maxima and sums, merged by verify_combine_kernel). 12 warps, balanced:
 * S = Q K^T is 3 x 4 tiles of 16 x 16 (one per warp), the softmax takes 4 rows per warp, and O += P V gives each
 * warp a 16-row x 64-column slice.
 *
 * Per 64-key tile: (1) stage K (exact integers for INT8 caches) and the tile's K and V scales; (2) S into shared
 * memory; (3) stage V while each warp runs the online softmax of its rows (P scaled by vscale / tile unit, as in
 * verify_attention_kernel); (4) scale each O accumulator row by its correction, then O += P V. sm70 accumulator
 * layout (probed): lane l holds rows r, r + 2 and columns c + {0, 1, 4, 5} with r = (l & 1) + 8 ((l >> 2) & 1)
 * + 4 ((l >> 4) & 1), c = 2 ((l >> 1) & 1) + 8 ((l >> 3) & 1); element i adds 2 to the row if bit 1 is set and
 * (i & 1) + 4 (i >> 2) to the column. The next tile's K and V are in registers while the current one computes.
 */
constexpr int kVW = 12;
constexpr int kVT = kVW * 32;
constexpr int kVRows = 48;
constexpr int kLdW = kHead + 8;
constexpr int kLdS = kTile + 4;
constexpr int kWmmaSmem = (kVRows * kHead + kTile * kLdW + kVRows * kTile) * static_cast<int>(sizeof(half)) +
                          (kVRows * kLdS + 2 * kTile + 3 * kVRows) * static_cast<int>(sizeof(float));

/*
 * Q and P (the A operands) are stored unpadded and XOR-swizzled in 16-byte chunks: chunk c of row r is at
 * c ^ swz(r). A row_major A fragment on sm70 (probed) gives lane l all 16 columns of row
 * (l & 3) + 8 ((l >> 2) & 1) + 4 ((l >> 4) & 1), and an 8-lane load phase reads rows {0-3, 8-11} or {4-7, 12-15};
 * swz takes distinct values on each, so the 16-byte loads hit distinct banks (load_matrix_sync needs a plain
 * stride, which puts rows r and r + 8 on the same banks).
 */
__device__ __forceinline__ int swz(int r) { return (r & 3) | ((r & 8) >> 1); }

__device__ __forceinline__ int a_row(int lane) { return (lane & 3) + 8 * ((lane >> 2) & 1) + 4 * ((lane >> 4) & 1); }

template <typename Frag>
__device__ __forceinline__ void load_a_swizzled(Frag &a, const half *base, int row0, int chunks_per_row, int k) {
  const int r = row0 + a_row(threadIdx.x & 31);
  const uint4 *rowp = reinterpret_cast<const uint4 *>(base) + r * chunks_per_row;
  const int c = k / 8;
  const uint4 v0 = rowp[c ^ swz(r)], v1 = rowp[(c + 1) ^ swz(r)];
  const half *h0 = reinterpret_cast<const half *>(&v0);
  const half *h1 = reinterpret_cast<const half *>(&v1);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    a.x[i] = h0[i];
    a.x[8 + i] = h1[i];
  }
}

__device__ __forceinline__ int acc_row(int lane, int i) {
  return (lane & 1) + 8 * ((lane >> 2) & 1) + 4 * ((lane >> 4) & 1) + 2 * ((i >> 1) & 1);
}

__device__ __forceinline__ int acc_col(int lane, int i) {
  return 2 * ((lane >> 1) & 1) + 8 * ((lane >> 3) & 1) + (i & 1) + 4 * (i >> 2);
}

template <typename T>
struct WideTile {
  static constexpr int kPer = 16 / sizeof(T);
  static constexpr int kTotal = kTile * kHead / kPer;
  static constexpr int kChunks = (kTotal + kVT - 1) / kVT;
  uint4 v[kChunks];
  float s[kChunks];
};

template <typename T>
__device__ __forceinline__ void wide_load(WideTile<T> &raw, const T *cache, const float *scales, const int *table,
                                          int k0, int n, int block_size, int hkv, int kh) {
  constexpr int kPer = WideTile<T>::kPer;
#pragma unroll
  for (int i = 0; i < WideTile<T>::kChunks; ++i) {
    const int e = threadIdx.x + i * kVT;
    const int j = e / (kHead / kPer);
    const int c = (e % (kHead / kPer)) * kPer;
    raw.v[i] = make_uint4(0, 0, 0, 0);
    raw.s[i] = 0.0f;
    if (e < WideTile<T>::kTotal && j < n) {
      const int pos = k0 + j;
      const int64_t vec = (static_cast<int64_t>(table[pos / block_size]) * block_size + pos % block_size) * hkv + kh;
      raw.v[i] = *reinterpret_cast<const uint4 *>(cache + vec * kHead + c);
      raw.s[i] = sizeof(T) == 1 ? scales[vec] : 1.0f;
    }
  }
}

template <typename T>
__device__ __forceinline__ void wide_store(half *dst, float *scales_out, const WideTile<T> &raw) {
  constexpr int kPer = WideTile<T>::kPer;
#pragma unroll
  for (int i = 0; i < WideTile<T>::kChunks; ++i) {
    const int e = threadIdx.x + i * kVT;
    if (e >= WideTile<T>::kTotal) break;
    const int j = e / (kHead / kPer);
    const int c = (e % (kHead / kPer)) * kPer;
    if (c == 0) scales_out[j] = raw.s[i];
    if constexpr (sizeof(T) == 2) {
      *reinterpret_cast<uint4 *>(dst + j * kLdW + c) = raw.v[i];
    } else {
      const uint32_t w[4] = {raw.v[i].x, raw.v[i].y, raw.v[i].z, raw.v[i].w};
      const uint2 h0 = int8x4_to_half_exact(w[0]), h1 = int8x4_to_half_exact(w[1]);
      const uint2 h2 = int8x4_to_half_exact(w[2]), h3 = int8x4_to_half_exact(w[3]);
      *reinterpret_cast<uint4 *>(dst + j * kLdW + c) = make_uint4(h0.x, h0.y, h1.x, h1.y);
      *reinterpret_cast<uint4 *>(dst + j * kLdW + c + 8) = make_uint4(h2.x, h2.y, h3.x, h3.y);
    }
  }
}

template <typename T>
__global__ void __launch_bounds__(kVT, 1) verify_attention_wmma(
    float *partial_o, float *partial_ml, const half *q, const T *k_cache, const T *v_cache, const float *k_scale,
    const float *v_scale, const int *block_tables, int table_width, const int *query_start, const int *context_lens,
    int block_size, int hq, int hkv, int splits, int min_split_keys, float scale) {
  using namespace nvcuda;
  extern __shared__ __align__(16) unsigned char wsm[];
  half *qs = reinterpret_cast<half *>(wsm);
  half *kv = qs + kVRows * kHead;
  half *ps = kv + kTile * kLdW;
  float *ss = reinterpret_cast<float *>(ps + kVRows * kTile);
  float *kscale = ss + kVRows * kLdS;
  float *vscale = kscale + kTile;
  float *m_s = vscale + kTile;
  float *l_s = m_s + kVRows;
  float *fac_s = l_s + kVRows;

  const int split = blockIdx.x, b = blockIdx.y, kh = blockIdx.z;
  const int group = hq / hkv;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int t0 = query_start[b];
  const int tokens = query_start[b + 1] - t0;
  const int rows = group * tokens;
  if (tokens <= 0 || rows > kVRows) return;
  const int ctx = context_lens[b];
  const int total = ctx + tokens;
  const int seq_blocks = (total + block_size - 1) / block_size;
  const int used = max(1, min(splits, (total + min_split_keys - 1) / min_split_keys));
  const int per = (seq_blocks + used - 1) / used;
  const int first = split < used ? split * per * block_size : total;
  const int last = min(total, first + per * block_size);
  const int *table = block_tables + static_cast<int64_t>(b) * table_width;
  auto pidx = [&](int r) {
    return (static_cast<int64_t>(t0 + r / group) * hq + kh * group + r % group) * splits + split;
  };
  if (first >= last) {
    for (int r = threadIdx.x; r < rows; r += kVT) {
      partial_ml[pidx(r) * 2] = -FLT_MAX;
      partial_ml[pidx(r) * 2 + 1] = 0.0f;
    }
    return;
  }
  const int row_tiles = (rows + 15) / 16;
  for (int e = threadIdx.x; e < kVRows * (kHead / 8); e += kVT) {
    const int r = e / (kHead / 8);
    const int c = (e % (kHead / 8)) * 8;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (r < rows)
      v = *reinterpret_cast<const uint4 *>(q + (static_cast<int64_t>(t0 + r / group) * hq + kh * group + r % group) *
                                                   kHead + c);
    reinterpret_cast<uint4 *>(qs)[r * (kHead / 8) + ((c / 8) ^ swz(r))] = v;
  }
  for (int r = threadIdx.x; r < kVRows; r += kVT) {
    m_s[r] = -FLT_MAX;
    l_s[r] = 0.0f;
  }

  const int qk_rt = warp >> 2, qk_ct = warp & 3;
  const int pv_rt = warp >> 2, pv_cols = (warp & 3) * (kHead / 4);
  const bool active = qk_rt < row_tiles;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> o[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) wmma::fill_fragment(o[j], 0.0f);
  float unit = 1.0f;

  WideTile<T> rk, rv;
  wide_load<T>(rk, k_cache, k_scale, table, first, min(kTile, last - first), block_size, hkv, kh);
  wide_load<T>(rv, v_cache, v_scale, table, first, min(kTile, last - first), block_size, hkv, kh);
  for (int k0 = first; k0 < last; k0 += kTile) {
    const int k1 = k0 + kTile;
    const int n1 = min(kTile, last - k1);
    __syncthreads();
    wide_store<T>(kv, kscale, rk);
#pragma unroll
    for (int i = 0; i < WideTile<T>::kChunks; ++i) {
      const int e = threadIdx.x + i * kVT;
      if (e < WideTile<T>::kTotal && (e % (kHead / WideTile<T>::kPer)) == 0) vscale[e / (kHead / WideTile<T>::kPer)] = rv.s[i];
    }
    __syncthreads();
    if (k1 < last) wide_load<T>(rk, k_cache, k_scale, table, k1, n1, block_size, hkv, kh);

    if (active) {
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> s0, s1;
      wmma::fill_fragment(s0, 0.0f);
      wmma::fill_fragment(s1, 0.0f);
#pragma unroll 4
      for (int k = 0; k < kHead; k += 32) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a0, a1;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b0, b1;
        load_a_swizzled(a0, qs, qk_rt * 16, kHead / 8, k);
        wmma::load_matrix_sync(b0, kv + qk_ct * 16 * kLdW + k, kLdW);
        load_a_swizzled(a1, qs, qk_rt * 16, kHead / 8, k + 16);
        wmma::load_matrix_sync(b1, kv + qk_ct * 16 * kLdW + k + 16, kLdW);
        wmma::mma_sync(s0, a0, b0, s0);
        wmma::mma_sync(s1, a1, b1, s1);
      }
#pragma unroll
      for (int i = 0; i < s0.num_elements; ++i) s0.x[i] += s1.x[i];
      wmma::store_matrix_sync(ss + qk_rt * 16 * kLdS + qk_ct * 16, s0, kLdS, wmma::mem_row_major);
    }
    __syncthreads();

    wide_store<T>(kv, vscale, rv);
    if (k1 < last) wide_load<T>(rv, v_cache, v_scale, table, k1, n1, block_size, hkv, kh);
    float tile_unit = fmaxf(vscale[lane], vscale[lane + 32]);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) tile_unit = fmaxf(tile_unit, __shfl_xor_sync(0xffffffffu, tile_unit, off));
    tile_unit = tile_unit > 0.0f ? tile_unit : 1.0f;
    const float rescale = unit / tile_unit;
    {
      const int r = warp * 4 + (lane >> 3);
      const int c0 = (lane & 7) * 8;
      if (r < row_tiles * 16) {
        const int limit = ctx + r / group;
        float v[8];
        const float4 *src = reinterpret_cast<const float4 *>(ss + r * kLdS + c0);
        const float4 f0 = src[0], f1 = src[1];
        v[0] = f0.x; v[1] = f0.y; v[2] = f0.z; v[3] = f0.w;
        v[4] = f1.x; v[5] = f1.y; v[6] = f1.z; v[7] = f1.w;
        float mx = -FLT_MAX;
#pragma unroll
        for (int c = 0; c < 8; ++c) {
          const int key = k0 + c0 + c;
          const bool ok = r < rows && key < last && key <= limit;
          v[c] = ok ? v[c] * kscale[c0 + c] * scale : -FLT_MAX;
          mx = fmaxf(mx, v[c]);
        }
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 4));
        const float m_old = r < rows ? m_s[r] : -FLT_MAX;
        const float m_new = fmaxf(m_old, mx);
        const float corr = (m_old == -FLT_MAX) ? 0.0f : expf(m_old - m_new);
        float sum = 0.0f;
        half ph[8];
#pragma unroll
        for (int c = 0; c < 8; ++c) {
          const float p = (v[c] == -FLT_MAX) ? 0.0f : expf(v[c] - m_new);
          sum += __half2float(__float2half_rn(p));
          ph[c] = __float2half_rn(p * (vscale[c0 + c] / tile_unit));
        }
        reinterpret_cast<uint4 *>(ps)[r * (kTile / 8) + ((c0 / 8) ^ swz(r))] = *reinterpret_cast<uint4 *>(ph);
        sum += __shfl_xor_sync(0xffffffffu, sum, 1);
        sum += __shfl_xor_sync(0xffffffffu, sum, 2);
        sum += __shfl_xor_sync(0xffffffffu, sum, 4);
        if ((lane & 7) == 0 && r < rows) {
          m_s[r] = m_new;
          l_s[r] = l_s[r] * corr + sum;
          fac_s[r] = corr * rescale;
        }
      }
    }
    unit = tile_unit;
    __syncthreads();

    if (active) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int row = pv_rt * 16 + acc_row(lane, i);
        const float f = row < rows ? fac_s[row] : 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) o[j].x[i] *= f;
      }
#pragma unroll
      for (int k = 0; k < kTile; k += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
        load_a_swizzled(a, ps, pv_rt * 16, kTile / 8, k);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bv;
          wmma::load_matrix_sync(bv, kv + k * kLdW + pv_cols + 16 * j, kLdW);
          wmma::mma_sync(o[j], a, bv, o[j]);
        }
      }
    }
  }

  if (active) {
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int row = pv_rt * 16 + acc_row(lane, i);
        if (row < rows) partial_o[pidx(row) * kHead + pv_cols + 16 * j + acc_col(lane, i)] = o[j].x[i] * unit;
      }
  }
  for (int r = threadIdx.x; r < rows; r += kVT) {
    partial_ml[pidx(r) * 2] = m_s[r];
    partial_ml[pidx(r) * 2 + 1] = l_s[r];
  }
}

/*
 * flash_attention_wmma: prefill attention (one block per 64-token query tile and query head) on WMMA tiles: K and V
 * dequantized while staged, P rounded to FP16, row sums of the rounded P, gated FP16 output. 8 warps: S = Q K^T is 4 x 4 tiles of 16 x 16 (two per warp, sharing the A operand), the
 * softmax takes 8 rows per warp, and O += P V gives each warp a 16-row x 128-column slice. Q and P are swizzled as in
 * verify_attention_wmma; the next K and V tiles are in registers while the current one computes.
 */
constexpr int kPT = kWarps * 32;
constexpr int kPrefillSmem = (kTile * kHead + kTile * kLdW + kTile * kTile) * static_cast<int>(sizeof(half)) +
                             (kTile * kLdS + 3 * kTile) * static_cast<int>(sizeof(float));

template <typename T>
struct PrefillTile {
  static constexpr int kPer = 16 / sizeof(T);
  static constexpr int kChunks = kTile * kHead / kPer / kPT;
  uint4 v[kChunks];
  float s[kChunks];
};

template <typename T>
__device__ __forceinline__ void prefill_load(PrefillTile<T> &raw, const T *cache, const float *scales, const int *table,
                                             int k0, int n, int block_size, int hkv, int kh) {
  constexpr int kPer = PrefillTile<T>::kPer;
#pragma unroll
  for (int i = 0; i < PrefillTile<T>::kChunks; ++i) {
    const int e = threadIdx.x + i * kPT;
    const int j = e / (kHead / kPer);
    const int c = (e % (kHead / kPer)) * kPer;
    raw.v[i] = make_uint4(0, 0, 0, 0);
    raw.s[i] = 0.0f;
    if (j < n) {
      const int pos = k0 + j;
      const int64_t vec = (static_cast<int64_t>(table[pos / block_size]) * block_size + pos % block_size) * hkv + kh;
      raw.v[i] = *reinterpret_cast<const uint4 *>(cache + vec * kHead + c);
      raw.s[i] = sizeof(T) == 1 ? scales[vec] : 1.0f;
    }
  }
}

template <typename T>
__device__ __forceinline__ void prefill_store(half *dst, const PrefillTile<T> &raw) {
  constexpr int kPer = PrefillTile<T>::kPer;
#pragma unroll
  for (int i = 0; i < PrefillTile<T>::kChunks; ++i) {
    const int e = threadIdx.x + i * kPT;
    const int j = e / (kHead / kPer);
    const int c = (e % (kHead / kPer)) * kPer;
    if constexpr (sizeof(T) == 2) {
      *reinterpret_cast<uint4 *>(dst + j * kLdW + c) = raw.v[i];
    } else {
      half h[16];
      int8x4_to_half(raw.v[i].x, raw.s[i], h);
      int8x4_to_half(raw.v[i].y, raw.s[i], h + 4);
      int8x4_to_half(raw.v[i].z, raw.s[i], h + 8);
      int8x4_to_half(raw.v[i].w, raw.s[i], h + 12);
      *reinterpret_cast<uint4 *>(dst + j * kLdW + c) = *reinterpret_cast<uint4 *>(h);
      *reinterpret_cast<uint4 *>(dst + j * kLdW + c + 8) = *reinterpret_cast<uint4 *>(h + 8);
    }
  }
}

template <typename T>
__global__ void __launch_bounds__(kPT, 1) flash_attention_wmma(
    half *out, const half *q, const T *k_cache, const T *v_cache, const float *k_scale, const float *v_scale,
    const half *qkv, int ld, const int *tiles, const int *block_tables, int table_width, const int *query_start,
    const int *context_lens, int block_size, int hq, int hkv, float scale) {
  using namespace nvcuda;
  extern __shared__ __align__(16) unsigned char fsm[];
  half *qs = reinterpret_cast<half *>(fsm);
  half *kv = qs + kTile * kHead;
  half *ps = kv + kTile * kLdW;
  float *ss = reinterpret_cast<float *>(ps + kTile * kTile);
  float *m_s = ss + kTile * kLdS;
  float *l_s = m_s + kTile;
  float *fac_s = l_s + kTile;

  const int b = tiles[2 * blockIdx.x];
  const int q0 = tiles[2 * blockIdx.x + 1];
  const int head = blockIdx.y;
  const int kh = head / (hq / hkv);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int t0 = query_start[b];
  const int n_new = query_start[b + 1] - t0;
  const int ctx = context_lens[b];
  const int rows = min(kTile, n_new - q0);
  const int key_end = ctx + q0 + rows;
  const int *table = block_tables + static_cast<int64_t>(b) * table_width;
  const int row_tiles = (rows + 15) / 16;
  for (int e = threadIdx.x; e < kTile * (kHead / 8); e += kPT) {
    const int r = e / (kHead / 8);
    const int c = (e % (kHead / 8)) * 8;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (r < rows) v = *reinterpret_cast<const uint4 *>(q + (static_cast<int64_t>(t0 + q0 + r) * hq + head) * kHead + c);
    reinterpret_cast<uint4 *>(qs)[r * (kHead / 8) + ((c / 8) ^ swz(r))] = v;
  }
  for (int r = threadIdx.x; r < kTile; r += kPT) {
    m_s[r] = -FLT_MAX;
    l_s[r] = 0.0f;
  }
  const int rt = warp >> 1, ct = (warp & 1) * 2, cols = (warp & 1) * (kHead / 2);
  const bool active = rt < row_tiles;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> o[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) wmma::fill_fragment(o[j], 0.0f);

  PrefillTile<T> rk, rv;
  prefill_load<T>(rk, k_cache, k_scale, table, 0, min(kTile, key_end), block_size, hkv, kh);
  prefill_load<T>(rv, v_cache, v_scale, table, 0, min(kTile, key_end), block_size, hkv, kh);
  for (int k0 = 0; k0 < key_end; k0 += kTile) {
    const int k1 = k0 + kTile;
    const int n1 = min(kTile, key_end - k1);
    __syncthreads();
    prefill_store<T>(kv, rk);
    __syncthreads();
    if (k1 < key_end) prefill_load<T>(rk, k_cache, k_scale, table, k1, n1, block_size, hkv, kh);
    if (active) {
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> s0, s1, s2, s3;
      wmma::fill_fragment(s0, 0.0f);
      wmma::fill_fragment(s1, 0.0f);
      wmma::fill_fragment(s2, 0.0f);
      wmma::fill_fragment(s3, 0.0f);
#pragma unroll 2
      for (int k = 0; k < kHead; k += 32) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a, a2;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b0, b1, b2, b3;
        load_a_swizzled(a, qs, rt * 16, kHead / 8, k);
        load_a_swizzled(a2, qs, rt * 16, kHead / 8, k + 16);
        wmma::load_matrix_sync(b0, kv + ct * 16 * kLdW + k, kLdW);
        wmma::load_matrix_sync(b1, kv + (ct + 1) * 16 * kLdW + k, kLdW);
        wmma::load_matrix_sync(b2, kv + ct * 16 * kLdW + k + 16, kLdW);
        wmma::load_matrix_sync(b3, kv + (ct + 1) * 16 * kLdW + k + 16, kLdW);
        wmma::mma_sync(s0, a, b0, s0);
        wmma::mma_sync(s1, a, b1, s1);
        wmma::mma_sync(s2, a2, b2, s2);
        wmma::mma_sync(s3, a2, b3, s3);
      }
#pragma unroll
      for (int i = 0; i < s0.num_elements; ++i) {
        s0.x[i] += s2.x[i];
        s1.x[i] += s3.x[i];
      }
      wmma::store_matrix_sync(ss + rt * 16 * kLdS + ct * 16, s0, kLdS, wmma::mem_row_major);
      wmma::store_matrix_sync(ss + rt * 16 * kLdS + (ct + 1) * 16, s1, kLdS, wmma::mem_row_major);
    }
    __syncthreads();
    prefill_store<T>(kv, rv);
    if (k1 < key_end) prefill_load<T>(rv, v_cache, v_scale, table, k1, n1, block_size, hkv, kh);
    {
      const int r = warp * 8 + (lane >> 2);
      const int c0 = (lane & 3) * 16;
      if (warp * 8 < row_tiles * 16) {
        const int limit = ctx + q0 + r;
        float v[16];
        const float4 *src = reinterpret_cast<const float4 *>(ss + r * kLdS + c0);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const float4 f = src[i];
          v[4 * i] = f.x;
          v[4 * i + 1] = f.y;
          v[4 * i + 2] = f.z;
          v[4 * i + 3] = f.w;
        }
        float mx = -FLT_MAX;
#pragma unroll
        for (int c = 0; c < 16; ++c) {
          const int key = k0 + c0 + c;
          const bool ok = r < rows && key < key_end && key <= limit;
          v[c] = ok ? v[c] * scale : -FLT_MAX;
          mx = fmaxf(mx, v[c]);
        }
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
        const float m_old = m_s[r];
        const float m_new = fmaxf(m_old, mx);
        const float corr = (m_old == -FLT_MAX) ? 0.0f : expf(m_old - m_new);
        float sum = 0.0f;
        half ph[16];
#pragma unroll
        for (int c = 0; c < 16; ++c) {
          ph[c] = __float2half_rn((v[c] == -FLT_MAX) ? 0.0f : expf(v[c] - m_new));
          sum += __half2float(ph[c]);
        }
        uint4 *prow = reinterpret_cast<uint4 *>(ps) + r * (kTile / 8);
        prow[(c0 / 8) ^ swz(r)] = *reinterpret_cast<uint4 *>(ph);
        prow[(c0 / 8 + 1) ^ swz(r)] = *reinterpret_cast<uint4 *>(ph + 8);
        sum += __shfl_xor_sync(0xffffffffu, sum, 1);
        sum += __shfl_xor_sync(0xffffffffu, sum, 2);
        if ((lane & 3) == 0) {
          m_s[r] = m_new;
          l_s[r] = l_s[r] * corr + sum;
          fac_s[r] = corr;
        }
      }
    }
    __syncthreads();
    if (active) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float f = fac_s[rt * 16 + acc_row(lane, i)];
#pragma unroll
        for (int j = 0; j < 8; ++j) o[j].x[i] *= f;
      }
#pragma unroll
      for (int k = 0; k < kTile; k += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
        load_a_swizzled(a, ps, rt * 16, kTile / 8, k);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bv;
          wmma::load_matrix_sync(bv, kv + k * kLdW + cols + 16 * j, kLdW);
          wmma::mma_sync(o[j], a, bv, o[j]);
        }
      }
    }
  }
  if (!active) return;
#pragma unroll
  for (int j = 0; j < 8; ++j)
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int row = rt * 16 + acc_row(lane, i);
      if (row >= rows) continue;
      const int col = cols + 16 * j + acc_col(lane, i);
      const int64_t t = t0 + q0 + row;
      const float a = __half2float(__float2half_rn(o[j].x[i] / l_s[row]));
      const float g = __half2float(qkv[t * ld + head * 2 * kHead + kHead + col]);
      out[(t * hq + head) * kHead + col] = __float2half_rn(a / (1.0f + expf(-g)));
    }
}


/*
 * flash_attention_mma: prefill attention for INT8 caches on mma.m8n8k4 with shared-memory operands. On sm70 a
 * WMMA fragment load moves twice the data it holds, and the S tile of flash_attention_wmma gives each warp one
 * 16 x 16 tile per pair of loads; here the A operand of m8n8k4 is one broadcast load for the four quad-pairs and a
 * warp computes larger tiles, so shared-memory traffic per multiply-add is about a third.
 *
 * One block of 8 warps per (64-token query tile of one sequence, query head); 64-key tiles. Shared memory: Q and
 * the K-or-V tile [64, 256] FP16, P [64, 64] FP16, the second half of S [64, 64] FP32, and row statistics. The
 * rows of Q, K and P are XOR-swizzled in 8-byte chunks with f(r) = (r & 3) | (((r >> 3) & 3) << 2), the rows of V
 * with g(r) = (r & 1) | ((r & 2) << 2), so that the operand loads of a warp (A: rows r0 + idx; B of row.col:
 * rows 8 quad + idx; B of row.row: rows k + (lane & 3), columns 8 quad + 4 (lane >> 4)) hit distinct banks.
 *
 * Per key tile: (1) stage K (dequantized with its scales); (2) S = Q K^T, warp w: rows 16 (w % 4) .. + 16, all 64
 * keys, half w / 4 of the head dimension; warps 4..7 store their partial S; (3) warps 0..3 add it and run the
 * online softmax of their rows in registers (P rounded to FP16, row sums of the rounded P), while warps 4..7
 * stage V; (4) O += P V, warp w: rows 32 (w % 2) .. + 32, columns 64 (w / 2) .. + 64. The next K and V tiles are
 * in registers while the current one computes. The epilogue is as in flash_attention_wmma: O / l rounded to FP16,
 * times sigmoid of the gate.
 */
constexpr int kLdSM = kTile + 4;
constexpr int kMmaSmem = (2 * kTile * kHead + kTile * kTile) * static_cast<int>(sizeof(half)) +
                         (kTile * kLdSM + 3 * kTile) * static_cast<int>(sizeof(float));

__device__ __forceinline__ int swz_f(int r) { return (r & 3) | (((r >> 3) & 3) << 2); }

__device__ __forceinline__ int swz_g(int r) { return (r & 1) | ((r & 2) << 2); }

template <int Cols>
__device__ __forceinline__ uint2 lds_chunk(const half *base, int r, int chunk, int s) {
  return *reinterpret_cast<const uint2 *>(base + r * Cols + ((chunk ^ s) << 2));
}

template <int Cols>
__device__ __forceinline__ void sts_pair(half *base, int r, int pair, int s, uint4 v) {
  *reinterpret_cast<uint2 *>(base + r * Cols + (((2 * pair) ^ s) << 2)) = make_uint2(v.x, v.y);
  *reinterpret_cast<uint2 *>(base + r * Cols + (((2 * pair + 1) ^ s) << 2)) = make_uint2(v.z, v.w);
}

template <int Threads>
struct Int8Tile {
  static constexpr int kChunks = kTile * kHead / 16 / Threads;
  uint4 v[kChunks];
  float s[kChunks];
};

template <int Threads>
__device__ __forceinline__ void int8_load(Int8Tile<Threads> &raw, int t, const int8_t *cache, const float *scales,
                                          const int *table, int k0, int n, int block_size, int hkv, int kh) {
#pragma unroll
  for (int i = 0; i < Int8Tile<Threads>::kChunks; ++i) {
    const int e = t + i * Threads;
    const int j = e / (kHead / 16);
    const int c = (e % (kHead / 16)) * 16;
    raw.v[i] = make_uint4(0, 0, 0, 0);
    raw.s[i] = 0.0f;
    if (j < n) {
      const int pos = k0 + j;
      const int64_t vec = (static_cast<int64_t>(table[pos / block_size]) * block_size + pos % block_size) * hkv + kh;
      raw.v[i] = *reinterpret_cast<const uint4 *>(cache + vec * kHead + c);
      raw.s[i] = scales[vec];
    }
  }
}

template <int Threads, bool IsV>
__device__ __forceinline__ void int8_store(half *dst, const Int8Tile<Threads> &raw, int t) {
#pragma unroll
  for (int i = 0; i < Int8Tile<Threads>::kChunks; ++i) {
    const int e = t + i * Threads;
    const int j = e / (kHead / 16);
    const int c = (e % (kHead / 16)) * 16;
    half h[16];
    int8x4_to_half(raw.v[i].x, raw.s[i], h);
    int8x4_to_half(raw.v[i].y, raw.s[i], h + 4);
    int8x4_to_half(raw.v[i].z, raw.s[i], h + 8);
    int8x4_to_half(raw.v[i].w, raw.s[i], h + 12);
    const int s = IsV ? swz_g(j) : swz_f(j);
    sts_pair<kHead>(dst, j, c / 8, s, *reinterpret_cast<uint4 *>(h));
    sts_pair<kHead>(dst, j, c / 8 + 1, s, *reinterpret_cast<uint4 *>(h + 8));
  }
}

__global__ void __launch_bounds__(kPT, 1) flash_attention_mma(
    half *out, const half *q, const int8_t *k_cache, const int8_t *v_cache, const float *k_scale,
    const float *v_scale, const half *qkv, int ld, const int *tiles, const int *block_tables, int table_width,
    const int *query_start, const int *context_lens, int block_size, int hq, int hkv, float scale) {
  extern __shared__ __align__(16) unsigned char msm[];
  half *qs = reinterpret_cast<half *>(msm);
  half *kv = qs + kTile * kHead;
  half *ps = kv + kTile * kHead;
  float *sp = reinterpret_cast<float *>(ps + kTile * kTile);
  float *m_s = sp + kTile * kLdSM;
  float *l_s = m_s + kTile;
  float *fac_s = l_s + kTile;

  const int b = tiles[2 * blockIdx.x];
  const int q0 = tiles[2 * blockIdx.x + 1];
  const int head = blockIdx.y;
  const int kh = head / (hq / hkv);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int quad = (lane >> 2) & 3;
  const int idx = lane_idx();
  const int t0 = query_start[b];
  const int n_new = query_start[b + 1] - t0;
  const int ctx = context_lens[b];
  const int rows = min(kTile, n_new - q0);
  const int key_end = ctx + q0 + rows;
  const int *table = block_tables + static_cast<int64_t>(b) * table_width;

  for (int e = threadIdx.x; e < kTile * (kHead / 8); e += kPT) {
    const int r = e / (kHead / 8);
    const int pair = e % (kHead / 8);
    uint4 v = make_uint4(0, 0, 0, 0);
    if (r < rows) v = *reinterpret_cast<const uint4 *>(q + (static_cast<int64_t>(t0 + q0 + r) * hq + head) * kHead + 8 * pair);
    sts_pair<kHead>(qs, r, pair, swz_f(r), v);
  }
  for (int r = threadIdx.x; r < kTile; r += kPT) {
    m_s[r] = -FLT_MAX;
    l_s[r] = 0.0f;
  }

  const int qk_r0 = 16 * (warp & 3), qk_half = warp >> 2;
  const int pv_r0 = 32 * (warp & 1), pv_c0 = 64 * (warp >> 1);
  const int vk = lane & 3;
  const int vn = 8 * quad + 4 * (lane >> 4);
  float o[4][2][8];
#pragma unroll
  for (int t = 0; t < 4; ++t)
#pragma unroll
    for (int n = 0; n < 2; ++n)
#pragma unroll
      for (int j = 0; j < 8; ++j) o[t][n][j] = 0.0f;

  Int8Tile<kPT> rk;
  Int8Tile<kPT / 2> rv;
  const int vt = threadIdx.x - kPT / 2;
  int8_load<kPT>(rk, threadIdx.x, k_cache, k_scale, table, 0, min(kTile, key_end), block_size, hkv, kh);
  if (warp >= 4) int8_load<kPT / 2>(rv, vt, v_cache, v_scale, table, 0, min(kTile, key_end), block_size, hkv, kh);

  for (int k0 = 0; k0 < key_end; k0 += kTile) {
    const int k1 = k0 + kTile;
    const int n1 = min(kTile, key_end - k1);
    __syncthreads();
    int8_store<kPT, false>(kv, rk, threadIdx.x);
    __syncthreads();
    if (k1 < key_end) int8_load<kPT>(rk, threadIdx.x, k_cache, k_scale, table, k1, n1, block_size, hkv, kh);

    float sacc[2][2][8];
#pragma unroll
    for (int t = 0; t < 2; ++t)
#pragma unroll
      for (int n = 0; n < 2; ++n)
#pragma unroll
        for (int j = 0; j < 8; ++j) sacc[t][n][j] = 0.0f;
#pragma unroll 4
    for (int c = qk_half * (kHead / 8); c < (qk_half + 1) * (kHead / 8); ++c) {
      uint2 a[2], bk[2];
#pragma unroll
      for (int t = 0; t < 2; ++t) {
        const int r = qk_r0 + 8 * t + idx;
        a[t] = lds_chunk<kHead>(qs, r, c, swz_f(r));
      }
#pragma unroll
      for (int n = 0; n < 2; ++n) {
        const int r = 32 * n + 8 * quad + idx;
        bk[n] = lds_chunk<kHead>(kv, r, c, swz_f(r));
      }
#pragma unroll
      for (int t = 0; t < 2; ++t)
#pragma unroll
        for (int n = 0; n < 2; ++n) mma_8x8x4(sacc[t][n], a[t], bk[n]);
    }
    if (qk_half == 1) {
#pragma unroll
      for (int t = 0; t < 2; ++t)
#pragma unroll
        for (int n = 0; n < 2; ++n)
#pragma unroll
          for (int j = 0; j < 8; ++j)
            sp[(qk_r0 + 8 * t + frag_row(j)) * kLdSM + 32 * n + frag_col(j)] = sacc[t][n][j];
    }
    __syncthreads();

    if (qk_half == 0) {
#pragma unroll
      for (int t = 0; t < 2; ++t) {
#pragma unroll
        for (int hb = 0; hb < 2; ++hb) {
          const int r = qk_r0 + 8 * t + (idx & 1) + 2 * hb + 4 * (idx >> 2);
          const int limit = ctx + q0 + r;
          float v[2][4];
          float mx = -FLT_MAX;
#pragma unroll
          for (int n = 0; n < 2; ++n)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              const int j = 2 * hb + (e & 1) + 4 * (e >> 1);
              const int col = 32 * n + frag_col(j);
              const int key = k0 + col;
              const float x = sacc[t][n][j] + sp[r * kLdSM + col];
              const bool ok = r < rows && key < key_end && key <= limit;
              v[n][e] = ok ? x * scale : -FLT_MAX;
              mx = fmaxf(mx, v[n][e]);
            }
          mx = row_reduce_max(mx);
          const float m_old = m_s[r];
          const float l_old = l_s[r];
          const float m_new = fmaxf(m_old, mx);
          const float corr = (m_old == -FLT_MAX) ? 0.0f : expf(m_old - m_new);
          float sum = 0.0f;
#pragma unroll
          for (int n = 0; n < 2; ++n)
#pragma unroll
            for (int e = 0; e < 4; e += 2) {
              const int j = 2 * hb + 4 * (e >> 1);
              const int col = 32 * n + frag_col(j);
              const half p0 = __float2half_rn(v[n][e] == -FLT_MAX ? 0.0f : expf(v[n][e] - m_new));
              const half p1 = __float2half_rn(v[n][e + 1] == -FLT_MAX ? 0.0f : expf(v[n][e + 1] - m_new));
              sum += __half2float(p0) + __half2float(p1);
              *reinterpret_cast<half2 *>(ps + r * kTile + (((col >> 2) ^ swz_f(r)) << 2) + (col & 3)) =
                  __halves2half2(p0, p1);
            }
          sum = row_reduce_sum(sum);
          __syncwarp();
          if ((lane & 0xe) == 0) {
            m_s[r] = m_new;
            l_s[r] = l_old * corr + sum;
            fac_s[r] = corr;
          }
        }
      }
    } else {
      int8_store<kPT / 2, true>(kv, rv, vt);
      if (k1 < key_end) int8_load<kPT / 2>(rv, vt, v_cache, v_scale, table, k1, n1, block_size, hkv, kh);
    }
    __syncthreads();

#pragma unroll
    for (int t = 0; t < 4; ++t)
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float f = fac_s[pv_r0 + 8 * t + frag_row(j)];
#pragma unroll
        for (int n = 0; n < 2; ++n) o[t][n][j] *= f;
      }
#pragma unroll 4
    for (int k = 0; k < kTile; k += 4) {
      uint2 a[4], bv[2];
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        const int r = pv_r0 + 8 * t + idx;
        a[t] = lds_chunk<kTile>(ps, r, k >> 2, swz_f(r));
      }
#pragma unroll
      for (int n = 0; n < 2; ++n) bv[n] = lds_chunk<kHead>(kv, k + vk, (pv_c0 + 32 * n + vn) >> 2, swz_g(k + vk));
#pragma unroll
      for (int t = 0; t < 4; ++t)
#pragma unroll
        for (int n = 0; n < 2; ++n) mma_8x8x4_rr(o[t][n], a[t], bv[n]);
    }
  }

#pragma unroll
  for (int t = 0; t < 4; ++t)
#pragma unroll
    for (int n = 0; n < 2; ++n)
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int row = pv_r0 + 8 * t + frag_row(j);
        if (row >= rows) continue;
        const int col = pv_c0 + 32 * n + frag_col(j);
        const int64_t tok = t0 + q0 + row;
        const float a = __half2float(__float2half_rn(o[t][n][j] / l_s[row]));
        const float g = __half2float(qkv[tok * ld + head * 2 * kHead + kHead + col]);
        out[(tok * hq + head) * kHead + col] = __float2half_rn(a / (1.0f + expf(-g)));
      }
}

/*
 * flash_attention_splitd: prefill attention for INT8 caches with the structure of 1Cat-vLLM's sm70 D256 split-D
 * kernel (Apache-2.0): one block of 8 warps per tile of 64 query rows and 32-key tiles (kPairs = 4; kPairs = 3 is
 * the same with 6 warps for tiles of up to 48 rows, the speculative verify steps: the 256 staging chunks of a key
 * tile then take two passes for warps 0 and 1). A tile (sequence, first
 * token q0, first head h0, heads g and tokens n, as int4 {b, q0, h0, g << 8 | n}) holds row r = token q0 + r / g
 * of head h0 + r % g, n * g <= 64 rows: g = 1 for 64 tokens of one head, g = hq / hkv to pack the heads of one KV
 * head for a few tokens. The warps form 4 pairs; pair p owns query rows 16 p .. 16 p + 15.
 *   S = Q K^T: warp (p, n) takes the pair's 16 rows and the 32 keys over head dimensions 128 n .. 128 n + 127, so
 *   that each K fragment feeds two mma (shared-memory bandwidth bounds this kernel); the Q fragments of the first
 *   kSdQReg k steps stay in registers. The pair adds its partial sums through shared memory: warp n keeps rows
 *   16 p + 8 n .. + 7. It runs the online softmax of these 8 rows on the accumulators (base 2, the scale folded into
 *   the key scales) and writes P and the rows' rescale factors to shared memory. Only the tiles on the diagonal apply
 *   the causal mask.
 *   O += P V: warp (p, n) takes the pair's 16 rows and head dimensions 128 n .. 128 n + 127, so O is 64 registers
 *   per thread; P is loaded once per tile. O is rescaled only when a row maximum of the warp changed.
 * A whole key tile of K and of V is staged in shared memory as exact FP16 integers; the K scale multiplies each
 * score column in FP32 and the V scale times 256 each P column (as verify_int8). Two barriers per tile: after the
 * first, V of the tile is stored and S computed; after the second (P ready), K of the next tile is stored and P V
 * computed. The INT8 K and V of the next tile load into registers a whole tile ahead (8 words each per thread).
 * Shared-memory layouts keep operand loads free of bank conflicts with offsets fixed at compile time: Q, K and V
 * rows have a pitch of 260 halves, P rows 36. Key k is K row (k & 3) | (k >> 3) << 2 | ((k >> 2) & 1) << 4, so that
 * each half-warp of S reads 16 consecutive rows, and V row (k & 1) | ((k >> 1) & 1) << 3 | ((k >> 2) & 3) << 1 |
 * (k >> 4) << 4, so that the 4 keys and 4 column chunks of a half-warp of P V fall in distinct banks. A thread stages
 * 32 consecutive bytes of one key (two 16-byte loads); each half-warp stages 8 keys whose rows differ modulo 8,
 * two parts of each, so that these stores are free of bank conflicts too.
 * Split over keys (grid z = splits > 1): a query tile with ntiles key tiles uses u = min(splits, ceil(ntiles /
 * min_split_tiles)) splits (short_split_tiles instead of min_split_tiles up to 128 key tiles, if it is not 0); split s < u takes the key tiles ntiles * s / u .. ntiles * (s + 1) / u, and writes the
 * unnormalized O and the row maxima (natural log) and sums in the layout of seventy_verify_attention, which
 * verify_combine_kernel merges and gates; a split s >= u writes an empty part. A fixed grid thus fits every context
 * length (CUDA graphs: seventy_verify_attention_splitd).
 */
constexpr int kSdKeys = 32;
constexpr int kSdQPitch = kHead + 4;
constexpr int kSdKPitch = kHead + 4;
constexpr int kSdPPitch = kSdKeys + 4;
constexpr int kSdQReg = 4;
constexpr int sd_smem(int pairs) {
  return (16 * pairs * kSdQPitch + 2 * kSdKeys * kSdKPitch + 16 * pairs * kSdPPitch) * static_cast<int>(sizeof(half)) +
         (2 * kSdKeys + 2 * 16 * pairs + 64 * pairs * 8) * static_cast<int>(sizeof(float));
}

__device__ __forceinline__ float ex2_approx(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ int sd_krow(int k) { return (k & 3) | ((k >> 3) << 2) | (((k >> 2) & 1) << 4); }

__device__ __forceinline__ int sd_vrow(int k) {
  return (k & 1) | (((k >> 1) & 1) << 3) | (((k >> 2) & 3) << 1) | ((k >> 4) << 4);
}

__device__ __forceinline__ int sd_krow_inv(int r) { return (r & 3) | (((r >> 2) & 3) << 3) | (((r >> 4) & 1) << 2); }

__device__ __forceinline__ int sd_vrow_inv(int r) {
  return (r & 1) | (((r >> 3) & 1) << 1) | (((r >> 1) & 3) << 2) | ((r >> 4) << 4);
}

template <int kPairs>
__global__ void __launch_bounds__(kPairs * 64, 1) flash_attention_splitd(
    half *out, const half *q, const int8_t *k_cache, const int8_t *v_cache, const float *k_scale,
    const float *v_scale, const half *qkv, int ld, const int *tiles, int num_tiles, const int *block_tables,
    int table_width, const int *query_start, const int *context_lens, int block_shift, int hq, int hkv, float scale,
    int min_split_tiles, int short_split_tiles, int min_context, float *partial_o, float *partial_ml) {
  constexpr int kRowsT = 16 * kPairs;
  constexpr int kThr = 64 * kPairs;
  constexpr int kChunks = (8 * kSdKeys + kThr - 1) / kThr;
  extern __shared__ __align__(16) unsigned char sdm[];
  half *qs = reinterpret_cast<half *>(sdm);
  half *kbuf = qs + kRowsT * kSdQPitch;
  half *vbuf = kbuf + kSdKeys * kSdKPitch;
  half *ps = vbuf + kSdKeys * kSdKPitch;
  float *ksc = reinterpret_cast<float *>(ps + kRowsT * kSdPPitch);
  float *vsc = ksc + kSdKeys;
  float *alpha = vsc + kSdKeys;
  float *lsum = alpha + kRowsT;
  float *xch = lsum + kRowsT;

  const int tix = num_tiles - 1 - blockIdx.x;
  const int4 tile = reinterpret_cast<const int4 *>(tiles)[tix];
  const int b = tile.x, q0 = tile.y, head0 = tile.z, grp = tile.w >> 8, ntok = tile.w & 255;
  const int kh = head0 / (hq / hkv);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int quad = (lane >> 2) & 3;
  const int idx = lane_idx();
  const int pair = warp >> 1, nw = warp & 1;
  const int t0 = query_start[b];
  const int ctx = context_lens[b];
  if (ctx < min_context) return;
  const int rows = ntok * grp;
  const int key_end = ctx + q0 + ntok;
  const int ntiles = (key_end + kSdKeys - 1) / kSdKeys;
  const int splits = gridDim.z, split = blockIdx.z;
  const int mst = ntiles <= 128 && short_split_tiles > 0 ? short_split_tiles : min_split_tiles;
  const int used = min(splits, max(1, (ntiles + mst - 1) / mst));
  if (split >= used) {
    for (int r = tid; r < rows; r += kThr) {
      const int64_t pidx = (static_cast<int64_t>(t0 + q0 + r / grp) * hq + head0 + r % grp) * splits + split;
      partial_ml[pidx * 2] = -FLT_MAX;
      partial_ml[pidx * 2 + 1] = 0.0f;
    }
    return;
  }
  const int t_begin = ntiles * split / used, t_end = ntiles * (split + 1) / used;
  const int *table = block_tables + static_cast<int64_t>(b) * table_width;
  const int bmask = (1 << block_shift) - 1;
  const int last_block = (key_end - 1) >> block_shift;
  const float kscale = scale * 1.4426950408889634f;

  for (int e = tid; e < kRowsT * (kHead / 8); e += kThr) {
    const int r = e / (kHead / 8);
    const int pr = e % (kHead / 8);
    uint4 val = make_uint4(0, 0, 0, 0);
    if (r < rows)
      val = *reinterpret_cast<const uint4 *>(
          q + (static_cast<int64_t>(t0 + q0 + r / grp) * hq + head0 + r % grp) * kHead + 8 * pr);
    *reinterpret_cast<uint2 *>(qs + r * kSdQPitch + 8 * pr) = make_uint2(val.x, val.y);
    *reinterpret_cast<uint2 *>(qs + r * kSdQPitch + 8 * pr + 4) = make_uint2(val.z, val.w);
  }

  int kkey[kChunks], vkey[kChunks];
  half *kst[kChunks], *vst[kChunks];
  const int8_t *kbase[kChunks], *vbase[kChunks];
#pragma unroll
  for (int c = 0; c < kChunks; ++c) {
    const int chunk = min(tid + c * kThr, 8 * kSdKeys - 1);
    const int vwarp = chunk >> 5, vlane = chunk & 31;
    const int part = 4 * (vwarp >> 2) + (vlane >> 3);
    const int srow = 8 * (vwarp & 3) + (vlane & 7);
    kkey[c] = sd_krow_inv(srow);
    vkey[c] = sd_vrow_inv(srow);
    kst[c] = kbuf + srow * kSdKPitch + 32 * part;
    vst[c] = vbuf + srow * kSdKPitch + 32 * part;
    kbase[c] = k_cache + kh * kHead + 32 * part;
    vbase[c] = v_cache + kh * kHead + 32 * part;
  }
  const bool chunk_on[2] = {true, kChunks > 1 && tid + kThr < 8 * kSdKeys};
  const int scale_key = tid & 31;
  int blk_k[kChunks], blk_v[kChunks], blk_s = 0;
  auto lookup = [&](int t) {
#pragma unroll
    for (int c = 0; c < kChunks; ++c) {
      blk_k[c] = __ldg(table + min((t * kSdKeys + kkey[c]) >> block_shift, last_block));
      blk_v[c] = __ldg(table + min((t * kSdKeys + vkey[c]) >> block_shift, last_block));
    }
    if (tid < 2 * kSdKeys) blk_s = __ldg(table + min((t * kSdKeys + scale_key) >> block_shift, last_block));
  };
  const float *const sbase = (tid < kSdKeys ? k_scale : v_scale) + kh;
  const int8_t *kptr[kChunks], *vptr[kChunks];
  const float *sptr = sbase;
  bool k_ok[kChunks], v_ok[kChunks], s_ok = false;
  auto set_tile = [&](int t) {
#pragma unroll
    for (int c = 0; c < kChunks; ++c) {
      const int kpos = t * kSdKeys + kkey[c], vpos = t * kSdKeys + vkey[c];
      k_ok[c] = chunk_on[c] && kpos < key_end;
      v_ok[c] = chunk_on[c] && vpos < key_end;
      kptr[c] = kbase[c] + static_cast<int64_t>(((blk_k[c] << block_shift) + (kpos & bmask)) * hkv) * kHead;
      vptr[c] = vbase[c] + static_cast<int64_t>(((blk_v[c] << block_shift) + (vpos & bmask)) * hkv) * kHead;
    }
    const int spos = t * kSdKeys + scale_key;
    s_ok = tid < 2 * kSdKeys && spos < key_end;
    sptr = sbase + ((blk_s << block_shift) + (spos & bmask)) * hkv;
  };
  uint4 rk[kChunks][2], rv[kChunks][2];
  float rs = 0.0f;
  auto load_k = [&]() {
#pragma unroll
    for (int c = 0; c < kChunks; ++c) {
      rk[c][0] = rk[c][1] = make_uint4(0, 0, 0, 0);
      if (k_ok[c]) {
        rk[c][0] = *reinterpret_cast<const uint4 *>(kptr[c]);
        rk[c][1] = *reinterpret_cast<const uint4 *>(kptr[c] + 16);
      }
    }
    rs = s_ok ? *sptr : 0.0f;
  };
  auto load_v = [&]() {
#pragma unroll
    for (int c = 0; c < kChunks; ++c) {
      rv[c][0] = rv[c][1] = make_uint4(0, 0, 0, 0);
      if (v_ok[c]) {
        rv[c][0] = *reinterpret_cast<const uint4 *>(vptr[c]);
        rv[c][1] = *reinterpret_cast<const uint4 *>(vptr[c] + 16);
      }
    }
  };
  auto store8 = [](half *dst, const uint4 (&w)[2]) {
    const uint32_t *u = reinterpret_cast<const uint32_t *>(w);
#pragma unroll
    for (int i = 0; i < 8; ++i) *reinterpret_cast<uint2 *>(dst + 4 * i) = int8x4_to_half_exact(u[i]);
  };
  auto store_k = [&]() {
#pragma unroll
    for (int c = 0; c < kChunks; ++c)
      if (chunk_on[c]) store8(kst[c], rk[c]);
    if (tid < kSdKeys) ksc[tid] = rs * kscale;
    else if (tid < 2 * kSdKeys) vsc[tid - kSdKeys] = rs * 256.0f;
  };
  auto store_v = [&]() {
#pragma unroll
    for (int c = 0; c < kChunks; ++c)
      if (chunk_on[c]) store8(vst[c], rv[c]);
  };

  float o[2][2][2][8];
#pragma unroll
  for (int a = 0; a < 2; ++a)
#pragma unroll
    for (int t = 0; t < 2; ++t)
#pragma unroll
      for (int g = 0; g < 2; ++g)
#pragma unroll
        for (int j = 0; j < 8; ++j) o[a][t][g][j] = 0.0f;
  const int qk_r0 = 16 * pair + 8 * nw;
  const int pv_r0 = 16 * pair;
  float m_row[2] = {-INFINITY, -INFINITY}, l_row[2] = {0.0f, 0.0f};
  const half *const qa = qs + (pv_r0 + idx) * kSdQPitch + 128 * nw;
  const half *const kb = kbuf + sd_krow(8 * quad + idx) * kSdKPitch + 128 * nw;
  float4 *const xsend = reinterpret_cast<float4 *>(xch) + warp * 64 + lane;
  const float4 *const xrecv = reinterpret_cast<const float4 *>(xch) + (warp ^ 1) * 64 + lane;
  const half *const pa = ps + (pv_r0 + idx) * kSdPPitch;
  const int vk = lane & 3;
  const int vchunk = 2 * quad + (lane >> 4);
  const half *const vb0 = vbuf + sd_vrow(vk) * kSdKPitch + 128 * nw + 4 * vchunk;
  const half *const vb1 = vb0 + 32;
  const int col0 = 8 * quad + 2 * ((idx >> 1) & 1);
  const int row0 = qk_r0 + frag_row(0), row1 = qk_r0 + frag_row(2);
  const int tok0 = row0 / grp, tok1 = row1 / grp;
  half *const pst0 = ps + row0 * kSdPPitch + col0;
  half *const pst1 = ps + row1 * kSdPPitch + col0;

  if (t_begin < t_end) {
    lookup(t_begin);
    set_tile(t_begin);
    load_k();
    if (t_begin + 1 < t_end) lookup(t_begin + 1);
  }
  __syncthreads();
  uint2 qreg[kSdQReg][2];
#pragma unroll
  for (int c = 0; c < kSdQReg; ++c) {
    qreg[c][0] = *reinterpret_cast<const uint2 *>(qa + 4 * c);
    qreg[c][1] = *reinterpret_cast<const uint2 *>(qa + 8 * kSdQPitch + 4 * c);
  }
  if (t_begin < t_end) {
    store_k();
    load_v();
    if (t_begin + 1 < t_end) {
      set_tile(t_begin + 1);
      load_k();
      if (t_begin + 2 < t_end) lookup(t_begin + 2);
    }
  }

  for (int t = t_begin; t < t_end; ++t) {
    __syncthreads();
    store_v();
    if (t + 1 < t_end) load_v();
    float sacc[2][8];
#pragma unroll
    for (int j = 0; j < 8; ++j) sacc[0][j] = sacc[1][j] = 0.0f;
#pragma unroll
    for (int c = 0; c < kSdQReg; ++c) {
      const uint2 bk = *reinterpret_cast<const uint2 *>(kb + 4 * c);
      mma_8x8x4(sacc[0], qreg[c][0], bk);
      mma_8x8x4(sacc[1], qreg[c][1], bk);
    }
#pragma unroll
    for (int c = kSdQReg; c < kHead / 8; ++c) {
      const uint2 a0 = *reinterpret_cast<const uint2 *>(qa + 4 * c);
      const uint2 a1 = *reinterpret_cast<const uint2 *>(qa + 8 * kSdQPitch + 4 * c);
      const uint2 bk = *reinterpret_cast<const uint2 *>(kb + 4 * c);
      mma_8x8x4(sacc[0], a0, bk);
      mma_8x8x4(sacc[1], a1, bk);
    }
    {
      float send[8];
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        send[j] = nw ? sacc[0][j] : sacc[1][j];
        sacc[0][j] = nw ? sacc[1][j] : sacc[0][j];
      }
      xsend[0] = make_float4(send[0], send[1], send[2], send[3]);
      xsend[32] = make_float4(send[4], send[5], send[6], send[7]);
      asm volatile("bar.sync %0, 64;" ::"r"(1 + pair) : "memory");
      const float4 r0 = xrecv[0], r1 = xrecv[32];
      sacc[0][0] += r0.x;
      sacc[0][1] += r0.y;
      sacc[0][2] += r0.z;
      sacc[0][3] += r0.w;
      sacc[0][4] += r1.x;
      sacc[0][5] += r1.y;
      sacc[0][6] += r1.z;
      sacc[0][7] += r1.w;
    }

    {
      const float2 ks01 = *reinterpret_cast<const float2 *>(ksc + col0);
      const float2 ks45 = *reinterpret_cast<const float2 *>(ksc + col0 + 4);
      const float2 vs01 = *reinterpret_cast<const float2 *>(vsc + col0);
      const float2 vs45 = *reinterpret_cast<const float2 *>(vsc + col0 + 4);
      const float ks[4] = {ks01.x, ks01.y, ks45.x, ks45.y};
      const float vs[4] = {vs01.x, vs01.y, vs45.x, vs45.y};
      float *sc = sacc[0];
#pragma unroll
      for (int j = 0; j < 8; ++j) sc[j] *= ks[(j & 1) | ((j >> 1) & 2)];
      if ((t + 1) * kSdKeys - 1 > ctx + q0) {
        const int lim0 = ctx + q0 + tok0 - t * kSdKeys - col0;
        const int lim1 = ctx + q0 + tok1 - t * kSdKeys - col0;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const int dc = (j & 1) + 4 * (j >> 2);
          if (dc > (((j >> 1) & 1) ? lim1 : lim0)) sc[j] = -INFINITY;
        }
      }
      float mx[2] = {-INFINITY, -INFINITY};
#pragma unroll
      for (int j = 0; j < 8; ++j) mx[(j >> 1) & 1] = fmaxf(mx[(j >> 1) & 1], sc[j]);
      float corr[2], sum[2] = {0.0f, 0.0f}, m_use[2];
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const float m_new = fmaxf(m_row[h], row_reduce_max(mx[h]));
        m_use[h] = m_new == -INFINITY ? 0.0f : m_new;
        corr[h] = ex2_approx(m_row[h] - m_use[h]);
        m_row[h] = m_new;
      }
#pragma unroll
      for (int j = 0; j < 8; j += 2) {
        const int h = (j >> 1) & 1;
        const float p0 = ex2_approx(sc[j] - m_use[h]);
        const float p1 = ex2_approx(sc[j + 1] - m_use[h]);
        sum[h] += p0 + p1;
        const half2 pp = __floats2half2_rn(p0 * vs[(j >> 1) & 2], p1 * vs[((j >> 1) & 2) | 1]);
        *reinterpret_cast<half2 *>((h ? pst1 : pst0) + 4 * (j >> 2)) = pp;
      }
#pragma unroll
      for (int h = 0; h < 2; ++h) l_row[h] = l_row[h] * corr[h] + sum[h];
      if ((lane & 0xe) == 0) {
        alpha[row0] = corr[0];
        alpha[row1] = corr[1];
      }
    }
    __syncthreads();
    if (t + 1 < t_end) {
      store_k();
      if (t + 2 < t_end) {
        set_tile(t + 2);
        load_k();
        if (t + 3 < t_end) lookup(t + 3);
      }
    }
    {
      float f[2][2];
#pragma unroll
      for (int tt = 0; tt < 2; ++tt)
#pragma unroll
        for (int h = 0; h < 2; ++h) f[tt][h] = alpha[pv_r0 + 8 * tt + frag_row(2 * h)];
      const bool same = f[0][0] == 1.0f && f[0][1] == 1.0f && f[1][0] == 1.0f && f[1][1] == 1.0f;
      if (!__all_sync(0xffffffffu, same)) {
#pragma unroll
        for (int a = 0; a < 2; ++a)
#pragma unroll
          for (int tt = 0; tt < 2; ++tt)
#pragma unroll
            for (int g = 0; g < 2; ++g)
#pragma unroll
              for (int j = 0; j < 8; ++j) o[a][tt][g][j] *= f[tt][(j >> 1) & 1];
      }
    }
    uint2 pf[kSdKeys / 4][2];
#pragma unroll
    for (int kk = 0; kk < kSdKeys / 4; ++kk)
#pragma unroll
      for (int tt = 0; tt < 2; ++tt)
        pf[kk][tt] = *reinterpret_cast<const uint2 *>(pa + 8 * tt * kSdPPitch + 4 * kk);
#pragma unroll
    for (int dl = 0; dl < 2; ++dl)
#pragma unroll
      for (int kk = 0; kk < kSdKeys / 4; ++kk) {
        const uint2 b0 = *reinterpret_cast<const uint2 *>(vb0 + 64 * dl + sd_vrow(4 * kk) * kSdKPitch);
        const uint2 b1 = *reinterpret_cast<const uint2 *>(vb1 + 64 * dl + sd_vrow(4 * kk) * kSdKPitch);
#pragma unroll
        for (int tt = 0; tt < 2; ++tt) {
          mma_8x8x4_rr(o[dl][tt][0], pf[kk][tt], b0);
          mma_8x8x4_rr(o[dl][tt][1], pf[kk][tt], b1);
        }
      }
  }

#pragma unroll
  for (int h = 0; h < 2; ++h) l_row[h] = row_reduce_sum(l_row[h]);
  constexpr float kInv256 = 1.0f / 256.0f;
  if (splits > 1) {
    if ((lane & 0xe) == 0) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int row = h ? row1 : row0;
        if (row >= rows) continue;
        const int64_t pidx = (static_cast<int64_t>(t0 + q0 + row / grp) * hq + head0 + row % grp) * splits + split;
        partial_ml[pidx * 2] = l_row[h] > 0.0f ? m_row[h] * 0.6931471805599453f : -FLT_MAX;
        partial_ml[pidx * 2 + 1] = l_row[h];
      }
    }
#pragma unroll
    for (int dl = 0; dl < 2; ++dl)
#pragma unroll
      for (int tt = 0; tt < 2; ++tt)
#pragma unroll
        for (int g = 0; g < 2; ++g)
#pragma unroll
          for (int j = 0; j < 8; j += 2) {
            const int row = pv_r0 + 8 * tt + frag_row(j);
            if (row >= rows) continue;
            const int col = 128 * nw + 64 * dl + 32 * g + frag_col(j);
            const int64_t pidx =
                (static_cast<int64_t>(t0 + q0 + row / grp) * hq + head0 + row % grp) * splits + split;
            *reinterpret_cast<float2 *>(partial_o + pidx * kHead + col) =
                make_float2(o[dl][tt][g][j] * kInv256, o[dl][tt][g][j + 1] * kInv256);
          }
    return;
  }
  if ((lane & 0xe) == 0) {
    lsum[row0] = l_row[0];
    lsum[row1] = l_row[1];
  }
  __syncthreads();
#pragma unroll
  for (int dl = 0; dl < 2; ++dl)
#pragma unroll
    for (int tt = 0; tt < 2; ++tt)
#pragma unroll
      for (int g = 0; g < 2; ++g)
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const int row = pv_r0 + 8 * tt + frag_row(j);
          if (row >= rows) continue;
          const int col = 128 * nw + 64 * dl + 32 * g + frag_col(j);
          const int64_t tok = t0 + q0 + row / grp;
          const int head = head0 + row % grp;
          const float l = lsum[row];
          const float a = __half2float(__float2half_rn(l > 0.0f ? o[dl][tt][g][j] * kInv256 / l : 0.0f));
          const float gt = __half2float(qkv[tok * ld + head * 2 * kHead + kHead + col]);
          out[(tok * hq + head) * kHead + col] = __float2half_rn(a / (1.0f + expf(-gt)));
        }
}

template <typename T>
int launch(half *out, const half *q, const T *k, const T *v, const float *ks, const float *vs, const half *qkv,
           int ld, const int *tiles, int num_tiles, const int *block_tables, int table_width, const int *query_start,
           const int *context_lens, int block_size, int hq, int hkv, float scale, cudaStream_t stream) {
  if constexpr (sizeof(T) == 1) {
    static const bool wmma_only = [] {
      const char *v = getenv("SEVENTY_PREFILL_ATTN");
      return v != nullptr && strcmp(v, "wmma") == 0;
    }();
    if (!wmma_only) {
      const cudaError_t e = cudaFuncSetAttribute(flash_attention_mma, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                 kMmaSmem);
      if (e != cudaSuccess) return static_cast<int>(e);
      flash_attention_mma<<<dim3(num_tiles, hq), kPT, kMmaSmem, stream>>>(
          out, q, reinterpret_cast<const int8_t *>(k), reinterpret_cast<const int8_t *>(v), ks, vs, qkv, ld, tiles,
          block_tables, table_width, query_start, context_lens, block_size, hq, hkv, scale);
      return launch_status();
    }
  }
  const cudaError_t e =
      cudaFuncSetAttribute(flash_attention_wmma<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, kPrefillSmem);
  if (e != cudaSuccess) return static_cast<int>(e);
  flash_attention_wmma<T><<<dim3(num_tiles, hq), kPT, kPrefillSmem, stream>>>(
      out, q, k, v, ks, vs, qkv, ld, tiles, block_tables, table_width, query_start, context_lens, block_size, hq, hkv,
      scale);
  return launch_status();
}

}  // namespace
}  // namespace seventy

SEVENTY_API int seventy_flash_attention(void *out, const void *q, const void *k_cache, const void *v_cache,
                                        const float *k_scale, const float *v_scale, int kv_type, const void *qkv,
                                        int ld, const int *tiles, int num_tiles, const int *block_tables,
                                        int table_width, const int *query_start, const int *context_lens,
                                        int block_size, int hq, int hkv, int head_dim, float scale,
                                        cudaStream_t stream) {
  using namespace seventy;
  if (out == nullptr || q == nullptr || k_cache == nullptr || v_cache == nullptr || qkv == nullptr ||
      tiles == nullptr || block_tables == nullptr || query_start == nullptr || context_lens == nullptr)
    return SEVENTY_ERR_INVALID_ARGUMENT;
  if (head_dim != kHead || hq % hkv || num_tiles < 0) return SEVENTY_ERR_INVALID_ARGUMENT;
  if (kv_type != 0 && (kv_type != 1 || k_scale == nullptr || v_scale == nullptr)) return SEVENTY_ERR_INVALID_ARGUMENT;
  if (num_tiles == 0) return SEVENTY_OK;
  if (kv_type == 1)
    return launch<int8_t>(static_cast<half *>(out), static_cast<const half *>(q), static_cast<const int8_t *>(k_cache),
                          static_cast<const int8_t *>(v_cache), k_scale, v_scale, static_cast<const half *>(qkv), ld,
                          tiles, num_tiles, block_tables, table_width, query_start, context_lens, block_size, hq,
                          hkv, scale, stream);
  return launch<half>(static_cast<half *>(out), static_cast<const half *>(q), static_cast<const half *>(k_cache),
                      static_cast<const half *>(v_cache), nullptr, nullptr, static_cast<const half *>(qkv), ld, tiles,
                      num_tiles, block_tables, table_width, query_start, context_lens, block_size, hq, hkv, scale,
                      stream);
}

namespace seventy {
namespace {
__global__ void verify_combine_kernel(half *out, const float *partial_o, const float *partial_ml, const half *qkv,
                                      int ld, int hq, int splits) {
  const int t = blockIdx.x;
  const int head = blockIdx.y;
  const int d = threadIdx.x;
  const int64_t base = (static_cast<int64_t>(t) * hq + head) * splits;
  float m = -FLT_MAX;
#pragma unroll 8
  for (int s = 0; s < splits; ++s) m = fmaxf(m, partial_ml[(base + s) * 2]);
  float l = 0.0f, o = 0.0f;
#pragma unroll 8
  for (int s = 0; s < splits; ++s) {
    const float2 ml = *reinterpret_cast<const float2 *>(partial_ml + (base + s) * 2);
    if (ml.x != -FLT_MAX) {
      const float w = expf(ml.x - m);
      l += ml.y * w;
      o += partial_o[(base + s) * kHead + d] * w;
    }
  }
  const float a = __half2float(__float2half_rn(o / l));
  const float g = __half2float(qkv[static_cast<int64_t>(t) * ld + head * 2 * kHead + kHead + d]);
  out[(static_cast<int64_t>(t) * hq + head) * kHead + d] = __float2half_rn(a / (1.0f + expf(-g)));
}
}  // namespace
}  // namespace seventy

SEVENTY_API int seventy_verify_attention(void *out, const void *q, const void *k_cache, const void *v_cache,
                                         const float *k_scale, const float *v_scale, int kv_type, const void *qkv,
                                         int ld, const int *block_tables, int table_width, const int *query_start,
                                         const int *context_lens, int num_seqs, int tokens, int max_query_len,
                                         int block_size, int hq, int hkv, int head_dim, int splits,
                                         int min_split_keys, float scale, void *workspace, int64_t workspace_bytes,
                                         cudaStream_t stream) {
  using namespace seventy;
  if (out == nullptr || q == nullptr || k_cache == nullptr || v_cache == nullptr || qkv == nullptr ||
      block_tables == nullptr || query_start == nullptr || context_lens == nullptr || workspace == nullptr)
    return SEVENTY_ERR_INVALID_ARGUMENT;
  if (head_dim != kHead || hq % hkv || (hq / hkv) * max_query_len > kTile || splits < 1 || block_size < 1)
    return SEVENTY_ERR_INVALID_ARGUMENT;
  if (kv_type != 0 && (kv_type != 1 || k_scale == nullptr || v_scale == nullptr)) return SEVENTY_ERR_INVALID_ARGUMENT;
  const int64_t need = static_cast<int64_t>(tokens) * hq * splits * (kHead + 2) * static_cast<int64_t>(sizeof(float));
  if (workspace_bytes < need) return SEVENTY_ERR_INVALID_ARGUMENT;
  if (num_seqs == 0 || tokens == 0) return SEVENTY_OK;
  auto *po = static_cast<float *>(workspace);
  auto *pml = po + static_cast<int64_t>(tokens) * hq * splits * kHead;
  const dim3 grid(splits, num_seqs, hkv);
  cudaError_t e;
  const bool wide = (hq / hkv) * max_query_len <= kVRows;
  constexpr int kVerifySmem =
      (kTile * kLdQ + kTile * kLdVR + kTile * kLdV) * static_cast<int>(sizeof(half)) + 2 * kTile * 4;
  if (kv_type == 1) {
    static const bool legacy = [] {
      const char *v = getenv("SEVENTY_VERIFY_ATTN");
      return v != nullptr && strcmp(v, "wmma") == 0;
    }();
    if (!legacy && verify_int8_supported(hq, hkv, max_query_len, block_size)) {
      const int code = launch_verify_int8(po, pml, static_cast<const half *>(q), static_cast<const int8_t *>(k_cache),
                                          static_cast<const int8_t *>(v_cache), k_scale, v_scale, block_tables,
                                          table_width, nullptr, query_start, context_lens, num_seqs, block_size, hq,
                                          hkv, splits, min_split_keys, scale, stream);
      if (code != 0) return code;
    } else if (wide) {
      e = cudaFuncSetAttribute(verify_attention_wmma<int8_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, kWmmaSmem);
      if (e != cudaSuccess) return static_cast<int>(e);
      verify_attention_wmma<int8_t><<<grid, kVT, kWmmaSmem, stream>>>(
          po, pml, static_cast<const half *>(q), static_cast<const int8_t *>(k_cache),
          static_cast<const int8_t *>(v_cache), k_scale, v_scale, block_tables, table_width, query_start,
          context_lens, block_size, hq, hkv, splits, max(1, min_split_keys), scale);
    } else {
      e = cudaFuncSetAttribute(verify_attention_kernel<int8_t>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                               kVerifySmem);
      if (e != cudaSuccess) return static_cast<int>(e);
      verify_attention_kernel<int8_t><<<grid, kWarps * 32, kVerifySmem, stream>>>(
          po, pml, static_cast<const half *>(q), static_cast<const int8_t *>(k_cache),
          static_cast<const int8_t *>(v_cache), k_scale, v_scale, block_tables, table_width, query_start,
          context_lens, block_size, hq, hkv, splits, max(1, min_split_keys), scale);
    }
  } else {
    if (wide) {
      e = cudaFuncSetAttribute(verify_attention_wmma<half>, cudaFuncAttributeMaxDynamicSharedMemorySize, kWmmaSmem);
      if (e != cudaSuccess) return static_cast<int>(e);
      verify_attention_wmma<half><<<grid, kVT, kWmmaSmem, stream>>>(
          po, pml, static_cast<const half *>(q), static_cast<const half *>(k_cache),
          static_cast<const half *>(v_cache), nullptr, nullptr, block_tables, table_width, query_start, context_lens,
          block_size, hq, hkv, splits, max(1, min_split_keys), scale);
    } else {
      e = cudaFuncSetAttribute(verify_attention_kernel<half>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                               kVerifySmem);
      if (e != cudaSuccess) return static_cast<int>(e);
      verify_attention_kernel<half><<<grid, kWarps * 32, kVerifySmem, stream>>>(
          po, pml, static_cast<const half *>(q), static_cast<const half *>(k_cache),
          static_cast<const half *>(v_cache), nullptr, nullptr, block_tables, table_width, query_start, context_lens,
          block_size, hq, hkv, splits, max(1, min_split_keys), scale);
    }
  }
  verify_combine_kernel<<<dim3(tokens, hq), kHead, 0, stream>>>(static_cast<half *>(out), po, pml,
                                                                static_cast<const half *>(qkv), ld, hq, splits);
  return launch_status();
}

SEVENTY_API int seventy_grouped_attention(void *out, const void *q, const void *k_cache, const void *v_cache,
                                          const float *k_scale, const float *v_scale, const void *qkv, int ld,
                                          const int *block_tables, int table_width, const int *group_rows,
                                          const int *group_start, const int *group_context, int num_groups,
                                          int tokens, int max_group_tokens, int block_size, int hq, int hkv,
                                          int head_dim, int splits, int min_split_keys, float scale, void *workspace,
                                          int64_t workspace_bytes, cudaStream_t stream) {
  using namespace seventy;
  if (out == nullptr || q == nullptr || k_cache == nullptr || v_cache == nullptr || k_scale == nullptr ||
      v_scale == nullptr || qkv == nullptr || block_tables == nullptr || group_rows == nullptr ||
      group_start == nullptr || group_context == nullptr || workspace == nullptr)
    return SEVENTY_ERR_INVALID_ARGUMENT;
  if (head_dim != kHead || splits < 1 || !verify_int8_supported(hq, hkv, max_group_tokens, block_size))
    return SEVENTY_ERR_INVALID_ARGUMENT;
  const int64_t need = static_cast<int64_t>(tokens) * hq * splits * (kHead + 2) * static_cast<int64_t>(sizeof(float));
  if (workspace_bytes < need) return SEVENTY_ERR_INVALID_ARGUMENT;
  if (num_groups == 0 || tokens == 0) return SEVENTY_OK;
  auto *po = static_cast<float *>(workspace);
  auto *pml = po + static_cast<int64_t>(tokens) * hq * splits * kHead;
  const int code = launch_verify_int8(po, pml, static_cast<const half *>(q), static_cast<const int8_t *>(k_cache),
                                      static_cast<const int8_t *>(v_cache), k_scale, v_scale, block_tables,
                                      table_width, group_rows, group_start, group_context, num_groups, block_size, hq,
                                      hkv, splits, min_split_keys, scale, stream);
  if (code != 0) return code;
  verify_combine_kernel<<<dim3(tokens, hq), kHead, 0, stream>>>(static_cast<half *>(out), po, pml,
                                                                static_cast<const half *>(qkv), ld, hq, splits);
  return launch_status();
}

SEVENTY_API int seventy_prefill_attention(void *out, const void *q, const void *k_cache, const void *v_cache,
                                          const float *k_scale, const float *v_scale, const void *qkv, int ld,
                                          const int *tiles, int num_tiles, const int *block_tables, int table_width,
                                          const int *query_start, const int *context_lens, int tokens,
                                          int block_size, int hq, int hkv, int head_dim, int splits,
                                          int min_split_tiles, int short_split_tiles, int tile_rows, float scale,
                                          void *workspace, int64_t workspace_bytes, cudaStream_t stream) {
  using namespace seventy;
  if (out == nullptr || q == nullptr || k_cache == nullptr || v_cache == nullptr || k_scale == nullptr ||
      v_scale == nullptr || qkv == nullptr || tiles == nullptr || block_tables == nullptr || query_start == nullptr ||
      context_lens == nullptr)
    return SEVENTY_ERR_INVALID_ARGUMENT;
  if (head_dim != kHead || hkv < 1 || hq % hkv || num_tiles < 0 || splits < 1 || splits > 65535 || block_size < 1 ||
      (block_size & (block_size - 1)) || min_split_tiles < 1 || short_split_tiles < 0 ||
      (tile_rows != 64 && tile_rows != 48))
    return SEVENTY_ERR_INVALID_ARGUMENT;
  const int64_t need =
      splits > 1 ? static_cast<int64_t>(tokens) * hq * splits * (kHead + 2) * static_cast<int64_t>(sizeof(float)) : 0;
  if (splits > 1 && (workspace == nullptr || workspace_bytes < need)) return SEVENTY_ERR_INVALID_ARGUMENT;
  if (num_tiles == 0 || tokens == 0) return SEVENTY_OK;
  int shift = 0;
  while ((1 << shift) < block_size) ++shift;
  auto *po = static_cast<float *>(workspace);
  float *pml = splits > 1 ? po + static_cast<int64_t>(tokens) * hq * splits * kHead : nullptr;
  const auto kernel = tile_rows == 48 ? flash_attention_splitd<3> : flash_attention_splitd<4>;
  const int smem = tile_rows == 48 ? sd_smem(3) : sd_smem(4);
  const cudaError_t e = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  if (e != cudaSuccess) return static_cast<int>(e);
  kernel<<<dim3(num_tiles, 1, splits), 4 * tile_rows, smem, stream>>>(
      static_cast<half *>(out), static_cast<const half *>(q), static_cast<const int8_t *>(k_cache),
      static_cast<const int8_t *>(v_cache), k_scale, v_scale, static_cast<const half *>(qkv), ld, tiles, num_tiles,
      block_tables, table_width, query_start, context_lens, shift, hq, hkv, scale, min_split_tiles,
      short_split_tiles, 0, po, pml);
  if (splits > 1)
    verify_combine_kernel<<<dim3(tokens, hq), kHead, 0, stream>>>(static_cast<half *>(out), po, pml,
                                                                  static_cast<const half *>(qkv), ld, hq, splits);
  return launch_status();
}

SEVENTY_API int seventy_verify_attention_hybrid(void *out, const void *q, const void *k_cache, const void *v_cache,
                                                const float *k_scale, const float *v_scale, const void *qkv, int ld,
                                                const int *tiles, int num_tiles, const int *block_tables,
                                                int table_width, const int *query_start, const int *context_lens,
                                                int num_seqs, int tokens, int max_query_len, int block_size, int hq,
                                                int hkv, int splits, int min_split_tiles, int short_split_tiles,
                                                int min_split_keys, int split_context, float scale, void *workspace,
                                                int64_t workspace_bytes, cudaStream_t stream) {
  using namespace seventy;
  if (out == nullptr || q == nullptr || k_cache == nullptr || v_cache == nullptr || k_scale == nullptr ||
      v_scale == nullptr || qkv == nullptr || tiles == nullptr || block_tables == nullptr || query_start == nullptr ||
      context_lens == nullptr || workspace == nullptr)
    return SEVENTY_ERR_INVALID_ARGUMENT;
  if (hkv < 1 || hq % hkv || (hq / hkv) * max_query_len > 48 || splits < 2 || splits > 65535 || block_size < 1 ||
      (block_size & (block_size - 1)) || min_split_tiles < 1 || short_split_tiles < 0 || split_context < 0 ||
      !verify_int8_supported(hq, hkv, max_query_len, block_size))
    return SEVENTY_ERR_INVALID_ARGUMENT;
  const int64_t need = static_cast<int64_t>(tokens) * hq * splits * (kHead + 2) * static_cast<int64_t>(sizeof(float));
  if (workspace_bytes < need) return SEVENTY_ERR_INVALID_ARGUMENT;
  if (num_seqs == 0 || tokens == 0 || num_tiles == 0) return SEVENTY_OK;
  int shift = 0;
  while ((1 << shift) < block_size) ++shift;
  auto *po = static_cast<float *>(workspace);
  float *pml = po + static_cast<int64_t>(tokens) * hq * splits * kHead;
  const cudaError_t e =
      cudaFuncSetAttribute(flash_attention_splitd<3>, cudaFuncAttributeMaxDynamicSharedMemorySize, sd_smem(3));
  if (e != cudaSuccess) return static_cast<int>(e);
  flash_attention_splitd<3><<<dim3(num_tiles, 1, splits), 192, sd_smem(3), stream>>>(
      static_cast<half *>(out), static_cast<const half *>(q), static_cast<const int8_t *>(k_cache),
      static_cast<const int8_t *>(v_cache), k_scale, v_scale, static_cast<const half *>(qkv), ld, tiles, num_tiles,
      block_tables, table_width, query_start, context_lens, shift, hq, hkv, scale, min_split_tiles,
      short_split_tiles, split_context, po, pml);
  const int code = launch_verify_int8(po, pml, static_cast<const half *>(q), static_cast<const int8_t *>(k_cache),
                                      static_cast<const int8_t *>(v_cache), k_scale, v_scale, block_tables,
                                      table_width, nullptr, query_start, context_lens, num_seqs, block_size, hq, hkv,
                                      splits, min_split_keys, scale, stream, split_context - 1);
  if (code != 0) return code;
  verify_combine_kernel<<<dim3(tokens, hq), kHead, 0, stream>>>(static_cast<half *>(out), po, pml,
                                                                static_cast<const half *>(qkv), ld, hq, splits);
  return launch_status();
}
