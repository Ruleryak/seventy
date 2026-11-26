#include "common.cuh"
#include "gguf_tables.cuh"

/*
 * Routed experts in GGUF formats (llama.cpp): the grouped expert product of a decode step (the plan of
 * csrc/moe.cu, slots of up to 8 pairs of one expert) and the dequantization of whole experts to FP16 for prefill.
 *
 * Block formats as in ggml (ggml/src/ggml-common.h and ggml-quants.c of llama.cpp, MIT License, Copyright (c)
 * 2023-2026 The ggml authors; tables in csrc/gguf_tables.cuh). Each format's Block::group decodes 8 consecutive
 * values (group g of a block) to FP32; seventy/gguf.py has the same decoding on the CPU. A weight row of k values is
 * k / Block::kValues blocks of Block::kBytes bytes; an expert is n such rows.
 *
 * gguf_moe_kernel: one block of 4 warps per (tile of kTileRows output rows, slot: one row per warp, so that a step
 * with few experts still has many warps). The slot's activation rows (up to
 * 8, gathered through the plan) wait in shared memory as FP16; warp w takes rows w, w + 4, ... of the tile, and lane
 * l the groups l, l + 32, ... of each row (consecutive lanes read consecutive groups of one block), decodes each
 * group and adds its 8 products per activation row in FP32; a warp sum gives each output. FP16 out (row stride
 * out_ld), or FP32 times pair_weight[pair] (row stride n). compact as in seventy_moe_mma: bit 0, x rows in plan
 * order; bit 1, out rows in plan order.
 */

namespace seventy {
namespace {

using namespace gguf;

__device__ __forceinline__ float half_at(const uint8_t *p) {
  const uint16_t u = static_cast<uint16_t>(p[0]) | static_cast<uint16_t>(p[1]) << 8;
  return __half2float(__ushort_as_half(u));
}

__device__ __forceinline__ uint32_t u32_at(const uint8_t *p) {
  return static_cast<uint32_t>(p[0]) | static_cast<uint32_t>(p[1]) << 8 | static_cast<uint32_t>(p[2]) << 16 |
         static_cast<uint32_t>(p[3]) << 24;
}

__device__ __forceinline__ uint32_t u16_at(const uint8_t *p) {
  return static_cast<uint32_t>(p[0]) | static_cast<uint32_t>(p[1]) << 8;
}

__device__ __forceinline__ void signed8(float (&v)[8], float scale, uint64_t grid, uint32_t signs) {
#pragma unroll
  for (int t = 0; t < 8; ++t) {
    const float g = static_cast<float>((grid >> (8 * t)) & 0xff);
    v[t] = (signs >> t) & 1 ? -scale * g : scale * g;
  }
}

struct Q8_0 {
  static constexpr int kValues = 32, kBytes = 34;
  static __device__ __forceinline__ void group(const uint8_t *b, int g, float (&v)[8]) {
    const float d = half_at(b);
#pragma unroll
    for (int t = 0; t < 8; ++t) v[t] = d * static_cast<float>(static_cast<int8_t>(b[2 + 8 * g + t]));
  }
};

struct Q4_0 {
  static constexpr int kValues = 32, kBytes = 18;
  static __device__ __forceinline__ void group(const uint8_t *b, int g, float (&v)[8]) {
    const float d = half_at(b);
    const uint8_t *q = b + 2 + 8 * (g & 1);
    const int shift = (g >> 1) * 4;
#pragma unroll
    for (int t = 0; t < 8; ++t) v[t] = d * static_cast<float>(((q[t] >> shift) & 15) - 8);
  }
};

struct Q2_0 {
  static constexpr int kValues = 64, kBytes = 18;
  static __device__ __forceinline__ void group(const uint8_t *b, int g, float (&v)[8]) {
    const float d = half_at(b);
    const uint32_t q = u16_at(b + 2 + 2 * g);
#pragma unroll
    for (int t = 0; t < 8; ++t) v[t] = d * static_cast<float>(static_cast<int>((q >> (2 * t)) & 3) - 1);
  }
};

struct IQ4_NL {
  static constexpr int kValues = 32, kBytes = 18;
  static __device__ __forceinline__ void group(const uint8_t *b, int g, float (&v)[8]) {
    const float d = half_at(b);
    const uint8_t *q = b + 2 + 8 * (g & 1);
    const int shift = (g >> 1) * 4;
#pragma unroll
    for (int t = 0; t < 8; ++t) v[t] = d * static_cast<float>(kValuesIq4nl[(q[t] >> shift) & 15]);
  }
};

struct IQ4_XS {
  static constexpr int kValues = 256, kBytes = 136;
  static __device__ __forceinline__ void group(const uint8_t *b, int g, float (&v)[8]) {
    const int ib = g >> 2, gg = g & 3;
    const uint32_t sh = u16_at(b + 2);
    const int ls = ((b[4 + (ib >> 1)] >> (4 * (ib & 1))) & 15) | (((sh >> (2 * ib)) & 3) << 4);
    const float dl = half_at(b) * static_cast<float>(ls - 32);
    const uint8_t *q = b + 8 + 16 * ib + 8 * (gg & 1);
    const int shift = (gg >> 1) * 4;
#pragma unroll
    for (int t = 0; t < 8; ++t) v[t] = dl * static_cast<float>(kValuesIq4nl[(q[t] >> shift) & 15]);
  }
};

__device__ __forceinline__ void scale_min_k4(int j, const uint8_t *s, int &sc, int &mn) {
  if (j < 4) {
    sc = s[j] & 63;
    mn = s[j + 4] & 63;
  } else {
    sc = (s[j + 4] & 15) | ((s[j - 4] >> 6) << 4);
    mn = (s[j + 4] >> 4) | ((s[j] >> 6) << 4);
  }
}

struct Q4_K {
  static constexpr int kValues = 256, kBytes = 144;
  static __device__ __forceinline__ void group(const uint8_t *b, int g, float (&v)[8]) {
    const int j = g >> 3, r = g & 7, h = r >> 2, off = (r & 3) * 8;
    int sc, mn;
    scale_min_k4(2 * j + h, b + 4, sc, mn);
    const float d = half_at(b) * static_cast<float>(sc), m = half_at(b + 2) * static_cast<float>(mn);
    const uint8_t *q = b + 16 + 32 * j + off;
#pragma unroll
    for (int t = 0; t < 8; ++t) v[t] = d * static_cast<float>((q[t] >> (4 * h)) & 15) - m;
  }
};

struct Q5_K {
  static constexpr int kValues = 256, kBytes = 176;
  static __device__ __forceinline__ void group(const uint8_t *b, int g, float (&v)[8]) {
    const int j = g >> 3, r = g & 7, h = r >> 2, off = (r & 3) * 8;
    int sc, mn;
    scale_min_k4(2 * j + h, b + 4, sc, mn);
    const float d = half_at(b) * static_cast<float>(sc), m = half_at(b + 2) * static_cast<float>(mn);
    const uint8_t *qh = b + 16 + off;
    const uint8_t *q = b + 48 + 32 * j + off;
#pragma unroll
    for (int t = 0; t < 8; ++t) {
      const int value = ((q[t] >> (4 * h)) & 15) + (((qh[t] >> (2 * j + h)) & 1) << 4);
      v[t] = d * static_cast<float>(value) - m;
    }
  }
};

struct Q6_K {
  static constexpr int kValues = 256, kBytes = 210;
  static __device__ __forceinline__ void group(const uint8_t *b, int g, float (&v)[8]) {
    const int n = g >> 4, rem = g & 15, kq = rem >> 2, l0 = (rem & 3) * 8;
    const uint8_t *ql = b + 64 * n + (kq & 1) * 32 + l0;
    const uint8_t *qh = b + 128 + 32 * n + l0;
    const float d = half_at(b + 208) * static_cast<float>(static_cast<int8_t>(b[192 + 8 * n + 2 * kq + l0 / 16]));
#pragma unroll
    for (int t = 0; t < 8; ++t) {
      const int value = ((ql[t] >> (4 * (kq >> 1))) & 15) | (((qh[t] >> (2 * kq)) & 3) << 4);
      v[t] = d * static_cast<float>(value - 32);
    }
  }
};

struct IQ2_XXS {
  static constexpr int kValues = 256, kBytes = 66;
  static __device__ __forceinline__ void group(const uint8_t *b, int g, float (&v)[8]) {
    const int ib = g >> 2, l = g & 3;
    const uint8_t *q = b + 2 + 8 * ib;
    const uint32_t aux = u32_at(q + 4);
    const float db = half_at(b) * (0.5f + static_cast<float>(aux >> 28)) * 0.25f;
    signed8(v, db, __ldg(&kIq2xxsGrid[q[l]]), __ldg(&kKsignsIq2xs[(aux >> (7 * l)) & 127]));
  }
};

struct IQ2_XS {
  static constexpr int kValues = 256, kBytes = 74;
  static __device__ __forceinline__ void group(const uint8_t *b, int g, float (&v)[8]) {
    const int ib = g >> 2, l = g & 3;
    const uint32_t q = u16_at(b + 2 + 2 * (4 * ib + l));
    const int s = b[66 + ib];
    const float db = half_at(b) * (0.5f + static_cast<float>(l < 2 ? (s & 15) : (s >> 4))) * 0.25f;
    signed8(v, db, __ldg(&kIq2xsGrid[q & 511]), __ldg(&kKsignsIq2xs[q >> 9]));
  }
};

struct IQ2_S {
  static constexpr int kValues = 256, kBytes = 82;
  static __device__ __forceinline__ void group(const uint8_t *b, int g, float (&v)[8]) {
    const int ib = g >> 2, l = g & 3;
    const int idx = b[2 + 4 * ib + l] | (((b[66 + ib] >> (2 * l)) & 3) << 8);
    const int s = b[74 + ib];
    const float db = half_at(b) * (0.5f + static_cast<float>(l < 2 ? (s & 15) : (s >> 4))) * 0.25f;
    signed8(v, db, __ldg(&kIq2sGrid[idx]), b[34 + 4 * ib + l]);
  }
};

struct IQ3_XXS {
  static constexpr int kValues = 256, kBytes = 98;
  static __device__ __forceinline__ void group(const uint8_t *b, int g, float (&v)[8]) {
    const int ib = g >> 2, l = g & 3;
    const uint32_t aux = u32_at(b + 66 + 4 * ib);
    const float db = half_at(b) * (0.5f + static_cast<float>(aux >> 28)) * 0.5f;
    const uint8_t *q = b + 2 + 8 * ib + 2 * l;
    const uint64_t grid = static_cast<uint64_t>(__ldg(&kIq3xxsGrid[q[0]])) |
                          static_cast<uint64_t>(__ldg(&kIq3xxsGrid[q[1]])) << 32;
    signed8(v, db, grid, __ldg(&kKsignsIq2xs[(aux >> (7 * l)) & 127]));
  }
};

struct IQ3_S {
  static constexpr int kValues = 256, kBytes = 110;
  static __device__ __forceinline__ void group(const uint8_t *b, int g, float (&v)[8]) {
    const int ib = g >> 2, l = g & 3;
    const int qh = b[66 + ib];
    const uint8_t *q = b + 2 + 8 * ib + 2 * l;
    const int i1 = q[0] | (((qh >> (2 * l)) & 1) << 8), i2 = q[1] | (((qh >> (2 * l + 1)) & 1) << 8);
    const int sc = (b[106 + (ib >> 1)] >> (4 * (ib & 1))) & 15;
    const float db = half_at(b) * static_cast<float>(1 + 2 * sc);
    const uint64_t grid = static_cast<uint64_t>(__ldg(&kIq3sGrid[i1])) |
                          static_cast<uint64_t>(__ldg(&kIq3sGrid[i2])) << 32;
    signed8(v, db, grid, b[74 + 4 * ib + l]);
  }
};

constexpr int kGgufThreads = 128;
constexpr int kTileRows = 16;
constexpr int kMaxSlotRows = 8;

template <class B>
__global__ void __launch_bounds__(kGgufThreads) gguf_moe_kernel(void *out, int out_ld, const half *x, int row_div,
                                                                int k, int n, const uint8_t *const *wptrs,
                                                                const uint8_t *wbase, int64_t w_stride,
                                                                const int32_t *order, const int32_t *slot_expert,
                                                                const int32_t *slot_start,
                                                                const int32_t *slot_count, const int32_t *num_slots,
                                                                const float *pair_weight, int compact) {
  extern __shared__ __align__(16) half xs[];
  __shared__ int pair_of[kMaxSlotRows];
  const int slot = blockIdx.y;
  if (slot >= *num_slots) return;
  const int e = slot_expert[slot], start = slot_start[slot];
  const int m = min(slot_count[slot], kMaxSlotRows);
  if (threadIdx.x < kMaxSlotRows) pair_of[threadIdx.x] = threadIdx.x < m ? order[start + threadIdx.x] : 0;
  __syncthreads();
  for (int i = threadIdx.x; i < m * (k / 8); i += kGgufThreads) {
    const int r = i / (k / 8), part = i % (k / 8);
    const int64_t xr = (compact & 1) ? start + r : pair_of[r] / row_div;
    reinterpret_cast<uint4 *>(xs + static_cast<int64_t>(r) * k)[part] =
        reinterpret_cast<const uint4 *>(x + xr * k)[part];
  }
  __syncthreads();
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int row_bytes = k / B::kValues * B::kBytes;
  const uint8_t *w = wptrs != nullptr ? wptrs[e] : wbase + static_cast<int64_t>(e) * w_stride;
  constexpr int kGroups = B::kValues / 8;
  const int groups = k / 8;
  for (int j = blockIdx.x * kTileRows + warp; j < min(n, (blockIdx.x + 1) * kTileRows); j += 4) {
    const uint8_t *row = w + static_cast<int64_t>(j) * row_bytes;
    float acc[kMaxSlotRows];
#pragma unroll
    for (int r = 0; r < kMaxSlotRows; ++r) acc[r] = 0.0f;
#pragma unroll 4
    for (int gi = lane; gi < groups; gi += 32) {
      float v[8];
      B::group(row + (gi / kGroups) * B::kBytes, gi % kGroups, v);
#pragma unroll
      for (int r = 0; r < kMaxSlotRows; ++r) {
        if (r >= m) break;
        const uint4 raw = reinterpret_cast<const uint4 *>(xs + static_cast<int64_t>(r) * k)[gi];
        const half2 *h = reinterpret_cast<const half2 *>(&raw);
        float s = acc[r];
#pragma unroll
        for (int t = 0; t < 4; ++t) {
          const float2 f = __half22float2(h[t]);
          s = fmaf(v[2 * t], f.x, fmaf(v[2 * t + 1], f.y, s));
        }
        acc[r] = s;
      }
    }
#pragma unroll
    for (int r = 0; r < kMaxSlotRows; ++r) {
      if (r >= m) break;
      float s = acc[r];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
      if (lane == 0) {
        const int p = pair_of[r];
        const int64_t orow = (compact & 2) ? start + r : p;
        if (pair_weight != nullptr)
          static_cast<float *>(out)[orow * n + j] = s * pair_weight[p];
        else
          static_cast<half *>(out)[orow * out_ld + j] = __float2half_rn(s);
      }
    }
  }
}

template <class B>
__global__ void __launch_bounds__(256) gguf_dequant_kernel(half *out, const uint8_t *rows, int64_t nrows, int k) {
  constexpr int kGroups = B::kValues / 8;
  const int groups = k / 8;
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= nrows * groups) return;
  const int64_t row = i / groups;
  const int gi = static_cast<int>(i % groups);
  float v[8];
  B::group(rows + row * (k / B::kValues * B::kBytes) + (gi / kGroups) * B::kBytes, gi % kGroups, v);
  uint4 packed;
  half2 *h = reinterpret_cast<half2 *>(&packed);
#pragma unroll
  for (int t = 0; t < 4; ++t) h[t] = __floats2half2_rn(v[2 * t], v[2 * t + 1]);
  reinterpret_cast<uint4 *>(out + row * k)[gi] = packed;
}

template <class B>
__global__ void gguf_gather_kernel(half *out, const int64_t *ids, int ids_ld, int h0, int heads, const uint8_t *rows,
                                   int64_t row_base, int dim) {
  constexpr int kGroups = B::kValues / 8;
  const int t = blockIdx.x, h = blockIdx.y, gi = threadIdx.x;
  if (gi >= dim / 8) return;
  const int64_t row = ids[static_cast<int64_t>(t) * ids_ld + h0 + h] - row_base;
  float v[8];
  B::group(rows + row * (dim / B::kValues * B::kBytes) + (gi / kGroups) * B::kBytes, gi % kGroups, v);
  uint4 packed;
  half2 *o = reinterpret_cast<half2 *>(&packed);
#pragma unroll
  for (int i = 0; i < 4; ++i) o[i] = __floats2half2_rn(v[2 * i], v[2 * i + 1]);
  reinterpret_cast<uint4 *>(out + (static_cast<int64_t>(t) * heads + h) * dim)[gi] = packed;
}

template <class F>
int dispatch(int type, F &&f) {
  switch (type) {
    case 2: f(Q4_0{}); return SEVENTY_OK;
    case 8: f(Q8_0{}); return SEVENTY_OK;
    case 12: f(Q4_K{}); return SEVENTY_OK;
    case 13: f(Q5_K{}); return SEVENTY_OK;
    case 14: f(Q6_K{}); return SEVENTY_OK;
    case 16: f(IQ2_XXS{}); return SEVENTY_OK;
    case 17: f(IQ2_XS{}); return SEVENTY_OK;
    case 18: f(IQ3_XXS{}); return SEVENTY_OK;
    case 20: f(IQ4_NL{}); return SEVENTY_OK;
    case 21: f(IQ3_S{}); return SEVENTY_OK;
    case 22: f(IQ2_S{}); return SEVENTY_OK;
    case 23: f(IQ4_XS{}); return SEVENTY_OK;
    case 42: f(Q2_0{}); return SEVENTY_OK;
    default: return SEVENTY_ERR_INVALID_ARGUMENT;
  }
}

template <class B>
int64_t row_bytes(int k) {
  return k % B::kValues ? -1 : static_cast<int64_t>(k / B::kValues) * B::kBytes;
}

}  // namespace
}  // namespace seventy

SEVENTY_API int64_t seventy_gguf_row_bytes(int type, int k) {
  using namespace seventy;
  int64_t bytes = -1;
  if (k <= 0 || dispatch(type, [&](auto b) { bytes = row_bytes<decltype(b)>(k); }) != SEVENTY_OK) return -1;
  return bytes;
}

SEVENTY_API int seventy_gguf_moe(void *out, int out_ld, const void *x, int row_div, int k, int n, int type,
                                 const void *wptrs, const void *wbase, int64_t w_stride, const int32_t *order,
                                 const int32_t *slot_expert, const int32_t *slot_start, const int32_t *slot_count,
                                 const int32_t *num_slots, int max_slots, const float *pair_weight, int compact,
                                 cudaStream_t stream) {
  using namespace seventy;
  if (out == nullptr || x == nullptr || (wptrs == nullptr && wbase == nullptr) || order == nullptr || k <= 0 ||
      k % 8 || n <= 0 || row_div < 1 || max_slots < 1 || (pair_weight == nullptr && out_ld < n))
    return SEVENTY_ERR_INVALID_ARGUMENT;
  if (seventy_gguf_row_bytes(type, k) < 0) return SEVENTY_ERR_INVALID_ARGUMENT;
  if (!aligned(x, 16)) return SEVENTY_ERR_MISALIGNED;
  const size_t smem = static_cast<size_t>(kMaxSlotRows) * k * sizeof(half);
  if (smem > 48 * 1024) return SEVENTY_ERR_INVALID_ARGUMENT;
  const dim3 grid((n + kTileRows - 1) / kTileRows, max_slots);
  const int code = dispatch(type, [&](auto b) {
    using B = decltype(b);
    gguf_moe_kernel<B><<<grid, kGgufThreads, smem, stream>>>(
        out, out_ld, static_cast<const half *>(x), row_div, k, n, static_cast<const uint8_t *const *>(wptrs),
        static_cast<const uint8_t *>(wbase), w_stride, order, slot_expert, slot_start, slot_count, num_slots,
        pair_weight, compact);
  });
  if (code != SEVENTY_OK) return code;
  return launch_status();
}

SEVENTY_API int seventy_gguf_dequant(void *out, const void *rows, int64_t nrows, int k, int type,
                                     cudaStream_t stream) {
  using namespace seventy;
  if (out == nullptr || rows == nullptr || nrows < 0 || k <= 0 || k % 8 || seventy_gguf_row_bytes(type, k) < 0)
    return SEVENTY_ERR_INVALID_ARGUMENT;
  if (nrows == 0) return SEVENTY_OK;
  const int64_t total = nrows * (k / 8);
  const int code = dispatch(type, [&](auto b) {
    using B = decltype(b);
    gguf_dequant_kernel<B><<<static_cast<unsigned>((total + 255) / 256), 256, 0, stream>>>(
        static_cast<half *>(out), static_cast<const uint8_t *>(rows), nrows, k);
  });
  if (code != SEVENTY_OK) return code;
  return launch_status();
}

SEVENTY_API int seventy_gguf_gather(void *out, const int64_t *ids, int ids_ld, int h0, int h1, int tokens,
                                    const void *rows, int64_t row_base, int dim, int type, cudaStream_t stream) {
  using namespace seventy;
  if (out == nullptr || ids == nullptr || rows == nullptr || h1 <= h0 || h0 < 0 || h1 > ids_ld || tokens < 0 ||
      dim <= 0 || dim % 8 || dim / 8 > 1024 || seventy_gguf_row_bytes(type, dim) < 0)
    return SEVENTY_ERR_INVALID_ARGUMENT;
  if (tokens == 0) return SEVENTY_OK;
  const int threads = (dim / 8 + 31) / 32 * 32;
  const int code = dispatch(type, [&](auto b) {
    using B = decltype(b);
    gguf_gather_kernel<B><<<dim3(tokens, h1 - h0), threads, 0, stream>>>(
        static_cast<half *>(out), ids, ids_ld, h0, h1 - h0, static_cast<const uint8_t *>(rows), row_base, dim);
  });
  if (code != SEVENTY_OK) return code;
  return launch_status();
}

