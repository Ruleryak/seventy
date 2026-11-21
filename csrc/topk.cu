#include "common.cuh"

/*
 * seventy_topk: the k largest values of every row (k <= 64), sorted by value descending and, among equal values, by
 * id ascending, so that the first is the argmax with the smallest id. Row r of x holds n FP32 values (row stride
 * ld); the id of element e is ids[r * ids_ld + e] if ids is given (negative ids mark padding), else id_offset + e.
 *
 * topk_kernel takes one segment of a row per block: thread t keeps the order-preserving 32-bit keys of elements
 * t + 256 i of its segment in registers, as 64-bit (key, inverted id) pairs once selected, so that one descending
 * order sorts by value and then id. Each warp sorts its 32 per-thread maxima (shuffles); the smallest over the 8
 * warps of each warp's ceil(k / 8)-th largest maximum is a lower bound of the k-th largest value (at least k values
 * reach it), and usually only a few more than k values do. These candidates, all
 * distinct pairs, are placed by rank (the count of larger candidates; one barrier, no sorting network) when there
 * are at most kTopRank of them, else sorted (bitonic). When more than kTopCand values reach the bound, three histogram
 * passes over the key bits (12, 12 and 8 bits, in shared memory) find the exact k-th largest key K and how many keys
 * equal to K are needed; when more keys equal K than are needed, three more passes over the inverted ids pick the
 * smallest ids among them. Rows longer than kTopSeg values are cut into segments whose top k go to a workspace, and a
 * second launch of the same kernel selects among them.
 */

namespace seventy {
namespace {

constexpr int kTopThreads = 256;
constexpr int kTopMaxK = 64;
constexpr int kTopSeg = 4096;
constexpr int kTopMaxSegs = 64;
constexpr int kTopBins = 4096;
constexpr int kTopCand = 1024;
constexpr int kTopRank = 512;

__device__ __forceinline__ unsigned long long warp_sort_desc(unsigned long long v) {
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int size = 2; size <= 32; size <<= 1) {
#pragma unroll
    for (int stride = size >> 1; stride > 0; stride >>= 1) {
      const unsigned long long o = __shfl_xor_sync(0xffffffffu, v, stride);
      const bool lower = (lane & stride) == 0;
      const bool desc = (lane & size) == 0;
      const bool keep_max = lower == desc;
      v = keep_max ? (v > o ? v : o) : (v < o ? v : o);
    }
  }
  return v;
}

template <typename T>
__device__ __forceinline__ void bitonic_desc(T *a, int n) {
#pragma unroll 1
  for (int size = 2; size <= n; size <<= 1) {
#pragma unroll 1
    for (int stride = size >> 1; stride > 0; stride >>= 1) {
#pragma unroll 1
      for (int p = threadIdx.x; p < n / 2; p += blockDim.x) {
        const int i = ((p & ~(stride - 1)) << 1) | (p & (stride - 1));
        const int j = i + stride;
        const bool desc = (i & size) == 0;
        const T x = a[i], y = a[j];
        const bool sw = (x < y) == desc;
        a[i] = sw ? y : x;
        a[j] = sw ? x : y;
      }
      __syncthreads();
    }
  }
}

__device__ __forceinline__ uint32_t f2key(float f) {
  const uint32_t u = __float_as_uint(f);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__device__ __forceinline__ float key2f(uint32_t k) {
  return __uint_as_float((k & 0x80000000u) ? (k & 0x7fffffffu) : ~k);
}

/*
 * One radix pass of the selection: the histogram of bits [shift, shift + bits) of the values v[i] of the elements
 * with match[i], then the bin B whose suffix count first reaches *remaining. Updates *prefix with B, lowers
 * *remaining by the count above B and returns the count of B.
 */
template <int kPer>
__device__ __forceinline__ uint32_t radix_pass(uint32_t *hist, uint32_t *warp_tot, uint32_t *shared2,
                                               const uint32_t (&v)[kPer], const bool (&match)[kPer], int shift,
                                               int bits, uint32_t *prefix, int *remaining) {
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
  const int nb = 1 << bits;
  const uint32_t m = static_cast<uint32_t>(nb - 1);
  for (int b = t; b < nb; b += kTopThreads) hist[b] = 0;
  __syncthreads();
#pragma unroll
  for (int i = 0; i < kPer; ++i)
    if (match[i]) atomicAdd(&hist[(v[i] >> shift) & m], 1u);
  __syncthreads();
  const int per = nb / kTopThreads;
  uint32_t local = 0;
  for (int b = 0; b < per; ++b) local += hist[t * per + b];
  uint32_t incl = local;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const uint32_t up = __shfl_down_sync(0xffffffffu, incl, o);
    if (lane + o < 32) incl += up;
  }
  if (lane == 0) warp_tot[warp] = incl;
  __syncthreads();
  for (int w = warp + 1; w < kTopThreads / 32; ++w) incl += warp_tot[w];
  const uint32_t excl = incl - local;
  const uint32_t need = static_cast<uint32_t>(*remaining);
  if (excl < need && incl >= need) {
    uint32_t acc = excl;
    for (int b = per - 1; b >= 0; --b) {
      const uint32_t c = hist[t * per + b];
      if (acc + c >= need) {
        shared2[0] = static_cast<uint32_t>(t * per + b);
        shared2[1] = acc;
        shared2[2] = c;
        break;
      }
      acc += c;
    }
  }
  __syncthreads();
  const uint32_t bin = shared2[0], above = shared2[1], count = shared2[2];
  __syncthreads();
  *prefix |= bin << shift;
  *remaining -= static_cast<int>(above);
  return count;
}

template <int kPer, bool kIds>
__global__ void __launch_bounds__(kTopThreads) topk_kernel(float *out_v, int *out_i, int64_t out_ld, const float *x,
                                                           int64_t ld, const int *ids, int64_t ids_ld, int n, int seg,
                                                           int k, int id_offset) {
  __shared__ uint32_t hist[kTopBins];
  __shared__ uint32_t warp_tot[kTopThreads / 32];
  __shared__ uint32_t shared2[4];
  __shared__ unsigned long long sel[kTopMaxK];
  const int row = blockIdx.y, s = blockIdx.x, t = threadIdx.x;
  const int start = s * seg, end = min(n, start + seg);
  const float *xr = x + static_cast<int64_t>(row) * ld;
  const int *ir = kIds ? ids + static_cast<int64_t>(row) * ids_ld : nullptr;
  uint32_t key[kPer];
  int idv[kIds ? kPer : 1];
#pragma unroll
  for (int i = 0; i < kPer; ++i) {
    const int e = start + t + kTopThreads * i;
    key[i] = 0;
    if (e < end) {
      key[i] = f2key(__ldg(xr + e));
      if constexpr (kIds) idv[i] = __ldg(ir + e);
    }
  }
  auto ok_at = [&](int i) {
    const bool in = start + t + kTopThreads * i < end;
    if constexpr (kIds) return in && idv[i] >= 0;
    return in;
  };
  auto inv_at = [&](int i) {
    if constexpr (kIds) return ~static_cast<uint32_t>(idv[i]);
    return ~static_cast<uint32_t>(id_offset + start + t + kTopThreads * i);
  };
  int valid = max(0, end - start);
  if constexpr (kIds) {
    valid = 0;
#pragma unroll
    for (int i = 0; i < kPer; ++i) valid += ok_at(i);
    valid = static_cast<int>(block_sum(static_cast<float>(valid), reinterpret_cast<float *>(hist)));
  }
  const int kk = min(k, valid);
  for (int i = t; i < kTopMaxK; i += kTopThreads) sel[i] = 0ull;
  if (t == 0) shared2[3] = 0;
  __syncthreads();
  unsigned long long *tmax = reinterpret_cast<unsigned long long *>(hist);
  unsigned long long *cand = tmax + kTopThreads;
  bool fast = false;
  if (kk > 0) {
    uint32_t mx = 0;
#pragma unroll
    for (int i = 0; i < kPer; ++i)
      if (ok_at(i)) mx = max(mx, key[i]);
    const unsigned long long sorted =
        warp_sort_desc((static_cast<unsigned long long>(mx) << 32) | static_cast<uint32_t>(t));
    const int q = (kk + kTopThreads / 32 - 1) / (kTopThreads / 32) - 1;
    if ((t & 31) == q) tmax[t >> 5] = sorted;
    __syncthreads();
    uint32_t bound = 0xffffffffu;
#pragma unroll
    for (int w = 0; w < kTopThreads / 32; ++w) bound = min(bound, static_cast<uint32_t>(tmax[w] >> 32));
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
      if (ok_at(i) && key[i] >= bound) {
        const uint32_t pos = atomicAdd(&shared2[3], 1u);
        if (pos < kTopCand) cand[pos] = (static_cast<unsigned long long>(key[i]) << 32) | inv_at(i);
      }
    }
    __syncthreads();
    const int c = static_cast<int>(shared2[3]);
    fast = c <= kTopCand;
    if (c <= kTopRank) {
      for (int i = t; i < c; i += kTopThreads) {
        const unsigned long long v = cand[i];
        int r = 0;
        for (int j = 0; j < c; ++j) r += cand[j] > v;
        if (r < kk) sel[r] = v;
      }
    } else if (fast) {
      int p2 = 2;
      while (p2 < c) p2 <<= 1;
      for (int i = c + t; i < p2; i += kTopThreads) cand[i] = 0ull;
      __syncthreads();
      bitonic_desc(cand, p2);
      for (int i = t; i < kk; i += kTopThreads) sel[i] = cand[i];
    }
    __syncthreads();
    if (t == 0) shared2[3] = 0;
    __syncthreads();
  }
  if (kk > 0 && !fast) {
    uint32_t prefix = 0;
    int remaining = kk;
    bool ok[kPer], match[kPer];
    uint32_t inv[kPer];
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
      ok[i] = ok_at(i);
      inv[i] = inv_at(i);
      match[i] = ok[i];
    }
    radix_pass<kPer>(hist, warp_tot, shared2, key, match, 20, 12, &prefix, &remaining);
#pragma unroll
    for (int i = 0; i < kPer; ++i) match[i] = ok[i] && (key[i] >> 20) == (prefix >> 20);
    radix_pass<kPer>(hist, warp_tot, shared2, key, match, 8, 12, &prefix, &remaining);
#pragma unroll
    for (int i = 0; i < kPer; ++i) match[i] = ok[i] && (key[i] >> 8) == (prefix >> 8);
    const uint32_t equal = radix_pass<kPer>(hist, warp_tot, shared2, key, match, 0, 8, &prefix, &remaining);
    const uint32_t kth = prefix;
    const int need = remaining;
    const bool tie = static_cast<int>(equal) > need;
    uint32_t iprefix = 0;
    if (tie) {
      int rem2 = need;
#pragma unroll
      for (int i = 0; i < kPer; ++i) match[i] = ok[i] && key[i] == kth;
      radix_pass<kPer>(hist, warp_tot, shared2, inv, match, 20, 12, &iprefix, &rem2);
#pragma unroll
      for (int i = 0; i < kPer; ++i) match[i] = ok[i] && key[i] == kth && (inv[i] >> 20) == (iprefix >> 20);
      radix_pass<kPer>(hist, warp_tot, shared2, inv, match, 8, 12, &iprefix, &rem2);
#pragma unroll
      for (int i = 0; i < kPer; ++i) match[i] = ok[i] && key[i] == kth && (inv[i] >> 8) == (iprefix >> 8);
      radix_pass<kPer>(hist, warp_tot, shared2, inv, match, 0, 8, &iprefix, &rem2);
    }
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
      if (!ok[i]) continue;
      const bool take = key[i] > kth || (key[i] == kth && (!tie || inv[i] >= iprefix));
      if (take) {
        const uint32_t pos = atomicAdd(&shared2[3], 1u);
        if (pos < kTopMaxK) sel[pos] = (static_cast<unsigned long long>(key[i]) << 32) | inv[i];
      }
    }
  }
  __syncthreads();
  if (!fast) bitonic_desc(sel, kTopMaxK);
  if (t < k) {
    const int64_t o = static_cast<int64_t>(row) * out_ld + static_cast<int64_t>(s) * k + t;
    if (t < kk) {
      out_v[o] = key2f(static_cast<uint32_t>(sel[t] >> 32));
      out_i[o] = static_cast<int>(~static_cast<uint32_t>(sel[t]));
    } else {
      out_v[o] = -INFINITY;
      out_i[o] = -1;
    }
  }
}

int launch_topk(float *out_v, int *out_i, int64_t out_ld, const float *x, int64_t ld, const int *ids, int64_t ids_ld,
                int rows, int n, int segs, int k, int id_offset, cudaStream_t stream) {
  const int seg = (n + segs - 1) / segs;
  const dim3 grid(segs, rows);
#define SEVENTY_TOPK_LAUNCH(P, I)                                                                               \
  topk_kernel<P, I><<<grid, kTopThreads, 0, stream>>>(out_v, out_i, out_ld, x, ld, I ? ids : nullptr, ids_ld, n, seg, \
                                                      k, id_offset)
  const bool with_ids = ids != nullptr;
  if (seg <= kTopThreads) {
    if (with_ids) SEVENTY_TOPK_LAUNCH(1, true); else SEVENTY_TOPK_LAUNCH(1, false);
  } else if (seg <= 4 * kTopThreads) {
    if (with_ids) SEVENTY_TOPK_LAUNCH(4, true); else SEVENTY_TOPK_LAUNCH(4, false);
  } else if (seg <= 8 * kTopThreads) {
    if (with_ids) SEVENTY_TOPK_LAUNCH(8, true); else SEVENTY_TOPK_LAUNCH(8, false);
  } else {
    if (with_ids) SEVENTY_TOPK_LAUNCH(16, true); else SEVENTY_TOPK_LAUNCH(16, false);
  }
#undef SEVENTY_TOPK_LAUNCH
  return launch_status();
}

}  // namespace
}  // namespace seventy

SEVENTY_API int64_t seventy_topk_workspace(int rows, int n, int k) {
  using namespace seventy;
  if (rows <= 0 || n <= kTopSeg) return 0;
  const int64_t segs = (n + kTopSeg - 1) / kTopSeg;
  return static_cast<int64_t>(rows) * segs * k * 2 * static_cast<int64_t>(sizeof(float));
}

SEVENTY_API int seventy_topk(float *values, int *ids_out, const float *x, int64_t ld, const int *ids, int64_t ids_ld,
                             int rows, int n, int k, int id_offset, void *workspace, int64_t workspace_bytes,
                             cudaStream_t stream) {
  using namespace seventy;
  if (values == nullptr || ids_out == nullptr || x == nullptr || rows < 0 || n < 1 || k < 1 || k > kTopMaxK ||
      ld < n || (ids != nullptr && ids_ld < n))
    return SEVENTY_ERR_INVALID_ARGUMENT;
  if (rows == 0) return SEVENTY_OK;
  if (n <= kTopSeg) return launch_topk(values, ids_out, k, x, ld, ids, ids_ld, rows, n, 1, k, id_offset, stream);
  const int segs = (n + kTopSeg - 1) / kTopSeg;
  if (segs > kTopMaxSegs || segs * k > kTopSeg) return SEVENTY_ERR_INVALID_ARGUMENT;
  if (workspace == nullptr || workspace_bytes < seventy_topk_workspace(rows, n, k)) return SEVENTY_ERR_INVALID_ARGUMENT;
  auto *cv = static_cast<float *>(workspace);
  auto *ci = reinterpret_cast<int *>(cv + static_cast<int64_t>(rows) * segs * k);
  const int code = launch_topk(cv, ci, static_cast<int64_t>(segs) * k, x, ld, ids, ids_ld, rows, n, segs, k, id_offset,
                               stream);
  if (code != SEVENTY_OK) return code;
  return launch_topk(values, ids_out, k, cv, static_cast<int64_t>(segs) * k, ci, static_cast<int64_t>(segs) * k, rows,
                     segs * k, 1, k, 0, stream);
}
