#include "common.cuh"

#include <cstdlib>
#include <cstring>

/*
 * Tensor-parallel all-reduce for small batches over peer-to-peer GPU memory (NVLink), fused with the residual add
 * and RMS norm that follow it in the model.
 *
 * Every rank allocates one buffer and shares it with CUDA IPC. Buffer layout: kMaxBlocks x kMaxRanks uint32 signal
 * flags, then two data areas (alternating between calls) of world slots of slot_bytes each. Block b of a call:
 *   1. takes epoch = ++counter[b] (the same on every rank, because every rank makes the same calls in order),
 *   2. writes its rows of the local partial sums into slot `rank` of every rank's data area (epoch parity),
 *   3. sets signal[b][rank] = epoch in every rank's buffer with a system-scope release store (after the block's
 *      barrier, so it orders the whole block's pushes) and waits with system-scope acquire loads until its own
 *      signal[b][p] == epoch for every p (a full system fence on each side cost about 4 us per call),
 *   4. adds the world slots of its rows in FP32 (rounded once to FP16), adds the residual, and writes the residual
 *      and the RMS-normed row. One block of 1024 threads per row; each thread holds 16-byte slices of the row in
 *      registers from the push to the norm (rows of up to 16384 values).
 * Rows map to blocks the same way on every rank, so a block only waits for the same block of the other ranks.
 * Two data areas are enough: a rank can only start call e + 2 after every peer passed call e + 1's barrier, which
 * every peer reaches only after it has read call e's data.
 */

namespace seventy {
namespace {

constexpr int kMaxRanks = 8;
constexpr int kMaxBlocks = 64;
constexpr int kThreads = 1024;
constexpr int kVecs = 2;
constexpr int kOneShotRows = 8;
constexpr int64_t kSignalBytes = 2 * static_cast<int64_t>(kMaxBlocks) * kMaxRanks * sizeof(uint32_t);

struct ArContext {
  char *buffers[kMaxRanks];
  uint32_t *counters;
  uint32_t *push_epochs;
  int64_t slot_bytes;
  int rank;
  int world;
};

/*
 * Push protocol (ar_norm_push_kernel). The data is its own flag: an empty slot holds the FP16 NaN 0x7f7f in every
 * half, a sender writes its 16-byte packs straight into the receiver's slot, and the receiver spins on its own
 * slots until no half is 0x7f7f, then writes 0x7f7f back. A payload half equal to 0x7f7f is sent as the NaN 0x7e00.
 * There is no separate flag store and no fence, so a call costs one NVLink write latency per phase. This is the
 * protocol of 1Cat-vLLM's sm70_cross_device_reduce_1stage_push (Apache-2.0), as the lab kernel for the fused
 * all-reduce and Gemma RMS norm used it. FP32 data (ar_sum_push_kernel) uses whole 32-bit words: empty is
 * 0x7f7f7f7f and a payload word equal to it is sent as 0x7f7f7f7e. The push slots have their own region of the
 * buffer (after the flag-protocol data areas), filled with 0x7f by seventy_ar_alloc, and every row block has its own
 * epoch bit (push_epochs, local), so the two halves of the region alternate per row from call to call. Row b of a
 * slot always takes the bytes [b * dim * 4, (b + 1) * dim * 4), for FP16 and FP32 data alike, so no call can write
 * into another row's bytes.
 */
__host__ __device__ inline int64_t push_slot_bytes(int64_t slot_bytes) { return slot_bytes; }

__host__ __device__ inline int64_t push_p1_offset(int world, int64_t slot_bytes) {
  return kSignalBytes + 2 * static_cast<int64_t>(world) * slot_bytes;
}

__host__ __device__ inline int64_t push_p2_offset(int world, int64_t slot_bytes) {
  return push_p1_offset(world, slot_bytes) + 2 * static_cast<int64_t>(world) * push_slot_bytes(slot_bytes);
}

__host__ __device__ inline int64_t push_end(int world, int64_t slot_bytes) {
  return push_p2_offset(world, slot_bytes) + 4 * push_slot_bytes(slot_bytes);
}

constexpr uint32_t kEmpty = 0x7f7f7f7fu;

__device__ __forceinline__ void st_vol(void *p, const uint4 &v) {
  asm volatile("st.volatile.global.v4.b32 [%4], {%0, %1, %2, %3};" ::"r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w),
               "l"(p));
}

__device__ __forceinline__ uint4 ld_vol(const void *p) {
  uint4 v;
  asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}

__device__ __forceinline__ bool is_empty(const uint4 &v) {
  const uint32_t w[4] = {v.x, v.y, v.z, v.w};
  bool e = false;
#pragma unroll
  for (int i = 0; i < 4; ++i) e |= ((w[i] & 0xffffu) == 0x7f7fu) | ((w[i] >> 16) == 0x7f7fu);
  return e;
}

__device__ __forceinline__ uint4 escape(uint4 v) {
  uint32_t *w = reinterpret_cast<uint32_t *>(&v);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    if ((w[i] & 0xffffu) == 0x7f7fu) w[i] = (w[i] & 0xffff0000u) | 0x7e00u;
    if ((w[i] >> 16) == 0x7f7fu) w[i] = (w[i] & 0x0000ffffu) | 0x7e000000u;
  }
  return v;
}

__device__ __forceinline__ uint4 wait_full(const void *p) {
  uint4 v;
  do {
    v = ld_vol(p);
  } while (is_empty(v));
  return v;
}

__device__ __forceinline__ bool is_empty32(const uint4 &v) {
  return v.x == kEmpty || v.y == kEmpty || v.z == kEmpty || v.w == kEmpty;
}

__device__ __forceinline__ uint4 escape32(uint4 v) {
  v.x = v.x == kEmpty ? kEmpty - 1 : v.x;
  v.y = v.y == kEmpty ? kEmpty - 1 : v.y;
  v.z = v.z == kEmpty ? kEmpty - 1 : v.z;
  v.w = v.w == kEmpty ? kEmpty - 1 : v.w;
  return v;
}

__device__ __forceinline__ uint4 wait_full32(const void *p) {
  uint4 v;
  do {
    v = ld_vol(p);
  } while (is_empty32(v));
  return v;
}

__device__ __forceinline__ void store_release(uint32_t *p, uint32_t v) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

__device__ __forceinline__ uint32_t load_acquire(const uint32_t *p) {
  uint32_t v;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ void unpack8(uint4 v, float (&f)[8]) {
  const half2 *h = reinterpret_cast<const half2 *>(&v);
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const float2 t = __half22float2(h[k]);
    f[2 * k] = t.x;
    f[2 * k + 1] = t.y;
  }
}

__device__ __forceinline__ uint4 pack8(const float (&f)[8]) {
  uint4 out;
  half2 *h = reinterpret_cast<half2 *>(&out);
#pragma unroll
  for (int k = 0; k < 4; ++k) h[k] = __floats2half2_rn(f[2 * k], f[2 * k + 1]);
  return out;
}

__global__ void __launch_bounds__(kThreads) ar_norm_kernel(ArContext ctx, half *out, half *residual, const half *x,
                                                           const float *weight, int rows, int dim, float eps) {
  __shared__ float scratch[32];
  __shared__ uint32_t epoch_s;
  const int b = blockIdx.x;
  if (threadIdx.x == 0) {
    epoch_s = ctx.counters[b] + 1;
    ctx.counters[b] = epoch_s;
  }
  __syncthreads();
  const uint32_t epoch = epoch_s;
  const int64_t signal_bytes = kSignalBytes;
  const int64_t area = static_cast<int64_t>(ctx.world) * ctx.slot_bytes;
  const int64_t data_off = signal_bytes + (epoch & 1) * area;
  const int vec_per_row = dim / 8;
  const int64_t slot_vecs = ctx.slot_bytes / static_cast<int64_t>(sizeof(uint4));

  for (int row = b; row < rows; row += gridDim.x) {
    const uint4 *src = reinterpret_cast<const uint4 *>(x) + static_cast<int64_t>(row) * vec_per_row;
#pragma unroll
    for (int j = 0; j < kVecs; ++j) {
      const int i = threadIdx.x + j * kThreads;
      if (i >= vec_per_row) break;
      const uint4 v = src[i];
      for (int p = 0; p < ctx.world; ++p) {
        uint4 *dst = reinterpret_cast<uint4 *>(ctx.buffers[p] + data_off) + ctx.rank * slot_vecs +
                     static_cast<int64_t>(row) * vec_per_row;
        dst[i] = v;
      }
    }
  }
  __syncthreads();
  if (threadIdx.x < ctx.world) {
    uint32_t *flag = reinterpret_cast<uint32_t *>(ctx.buffers[threadIdx.x]) + b * kMaxRanks + ctx.rank;
    store_release(flag, epoch);
    const uint32_t *mine = reinterpret_cast<const uint32_t *>(ctx.buffers[ctx.rank]) + b * kMaxRanks + threadIdx.x;
    while (load_acquire(mine) != epoch) {
    }
  }
  __syncthreads();

  const uint4 *local = reinterpret_cast<const uint4 *>(ctx.buffers[ctx.rank] + data_off);
  for (int row = b; row < rows; row += gridDim.x) {
    uint4 *res = residual != nullptr ? reinterpret_cast<uint4 *>(residual) + static_cast<int64_t>(row) * vec_per_row
                                     : nullptr;
    uint4 *dst = reinterpret_cast<uint4 *>(out) + static_cast<int64_t>(row) * vec_per_row;
    float r[kVecs][8];
    float sumsq = 0.0f;
#pragma unroll
    for (int j = 0; j < kVecs; ++j) {
      const int i = threadIdx.x + j * kThreads;
      if (i >= vec_per_row) break;
      float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
      for (int p = 0; p < ctx.world; ++p) {
        float f[8];
        unpack8(local[p * slot_vecs + static_cast<int64_t>(row) * vec_per_row + i], f);
#pragma unroll
        for (int k = 0; k < 8; ++k) acc[k] += f[k];
      }
      uint4 sum16 = pack8(acc);
      unpack8(sum16, r[j]);
      if (res != nullptr) {
        float old[8];
        unpack8(res[i], old);
#pragma unroll
        for (int k = 0; k < 8; ++k) r[j][k] += old[k];
        sum16 = pack8(r[j]);
        unpack8(sum16, r[j]);
        if (weight != nullptr) res[i] = sum16;
      }
      if (weight == nullptr) {
        dst[i] = sum16;
      } else {
#pragma unroll
        for (int k = 0; k < 8; ++k) sumsq += r[j][k] * r[j][k];
      }
    }
    if (weight == nullptr) continue;
    const float total = block_sum(sumsq, scratch);
    const float inv = 1.0f / sqrtf(total / dim + eps);
    const float4 *w4 = reinterpret_cast<const float4 *>(weight);
#pragma unroll
    for (int j = 0; j < kVecs; ++j) {
      const int i = threadIdx.x + j * kThreads;
      if (i >= vec_per_row) break;
      const float4 wa = w4[2 * i], wb = w4[2 * i + 1];
      const float w[8] = {wa.x, wa.y, wa.z, wa.w, wb.x, wb.y, wb.z, wb.w};
      float o[8];
#pragma unroll
      for (int k = 0; k < 8; ++k) o[k] = r[j][k] * inv * w[k];
      dst[i] = pack8(o);
    }
    __syncthreads();
  }
}

/*
 * Two-shot variant (ar_norm_2shot_kernel), one block per row b. Rows are split into world contiguous shards; the
 * home rank q of row b is b / ceil(rows / world).
 *   1. Every rank writes its partial row b into slot `rank` of the home rank's data area, then sets its
 *      signal[b][rank] = epoch (release); the home rank waits for all of them (acquire).
 *   2. The home rank adds the world slots in FP32 (rounded once to FP16), adds its residual row, writes the residual
 *      and the RMS-normed row, and writes the normed row (and, with gather_residual, the residual row) into slot q
 *      (and slot (q + 1) % world) of every other rank's data area, then sets their signal2[b][q] = epoch.
 *   3. Every other rank waits for signal2[b][q] and copies the row(s) to its out (and residual).
 * Half the NVLink traffic of ar_norm_kernel, but the residual is only correct on the home rank of each row unless
 * gather_residual is set: the model keeps the residual stream sharded between these calls (every call of a forward
 * takes the same path) and gathers it where it reads it. Slot reuse within a call: on another rank, row b of
 * slots q and (q + 1) % world is not written in step 1 (step 1 only writes to home ranks). Across calls the two data
 * areas alternate as in ar_norm_kernel; every barrier of a call depends on every rank having finished step 1 of
 * that call, so call e + 2 cannot write an area that call e still reads.
 */
__global__ void __launch_bounds__(kThreads) ar_norm_2shot_kernel(ArContext ctx, half *out, half *residual,
                                                                 const half *x, const float *weight, int rows,
                                                                 int dim, float eps, int gather_residual) {
  __shared__ float scratch[32];
  __shared__ uint32_t epoch_s;
  const int b = blockIdx.x;
  if (threadIdx.x == 0) {
    epoch_s = ctx.counters[b] + 1;
    ctx.counters[b] = epoch_s;
  }
  __syncthreads();
  const uint32_t epoch = epoch_s;
  const int64_t area = static_cast<int64_t>(ctx.world) * ctx.slot_bytes;
  const int64_t data_off = kSignalBytes + (epoch & 1) * area;
  const int vec_per_row = dim / 8;
  const int64_t slot_vecs = ctx.slot_bytes / static_cast<int64_t>(sizeof(uint4));
  const int per = (rows + ctx.world - 1) / ctx.world;
  const int home = b / per;
  const bool mine = home == ctx.rank;
  const int64_t row_off = static_cast<int64_t>(b) * vec_per_row;

  {
    const uint4 *src = reinterpret_cast<const uint4 *>(x) + row_off;
    uint4 *dst = reinterpret_cast<uint4 *>(ctx.buffers[home] + data_off) + ctx.rank * slot_vecs + row_off;
#pragma unroll
    for (int j = 0; j < kVecs; ++j) {
      const int i = threadIdx.x + j * kThreads;
      if (i < vec_per_row) dst[i] = src[i];
    }
  }
  __syncthreads();
  if (threadIdx.x == 0)
    store_release(reinterpret_cast<uint32_t *>(ctx.buffers[home]) + b * kMaxRanks + ctx.rank, epoch);
  if (mine && threadIdx.x < ctx.world) {
    const uint32_t *flag = reinterpret_cast<const uint32_t *>(ctx.buffers[ctx.rank]) + b * kMaxRanks + threadIdx.x;
    while (load_acquire(flag) != epoch) {
    }
  }
  __syncthreads();

  const int64_t signal2 = static_cast<int64_t>(kMaxBlocks) * kMaxRanks;
  uint4 *res = residual != nullptr ? reinterpret_cast<uint4 *>(residual) + row_off : nullptr;
  uint4 *dst = reinterpret_cast<uint4 *>(out) + row_off;
  if (mine) {
    const uint4 *local = reinterpret_cast<const uint4 *>(ctx.buffers[ctx.rank] + data_off) + row_off;
    float r[kVecs][8];
    uint4 rnew[kVecs];
    float sumsq = 0.0f;
#pragma unroll
    for (int j = 0; j < kVecs; ++j) {
      const int i = threadIdx.x + j * kThreads;
      if (i >= vec_per_row) break;
      float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
      for (int p = 0; p < ctx.world; ++p) {
        float f[8];
        unpack8(local[p * slot_vecs + i], f);
#pragma unroll
        for (int k = 0; k < 8; ++k) acc[k] += f[k];
      }
      uint4 sum16 = pack8(acc);
      unpack8(sum16, r[j]);
      if (res != nullptr) {
        float old[8];
        unpack8(res[i], old);
#pragma unroll
        for (int k = 0; k < 8; ++k) r[j][k] += old[k];
        sum16 = pack8(r[j]);
        unpack8(sum16, r[j]);
      }
      rnew[j] = sum16;
#pragma unroll
      for (int k = 0; k < 8; ++k) sumsq += r[j][k] * r[j][k];
    }
    const float total = block_sum(sumsq, scratch);
    const float inv = 1.0f / sqrtf(total / dim + eps);
    const float4 *w4 = reinterpret_cast<const float4 *>(weight);
#pragma unroll
    for (int j = 0; j < kVecs; ++j) {
      const int i = threadIdx.x + j * kThreads;
      if (i >= vec_per_row) break;
      const float4 wa = w4[2 * i], wb = w4[2 * i + 1];
      const float w[8] = {wa.x, wa.y, wa.z, wa.w, wb.x, wb.y, wb.z, wb.w};
      float o[8];
#pragma unroll
      for (int k = 0; k < 8; ++k) o[k] = r[j][k] * inv * w[k];
      const uint4 o16 = pack8(o);
      dst[i] = o16;
      if (res != nullptr) res[i] = rnew[j];
      for (int p = 0; p < ctx.world; ++p) {
        if (p == ctx.rank) continue;
        uint4 *g = reinterpret_cast<uint4 *>(ctx.buffers[p] + data_off) + row_off;
        g[home * slot_vecs + i] = o16;
        if (gather_residual) g[((home + 1) % ctx.world) * slot_vecs + i] = rnew[j];
      }
    }
    __syncthreads();
    if (threadIdx.x < ctx.world && threadIdx.x != ctx.rank)
      store_release(reinterpret_cast<uint32_t *>(ctx.buffers[threadIdx.x]) + signal2 + b * kMaxRanks + ctx.rank,
                    epoch);
    return;
  }
  if (threadIdx.x == 0) {
    const uint32_t *flag = reinterpret_cast<const uint32_t *>(ctx.buffers[ctx.rank]) + signal2 + b * kMaxRanks + home;
    while (load_acquire(flag) != epoch) {
    }
  }
  __syncthreads();
  const uint4 *g = reinterpret_cast<const uint4 *>(ctx.buffers[ctx.rank] + data_off) + row_off;
#pragma unroll
  for (int j = 0; j < kVecs; ++j) {
    const int i = threadIdx.x + j * kThreads;
    if (i >= vec_per_row) break;
    dst[i] = g[home * slot_vecs + i];
    if (gather_residual && res != nullptr) res[i] = g[((home + 1) % ctx.world) * slot_vecs + i];
  }
}

/*
 * ar_norm_push_kernel: the two-shot all-reduce and norm of ar_norm_2shot_kernel (same rows per home rank, same
 * rounding, same sharded residual) on the push protocol above. One block per row b; thread i handles 16-byte packs
 * i and i + 1024 of the row.
 *   1. Every rank writes its partial row b into slot `rank` of the home rank's phase-1 area.
 *   2. The home rank waits for its world slots, adds them in FP32 in rank order (rounded once to FP16), adds its
 *      residual row, writes the residual and the RMS-normed row, empties its slots, and writes the normed row (and,
 *      with gather_residual, the residual row) into every other rank's phase-2 area.
 *   3. Every other rank waits for the row(s) in its phase-2 area, copies them and empties the slots.
 */
__global__ void __launch_bounds__(kThreads) ar_norm_push_kernel(ArContext ctx, half *out, half *residual,
                                                                const half *x, const float *weight, int rows,
                                                                int dim, float eps, int gather_residual) {
  __shared__ float scratch[32];
  const int b = blockIdx.x;
  const uint32_t parity = ctx.push_epochs[b] & 1u;
  const int64_t ps = push_slot_bytes(ctx.slot_bytes);
  const int64_t p1 = push_p1_offset(ctx.world, ctx.slot_bytes) + parity * ctx.world * ps;
  const int64_t p2 = push_p2_offset(ctx.world, ctx.slot_bytes) + parity * 2 * ps;
  const int vec_per_row = dim / 8;
  const int64_t slot_vecs = ps / static_cast<int64_t>(sizeof(uint4));
  const int per = (rows + ctx.world - 1) / ctx.world;
  const int home = b / per;
  const bool mine = home == ctx.rank;
  const int64_t row_off = static_cast<int64_t>(b) * vec_per_row;
  const int64_t slot_off = static_cast<int64_t>(b) * (dim / 4);
  const uint4 empty = make_uint4(kEmpty, kEmpty, kEmpty, kEmpty);

  {
    const uint4 *src = reinterpret_cast<const uint4 *>(x) + row_off;
    uint4 *dst = reinterpret_cast<uint4 *>(ctx.buffers[home] + p1) + ctx.rank * slot_vecs + slot_off;
#pragma unroll
    for (int j = 0; j < kVecs; ++j) {
      const int i = threadIdx.x + j * kThreads;
      if (i < vec_per_row) st_vol(dst + i, escape(src[i]));
    }
  }
  uint4 *res = residual != nullptr ? reinterpret_cast<uint4 *>(residual) + row_off : nullptr;
  uint4 *dst = reinterpret_cast<uint4 *>(out) + row_off;
  if (mine) {
    uint4 *local = reinterpret_cast<uint4 *>(ctx.buffers[ctx.rank] + p1) + slot_off;
    float r[kVecs][8];
    uint4 rnew[kVecs];
    float sumsq = 0.0f;
#pragma unroll
    for (int j = 0; j < kVecs; ++j) {
      const int i = threadIdx.x + j * kThreads;
      if (i >= vec_per_row) break;
      float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
      for (int p = 0; p < ctx.world; ++p) {
        float f[8];
        unpack8(wait_full(local + p * slot_vecs + i), f);
#pragma unroll
        for (int k = 0; k < 8; ++k) acc[k] += f[k];
      }
      for (int p = 0; p < ctx.world; ++p) st_vol(local + p * slot_vecs + i, empty);
      uint4 sum16 = pack8(acc);
      unpack8(sum16, r[j]);
      if (res != nullptr) {
        float old[8];
        unpack8(res[i], old);
#pragma unroll
        for (int k = 0; k < 8; ++k) r[j][k] += old[k];
        sum16 = pack8(r[j]);
        unpack8(sum16, r[j]);
      }
      rnew[j] = sum16;
#pragma unroll
      for (int k = 0; k < 8; ++k) sumsq += r[j][k] * r[j][k];
    }
    const float total = block_sum(sumsq, scratch);
    const float inv = 1.0f / sqrtf(total / dim + eps);
    const float4 *w4 = reinterpret_cast<const float4 *>(weight);
#pragma unroll
    for (int j = 0; j < kVecs; ++j) {
      const int i = threadIdx.x + j * kThreads;
      if (i >= vec_per_row) break;
      const float4 wa = w4[2 * i], wb = w4[2 * i + 1];
      const float w[8] = {wa.x, wa.y, wa.z, wa.w, wb.x, wb.y, wb.z, wb.w};
      float o[8];
#pragma unroll
      for (int k = 0; k < 8; ++k) o[k] = r[j][k] * inv * w[k];
      const uint4 o16 = pack8(o);
      dst[i] = o16;
      if (res != nullptr) res[i] = rnew[j];
      const uint4 o_esc = escape(o16), r_esc = escape(rnew[j]);
      for (int p = 0; p < ctx.world; ++p) {
        if (p == ctx.rank) continue;
        uint4 *g = reinterpret_cast<uint4 *>(ctx.buffers[p] + p2) + slot_off;
        st_vol(g + i, o_esc);
        if (gather_residual) st_vol(g + slot_vecs + i, r_esc);
      }
    }
  } else {
    uint4 *g = reinterpret_cast<uint4 *>(ctx.buffers[ctx.rank] + p2) + slot_off;
#pragma unroll
    for (int j = 0; j < kVecs; ++j) {
      const int i = threadIdx.x + j * kThreads;
      if (i >= vec_per_row) break;
      dst[i] = wait_full(g + i);
      st_vol(g + i, empty);
      if (gather_residual) {
        const uint4 rv = wait_full(g + slot_vecs + i);
        st_vol(g + slot_vecs + i, empty);
        if (res != nullptr) res[i] = rv;
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) ctx.push_epochs[b] += 1;
}

/*
 * ar_norm_push_all_kernel: the two-shot all-reduce and norm with the structure of the lab kernel for the fused
 * all-reduce and norm (the critical path holds only the sum): the home rank of row b waits for its world slots,
 * adds them in FP32 in rank order, rounds once to FP16 and writes the sum into every other rank's phase-2 area at
 * once; then every rank adds the sum to its residual row, takes the RMS norm and writes both. Same rounding as
 * ar_norm_push_kernel (bitwise the same output), with the residual complete on every rank. Without weight, out is
 * the sum (plus the residual, which is then not written). Packs i and i + blockDim.x of the row per thread.
 */
__global__ void __launch_bounds__(kThreads) ar_norm_push_all_kernel(ArContext ctx, half *out, half *residual,
                                                                    const half *x, const float *weight, int rows,
                                                                    int dim, float eps) {
  __shared__ float scratch[32];
  const int b = blockIdx.x;
  const int nt = blockDim.x;
  const uint32_t parity = ctx.push_epochs[b] & 1u;
  const int64_t ps = push_slot_bytes(ctx.slot_bytes);
  const int64_t p1 = push_p1_offset(ctx.world, ctx.slot_bytes) + parity * ctx.world * ps;
  const int64_t p2 = push_p2_offset(ctx.world, ctx.slot_bytes) + parity * 2 * ps;
  const int vec_per_row = dim / 8;
  const int64_t slot_vecs = ps / static_cast<int64_t>(sizeof(uint4));
  const int per = (rows + ctx.world - 1) / ctx.world;
  const int home = b / per;
  const int64_t row_off = static_cast<int64_t>(b) * vec_per_row;
  const int64_t slot_off = static_cast<int64_t>(b) * (dim / 4);
  const uint4 empty = make_uint4(kEmpty, kEmpty, kEmpty, kEmpty);

  {
    const uint4 *src = reinterpret_cast<const uint4 *>(x) + row_off;
    uint4 *dst = reinterpret_cast<uint4 *>(ctx.buffers[home] + p1) + ctx.rank * slot_vecs + slot_off;
#pragma unroll
    for (int j = 0; j < kVecs; ++j) {
      const int i = threadIdx.x + j * nt;
      if (i < vec_per_row) st_vol(dst + i, escape(src[i]));
    }
  }
  float r[kVecs][8];
  if (home == ctx.rank) {
    uint4 *local = reinterpret_cast<uint4 *>(ctx.buffers[ctx.rank] + p1) + slot_off;
#pragma unroll
    for (int j = 0; j < kVecs; ++j) {
      const int i = threadIdx.x + j * nt;
      if (i >= vec_per_row) break;
      float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
      for (int p = 0; p < ctx.world; ++p) {
        float f[8];
        unpack8(wait_full(local + p * slot_vecs + i), f);
#pragma unroll
        for (int k = 0; k < 8; ++k) acc[k] += f[k];
      }
      const uint4 sum16 = pack8(acc);
      const uint4 s_esc = escape(sum16);
      for (int p = 0; p < ctx.world; ++p) {
        if (p == ctx.rank) continue;
        st_vol(reinterpret_cast<uint4 *>(ctx.buffers[p] + p2) + slot_off + i, s_esc);
      }
      for (int p = 0; p < ctx.world; ++p) st_vol(local + p * slot_vecs + i, empty);
      unpack8(sum16, r[j]);
    }
  } else {
    uint4 *g = reinterpret_cast<uint4 *>(ctx.buffers[ctx.rank] + p2) + slot_off;
#pragma unroll
    for (int j = 0; j < kVecs; ++j) {
      const int i = threadIdx.x + j * nt;
      if (i >= vec_per_row) break;
      unpack8(wait_full(g + i), r[j]);
      st_vol(g + i, empty);
    }
  }
  uint4 *res = residual != nullptr ? reinterpret_cast<uint4 *>(residual) + row_off : nullptr;
  uint4 *dst = reinterpret_cast<uint4 *>(out) + row_off;
  uint4 rnew[kVecs];
  float sumsq = 0.0f;
#pragma unroll
  for (int j = 0; j < kVecs; ++j) {
    const int i = threadIdx.x + j * nt;
    if (i >= vec_per_row) break;
    if (res != nullptr) {
      float old[8];
      unpack8(res[i], old);
#pragma unroll
      for (int k = 0; k < 8; ++k) r[j][k] += old[k];
    }
    rnew[j] = pack8(r[j]);
    unpack8(rnew[j], r[j]);
#pragma unroll
    for (int k = 0; k < 8; ++k) sumsq += r[j][k] * r[j][k];
  }
  if (weight == nullptr) {
#pragma unroll
    for (int j = 0; j < kVecs; ++j) {
      const int i = threadIdx.x + j * nt;
      if (i < vec_per_row) dst[i] = rnew[j];
    }
    __syncthreads();
    if (threadIdx.x == 0) ctx.push_epochs[b] += 1;
    return;
  }
  const float total = block_sum(sumsq, scratch);
  const float inv = 1.0f / sqrtf(total / dim + eps);
  const float4 *w4 = reinterpret_cast<const float4 *>(weight);
#pragma unroll
  for (int j = 0; j < kVecs; ++j) {
    const int i = threadIdx.x + j * nt;
    if (i >= vec_per_row) break;
    const float4 wa = w4[2 * i], wb = w4[2 * i + 1];
    const float w[8] = {wa.x, wa.y, wa.z, wa.w, wb.x, wb.y, wb.z, wb.w};
    float o[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) o[k] = r[j][k] * inv * w[k];
    dst[i] = pack8(o);
    if (res != nullptr) res[i] = rnew[j];
  }
  __syncthreads();
  if (threadIdx.x == 0) ctx.push_epochs[b] += 1;
}

/*
 * ar_norm_push_one_kernel: the one-shot form of ar_norm_push_all_kernel for up to kOneShotRows rows (V100, 4 GPUs,
 * 2 NVLinks per pair: 0.5 to 0.9 us faster per call from 1 to 8 rows, slower from 16): every rank writes its row b
 * into slot `rank` of every rank's phase-1 area at once, waits for all world slots of its own area, adds them in
 * FP32 in rank order, rounds once to FP16 and goes on as ar_norm_push_all_kernel. The sum is the same as the home
 * rank's there (bitwise the same output), with one NVLink write per call instead of two; each rank clears its own
 * slots, and the slots alternate per row with the epoch bit as in the two-shot forms, so the forms can follow each
 * other from call to call.
 */
__global__ void __launch_bounds__(kThreads) ar_norm_push_one_kernel(ArContext ctx, half *out, half *residual,
                                                                    const half *x, const float *weight, int rows,
                                                                    int dim, float eps) {
  __shared__ float scratch[32];
  const int b = blockIdx.x;
  const int nt = blockDim.x;
  const uint32_t parity = ctx.push_epochs[b] & 1u;
  const int64_t ps = push_slot_bytes(ctx.slot_bytes);
  const int64_t p1 = push_p1_offset(ctx.world, ctx.slot_bytes) + parity * ctx.world * ps;
  const int vec_per_row = dim / 8;
  const int64_t slot_vecs = ps / static_cast<int64_t>(sizeof(uint4));
  const int64_t row_off = static_cast<int64_t>(b) * vec_per_row;
  const int64_t slot_off = static_cast<int64_t>(b) * (dim / 4);
  const uint4 empty = make_uint4(kEmpty, kEmpty, kEmpty, kEmpty);
  {
    const uint4 *src = reinterpret_cast<const uint4 *>(x) + row_off;
#pragma unroll
    for (int j = 0; j < kVecs; ++j) {
      const int i = threadIdx.x + j * nt;
      if (i >= vec_per_row) break;
      const uint4 v = escape(src[i]);
      for (int p = 0; p < ctx.world; ++p)
        st_vol(reinterpret_cast<uint4 *>(ctx.buffers[p] + p1) + ctx.rank * slot_vecs + slot_off + i, v);
    }
  }
  float r[kVecs][8];
  uint4 *local = reinterpret_cast<uint4 *>(ctx.buffers[ctx.rank] + p1) + slot_off;
#pragma unroll
  for (int j = 0; j < kVecs; ++j) {
    const int i = threadIdx.x + j * nt;
    if (i >= vec_per_row) break;
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int p = 0; p < ctx.world; ++p) {
      float f[8];
      unpack8(wait_full(local + p * slot_vecs + i), f);
#pragma unroll
      for (int k = 0; k < 8; ++k) acc[k] += f[k];
    }
    for (int p = 0; p < ctx.world; ++p) st_vol(local + p * slot_vecs + i, empty);
    unpack8(pack8(acc), r[j]);
  }
  uint4 *res = residual != nullptr ? reinterpret_cast<uint4 *>(residual) + row_off : nullptr;
  uint4 *dst = reinterpret_cast<uint4 *>(out) + row_off;
  uint4 rnew[kVecs];
  float sumsq = 0.0f;
#pragma unroll
  for (int j = 0; j < kVecs; ++j) {
    const int i = threadIdx.x + j * nt;
    if (i >= vec_per_row) break;
    if (res != nullptr) {
      float old[8];
      unpack8(res[i], old);
#pragma unroll
      for (int k = 0; k < 8; ++k) r[j][k] += old[k];
    }
    rnew[j] = pack8(r[j]);
    unpack8(rnew[j], r[j]);
#pragma unroll
    for (int k = 0; k < 8; ++k) sumsq += r[j][k] * r[j][k];
  }
  if (weight == nullptr) {
#pragma unroll
    for (int j = 0; j < kVecs; ++j) {
      const int i = threadIdx.x + j * nt;
      if (i < vec_per_row) dst[i] = rnew[j];
    }
    __syncthreads();
    if (threadIdx.x == 0) ctx.push_epochs[b] += 1;
    return;
  }
  const float total = block_sum(sumsq, scratch);
  const float inv = 1.0f / sqrtf(total / dim + eps);
  const float4 *w4 = reinterpret_cast<const float4 *>(weight);
#pragma unroll
  for (int j = 0; j < kVecs; ++j) {
    const int i = threadIdx.x + j * nt;
    if (i >= vec_per_row) break;
    const float4 wa = w4[2 * i], wb = w4[2 * i + 1];
    const float w[8] = {wa.x, wa.y, wa.z, wa.w, wb.x, wb.y, wb.z, wb.w};
    float o[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) o[k] = r[j][k] * inv * w[k];
    dst[i] = pack8(o);
    if (res != nullptr) res[i] = rnew[j];
  }
  __syncthreads();
  if (threadIdx.x == 0) ctx.push_epochs[b] += 1;
}

/*
 * ar_sum_push_kernel: two-shot FP32 all-reduce of [rows, dim] on the push protocol (FP32 words), one block per row
 * b, same home ranks as ar_norm_push_kernel. The home rank adds the world partial rows in rank order, as
 * ar_sum_f32_kernel does, so the result is bitwise the same, and sends the sum to every other rank.
 */
__global__ void __launch_bounds__(kThreads) ar_sum_push_kernel(ArContext ctx, float *out, const float *x, int rows,
                                                               int dim) {
  const int b = blockIdx.x;
  const uint32_t parity = ctx.push_epochs[b] & 1u;
  const int64_t ps = push_slot_bytes(ctx.slot_bytes);
  const int64_t p1 = push_p1_offset(ctx.world, ctx.slot_bytes) + parity * ctx.world * ps;
  const int64_t p2 = push_p2_offset(ctx.world, ctx.slot_bytes) + parity * 2 * ps;
  const int vecs = dim / 4;
  const int64_t slot_vecs = ps / static_cast<int64_t>(sizeof(uint4));
  const int per = (rows + ctx.world - 1) / ctx.world;
  const int home = b / per;
  const int64_t off = static_cast<int64_t>(b) * vecs;
  const uint4 empty = make_uint4(kEmpty, kEmpty, kEmpty, kEmpty);
  const uint4 *src = reinterpret_cast<const uint4 *>(x) + off;
  uint4 *dst = reinterpret_cast<uint4 *>(out) + off;
  {
    uint4 *slot = reinterpret_cast<uint4 *>(ctx.buffers[home] + p1) + ctx.rank * slot_vecs + off;
    for (int i = threadIdx.x; i < vecs; i += kThreads) st_vol(slot + i, escape32(src[i]));
  }
  if (home == ctx.rank) {
    uint4 *local = reinterpret_cast<uint4 *>(ctx.buffers[ctx.rank] + p1) + off;
    for (int i = threadIdx.x; i < vecs; i += kThreads) {
      float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      for (int p = 0; p < ctx.world; ++p) {
        const uint4 v = wait_full32(local + p * slot_vecs + i);
        acc.x += __uint_as_float(v.x);
        acc.y += __uint_as_float(v.y);
        acc.z += __uint_as_float(v.z);
        acc.w += __uint_as_float(v.w);
      }
      for (int p = 0; p < ctx.world; ++p) st_vol(local + p * slot_vecs + i, empty);
      const uint4 sum = make_uint4(__float_as_uint(acc.x), __float_as_uint(acc.y), __float_as_uint(acc.z),
                                   __float_as_uint(acc.w));
      dst[i] = sum;
      const uint4 sum_esc = escape32(sum);
      for (int p = 0; p < ctx.world; ++p)
        if (p != ctx.rank) st_vol(reinterpret_cast<uint4 *>(ctx.buffers[p] + p2) + off + i, sum_esc);
    }
  } else {
    uint4 *g = reinterpret_cast<uint4 *>(ctx.buffers[ctx.rank] + p2) + off;
    for (int i = threadIdx.x; i < vecs; i += kThreads) {
      dst[i] = wait_full32(g + i);
      st_vol(g + i, empty);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) ctx.push_epochs[b] += 1;
}

/*
 * Plain FP32 all-reduce of [rows, dim] (the drafter's partial sums), with the same buffers, epochs and barrier as
 * ar_norm_kernel. Each rank's FP32 slice fits the slot because the slot is sized for max_rows FP32 rows.
 */
__global__ void __launch_bounds__(kThreads) ar_sum_f32_kernel(ArContext ctx, float *out, const float *x, int rows,
                                                              int dim) {
  __shared__ uint32_t epoch_s;
  const int b = blockIdx.x;
  if (threadIdx.x == 0) {
    epoch_s = ctx.counters[b] + 1;
    ctx.counters[b] = epoch_s;
  }
  __syncthreads();
  const uint32_t epoch = epoch_s;
  const int64_t signal_bytes = kSignalBytes;
  const int64_t area = static_cast<int64_t>(ctx.world) * ctx.slot_bytes;
  const int64_t data_off = signal_bytes + (epoch & 1) * area;
  const int vec_per_row = dim / 4;
  const int64_t slot_vecs = ctx.slot_bytes / static_cast<int64_t>(sizeof(float4));
  for (int row = b; row < rows; row += gridDim.x) {
    const float4 *src = reinterpret_cast<const float4 *>(x) + static_cast<int64_t>(row) * vec_per_row;
    for (int i = threadIdx.x; i < vec_per_row; i += kThreads) {
      const float4 v = src[i];
      for (int p = 0; p < ctx.world; ++p)
        reinterpret_cast<float4 *>(ctx.buffers[p] + data_off)[ctx.rank * slot_vecs + static_cast<int64_t>(row) *
                                                                   vec_per_row + i] = v;
    }
  }
  __syncthreads();
  if (threadIdx.x < ctx.world) {
    uint32_t *flag = reinterpret_cast<uint32_t *>(ctx.buffers[threadIdx.x]) + b * kMaxRanks + ctx.rank;
    store_release(flag, epoch);
    const uint32_t *mine = reinterpret_cast<const uint32_t *>(ctx.buffers[ctx.rank]) + b * kMaxRanks + threadIdx.x;
    while (load_acquire(mine) != epoch) {
    }
  }
  __syncthreads();
  const float4 *local = reinterpret_cast<const float4 *>(ctx.buffers[ctx.rank] + data_off);
  for (int row = b; row < rows; row += gridDim.x) {
    float4 *dst = reinterpret_cast<float4 *>(out) + static_cast<int64_t>(row) * vec_per_row;
    for (int i = threadIdx.x; i < vec_per_row; i += kThreads) {
      float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      for (int p = 0; p < ctx.world; ++p) {
        const float4 v = local[p * slot_vecs + static_cast<int64_t>(row) * vec_per_row + i];
        acc.x += v.x;
        acc.y += v.y;
        acc.z += v.z;
        acc.w += v.w;
      }
      dst[i] = acc;
    }
  }
}

}  // namespace
}  // namespace seventy

SEVENTY_API int64_t seventy_ar_buffer_bytes(int world, int64_t slot_bytes) {
  using namespace seventy;
  return push_end(world, slot_bytes);
}

SEVENTY_API int seventy_ar_alloc(int world, int64_t slot_bytes, void **ptr, void *handle) {
  using namespace seventy;
  if (world < 2 || world > kMaxRanks || slot_bytes <= 0 || slot_bytes % 32) return SEVENTY_ERR_INVALID_ARGUMENT;
  const int64_t bytes = push_end(world, slot_bytes);
  cudaError_t e = cudaMalloc(ptr, bytes);
  if (e != cudaSuccess) return static_cast<int>(e);
  const int64_t p1 = push_p1_offset(world, slot_bytes);
  e = cudaMemset(*ptr, 0, p1);
  if (e == cudaSuccess) e = cudaMemset(static_cast<char *>(*ptr) + p1, 0x7f, bytes - p1);
  if (e == cudaSuccess) e = cudaDeviceSynchronize();
  if (e != cudaSuccess) return static_cast<int>(e);
  e = cudaIpcGetMemHandle(static_cast<cudaIpcMemHandle_t *>(handle), *ptr);
  return static_cast<int>(e);
}

SEVENTY_API int seventy_ar_open(const void *handle, void **ptr) {
  cudaIpcMemHandle_t h;
  memcpy(&h, handle, sizeof(h));
  return static_cast<int>(cudaIpcOpenMemHandle(ptr, h, cudaIpcMemLazyEnablePeerAccess));
}

SEVENTY_API int seventy_ar_context(int rank, int world, void *const *buffers, int64_t slot_bytes, void **ctx_out) {
  using namespace seventy;
  if (world < 2 || world > kMaxRanks || rank < 0 || rank >= world || slot_bytes <= 0 || slot_bytes % 16)
    return SEVENTY_ERR_INVALID_ARGUMENT;
  auto *ctx = new ArContext();
  for (int p = 0; p < world; ++p) ctx->buffers[p] = static_cast<char *>(buffers[p]);
  ctx->rank = rank;
  ctx->world = world;
  ctx->slot_bytes = slot_bytes;
  cudaError_t e = cudaMalloc(reinterpret_cast<void **>(&ctx->counters), 2 * kMaxBlocks * sizeof(uint32_t));
  if (e == cudaSuccess) e = cudaMemset(ctx->counters, 0, 2 * kMaxBlocks * sizeof(uint32_t));
  ctx->push_epochs = ctx->counters + kMaxBlocks;
  if (e != cudaSuccess) {
    delete ctx;
    return static_cast<int>(e);
  }
  *ctx_out = ctx;
  return SEVENTY_OK;
}

SEVENTY_API int seventy_ar_norm(void *ctx_ptr, void *out, void *residual, const void *x, const float *weight, int rows,
                                int dim, float eps, cudaStream_t stream) {
  using namespace seventy;
  auto *ctx = static_cast<ArContext *>(ctx_ptr);
  if (ctx == nullptr || out == nullptr || x == nullptr || rows <= 0 || dim <= 0 || dim % 8 ||
      dim / 8 > kThreads * kVecs)
    return SEVENTY_ERR_INVALID_ARGUMENT;
  if (static_cast<int64_t>(rows) * dim * static_cast<int64_t>(sizeof(half)) > ctx->slot_bytes)
    return SEVENTY_ERR_INVALID_ARGUMENT;
  if (!aligned(x, 16) || !aligned(out, 16) || (residual != nullptr && !aligned(residual, 16)) ||
      (weight != nullptr && !aligned(weight, 16)))
    return SEVENTY_ERR_MISALIGNED;
  const int blocks = rows < kMaxBlocks ? rows : kMaxBlocks;
  ar_norm_kernel<<<blocks, kThreads, 0, stream>>>(*ctx, static_cast<half *>(out), static_cast<half *>(residual),
                                                  static_cast<const half *>(x), weight, rows, dim, eps);
  return launch_status();
}

SEVENTY_API int seventy_ar_sum_f32(void *ctx_ptr, float *out, const float *x, int rows, int dim, cudaStream_t stream) {
  using namespace seventy;
  auto *ctx = static_cast<ArContext *>(ctx_ptr);
  if (ctx == nullptr || out == nullptr || x == nullptr || rows <= 0 || dim <= 0 || dim % 4)
    return SEVENTY_ERR_INVALID_ARGUMENT;
  if (static_cast<int64_t>(rows) * dim * static_cast<int64_t>(sizeof(float)) > ctx->slot_bytes)
    return SEVENTY_ERR_INVALID_ARGUMENT;
  if (!aligned(x, 16) || !aligned(out, 16)) return SEVENTY_ERR_MISALIGNED;
  static const bool flags = [] {
    const char *v = getenv("SEVENTY_AR");
    return v != nullptr && strcmp(v, "flags") == 0;
  }();
  if (!flags && rows <= kMaxBlocks)
    ar_sum_push_kernel<<<rows, kThreads, 0, stream>>>(*ctx, out, x, rows, dim);
  else
    ar_sum_f32_kernel<<<rows < kMaxBlocks ? rows : kMaxBlocks, kThreads, 0, stream>>>(*ctx, out, x, rows, dim);
  return launch_status();
}

SEVENTY_API int seventy_ar_norm_2shot(void *ctx_ptr, void *out, void *residual, const void *x, const float *weight,
                                      int rows, int dim, float eps, int gather_residual, cudaStream_t stream) {
  using namespace seventy;
  auto *ctx = static_cast<ArContext *>(ctx_ptr);
  if (ctx == nullptr || out == nullptr || x == nullptr || rows <= 0 || rows > kMaxBlocks || dim <= 0 || dim % 8 ||
      dim / 8 > kThreads * kVecs || ctx->world < 2)
    return SEVENTY_ERR_INVALID_ARGUMENT;
  if (static_cast<int64_t>(rows) * dim * static_cast<int64_t>(sizeof(half)) > ctx->slot_bytes)
    return SEVENTY_ERR_INVALID_ARGUMENT;
  if (!aligned(x, 16) || !aligned(out, 16) || (residual != nullptr && !aligned(residual, 16)) ||
      (weight != nullptr && !aligned(weight, 16)))
    return SEVENTY_ERR_MISALIGNED;
  static const bool flags = [] {
    const char *v = getenv("SEVENTY_AR");
    return v != nullptr && strcmp(v, "flags") == 0;
  }();
  static const bool home_norm = [] {
    const char *v = getenv("SEVENTY_AR");
    return v != nullptr && strcmp(v, "home") == 0;
  }();
  const bool push = !flags && static_cast<int64_t>(rows) * dim * static_cast<int64_t>(sizeof(float)) <=
                                  push_slot_bytes(ctx->slot_bytes);
  if (weight == nullptr && (!push || home_norm)) return SEVENTY_ERR_INVALID_ARGUMENT;
  if (!push)
    ar_norm_2shot_kernel<<<rows, kThreads, 0, stream>>>(*ctx, static_cast<half *>(out),
                                                        static_cast<half *>(residual), static_cast<const half *>(x),
                                                        weight, rows, dim, eps, gather_residual);
  else if (home_norm)
    ar_norm_push_kernel<<<rows, kThreads, 0, stream>>>(*ctx, static_cast<half *>(out),
                                                       static_cast<half *>(residual), static_cast<const half *>(x),
                                                       weight, rows, dim, eps, gather_residual);
  else {
    int threads = (dim / 8 + 31) / 32 * 32;
    if (threads > kThreads) threads = kThreads;
    if (threads * kVecs < dim / 8) threads = kThreads;
    if (rows <= kOneShotRows)
      ar_norm_push_one_kernel<<<rows, threads, 0, stream>>>(*ctx, static_cast<half *>(out),
                                                            static_cast<half *>(residual),
                                                            static_cast<const half *>(x), weight, rows, dim, eps);
    else
      ar_norm_push_all_kernel<<<rows, threads, 0, stream>>>(*ctx, static_cast<half *>(out),
                                                            static_cast<half *>(residual),
                                                            static_cast<const half *>(x), weight, rows, dim, eps);
  }
  return launch_status();
}
