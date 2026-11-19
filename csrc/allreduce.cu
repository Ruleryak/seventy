#include "common.cuh"

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
constexpr int64_t kSignalBytes = 2 * static_cast<int64_t>(kMaxBlocks) * kMaxRanks * sizeof(uint32_t);

struct ArContext {
  char *buffers[kMaxRanks];
  uint32_t *counters;
  int64_t slot_bytes;
  int rank;
  int world;
};

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
  return kSignalBytes + 2 * static_cast<int64_t>(world) * slot_bytes;
}

SEVENTY_API int seventy_ar_alloc(int64_t bytes, void **ptr, void *handle) {
  cudaError_t e = cudaMalloc(ptr, bytes);
  if (e != cudaSuccess) return static_cast<int>(e);
  e = cudaMemset(*ptr, 0, bytes);
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
  cudaError_t e = cudaMalloc(reinterpret_cast<void **>(&ctx->counters), kMaxBlocks * sizeof(uint32_t));
  if (e == cudaSuccess) e = cudaMemset(ctx->counters, 0, kMaxBlocks * sizeof(uint32_t));
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
  const int blocks = rows < kMaxBlocks ? rows : kMaxBlocks;
  ar_sum_f32_kernel<<<blocks, kThreads, 0, stream>>>(*ctx, out, x, rows, dim);
  return launch_status();
}

SEVENTY_API int seventy_ar_norm_2shot(void *ctx_ptr, void *out, void *residual, const void *x, const float *weight,
                                      int rows, int dim, float eps, int gather_residual, cudaStream_t stream) {
  using namespace seventy;
  auto *ctx = static_cast<ArContext *>(ctx_ptr);
  if (ctx == nullptr || out == nullptr || x == nullptr || weight == nullptr || rows <= 0 || rows > kMaxBlocks ||
      dim <= 0 || dim % 8 || dim / 8 > kThreads * kVecs || ctx->world < 2)
    return SEVENTY_ERR_INVALID_ARGUMENT;
  if (static_cast<int64_t>(rows) * dim * static_cast<int64_t>(sizeof(half)) > ctx->slot_bytes)
    return SEVENTY_ERR_INVALID_ARGUMENT;
  if (!aligned(x, 16) || !aligned(out, 16) || (residual != nullptr && !aligned(residual, 16)) || !aligned(weight, 16))
    return SEVENTY_ERR_MISALIGNED;
  ar_norm_2shot_kernel<<<rows, kThreads, 0, stream>>>(*ctx, static_cast<half *>(out), static_cast<half *>(residual),
                                                      static_cast<const half *>(x), weight, rows, dim, eps,
                                                      gather_residual);
  return launch_status();
}
