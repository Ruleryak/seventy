#include "common.cuh"

/*
 * Tensor-parallel all-reduce for small batches over peer-to-peer GPU memory (NVLink), fused with the residual add
 * and RMS norm that follow it in the model.
 *
 * Every rank allocates one buffer and shares it with CUDA IPC. Buffer layout: kMaxBlocks x kMaxRanks uint32 signal
 * flags, then two data areas (alternating between calls) of world slots of slot_bytes each. Block b of a call:
 *   1. takes epoch = ++counter[b] (the same on every rank, because every rank makes the same calls in order),
 *   2. writes its rows of the local partial sums into slot `rank` of every rank's data area (epoch parity),
 *   3. after a system-wide fence, sets signal[b][rank] = epoch in every rank's buffer and waits until its own
 *      signal[b][p] == epoch for every p,
 *   4. adds the world slots of its rows in FP32 (rounded once to FP16), adds the residual, and writes the residual
 *      and the RMS-normed row.
 * Rows map to blocks the same way on every rank, so a block only waits for the same block of the other ranks.
 * Two data areas are enough: a rank can only start call e + 2 after every peer passed call e + 1's barrier, which
 * every peer reaches only after it has read call e's data.
 */

namespace seventy {
namespace {

constexpr int kMaxRanks = 8;
constexpr int kMaxBlocks = 64;
constexpr int kThreads = 256;

struct ArContext {
  char *buffers[kMaxRanks];
  uint32_t *counters;
  int64_t slot_bytes;
  int rank;
  int world;
};

__device__ __forceinline__ uint32_t load_volatile(const uint32_t *p) {
  uint32_t v;
  asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}

__device__ __forceinline__ void store_volatile(uint32_t *p, uint32_t v) {
  asm volatile("st.volatile.global.u32 [%0], %1;" ::"l"(p), "r"(v));
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
  const int64_t signal_bytes = static_cast<int64_t>(kMaxBlocks) * kMaxRanks * sizeof(uint32_t);
  const int64_t area = static_cast<int64_t>(ctx.world) * ctx.slot_bytes;
  const int64_t data_off = signal_bytes + (epoch & 1) * area;
  const int vec_per_row = dim / 8;

  for (int row = b; row < rows; row += gridDim.x) {
    const uint4 *src = reinterpret_cast<const uint4 *>(x + static_cast<int64_t>(row) * dim);
    for (int p = 0; p < ctx.world; ++p) {
      uint4 *dst = reinterpret_cast<uint4 *>(ctx.buffers[p] + data_off + ctx.rank * ctx.slot_bytes +
                                             static_cast<int64_t>(row) * dim * sizeof(half));
      for (int i = threadIdx.x; i < vec_per_row; i += kThreads) dst[i] = src[i];
    }
  }
  __syncthreads();
  if (threadIdx.x < ctx.world) {
    __threadfence_system();
    uint32_t *flag = reinterpret_cast<uint32_t *>(ctx.buffers[threadIdx.x]) + b * kMaxRanks + ctx.rank;
    store_volatile(flag, epoch);
    const uint32_t *mine = reinterpret_cast<const uint32_t *>(ctx.buffers[ctx.rank]) + b * kMaxRanks + threadIdx.x;
    while (load_volatile(mine) != epoch) {
    }
    __threadfence_system();
  }
  __syncthreads();

  const half *local = reinterpret_cast<const half *>(ctx.buffers[ctx.rank] + data_off);
  for (int row = b; row < rows; row += gridDim.x) {
    float sumsq = 0.0f;
    half *res = residual != nullptr ? residual + static_cast<int64_t>(row) * dim : nullptr;
    for (int i = threadIdx.x; i < dim / 2; i += kThreads) {
      float2 acc = make_float2(0.0f, 0.0f);
      for (int p = 0; p < ctx.world; ++p) {
        const half2 v = reinterpret_cast<const half2 *>(local + (ctx.slot_bytes / sizeof(half)) * p +
                                                        static_cast<int64_t>(row) * dim)[i];
        const float2 f = __half22float2(v);
        acc.x += f.x;
        acc.y += f.y;
      }
      float2 r = __half22float2(__floats2half2_rn(acc.x, acc.y));
      if (res != nullptr) {
        const float2 old = __half22float2(reinterpret_cast<half2 *>(res)[i]);
        r = __half22float2(__floats2half2_rn(r.x + old.x, r.y + old.y));
      }
      if (weight == nullptr) {
        reinterpret_cast<half2 *>(out + static_cast<int64_t>(row) * dim)[i] = __floats2half2_rn(r.x, r.y);
      } else {
        if (res != nullptr) reinterpret_cast<half2 *>(res)[i] = __floats2half2_rn(r.x, r.y);
        sumsq += r.x * r.x + r.y * r.y;
      }
    }
    if (weight == nullptr) continue;
    const float total = block_sum(sumsq, scratch);
    const float inv = 1.0f / sqrtf(total / dim + eps);
    const half *srow = res != nullptr ? res : nullptr;
    for (int i = threadIdx.x; i < dim / 2; i += kThreads) {
      float2 r;
      if (srow != nullptr) {
        r = __half22float2(reinterpret_cast<const half2 *>(srow)[i]);
      } else {
        float2 acc = make_float2(0.0f, 0.0f);
        for (int p = 0; p < ctx.world; ++p) {
          const float2 f = __half22float2(reinterpret_cast<const half2 *>(
              local + (ctx.slot_bytes / sizeof(half)) * p + static_cast<int64_t>(row) * dim)[i]);
          acc.x += f.x;
          acc.y += f.y;
        }
        r = __half22float2(__floats2half2_rn(acc.x, acc.y));
      }
      const float2 w = reinterpret_cast<const float2 *>(weight)[i];
      reinterpret_cast<half2 *>(out + static_cast<int64_t>(row) * dim)[i] =
          __floats2half2_rn(r.x * inv * w.x, r.y * inv * w.y);
    }
    __syncthreads();
  }
}

}  // namespace
}  // namespace seventy

SEVENTY_API int64_t seventy_ar_buffer_bytes(int world, int64_t slot_bytes) {
  using namespace seventy;
  return static_cast<int64_t>(kMaxBlocks) * kMaxRanks * sizeof(uint32_t) + 2 * static_cast<int64_t>(world) * slot_bytes;
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
  if (ctx == nullptr || out == nullptr || x == nullptr || rows <= 0 || dim <= 0 || dim % 8)
    return SEVENTY_ERR_INVALID_ARGUMENT;
  if (static_cast<int64_t>(rows) * dim * static_cast<int64_t>(sizeof(half)) > ctx->slot_bytes)
    return SEVENTY_ERR_INVALID_ARGUMENT;
  if (!aligned(x, 16)) return SEVENTY_ERR_MISALIGNED;
  const int blocks = rows < kMaxBlocks ? rows : kMaxBlocks;
  ar_norm_kernel<<<blocks, kThreads, 0, stream>>>(*ctx, static_cast<half *>(out), static_cast<half *>(residual),
                                                  static_cast<const half *>(x), weight, rows, dim, eps);
  return launch_status();
}
