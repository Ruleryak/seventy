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
  const int64_t signal_bytes = static_cast<int64_t>(kMaxBlocks) * kMaxRanks * sizeof(uint32_t);
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
    __threadfence_system();
    uint32_t *flag = reinterpret_cast<uint32_t *>(ctx.buffers[threadIdx.x]) + b * kMaxRanks + ctx.rank;
    store_volatile(flag, epoch);
    const uint32_t *mine = reinterpret_cast<const uint32_t *>(ctx.buffers[ctx.rank]) + b * kMaxRanks + threadIdx.x;
    while (load_volatile(mine) != epoch) {
    }
    __threadfence_system();
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
