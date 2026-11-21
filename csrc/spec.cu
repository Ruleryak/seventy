#include "common.cuh"

/*
 * Device kernels of speculative decoding (seventy/model/dflash2.py and seventy/engine/spec.py have the reference of
 * each one).
 *
 *   drafter_walk  the DFlash2 selector walk (Drafter.walk) for row b, one block of `rank` threads (thread r holds
 *                 rank component r). At step l the previous token p (the anchor at l = 0) gives
 *                 w[r] = pred[p, r] * hid[b, l, r]; candidate c scores values[b, l, c] + sum_r w[r] succ[ids[b, l, c], r]
 *                 in FP32. scaled = score / t (t = 1 for greedy rows, temperature 0); the choice is the first argmax of
 *                 scaled (plus gumbel[b, l, c] for sampled rows); probs[b, l] is one-hot (greedy) or softmax(scaled);
 *                 the chosen candidate's token feeds step l + 1. The reference forms every score of every previous
 *                 candidate; the walk needs only those of the previous choice.
 *   spec_accept   device_accept_top followed by cut_at_stops (seventy/engine/spec.py) for row b, one warp; lane j
 *                 holds ranks j and j + 32 of the target's top 64 at a position. Greedy rows (temperature 0) accept
 *                 the drafts that match the target's argmax. Sampled rows filter the top 64 by top-k, top-p and min-p
 *                 as the reference does, accept draft i while u_accept * q(draft) <= p(draft) and q(draft) > 0, and
 *                 sample the next token from max(p - q, 0) (or p) at the first rejected position by its CDF.
 */

namespace seventy {
namespace {

constexpr int kWalkMaxK = 32;

__global__ void drafter_walk_kernel(int64_t *tokens, float *probs, const int64_t *anchors, const float *values,
                                    const int64_t *ids, const float *hid, const half *pred, const half *succ,
                                    const float *temps, const float *gumbel, int n, int k, int rank) {
  __shared__ float part[kWalkMaxK][32];
  __shared__ float score[kWalkMaxK];
  __shared__ int64_t chosen;
  const int b = blockIdx.x, r = threadIdx.x, lane = r & 31, warp = r >> 5, warps = blockDim.x >> 5;
  const float temp = temps[b];
  const bool greedy = temp == 0.0f;
  const float safe = greedy ? 1.0f : temp;
  int64_t prev = anchors[b];
  for (int l = 0; l < n; ++l) {
    const int64_t row = static_cast<int64_t>(b) * n + l;
    const float w = __half2float(pred[prev * rank + r]) * hid[row * rank + r];
    float acc[kWalkMaxK];
#pragma unroll
    for (int c = 0; c < kWalkMaxK; ++c) {
      acc[c] = 0.0f;
      if (c < k) acc[c] = w * __half2float(succ[ids[row * k + c] * rank + r]);
    }
#pragma unroll
    for (int c = 0; c < kWalkMaxK; ++c) {
      if (c >= k) break;
      const float v = warp_sum(acc[c]);
      if (lane == 0) part[c][warp] = v;
    }
    __syncthreads();
    if (r < k) {
      float s = 0.0f;
      for (int j = 0; j < warps; ++j) s += part[r][j];
      score[r] = values[row * k + r] + s;
    }
    __syncthreads();
    if (warp == 0) {
      const bool in = lane < k;
      const float scaled = in ? score[lane] / safe : -INFINITY;
      const float noisy = in && !greedy && gumbel != nullptr ? scaled + gumbel[row * k + lane] : scaled;
      float best = noisy;
      int at = in ? lane : kWalkMaxK;
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        const float ob = __shfl_xor_sync(0xffffffffu, best, o);
        const int oa = __shfl_xor_sync(0xffffffffu, at, o);
        if (ob > best || (ob == best && oa < at)) {
          best = ob;
          at = oa;
        }
      }
      float p;
      if (greedy) {
        p = lane == at ? 1.0f : 0.0f;
      } else {
        float m = scaled;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        const float e = in ? expf(scaled - m) : 0.0f;
        p = e / warp_sum(e);
      }
      if (in) probs[row * k + lane] = p;
      if (lane == 0) {
        const int64_t tok = ids[row * k + at];
        tokens[row] = tok;
        chosen = tok;
      }
    }
    __syncthreads();
    prev = chosen;
  }
}

constexpr int kAccTop = 64;

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

__device__ __forceinline__ float warp_scan(float v) {
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const float u = __shfl_up_sync(0xffffffffu, v, o);
    if (lane >= o) v += u;
  }
  return v;
}

__global__ void __launch_bounds__(32) spec_accept_kernel(
    int64_t *n_out, int64_t *new_out, const int64_t *pred, const float *vals, const int64_t *idx,
    const int64_t *drafts, int64_t drafts_ld, const float *q, const int64_t *cand, const float *temp,
    const int64_t *top_k, const float *top_p, const float *min_p, const float *u_accept, const float *u_residual,
    const int64_t *stop, int stops, const int64_t *limit, int k, int c) {
  const int b = blockIdx.x, lane = threadIdx.x;
  const int k1 = k + 1;
  const float t = temp[b];
  const bool greedy = t == 0.0f;
  const float safe = t > 0.0f ? t : 1.0f;
  const int64_t tk = top_k[b];
  const float tp = top_p[b], mp = min_p[b];
  const int64_t *dr = drafts + static_cast<int64_t>(b) * drafts_ld;
  int n_greedy = 0;
  while (n_greedy < k && dr[n_greedy] == pred[static_cast<int64_t>(b) * k1 + n_greedy]) ++n_greedy;

  float p0 = 0.0f, p1 = 0.0f;
  auto probs_at = [&](int i) {
    const float *v = vals + (static_cast<int64_t>(b) * k1 + i) * kAccTop;
    float v0 = v[lane] / safe, v1 = v[lane + 32] / safe;
    if (lane >= tk) v0 = -INFINITY;
    if (lane + 32 >= tk) v1 = -INFINITY;
    const float m = warp_max(fmaxf(v0, v1));
    const float e0 = expf(v0 - m), e1 = expf(v1 - m);
    const float sum = warp_sum(e0 + e1);
    p0 = e0 / sum;
    p1 = e1 / sum;
    const float inc0 = warp_scan(p0);
    const float inc1 = warp_scan(p1) + __shfl_sync(0xffffffffu, inc0, 31);
    const float first = __shfl_sync(0xffffffffu, p0, 0);
    const bool keep0 = lane == 0 || ((inc0 - p0) < tp && p0 >= mp * first);
    const bool keep1 = (inc1 - p1) < tp && p1 >= mp * first;
    p0 = keep0 ? p0 : 0.0f;
    p1 = keep1 ? p1 : 0.0f;
    const float kept = warp_sum(p0 + p1);
    p0 /= kept;
    p1 /= kept;
  };

  int ns = k;
  bool have = false;
  for (int i = 0; i < k; ++i) {
    probs_at(i);
    const int64_t *ir = idx + (static_cast<int64_t>(b) * k1 + i) * kAccTop;
    const int64_t d = dr[i];
    const float pd = warp_sum((ir[lane] == d ? p0 : 0.0f) + (ir[lane + 32] == d ? p1 : 0.0f));
    const int64_t qrow = (static_cast<int64_t>(b) * k + i) * c;
    float qd = 0.0f;
    for (int j = lane; j < c; j += 32) qd += cand[qrow + j] == d ? q[qrow + j] : 0.0f;
    qd = warp_sum(qd);
    if (!(u_accept[static_cast<int64_t>(b) * k + i] * qd <= pd && qd > 0.0f)) {
      ns = i;
      have = true;
      break;
    }
  }
  if (!have) probs_at(ns);
  const int64_t *ir = idx + (static_cast<int64_t>(b) * k1 + ns) * kAccTop;
  float r0 = p0, r1 = p1;
  if (ns < k) {
    const int64_t qrow = (static_cast<int64_t>(b) * k + ns) * c;
    float qo0 = 0.0f, qo1 = 0.0f;
    const int64_t i0 = ir[lane], i1 = ir[lane + 32];
    for (int j = 0; j < c; ++j) {
      const int64_t cj = cand[qrow + j];
      const float qj = q[qrow + j];
      qo0 += i0 == cj ? qj : 0.0f;
      qo1 += i1 == cj ? qj : 0.0f;
    }
    r0 = fmaxf(p0 - qo0, 0.0f);
    r1 = fmaxf(p1 - qo1, 0.0f);
  }
  float rs = warp_sum(r0 + r1);
  if (!(ns < k && rs > 0.0f)) {
    r0 = p0;
    r1 = p1;
    rs = warp_sum(r0 + r1);
  }
  const float u = u_residual[static_cast<int64_t>(b) * k1 + ns];
  const float c0 = warp_scan(r0 / rs);
  const float c1 = warp_scan(r1 / rs) + __shfl_sync(0xffffffffu, c0, 31);
  int pick = __popc(__ballot_sync(0xffffffffu, c0 < u)) + __popc(__ballot_sync(0xffffffffu, c1 < u));
  pick = min(pick, kAccTop - 1);
  const int64_t sampled_next = idx[(static_cast<int64_t>(b) * k1 + ns) * kAccTop + pick];

  const int n = greedy ? n_greedy : ns;
  const int64_t nxt = greedy ? pred[static_cast<int64_t>(b) * k1 + n_greedy] : sampled_next;
  int64_t *nw = new_out + static_cast<int64_t>(b) * k1;
  for (int i = lane; i < k1; i += 32) {
    const int64_t v = i == n ? nxt : (i < k ? dr[i] : 0);
    nw[i] = v;
  }
  if (lane == 0) {
    int first = k1;
    for (int i = 0; i <= n && first == k1; ++i) {
      const int64_t v = i == n ? nxt : (i < k ? dr[i] : 0);
      for (int s2 = 0; s2 < stops; ++s2)
        if (stop[static_cast<int64_t>(b) * stops + s2] == v) {
          first = i;
          break;
        }
    }
    const int64_t lim = limit[b] - 1;
    int64_t out = n < first ? n : first;
    n_out[b] = out < lim ? out : lim;
  }
}

}  // namespace
}  // namespace seventy

SEVENTY_API int seventy_spec_accept(int64_t *n_out, int64_t *new_out, const int64_t *pred, const float *vals,
                                    const int64_t *idx, const int64_t *drafts, int64_t drafts_ld, const float *q,
                                    const int64_t *cand, const float *temp, const int64_t *top_k, const float *top_p,
                                    const float *min_p, const float *u_accept, const float *u_residual,
                                    const int64_t *stop, int stops, const int64_t *limit, int rows, int k, int top,
                                    int c, cudaStream_t stream) {
  using namespace seventy;
  if (n_out == nullptr || new_out == nullptr || pred == nullptr || vals == nullptr || idx == nullptr ||
      drafts == nullptr || q == nullptr || cand == nullptr || temp == nullptr || top_k == nullptr ||
      top_p == nullptr || min_p == nullptr || u_accept == nullptr || u_residual == nullptr || stop == nullptr ||
      limit == nullptr || rows < 0 || k < 1 || top != kAccTop || c < 1 || stops < 0 || drafts_ld < k)
    return SEVENTY_ERR_INVALID_ARGUMENT;
  if (rows == 0) return SEVENTY_OK;
  spec_accept_kernel<<<rows, 32, 0, stream>>>(n_out, new_out, pred, vals, idx, drafts, drafts_ld, q, cand, temp,
                                              top_k, top_p, min_p, u_accept, u_residual, stop, stops, limit, k, c);
  return launch_status();
}

SEVENTY_API int seventy_drafter_walk(int64_t *tokens, float *probs, const int64_t *anchors, const float *values,
                                     const int64_t *ids, const float *hid, const void *pred, const void *succ,
                                     const float *temps, const float *gumbel, int rows, int n, int k, int rank,
                                     cudaStream_t stream) {
  using namespace seventy;
  if (tokens == nullptr || probs == nullptr || anchors == nullptr || values == nullptr || ids == nullptr ||
      hid == nullptr || pred == nullptr || succ == nullptr || temps == nullptr || rows < 0 || n < 1 || k < 1 ||
      k > kWalkMaxK || rank < 32 || rank > 1024 || rank % 32)
    return SEVENTY_ERR_INVALID_ARGUMENT;
  if (rows == 0) return SEVENTY_OK;
  drafter_walk_kernel<<<rows, rank, 0, stream>>>(tokens, probs, anchors, values, ids, hid,
                                                 static_cast<const half *>(pred), static_cast<const half *>(succ),
                                                 temps, gumbel, n, k, rank);
  return launch_status();
}
