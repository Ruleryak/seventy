# Mixture-of-experts models (branch `moe`)

This branch adds Qwen mixture-of-experts models to seventy: Qwen3.5-MoE (for example Qwen3.6-35B-A3B) and
Qwen4-Exp (Qwen3.8 Flash-Next). The work is not finished. This page gives the state of the branch.

## The checkpoints

| Model | Checkpoint | Format | Why this one |
|---|---|---|---|
| Qwen3.8 Flash-Next | `nvidia/Qwen3.8-Flash-Next-FP4-Local` | ModelOpt mixed: routed experts NVFP4 trained with quantization-aware distillation (QAD), attention, Gated DeltaNet and shared experts MXFP8, n-gram table NVFP4 (also QAD), block scales CSF-coded | QAD: GPQA Diamond 91.5 against 92.2 for BF16 and ToolEvalBench 88.7 against 89.4 (nvidia's model card). The smallest checkpoint with all 512 experts (95 GiB). |
| Qwen3.8 Flash-Next | `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` | The same QAD text weights (byte for byte), plain block scales, BF16 vision tower | Read by the same code; its vision tower is not quantized. |
| Qwen3.6-35B-A3B | `nvidia/Qwen3.6-35B-A3B-NVFP4` | ModelOpt mixed: experts, shared experts and LM head NVFP4 (weights only), attention and Gated DeltaNet FP8 with one scale per tensor | No NVFP4 checkpoint of this model with quantization-aware training or distillation exists. This post-training quantization scores like BF16 in unsloth's comparison (MMLU-Pro 85.60 against 85.75, GPQA 87.12 against 86.36, AIME 2025 91.88 against 92.50). |
| Qwen3.6-35B-A3B | `Qwen/Qwen3.6-35B-A3B` | BF16 (full weights) | The full-weight model, on 2 or more 32 GB GPUs. |

Other Flash-Next checkpoints that were compared:

- `RadixArk/Qwen3.8-Flash-Next-NVFP4`: post-training quantization of the routed experts only (calibration: 128
  news articles); attention, Gated DeltaNet and shared experts stay BF16, so each token reads more bytes; 126 GiB.
- `nvidia/Qwen3.8-Flash-Next-NVFP4`: post-training quantization (ModelOpt), 124 GiB, scores close to FP8.
- `autotrust/Qwen3.8-Flash-Next-E384-FP4-*`: the QAD checkpoint with 384 of the 512 experts kept (80 GiB); it loses
  5 to 9 points on knowledge benchmarks (GPQA, MMLU). The V2 checkpoint is also fine-tuned for security tasks.
- GGUF checkpoints (the most downloaded, for example `ISTA-DASLab/...-GSQ-RCO-GGUF`) and AWQ or AutoRound INT4
  checkpoints: formats that seventy does not read.

## What the branch has

- The MoE MLP (`seventy/model/moe.py`): the router (softmax over all experts, top k, the k weights renormalized),
  the routed experts split over the GPUs (expert parallel: each GPU keeps a part of the experts) and the shared
  expert split like a dense MLP. The partial sums meet in the all-reduce after the MLP.
- MoE kernels. Up to 64 tokens (decode and speculative verify, in CUDA graphs, no host synchronization): one FP16
  product gives the router logits and the shared expert's gate logit (BF16 weights times a power of two are exact
  in FP16), the top k (`csrc/moe.cu`, one warp per token), a plan that sorts each GPU's (token, expert) pairs on the
  GPU, the grouped expert products, and `moe_finish` (the gated shared expert plus the token's picks). Prefill: the
  same products over a plan built with PyTorch, on rows for this GPU's pairs only. NVFP4 experts are repacked at
  load for the tensor cores (`seventy_moe_mma` in `csrc/nvfp4_gemm.cu`: 64-row slots for prefill, warps splitting
  k for small slots); FP16 experts (BF16 checkpoints) use `seventy_moe_gemm` and, in prefill, cuBLAS per expert.
- Checkpoint formats (`seventy/checkpoint.py`): ModelOpt NVFP4 (with CSF-coded block scales), FP8 with one scale per
  tensor or row (kept as FP8 for the FP8 GEMM), MXFP8 (kept as FP8 with one scale per row, see below), and the
  fused experts of BF16 checkpoints.
- 1, 2, 3 and 4 GPUs. With 2 KV heads and 3 or 4 GPUs, the GPUs share each KV head (`seventy/split.py`).
- The KV split (Flash-Next on 4 GPUs): the two GPUs that share a KV head keep half of its tokens each (even and odd
  cache slots), so the KV cache holds about 1.8 times the tokens. Each GPU attends over its half of every query's
  selected positions for all 12 query heads of the KV head (the GPUs computed all 12 before, for 6 each), and one
  kernel sends the partner's 6 heads over NVLink and merges the two halves (`seventy_ar_attn_merge` in
  `csrc/allreduce.cu`; without the peer-to-peer all-reduce, NCCL). A QSA selection always puts the positions of
  each parity at entries of the same parity, so each GPU reads only its own entries. The pooled indexer keys stay
  on every GPU. `SEVENTY_KV_SPLIT=0` turns the split off.
- The MTP drafter (`--draft-model mtp`) for both models (`seventy/model/mtp.py`, and `seventy/model/mtp_exp.py` for
  Flash-Next's MTP layer, which works on hyper-connection streams).
- The Qwen4-Exp text model (`seventy/model/qwen4_exp.py`): hyper-connections (`csrc/hyper.cu`), the n-gram
  embeddings (PLE: `csrc/ple.cu`, hashing, the table rows read from pinned host memory, the convolution and its
  state), the QSA indexer and sparse attention (`csrc/qsa.cu`), the sigmoid gate of its Gated DeltaNet layers.
  - QSA: the indexer queries and raw keys come from one product and one kernel; a block of 4 tokens gets its pooled
    key when it completes. The raw keys are not kept per token: the step's own and a ring of the last 32 per sequence
    and layer give the pooled keys (a block that rejected drafts completed is pooled again from the right keys).
    Decode selects with a score kernel and a top-k kernel; prefill scores the blocks with cuBLAS. The sparse
    attention kernel reads each query's selected positions in chunks from the paged cache.
  - Speculative verify: the Gated DeltaNet and PLE states wait for the accepted rows (`commit`).
- YaRN (`seventy/rope.py`): checkpoints whose config has YaRN rope parameters load with their longer context, and
  `--yarn-factor` applies YaRN to a checkpoint without it. Flash-Next is trained for 262,144 tokens; its model card
  gives up to 1,000,000 with YaRN. For example:

  ```bash
  seventy engine --model nvidia/Qwen3.8-Flash-Next-FP4-Local --yarn-factor 2 --max-model-len 524288
  ```

  YaRN applies to every request, short ones too.

## Results on 4 x V100-SXM2-32GB (local, 250 W)

The smoke checks: greedy and sampled answers, needles at 8K and 30K or 100K tokens, an image, 8 identical greedy
requests at the same time, one stream of 512 tokens. All pass, except where noted: the image check is
case-sensitive, and some runs read the image text correctly in lowercase.

| Model | GPUs | Settings | KV tokens | One stream | 8 identical requests |
|---|---:|---|---:|---:|---:|
| `nvidia/Qwen3.6-35B-A3B-NVFP4` | 1 | defaults | 282,592 | 128 tok/s | 642 tok/s |
| `nvidia/Qwen3.6-35B-A3B-NVFP4` | 1 | `--draft-model mtp` | 113,184 | 177 tok/s | 639 tok/s |
| `nvidia/Qwen3.6-35B-A3B-NVFP4` | 2 | defaults | 2,837,216 | 127 tok/s | 665 tok/s |
| `nvidia/Qwen3.6-35B-A3B-NVFP4` | 3 | defaults | 3,542,448 | 135 tok/s | 731 tok/s |
| `nvidia/Qwen3.6-35B-A3B-NVFP4` | 4 | defaults | 3,940,368 | 148 tok/s | 800 tok/s |
| `nvidia/Qwen3.6-35B-A3B-NVFP4` | 4 | `--draft-model mtp` | 3,508,096 | 261 tok/s | 1,002 tok/s |
| `Qwen/Qwen3.6-35B-A3B` (BF16) | 3 | defaults | 405,600 | 120 tok/s | 477 tok/s |
| `Qwen/Qwen3.6-35B-A3B` (BF16) | 4 | defaults | 1,607,696 | 132 tok/s | 578 tok/s |
| `nvidia/Qwen3.8-Flash-Next-FP4-Local` | 4 | defaults | 1,267,488 | 68 tok/s | 378 tok/s |
| `nvidia/Qwen3.8-Flash-Next-FP4-Local` | 4 | `--draft-model mtp` | 1,014,304 | 131 tok/s | 441 tok/s |
| `nvidia/Qwen3.8-Flash-Next-FP4-Local` | 4 | `--draft-model mtp --expert-offload 0.1` | 1,405,536 | 120 tok/s | 391 tok/s |
| `nvidia/Qwen3.8-Flash-Next-FP4-Local` | 4 | `--draft-model mtp --expert-offload 0.25` | 1,982,448 | 95 tok/s | 298 tok/s |
| `nvidia/Qwen3.8-Flash-Next-FP4-Local` | 3 | `--gpu-memory-utilization 0.97` | 69,392 | 65 tok/s | 362 tok/s |
| `nvidia/Qwen3.8-Flash-Next-FP4-Local` | 3 | `--draft-model mtp --expert-offload 0.25` | 371,184 | 87 tok/s | 336 tok/s |

Prefill: a 100K-token prompt takes 14 s with the 35B-A3B NVFP4 on 4 GPUs, 22 s with Flash-Next on 4 GPUs (MTP; 23 s
with `--expert-offload 0.25`) and 28 s on 3 GPUs (MTP, `--expert-offload 0.25`). With `--yarn-factor 2 --max-model-len 524288`, a 500K-token needle passes with Flash-Next on 4 GPUs (238 s,
before the tensor-core QSA attention kernel). On three 32 GB GPUs, Flash-Next with the default settings needs 0.4
GiB more than the GPUs have; `--expert-offload 0.25` makes room for MTP and 371K KV tokens.

Flash-Next on 4 GPUs with MTP, the README sweep method (sampled, thinking on, 512 tokens per request; tok/s, two
runs per cell):

| Context | 1 stream | 4 streams (per stream) | 8 streams (total) | Cold prefill |
|---|---:|---:|---:|---:|
| 2,000 | 98 / 147 | 72 / 80 | 516 / 546 | 2,682 tok/s |
| 32,000 | 124 / 107 | 101 / 83 | 507 / 507 | 3,357 tok/s |
| 64,000 | 115 / 134 | 88 / 100 | 480 / 505 | 3,260 tok/s |
| 100,000 | 136 / 127 | 79 / 75 | - | 3,121 tok/s |

The sweep ran before the tensor-core QSA attention kernel, which made the 100K prefill about a quarter faster.

Expert offload (`--expert-offload F`) keeps that share of each GPU's routed experts in pinned host memory. It frees
GPU memory for the KV cache, or lets a model run on GPUs that cannot hold all its experts. The MTP layer's experts
stay on the GPU. It works like the GPU expert caches of llama.cpp (`--moe-cache-mib`) and 1Cat-vLLM:

- Placement: the most used experts stay in GPU memory, by an expert profile. Flash-Next has a built-in profile
  (`seventy/profiles/`). A router trace of Flash-Next showed a strong skew: per layer, the most used quarter of the
  experts gets 71 % of the picks and the least used quarter 1.4 %.
- Cache: 8 GPU slots per layer and GPU (`SEVENTY_EXPERT_POOL`, out of the GPU share) take turns holding the other
  experts. The plan kernel of each decode step gives an expert in host memory the least recently used slot that
  the step does not use, and copies it from host memory in the same kernel (one block keeps PCIe busy). The
  expert products read every expert through pointer tables. All of it runs on the GPU, in the CUDA graphs.
- Prefill: before each layer's expert products, its experts in host memory are copied into one of two GPU banks
  on a copy stream, while the layer before computes (`PrefillStage`). A 100K-token prefill takes about as long as
  without offload.
- NUMA: each rank runs on the CPUs of its GPU's NUMA node, so that its pinned host memory is on that node. With 4
  V100s copying at once, each copy runs at 9.3 GB/s from local memory and at 4.7 to 6.4 GB/s from one node.
- `--expert-profile FILE` places the experts by FILE (when it exists and is for the same model) and adds the
  counts of the run to FILE when the engine stops (each GPU counts its experts' picks on the GPU). For example, a
  first run without offload learns the profile of a workload, and later runs use it:

  ```bash
  seventy engine --model nvidia/Qwen3.8-Flash-Next-FP4-Local --gpus 0,1,2,3 --expert-profile ~/fn-experts.json
  ```

  ```bash
  seventy engine --model nvidia/Qwen3.8-Flash-Next-FP4-Local --gpus 0,1,2,3 --expert-offload 0.25 --expert-profile ~/fn-experts.json
  ```

  Without a profile for the model, experts stay on the GPU by index, which misses about 8 times as often.

Decoding slows down when a step needs experts that are in host memory: each one crosses PCIe (2.8 MB per expert
and GPU for Flash-Next, 0.3 to 0.6 ms with 4 GPUs copying), and each layer waits for the GPU with the most misses.

The MXFP8 weights (attention, Gated DeltaNet and shared experts) are kept as FP8 with one scale per row
(`Checkpoint.mxfp8_rows`): each 32-value block moves by its scale's exponent below the row's largest, which rounds
0.01 to 0.02 % of the values (the smallest) to E4M3 subnormals and moves the weights by 3e-7 to 6e-7 relative.
`SEVENTY_MXFP8_ROWS=0` keeps them in FP16 instead (exact, twice the memory).

The engine option `--activation-reserve` (GiB, default 3) sets the memory each GPU keeps for activations and CUDA
graphs; with a smaller prefill budget, less is enough.

## Tests

- `tests/test_moe_reference.py` and `tests/test_flash_next_reference.py`: the first layers of the real checkpoints
  against transformers, in FP32 on the CPU.
- `tests/test_flash_next_verify.py`: a speculative verify and commit on the GPU path equal one-by-one decoding.
- `tests/test_moe_kernels.py`, `tests/test_qsa_kernels.py`, `tests/test_rope.py`, `tests/test_checkpoint_formats.py`,
  `tests/test_fp8_linear.py`: the kernels, the formats and YaRN against PyTorch and transformers references.
- `tests/test_qsa_kernels.py` and `tests/test_allreduce.py`: the two halves of a split KV head merge to the attention
  over the whole cache, on one GPU and between peer GPUs (2 and 4 GPUs).
- `tests/test_moe_kernels.py` and `tests/test_expert_profile.py`: offloaded experts through the cache slots and the
  prefill banks against the per-expert loop, the cache's pointers and slot contents, the use counts, and the
  profile files.

## Not done

- Offloaded experts that a step needs are copied to the GPU. Running them on the CPU instead (KTransformers,
  llama.cpp, AdaOffload, Fiddler) reads them from host memory at CPU memory speed, and only the activations cross
  PCIe.
- Flash-Next's DFlash2 drafter; context past 524,288 tokens (1,000,000 with `--yarn-factor 4` needs more KV cache
  than 4 GPUs have without offload).
- GGUF checkpoints (for example the IQ3 quantizations of Flash-Next).
- The panel's checks for MoE checkpoints (sizes of the expert weights per GPU).
