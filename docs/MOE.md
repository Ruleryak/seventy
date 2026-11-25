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
- Kernels (`csrc/moe.cu`): the router's top k, a plan that sorts the (token, expert) pairs of each GPU's experts on
  the GPU, and a grouped NVFP4 or FP16 GEMM over the plan. Decode and speculative steps (up to 64 tokens) have no
  host synchronization and run in CUDA graphs. Prefill runs one product per picked expert.
- Checkpoint formats (`seventy/checkpoint.py`): ModelOpt NVFP4 (with CSF-coded block scales), FP8 with one scale per
  tensor or row (kept as FP8 for the FP8 GEMM), MXFP8 (loaded as FP16), and the fused experts of BF16 checkpoints.
- 1, 2, 3 and 4 GPUs. With 2 KV heads and 3 or 4 GPUs, the GPUs share each KV head (`seventy/split.py`).
- The MTP drafter for MoE models.
- The Qwen4-Exp text model (`seventy/model/qwen4_exp.py`): hyper-connections (with kernels, `csrc/hyper.cu`), the
  n-gram embeddings (PLE) from the memory-mapped table in host memory, the QSA indexer and sparse attention (the
  attention kernels while every sequence fits in the indexer budget, else PyTorch over the selected tokens), the
  sigmoid gate of its Gated DeltaNet layers. On the GPUs, QSA (`csrc/qsa.cu`: pooled block keys, block scores,
  top-k selection, sparse attention over the paged cache) and PLE (`csrc/ple.cu`: hashing, the table rows from
  pinned host memory, the convolution and its state) run in kernels, and decode steps run in CUDA graphs.

## Results on 4 x V100-SXM2-32GB (local, 250 W), without speed sweeps

| Model | GPUs | Drafter | KV tokens | Smoke checks | One stream |
|---|---:|---|---:|---|---:|
| `nvidia/Qwen3.6-35B-A3B-NVFP4` | 1 | none | 286,032 | pass (the image text comes back in lowercase) | 103 tok/s |
| `nvidia/Qwen3.6-35B-A3B-NVFP4` | 2 | none | 2,711,392 | pass (lowercase image text) | 73 tok/s (older kernels) |
| `nvidia/Qwen3.6-35B-A3B-NVFP4` | 3 | none | 3,419,856 | pass | 77 tok/s (older kernels) |
| `nvidia/Qwen3.6-35B-A3B-NVFP4` | 4 | none | 3,946,416 | pass, also a 250K-token needle | 108 tok/s |
| `nvidia/Qwen3.6-35B-A3B-NVFP4` | 4 | MTP | 3,516,528 | pass | 168 tok/s |
| `Qwen/Qwen3.6-35B-A3B` (BF16) | 3 | none | 408,816 | pass (lowercase image text) | 95 tok/s |
| `Qwen/Qwen3.6-35B-A3B` (BF16) | 4 | none | 1,614,544 | pass (lowercase image text) | 102 tok/s |
| `nvidia/Qwen3.8-Flash-Next-FP4-Local` | 4 | none | 490,128 | pass, needles at 8K and 30K (lowercase image text); a 250K-token needle passes (max-model-len 262144) | 50.9 tok/s |
| `nvidia/Qwen3.8-Flash-Next-FP4-Local` | 3 | none | 125,712 (4 streams, `--gpu-memory-utilization 0.97`) | pass (lowercase image text) | 48.8 tok/s |
| `nvidia/Qwen3.8-Flash-Next-FP4-Local` | 3 | none | 262,144 (1 stream, no prefix cache, 0.97, `--prefill-budget 2048 --activation-reserve 1.5`) | pass, a 250K-token needle passes | 49.2 tok/s |

Flash-Next with 8 identical requests at the same time: 192 tok/s in total on 4 GPUs, all 8 texts the same. With
the default settings, the weights and the activation reserve take 0.3 GiB more than three 32 GB GPUs have. Prefill
of a 250K-token prompt took 307 s on 4 GPUs and 390 s on 3 GPUs (the MoE prefill runs one product per expert).

The MXFP8 weights (attention, Gated DeltaNet and shared experts) are kept as FP8 with one scale per row
(`Checkpoint.mxfp8_rows`): each 32-value block moves by its scale's exponent below the row's largest, which rounds
0.01 to 0.02 % of the values (the smallest) to E4M3 subnormals and moves the weights by 3e-7 to 6e-7 relative.
`SEVENTY_MXFP8_ROWS=0` keeps them in FP16 instead (exact, twice the memory).

The engine option `--activation-reserve` (GiB, default 3) sets the memory each GPU keeps for activations and CUDA
graphs; with a smaller prefill budget, less is enough.

The smoke checks: greedy and sampled answers, needles at 8K and 30K or 100K tokens, an image, 8 identical greedy
requests at the same time, one stream of 512 tokens.

## Tests

- `tests/test_moe_reference.py` and `tests/test_flash_next_reference.py`: the first layers of the real checkpoints
  against transformers, in FP32 on the CPU.
- `tests/test_moe_kernels.py`, `tests/test_checkpoint_formats.py`, `tests/test_fp8_linear.py`: the kernels and the
  formats against PyTorch references.

## Not done

- Flash-Next: its MTP layer and its DFlash2 drafter; faster prefill (grouped expert GEMMs for long chunks).
- Faster MoE decode: the shared expert in the grouped GEMM, fewer small kernels per layer, MXFP8 GEMMs.
- The panel's checks for MoE checkpoints (sizes of the expert weights per GPU).
