# seventy design

seventy is an inference engine for one model family on one GPU generation: Qwen3.5-family models on NVIDIA
V100 (sm70) GPUs. Every part is chosen for that pair. This document is the reference for the code layout, the
numerics, the memory model, and the milestones.

## Scope of version 1

| Area | Version 1 |
|---|---|
| Architecture | `Qwen3_5ForConditionalGeneration` (`model_type: qwen3_5`): Qwen3.5, Qwen3.6, Qwen3.8. Gated DeltaNet layers plus gated full-attention layers, and the vision tower. |
| Weights | NVFP4 (`compressed-tensors`, format `nvfp4-pack-quantized`, used weight-only) and BF16 (computed in FP16). |
| Speculative decoding | DFlash2 drafters (`DFlash2DraftModel`). |
| GPUs | V100 16 GB and 32 GB; 1, 2 or 4 GPUs in tensor parallel; any set of GPU ids; NVLink or PCIe. |
| API | OpenAI-compatible HTTP: chat completions (streaming, tools, thinking on or off, images), completions with logprobs and prompt logprobs, tokenize, models, Prometheus metrics. |
| Control panel | Startup screen, GPU picker, memory plan, live dashboard with history, model downloads, engine log, crash reports, automatic restart, API key. |

Other architectures (for example a Qwen4 that changes the layer structure, or a mixture-of-experts model) are
added as new model modules. The engine core does not assume one architecture.

## Stack

- Python 3.12 and PyTorch 2.10 built for CUDA 12.9 (`https://download.pytorch.org/whl/cu129`). CUDA 12.9 is
  the last toolkit that compiles for sm70, and driver branch 580 is the last driver branch for Volta. The pin is
  a property of the hardware, not a choice that can float.
- Our kernels are CUDA C++ in `csrc/`. They build into one shared library, `libseventy_kernels.so`, with a C ABI.
  Python loads it with `ctypes` and passes device pointers and the current stream. The library does not link
  against PyTorch or Python, so one wheel (`py3-none-manylinux_2_28_x86_64`) works with any Python 3.12 build of
  the pinned PyTorch. It links the shared CUDA runtime (`libcudart.so.12`) that PyTorch installs, and needs
  glibc 2.28 or later, the same as PyTorch.
- The wheel pins PyTorch (`torch==2.10.0+cu129`) and each other dependency to the version that the release was
  tested with.
- No Triton, no TileLang, no `torch.compile`. Decode and verify steps run as CUDA graphs. This removes compile
  caches and the minutes they add to a cold start.
- PyTorch provides tensors, the caching allocator, cuBLAS (dense FP16 GEMMs), NCCL (`torch.distributed`), and
  CUDA graphs.

## Code layout

| Path | Contents |
|---|---|
| `csrc/` | CUDA kernels and the C ABI (`seventy.h`). |
| `seventy/config.py` | Model configuration: parse, check, derive sizes. |
| `seventy/checkpoint.py` | safetensors index, tensor lookup, slices for tensor parallel. |
| `seventy/nvfp4.py` | The NVFP4 format: unpack, dequantize, and the layout our GEMM kernels read. |
| `seventy/ops/` | One Python function per operation. Each has a PyTorch reference implementation (`reference.py`) and a kernel path. CPU tensors always use the reference. |
| `seventy/model/` | Model definitions: the Qwen3.5 text model, the vision tower, the DFlash2 drafter. |
| `seventy/engine/` | Scheduler, KV and state cache, model runner, sampler, speculative decoding, workers. |
| `seventy/api/` | HTTP server, chat template, tool-call and reasoning parsers. |
| `seventy/panel/` | Control panel server and web UI. |
| `tests/` | Unit tests (CPU), kernel tests (GPU, small memory), model tests (slow). |
| `bench/` | Benchmarks and serving replays. |

## Processes

```
panel (no GPU)  ──starts, watches──>  engine: one process per GPU (rank 0 schedules)
                                          ^
API server process ──requests/tokens──────┘
```

- The panel never touches a GPU. It starts the engine, reads its metrics, writes crash reports, and restarts it
  with the last good settings.
- The engine runs one process per GPU. Rank 0 owns the scheduler and sends each step's plan to the other ranks.
  All ranks run the same model step on their weight shards.
- The API server is a separate process, so tokenizing, detokenizing and parsing never delay a GPU step.

## Control panel

`seventy panel` (default 127.0.0.1:8270) does not load PyTorch, so it runs while the engine is stopped.

- Home folder: `$XDG_STATE_HOME/seventy` (else `~/.local/state/seventy`, or `--home`), mode 700. It holds the
  settings, one run folder per start (`runs/<time>/`, with the settings and the engine log), error reports,
  the measured memory profiles and the dashboard history (`history.sqlite`, 10-second averages, 30 days).
- Start: the planner predicts the KV cache size with the engine's own memory model (`seventy/memory.py`); after
  a start it uses what the engine measured with the same model, drafter, GPU count, GPU memory size and KV format.
  On GPUs with different memory sizes, every GPU gets the KV cache of the smallest GPU (a warning, not an error).
  The engine prints `seventy-stage:` lines; the panel shows them.
- Stop: SIGINT to the API process; the API process stops the engine process, which stops the tensor-parallel
  workers (they ignore SIGINT, so all ranks tear down together). SIGKILL to the whole session after 60 s.
- Crash: an exit nobody asked for, a dead tensor-parallel worker (the engine ends itself), or 8 failed health
  checks. The panel kills what is left of the session, writes a report and, if allowed, starts the last good
  settings again (3 times in 10 minutes at most).
- Access: on an address other than loopback, every `/api/` request needs the UI token (`ui_token` in the home);
  the panel prints the URL with the token. The engine's API key goes to the engine in its environment, never
  on the command line.
- Downloads use the endpoint in the panel settings, or `HF_ENDPOINT`, or the public Hub.

## Numerics

| Quantity | Format |
|---|---|
| Activations, residual stream | FP16 |
| Norms, softmax, rotary, gates, logits | FP32 math |
| NVFP4 weights | packed E2M1, FP8 E4M3 group scales (16 values per group), one FP32 global scale per tensor; dequantized to FP16 inside the GEMM |
| BF16 weights | converted to FP16 at load |
| Norm weights | FP32, with the `1 + w` offset of zero-centered norms folded in at load |
| Gated DeltaNet state | FP32 |
| KV cache (attention layers) | INT8 with one FP32 scale per token and head (default), or FP16 |
| Vision tower | FP16 weights and GEMM inputs, FP32 residual stream (an outlier channel reaches about 7600) |

## Images and videos

- The API process fetches (`data:` or `http(s):` URLs only), decodes and resizes each image: the checkpoint's
  `preprocessor_config.json` limits, antialiased bicubic on the uint8 image (the same operation as the reference
  processor, which the tests check pixel for pixel). `--max-image-pixels` sets a lower limit.
- The chat template's one image pad token per image becomes one pad per 2 x 2 patch block. Rotary positions follow
  the multimodal rules (`seventy/multimodal.py`); every token after the last image has position index + delta.
- The engine encodes an image on every rank just before the first prefill chunk that reaches its tokens, and
  drops the embeddings when prefill has passed them. The vision tower is split over the tensor-parallel ranks.
- The prefix cache matches image tokens by a digest of the resized pixels.
- Videos (`video_url` parts, one per request by default): OpenCV decodes the frames, which are sampled at 2 per
  second (4 to 768) and resized together (`video_preprocessor_config.json`). Each pair of frames is one group: a
  "<t seconds>" timestamp, then its own vision block in the prompt; the vision tower encodes a group like an image
  whose two temporal frames differ.

## Memory model

The planner and the engine use one memory model. Per GPU:

`weights (shard) + workspace + CUDA graph pools + per-request state x requests + KV bytes per token x tokens`

- Per-request state: Gated DeltaNet state and convolution state for each linear-attention layer, plus the
  drafter state. This is independent of context length.
- KV bytes per token: for each full-attention layer, K and V for the local KV heads, plus scales.
- The engine measures the real peak of each piece at startup and stores it, so the panel can show what fits
  before it starts the engine.

## Correctness

- Every kernel has a PyTorch reference in `seventy/ops/reference.py`. Kernel tests compare against it on small
  shapes that use little GPU memory.
- Model tests compare the full model against the `transformers` implementation (test dependency only) on the
  CPU, and against the output of a known-good server.
- Exactness proofs use one stream, where the engine is deterministic. With more streams, compare against a rerun
  of the same settings.
- Speed claims come from step timings in the real server, never from microbenchmarks alone.

## Milestones of version 1

1. Repository, build, wheel, test harness, configuration and checkpoint reading.
2. Correct output of the text model on one GPU, without CUDA graphs, checked against the reference.
3. Kernels: NVFP4 GEMMs for every batch size, Gated DeltaNet prefill and decode, paged attention with INT8 KV,
   norms, rotary, convolution.
4. Tensor parallel on 2 and 4 GPUs, CUDA graphs, continuous batching, prefix cache with state snapshots.
5. DFlash2 drafting and verification, sparse sampling.
6. HTTP API, parsers, vision, metrics, then the control panel.
7. Side-by-side comparison with a known-good server: decode at 1 to 8 streams near 100K context, replay
   throughput, time to first token, draft acceptance, API features, vision.
8. Release: wheel with precompiled kernels on a GitHub release.

A small slowdown of one part against the comparison server is acceptable for version 1. If a part is within a few
percent and more gain needs a long effort, the work moves to the next part and comes back in a later round.
