# seventy

seventy is an inference engine for Qwen3.5-family models (Qwen3.5, Qwen3.6, Qwen3.8) on NVIDIA V100 GPUs.
It serves an OpenAI-compatible API and has a control panel in the browser.

**Status:** version 1.1.0. The tests ran on 4 Tesla V100-SXM2-32GB GPUs with NVLink. The installation and a short check also ran on 4 Tesla V100-PCIE-16GB GPUs without peer-to-peer access (Ubuntu 22.04).

- The design and the plan for version 1 are in [docs/DESIGN.md](docs/DESIGN.md).
- To build and test the code, see [docs/DEVELOPING.md](docs/DEVELOPING.md).

## Models

seventy serves checkpoints with the `Qwen3_5ForConditionalGeneration` architecture, with NVFP4 weights
(compressed-tensors) or BF16 or FP16 weights. A DFlash2 drafter makes decoding faster (speculative decoding).

Tested models:

| Model | Use |
|---|---|
| `QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4` | target model (text, images, videos) |
| `incoai/Qwen3.8-27B-DFlash2` | drafter for the model above |

To check a checkpoint folder before you start it:

```
seventy check-model /path/to/checkpoint
```

## Speed

These charts show the speed of seventy on 4 Tesla V100-SXM2-32GB GPUs with NVLink (tensor parallel 4). The engine
uses the NVFP4 27B model, the DFlash2 drafter and the INT8 KV cache. Thinking is on. Each request makes 512 tokens.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/speed-total-dark.svg">
  <img alt="Total decode speed in tokens per second for 1, 4 and 8 streams from 2K to 256K tokens of context" src="docs/images/speed-total-light.svg">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/speed-stream-dark.svg">
  <img alt="Decode speed per stream in tokens per second for 1, 4 and 8 streams from 2K to 256K tokens of context" src="docs/images/speed-stream-light.svg">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/speed-prefill-dark.svg">
  <img alt="Prefill speed in tokens per second for prompts from 2K to 256K tokens" src="docs/images/speed-prefill-light.svg">
</picture>

- A stream is one request. With 4 streams, 4 requests run at the same time.
- In the decode tests, the prompt is in the prefix cache. Thus the decode tests measure only the generation of tokens.
- In the prefill test, the prompt is not in the cache.
- Each test runs two times. The bar at a point shows the range of the runs. The speed of one stream changes most
  from run to run, because the number of draft tokens that the model accepts changes with the text.
- To measure your own GPUs, see "Speed charts" in [docs/DEVELOPING.md](docs/DEVELOPING.md).

<details>
<summary>The numbers in tables</summary>

| Context (tokens) | Streams | Tokens/s per stream | Tokens/s in total |
|---:|---:|---:|---:|
| 2,000 | 1 | 250 | 250 |
| 2,000 | 4 | 197 | 781 |
| 2,000 | 8 | 111 | 877 |
| 32,000 | 1 | 216 | 216 |
| 32,000 | 4 | 140 | 558 |
| 32,000 | 8 | 86 | 668 |
| 100,000 | 1 | 190 | 191 |
| 100,000 | 4 | 96 | 383 |
| 100,000 | 8 | 58 | 467 |
| 128,000 | 1 | 177 | 178 |
| 128,000 | 4 | 100 | 398 |
| 128,000 | 8 | 49 | 388 |
| 192,000 | 1 | 164 | 164 |
| 192,000 | 4 | 104 | 392 |
| 192,000 | 8 | 43 | 335 |
| 256,000 | 1 | 138 | 138 |
| 256,000 | 4 | 62 | 250 |
| 256,000 | 8 | 38 | 301 |

| Prompt (tokens) | Prefill tokens/s | Time to the first token |
|---:|---:|---:|
| 2,000 | 3,839 | 0.5 s |
| 32,000 | 4,087 | 7.8 s |
| 100,000 | 3,102 | 32.2 s |
| 128,000 | 2,832 | 45.2 s |
| 192,000 | 2,369 | 81.1 s |
| 256,000 | 2,035 | 125.8 s |

</details>

## Quickstart

Start with the control panel to download a model, choose your GPUs, and run the engine.

You need Linux x86_64 (glibc 2.28 or later) and NVIDIA V100 GPUs with a working NVIDIA driver; see [Requirements](#requirements). The wheel includes the compiled kernels; no CUDA toolkit or compiler is required. Python 3.12 is installed automatically by uv if needed.

### 1. Install uv

Skip this step if you already have a recent version of [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

### 2. Install seventy

Run as your normal user:

```bash
uv venv --python 3.12 ~/seventy-env &&
uv pip install --python ~/seventy-env --torch-backend=cu129 \
  "seventy @ https://github.com/Ruleryak/seventy/releases/download/v1.1.0/seventy-1.1.0-py3-none-manylinux_2_28_x86_64.whl" &&
~/seventy-env/bin/seventy info
```

Check the reported GPUs, driver, and kernel library before continuing.

### 3. Start the control panel

If your browser is on the GPU machine:

```bash
~/seventy-env/bin/seventy panel
```

Open [http://127.0.0.1:8270/](http://127.0.0.1:8270/).

If your browser is on another computer, use this command instead:

```bash
~/seventy-env/bin/seventy panel --host 0.0.0.0
```

Open the URL printed in the terminal, including its access token.

In the panel:

1. Use **Models** to download `QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4`.
2. Select the model and the GPUs you want to use, review the memory plan, and start the engine.

The drafter is optional. To use it, also download `incoai/Qwen3.8-27B-DFlash2` and select it before starting the engine. It requires additional GPU memory.

Keep the panel running. To run it in the background and start it at boot, see [Run the panel as a service](#run-the-panel-as-a-service).

### 4. Send a request

Once the engine is ready, open another terminal on the GPU machine:

```bash
curl --fail-with-body -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3.8-27B-QUASAR-NVFP4",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

## Full installation and startup

If you completed the quickstart, seventy is already installed. The sections below cover alternative installation, command-line startup, downloads, and background operation.

### Requirements

- Linux x86_64 with glibc 2.28 or later, for example Ubuntu 20.04 or later, Debian 11 or later, or RHEL 8 or later. Tested on Ubuntu 24.04 LTS.
- 1, 2 or 4 NVIDIA V100 GPUs (16 GB or 32 GB). With more than one GPU, NVLink between the GPUs is recommended. Without NVLink or PCIe peer-to-peer access, tensor parallel uses NCCL for every all-reduce, which is slower. `seventy info` shows the peer access. GPUs with different memory sizes can work together. Each GPU then gets the KV cache size of the smallest GPU.
- A working NVIDIA driver. `nvidia-smi` must be able to list the GPUs. Tested with driver 580.178.04 (branch 580, the last branch for Volta). The CUDA 12.9 libraries need driver 525.60.13 or later; drivers before branch 580 are not tested.
- Enough GPU memory for your model and settings. The panel calculates a memory plan before startup.
- Internet access for installation and model downloads, and enough disk space for the downloaded files: approximately 20 GB for the NVFP4 27B model and 4 GB for its DFlash2 drafter.
- Bash and curl for the commands below. systemd is needed only for the service instructions.

The uv installation method obtains Python 3.12 if necessary. The pip alternative requires an existing Python 3.12 installation with `venv` support.

No local CUDA toolkit or compiler is required. The NVIDIA driver is still required.

### Install with uv

If uv is not installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

Create the environment and install the release:

```bash
uv venv --python 3.12 ~/seventy-env &&
uv pip install --python ~/seventy-env --torch-backend=cu129 \
  "seventy @ https://github.com/Ruleryak/seventy/releases/download/v1.1.0/seventy-1.1.0-py3-none-manylinux_2_28_x86_64.whl"
```

All examples use the executables inside `~/seventy-env`, so you do not need to activate the environment or change your system Python.

### Alternative: install with pip

Use this instead of the uv installation method:

```bash
python3.12 -m venv ~/seventy-env &&
~/seventy-env/bin/python -m pip install \
  "seventy @ https://github.com/Ruleryak/seventy/releases/download/v1.1.0/seventy-1.1.0-py3-none-manylinux_2_28_x86_64.whl" \
  --extra-index-url https://download.pytorch.org/whl/cu129
```

### Check the installation

```bash
~/seventy-env/bin/seventy info
```

This reports the GPUs, NVIDIA driver, and kernel library. Resolve any reported compatibility or loading errors before starting the engine.

The commands above install version **1.1.0**, with the versions of PyTorch and the other libraries that were tested with it. Running the install command again does not select a newer seventy release; use the wheel URL from the release you want to install.

### Start with the control panel

For a browser on the GPU machine:

```bash
~/seventy-env/bin/seventy panel
```

Open [http://127.0.0.1:8270/](http://127.0.0.1:8270/).

To listen on all network interfaces for access from another computer:

```bash
~/seventy-env/bin/seventy panel --host 0.0.0.0
```

Open the printed URL with its access token. Treat that URL as a password: anyone with the token and network access can use the panel. The panel token and the engine's API key are separate credentials.

The panel downloads models, calculates the memory plan, starts and stops the engine, and displays generation speed, cache use, and GPU activity. If the engine fails, the panel saves an error report and attempts to restart it using the last good settings.

After you have successfully run an engine through the panel, subsequent panel starts use the last good settings to start the engine again.

Settings, logs, and history are stored in `~/.local/state/seventy`. Use `--home` to choose another directory.

### Start without the panel

Use either a Hugging Face model ID or a local model directory for `--model` and `--draft-model`. Model IDs use the Hugging Face cache and download missing files on first use.

Start on one 32 GB GPU without a drafter:

```bash
~/seventy-env/bin/seventy engine \
  --model QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4 \
  --gpus 0
```

The 27B model does not fit on one 16 GB GPU. With 16 GB GPUs, use four GPUs. Two 16 GB GPUs can run the model only with one stream, no prefix cache and no drafter:

```bash
~/seventy-env/bin/seventy engine \
  --model QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4 \
  --gpus 0,1 \
  --max-num-seqs 1 --no-prefix-cache \
  --gpu-memory-utilization 0.97 --prefill-budget 2048
```

Start on four GPUs with a drafter:

```bash
~/seventy-env/bin/seventy engine \
  --model QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4 \
  --draft-model incoai/Qwen3.8-27B-DFlash2 \
  --gpus 0,1,2,3
```

The API listens on port **8000**. Use `--port` to change it. The engine runs in the foreground; press **Ctrl+C** to stop it.

See all engine options:

```bash
~/seventy-env/bin/seventy engine --help
```

### Download models in advance

You can prefetch the same models used by the panel and the command-line engine:

```bash
~/seventy-env/bin/hf download QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4
```

For the optional drafter:

```bash
~/seventy-env/bin/hf download incoai/Qwen3.8-27B-DFlash2
```

These commands use the Hugging Face cache. The output of each command ends with `path:` and the downloaded directory. You can also give that directory to the engine.

To keep a model in a named directory:

```bash
~/seventy-env/bin/hf download QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4 \
  --local-dir "$HOME/models/seventy/qwen-27b" &&
~/seventy-env/bin/seventy engine \
  --model "$HOME/models/seventy/qwen-27b" \
  --gpus 0
```

### Use a Hugging Face mirror

Set `HF_ENDPOINT` before starting a download, the panel, or the engine. Enter your mirror's full URL when prompted:

```bash
read -r -p 'Hugging Face mirror URL: ' HF_ENDPOINT
export HF_ENDPOINT
```

Then run the relevant command in that terminal. This setting applies to programs started from that shell. For the systemd service, configure `HF_ENDPOINT` with an `Environment=` entry in the service's `[Service]` section.

### Allow API access from another computer

Set an API key and bind the engine to all network interfaces. This example uses GPUs 2 and 3:

```bash
read -r -s -p 'API key: ' SEVENTY_API_KEY
printf '\n'
if [ -z "$SEVENTY_API_KEY" ]; then
  printf 'Error: the API key cannot be empty. Engine not started.\n' >&2
else
  export SEVENTY_API_KEY
  ~/seventy-env/bin/seventy engine \
    --model QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4 \
    --draft-model incoai/Qwen3.8-27B-DFlash2 \
    --gpus 2,3 \
    --host 0.0.0.0
fi
```

Use a nonempty key. Configure your client with the GPU machine's address, port **8000**, the `/v1` base path, and the same API key. Authenticated requests use the `Authorization: Bearer` header.

Opening the panel to the network does not automatically open the engine API to the network.

### Run the panel as a service

First configure and successfully start the engine through the panel so it has settings to restore. Stop the foreground panel with **Ctrl+C** before starting the service.

Create the user service:

```bash
mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/seventy.service <<'EOF'
[Unit]
Description=seventy control panel

[Service]
Type=exec
ExecStart=%h/seventy-env/bin/seventy panel
Environment=PYTHONUNBUFFERED=1
KillMode=mixed
TimeoutStopSec=90
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF
```

This unit uses local browser access. For access from another computer, change its `ExecStart` line to:

```ini
ExecStart=%h/seventy-env/bin/seventy panel --host 0.0.0.0
```

Start it now and enable startup without an interactive login:

```bash
systemctl --user daemon-reload &&
systemctl --user enable --now seventy.service &&
sudo loginctl enable-linger "$USER"
```

The panel restores the last good engine settings when it starts. In network mode, retrieve its token URL from the service log:

```bash
journalctl --user -u seventy.service -b -n 50 --no-pager
```

Useful service commands:

| Action | Command |
| --- | --- |
| Show status | `systemctl --user status seventy.service` |
| Follow logs | `journalctl --user -u seventy.service -f` |
| Stop the panel and its engine | `systemctl --user stop seventy.service` |
| Restart the panel and its engine | `systemctl --user restart seventy.service` |
| Stop and disable automatic startup | `systemctl --user disable --now seventy.service` |

After editing the service file, apply the changes:

```bash
systemctl --user daemon-reload &&
systemctl --user restart seventy.service
```

## Use the API

For a client on the GPU machine, the API base URL is `http://127.0.0.1:8000/v1`. For a remote client, use the GPU machine's address after enabling network access.

The example model name is `Qwen3.8-27B-QUASAR-NVFP4`. Query `/v1/models` to see the model name advertised by your running engine.

Send a chat request to the local engine started by the quickstart:

```bash
curl --fail-with-body -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3.8-27B-QUASAR-NVFP4",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

To disable thinking for one request, include `"chat_template_kwargs": {"enable_thinking": false}` in the JSON body, as shown in the quickstart. For streaming, include `"stream": true` and use curl's `--no-buffer` option.

| Endpoint | Purpose |
| --- | --- |
| `/v1/chat/completions` | OpenAI-compatible chat, including streaming, tools, images, and videos |
| `/v1/completions` | OpenAI-compatible text completions |
| `/v1/models` | Available model information |
| `/tokenize` | Tokenization |
| `/detokenize` | Detokenization |
| `/health` | Engine health |
| `/metrics` | Prometheus metrics |

## Credits

seventy uses methods from these projects. Each source file that holds such a method names the project in its header.

| Project | What seventy uses | License |
|---|---|---|
| [1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM), a fork of [vLLM](https://github.com/vllm-project/vllm) | The push protocol of its tensor-parallel all-reduce for sm70: the data in a slot is also its ready flag (`csrc/allreduce.cu`); the split-D structure of its sm70 D256 prefill attention (`flash_attention_splitd` in `csrc/flash_attention.cu`) | Apache-2.0 |
| [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) | The split of the chunked gated delta rule into a sequential state recurrence and parallel chunk outputs (`csrc/gdn_chunk.cu`) | MIT |

## License

seventy is licensed under the Apache License 2.0. See [LICENSE](LICENSE). The libraries that seventy installs
(PyTorch, FastAPI and others) have their own licenses. Model weights are not part of seventy.
