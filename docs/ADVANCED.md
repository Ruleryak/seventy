# Advanced installation, startup and tuning

If you completed the [quickstart](../README.md#quickstart), seventy is already installed. This page gives the
requirements, other installation methods, startup without the panel, the engine options, model downloads,
network access, and the panel as a service.

## Requirements

- Linux x86_64 with glibc 2.28 or later, for example Ubuntu 20.04 or later, Debian 11 or later, or RHEL 8 or later. Version 1.1.4 has the same kernel library as version 1.1.1, which was tested on Ubuntu 20.04, 22.04 and 24.04, and on Rocky Linux 8.10 (glibc 2.28).
- 1, 2, 3 or 4 NVIDIA V100 or V100S GPUs (16 GB or 32 GB). On a machine with more GPUs, choose 1, 2, 3 or 4 of them with `--gpus`. For three GPUs, see [Three GPUs](#three-gpus). With more than one GPU, NVLink between the GPUs is recommended. Without NVLink or PCIe peer-to-peer access, tensor parallel uses NCCL for every all-reduce, which is slower. `seventy info` shows the peer access. GPUs with different memory sizes can work together. Each GPU then gets the KV cache size of the smallest GPU. GPUs other than the V100 are not supported: the engine and the panel stop with an error.
- A working NVIDIA driver. `nvidia-smi` must be able to list the GPUs. Tested with drivers 580.178.04 (branch 580, the last branch for Volta) and 570.133.20. The CUDA 12.9 libraries need driver 525.60.13 or later; drivers before branch 570 are not tested.
- Enough GPU memory for your model and settings. The panel calculates a memory plan before startup.
- Internet access for the installation and the model downloads, and enough disk space: approximately 8 GB for the seventy environment (PyTorch and the CUDA libraries), up to 8 GB more for the download cache of uv or pip, 20 GB for the NVFP4 27B model, and 4 GB for its DFlash2 drafter.
- Bash and curl for the commands below. systemd is needed only for the service instructions.

The uv installation method obtains Python 3.12 if necessary. The pip alternative requires an existing Python 3.12 installation with `venv` support.

No local CUDA toolkit or compiler is required. The NVIDIA driver is still required.

## Install with uv

If uv is not installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

Create the environment and install the release:

```bash
uv venv --python 3.12 ~/seventy-env &&
uv pip install --python ~/seventy-env --torch-backend=cu129 \
  "seventy @ https://github.com/Ruleryak/seventy/releases/download/v1.1.4/seventy-1.1.4-py3-none-manylinux_2_28_x86_64.whl"
```

All examples use the executables inside `~/seventy-env`, so you do not need to activate the environment or change your system Python.

## Alternative: install with pip

Use this instead of the uv installation method:

```bash
python3.12 -m venv ~/seventy-env &&
~/seventy-env/bin/python -m pip install \
  "seventy @ https://github.com/Ruleryak/seventy/releases/download/v1.1.4/seventy-1.1.4-py3-none-manylinux_2_28_x86_64.whl" \
  --extra-index-url https://download.pytorch.org/whl/cu129
```

## Check the installation

```bash
~/seventy-env/bin/seventy info
```

This reports the GPUs, NVIDIA driver, and kernel library. Resolve any reported compatibility or loading errors before starting the engine.

The commands above install version **1.1.4**, with the versions of PyTorch and the other libraries that were tested with it. Running the install command again does not select a newer seventy release; use the wheel URL from the release you want to install.

## Start with the control panel

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

## Start without the panel

Use either a Hugging Face model ID or a local model directory for `--model` and `--draft-model`. Model IDs use the Hugging Face cache and download missing files on first use.

Start on one 32 GB GPU without a drafter. For a larger KV cache on one GPU, see [One GPU](#one-gpu).

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

One 16 GB GPU can run a smaller model, for example `Qwen/Qwen3.5-4B` (BF16), with 2 streams. The KV cache then holds
about 41,000 tokens:

```bash
~/seventy-env/bin/seventy engine \
  --model Qwen/Qwen3.5-4B \
  --gpus 0 \
  --max-num-seqs 2
```

With the default of 8 streams, the state of the streams and of their prefix-cache entries leaves no memory for the
KV cache on one 16 GB GPU, and the engine stops with a message that names these options. The MTP drafter also takes
KV cache memory: `Qwen/Qwen3.5-9B` on two 16 GB GPUs holds about 140,000 tokens without a drafter and about 45,000
with `--draft-model mtp`.

Start on four GPUs with a drafter:

```bash
~/seventy-env/bin/seventy engine \
  --model QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4 \
  --draft-model incoai/Qwen3.8-27B-DFlash2 \
  --gpus 0,1,2,3
```

Start on three GPUs with a drafter (this needs a version later than 1.1.4, see [Three GPUs](#three-gpus)):

```bash
~/seventy-env/bin/seventy engine \
  --model QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4 \
  --draft-model incoai/Qwen3.8-27B-DFlash2 \
  --gpus 0,1,2
```

The API listens on port **8000**. Use `--port` to change it. The engine runs in the foreground; press **Ctrl+C** to stop it.

See all engine options:

```bash
~/seventy-env/bin/seventy engine --help
```

## Tune the engine

The control panel sets these options from its memory plan. For the engine command, add them to the command
line. To see all options:

```bash
~/seventy-env/bin/seventy engine --help
```

| Option | Default | Effect |
|---|---|---|
| `--max-num-seqs` | `8` | The number of requests that run at the same time (streams). Each stream keeps a state in GPU memory. With the prefix cache, the engine keeps 3 more states for each stream. Fewer streams leave more memory for the KV cache. |
| `--max-model-len` | the model's limit (262,144 tokens) | The maximum length of one request: the prompt and the output together. |
| `--gpu-memory-utilization` | `0.90` | The part of each GPU's memory that seventy can use. The KV cache gets the memory that remains after the weights, the stream states, and 3 GiB for activations. |
| `--kv-cache-dtype` | `int8` | With `fp16`, the KV cache is in FP16. Each token then needs two times the memory. |
| `--cache-entries` | three per stream | The number of earlier prompts and replies that the prefix cache keeps, besides the running requests. Each one keeps a state in GPU memory, like a stream. More entries help many clients that come back to their conversations. |
| `--mtp-tokens` | `4` | With `--draft-model mtp`: the number of tokens that the MTP layer drafts for each step. |
| `--no-prefix-cache` | the prefix cache is on | Stops the reuse of earlier prompt text, for example a system prompt or the earlier turns of a chat. This releases the 3 extra states for each stream, but the engine must read repeated prompt text again. |
| `--prefill-budget` | `8192` | The number of prompt tokens in one engine step. A smaller value needs less activation memory, but the engine reads long prompts more slowly. |
| `--no-cuda-graphs` | CUDA graphs are on | Runs without CUDA graphs. Decoding is slower. Use it only to find a problem. |
| `--default-chat-template-kwargs` | none | Chat template settings for all requests, for example `'{"enable_thinking": false}'`. The `chat_template_kwargs` of a request replace them. |
| `--max-tokens-default` | the remaining context | The maximum number of output tokens for a request that does not set `max_tokens`. |
| `--max-image-pixels` | the model's limit | The engine makes larger images smaller, to this number of pixels. |
| `--max-images-per-request` | `20` | The maximum number of images in one request. |
| `--max-videos-per-request` | `1` | The maximum number of videos in one request. |

The KV cache holds the context of all running requests together. A prompt that is larger than the KV cache gets
an error. The engine prints the size of the KV cache at startup, for example `cache 188256 KV tokens on each GPU`.
The panel shows it in its memory plan.

### Three GPUs

Three GPUs need a seventy version later than 1.1.4. Until that version is released, install seventy from the main
branch (see [docs/DEVELOPING.md](DEVELOPING.md)). Version 1.1.4 stops with "tensor parallel needs 1, 2, 4 or 8 GPUs".

The 27B model has 4 KV heads, and 3 GPUs cannot hold equal parts of them. Thus each of the 3 GPUs keeps 2 KV heads:
GPU 0 keeps heads 0 and 1, GPU 1 keeps heads 1 and 2, and GPU 2 keeps heads 2 and 3. Each GPU computes the query
heads of its KV heads, but adds only its own 8 of the 24 query heads to the output. The Gated DeltaNet heads, the
MLP and the vocabulary split in the same way into parts that are almost equal.

- The KV cache of each GPU holds the same bytes per token as with 2 GPUs, but the weights take less memory. On
  three 32 GB GPUs with DFlash2 and the default settings, the KV cache holds about 938,000 tokens (about 665,000
  on two GPUs).
- Decoding is faster than on 2 GPUs and slower than on 4 GPUs. On 32 GB GPUs with NVLink at 250 W, 8 streams at
  2,000 tokens of context made about 840 tokens per second in total (640 on 2 GPUs, 1,110 on 4 GPUs), and at
  32,000 tokens about 570 (470 on 2 GPUs, 810 on 4 GPUs).
- At long context, the gain over 2 GPUs is small: each GPU reads the keys and values of 2 KV heads, as with
  2 GPUs. At 128,000 tokens, 1 stream made about 110 tokens per second on 3 GPUs and on 2 GPUs (150 on 4 GPUs), and
  4 streams about 230 in total (210 on 2 GPUs, 360 on 4 GPUs).

### One GPU

On one GPU, the stream states use a large part of the memory. Each stream needs about 147 MiB on one GPU
(73 MiB on each of 2 GPUs, 37 MiB on each of 4 GPUs). With the prefix cache, each stream needs four times that.
On one 32 GB V100, the KV cache holds about 60,000 tokens with the default settings, and about 190,000 tokens
with one stream. The numbers change a little with the driver and with other programs on the GPU.

To start on one 32 GB GPU with one stream:

```bash
~/seventy-env/bin/seventy engine \
  --model QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4 \
  --gpus 0 \
  --max-num-seqs 1
```

### Draft with the MTP layer of the model

The Qwen3.8 27B model has its own MTP layer (multi-token prediction). The engine can use this layer as the
drafter, in place of DFlash2. Then you do not download a drafter.

On 4 V100 GPUs with NVLink, compared with DFlash2:

- With 8 streams and contexts from 2K to 32K tokens, MTP gives approximately 15% to 35% more tokens per second
  in total. At 100K tokens, the two drafters are approximately equal.
- With 1 stream, DFlash2 is usually faster.
- For agent work with long prompts (approximately 77K tokens), DFlash2 is faster: by approximately 30% with 4
  agents, and by approximately 5% with 8 agents.

Use MTP when many short requests run at the same time. For other work, use DFlash2.

To start on four GPUs with the MTP drafter:

```bash
~/seventy-env/bin/seventy engine \
  --model QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4 \
  --draft-model mtp \
  --gpus 0,1,2,3
```

To use the MTP drafter in the control panel:

1. On the **Start** page, in **2. Model**, set **Drafter** to **MTP layer of the model**.
2. Optional: change **MTP draft tokens per step** (1 to 7, default 4). This is the same as `--mtp-tokens`.

If the model has no MTP layer, the panel disables this option.

## Download models in advance

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

## Use a Hugging Face mirror

Set `HF_ENDPOINT` before starting a download, the panel, or the engine. Enter your mirror's full URL when prompted:

```bash
read -r -p 'Hugging Face mirror URL: ' HF_ENDPOINT
export HF_ENDPOINT
```

Then run the relevant command in that terminal. This setting applies to programs started from that shell. For the systemd service, configure `HF_ENDPOINT` with an `Environment=` entry in the service's `[Service]` section.

## Allow API access from another computer

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

## Run the panel as a service

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
