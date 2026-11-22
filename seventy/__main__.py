"""Command line: `seventy <command>`."""

from __future__ import annotations

import argparse
import platform
import sys

from . import __version__


def cmd_info(args: argparse.Namespace) -> int:
    import os

    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    import torch

    from .ops import _lib

    print(f"seventy {__version__}")
    print(f"python {platform.python_version()}, torch {torch.__version__}, CUDA {torch.version.cuda}")
    try:
        from .panel.hardware import driver_version

        print(f"NVIDIA driver {driver_version()}")
    except Exception as e:
        print(f"NVIDIA driver: not found ({type(e).__name__}: {e})")
    if not torch.cuda.is_available():
        print("GPUs: none visible")
    else:
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            note = "" if (p.major, p.minor) == (7, 0) else " (not a V100: seventy does not run on this GPU)"
            print(f"GPU {i}: {p.name}, {p.total_memory / 2**30:.1f} GiB, sm{p.major}{p.minor}{note}")
        n = torch.cuda.device_count()
        if n > 1:
            missing = [f"{a}-{b}" for a in range(n) for b in range(a + 1, n)
                       if not (torch.cuda.can_device_access_peer(a, b) and torch.cuda.can_device_access_peer(b, a))]
            if missing:
                print(f"GPU peer access: none for {', '.join(missing)} (tensor parallel over these GPUs uses NCCL for "
                      f"every all-reduce, which is slower)")
            else:
                print("GPU peer access: all pairs")
    if _lib.available():
        print(f"kernels: {_lib.path()} (ABI {_lib.ABI_VERSION})")
    else:
        try:
            _lib.lib()
        except Exception as e:
            print(f"kernels: not available: {e}")
    return 0


def cmd_check_model(args: argparse.Namespace) -> int:
    from .config import UnsupportedModel, load_config

    try:
        cfg = load_config(args.path)
    except UnsupportedModel as e:
        print("seventy cannot serve this model:")
        for reason in e.reasons:
            print(f"  - {reason}")
        return 1
    t = cfg.text
    print(f"{cfg.architecture}: {t.num_layers} layers ({len(t.linear_layers)} Gated DeltaNet, "
          f"{len(t.attention_layers)} attention), hidden {t.hidden_size}, weights {cfg.quant.method}, "
          f"vision {'yes' if cfg.vision else 'no'}")
    return 0


def _gpu_problems(tp: int) -> list[str]:
    """Why the first tp visible GPUs cannot run the engine, from NVML (no CUDA context in this process). Empty
    when they can, or when NVML or the GPU ids are not available (the engine then finds the problems itself)."""
    import os

    from .panel.hardware import gpu_set_problems, inventory

    inv = inventory()
    if inv.error is not None:
        return []
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        ids = list(range(tp))
    else:
        parts = [p.strip() for p in visible.split(",") if p.strip()]
        if not all(p.isdigit() for p in parts):
            return []
        ids = [int(p) for p in parts]
        if len(ids) < tp:
            return [f"tensor parallel {tp} needs {tp} GPUs, but only {len(ids)} are given"]
        ids = ids[:tp]
    return gpu_set_problems(inv, ids)


def cmd_engine(args: argparse.Namespace) -> int:
    import json
    import os

    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    from . import memory
    from .api.server import ServerConfig, run
    from .config import UnsupportedModel, has_mtp, load_config
    from .engine.engine import EngineConfig
    from .hub import ModelNotFound, resolve, served_name

    tp = args.tensor_parallel or (len(args.gpus.split(",")) if args.gpus else 1)
    problems = _gpu_problems(tp)
    if problems:
        print("seventy cannot start on these GPUs:\n" + "\n".join(f"  - {p}" for p in problems), file=sys.stderr)
        return 1
    try:
        model = resolve(args.model)
        draft_model = (resolve(args.draft_model, "drafter") if args.draft_model and args.draft_model != "mtp"
                       else args.draft_model)
    except ModelNotFound as e:
        print(f"seventy: {e}", file=sys.stderr)
        return 1
    try:
        cfg = load_config(model)
    except UnsupportedModel as e:
        print("seventy cannot serve this model:\n" + "\n".join(f"  - {r}" for r in e.reasons))
        return 1
    if draft_model == "mtp":
        if not has_mtp(model):
            print("seventy: --draft-model mtp needs a checkpoint with an MTP layer (mtp.* tensors)", file=sys.stderr)
            return 1
    shape = memory.DrafterShape.from_checkpoint(draft_model) if draft_model and draft_model != "mtp" else None
    problems = memory.split_problems(cfg, tp, shape)
    if problems:
        print(f"seventy cannot split the model over {tp} GPUs:\n" + "\n".join(f"  - {p}" for p in problems),
              file=sys.stderr)
        return 1
    engine = EngineConfig(
        model=model, max_num_seqs=args.max_num_seqs,
        max_model_len=min(args.max_model_len or cfg.text.max_position_embeddings, cfg.text.max_position_embeddings),
        gpu_memory_utilization=args.gpu_memory_utilization, prefill_budget=args.prefill_budget,
        kv_dtype=args.kv_cache_dtype, tensor_parallel=tp, cuda_graphs=not args.no_cuda_graphs,
        draft_model=draft_model, mtp_tokens=args.mtp_tokens, prefix_cache=not args.no_prefix_cache,
        cache_entries=args.cache_entries,
    )
    server = ServerConfig(
        engine=engine, served_model_name=args.served_model_name or served_name(args.model, model),
        api_key=args.api_key or os.environ.get("SEVENTY_API_KEY") or None,
        default_chat_template_kwargs=json.loads(args.default_chat_template_kwargs or "{}"),
        max_tokens_default=args.max_tokens_default, max_image_pixels=args.max_image_pixels,
        max_images_per_request=args.max_images_per_request, max_videos_per_request=args.max_videos_per_request,
    )
    run(server, args.host, args.port)
    return 0


def cmd_panel(args: argparse.Namespace) -> int:
    from pathlib import Path

    from .panel.app import run

    run(args.host, args.port, Path(args.home) if args.home else None)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="seventy", description=__doc__)
    parser.add_argument("--version", action="version", version=f"seventy {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("info", help="show versions, GPUs and the kernel library").set_defaults(fn=cmd_info)
    check = sub.add_parser("check-model", help="check whether seventy can serve a checkpoint folder")
    check.add_argument("path")
    check.set_defaults(fn=cmd_check_model)
    eng = sub.add_parser("engine", help="serve a model with an OpenAI-compatible API")
    eng.add_argument("--model", required=True,
                     help="checkpoint folder, or a Hugging Face model ID (downloaded into the Hugging Face cache)")
    eng.add_argument("--draft-model", help="drafter for speculative decoding: a DFlash2 folder or model ID, or mtp "
                     "(the model's own MTP layer)")
    eng.add_argument("--mtp-tokens", type=int, default=4, help="tokens that the MTP drafter drafts per step")
    eng.add_argument("--served-model-name", help="the model name in the API (default: the name part of the model ID, "
                     "or the folder name)")
    eng.add_argument("--host", default="127.0.0.1")
    eng.add_argument("--port", type=int, default=8000)
    eng.add_argument("--api-key", help="require this key (also SEVENTY_API_KEY)")
    eng.add_argument("--gpus", help="GPU ids to use, for example 0,1,2,3 (default: all visible)")
    eng.add_argument("--tensor-parallel", type=int, help="GPUs to split the model over (default: the number of --gpus)")
    eng.add_argument("--max-model-len", type=int)
    eng.add_argument("--max-num-seqs", type=int, default=8)
    eng.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    eng.add_argument("--prefill-budget", type=int, default=8192, help="prompt tokens per engine step")
    eng.add_argument("--kv-cache-dtype", choices=("int8", "fp16"), default="int8")
    eng.add_argument("--no-cuda-graphs", action="store_true")
    eng.add_argument("--no-prefix-cache", action="store_true")
    eng.add_argument("--cache-entries", type=int, default=0,
                     help="prefix-cache entries (state slots) besides the running streams (default: three per stream)")
    eng.add_argument("--default-chat-template-kwargs", help='JSON, for example {"enable_thinking": false}')
    eng.add_argument("--max-tokens-default", type=int)
    eng.add_argument("--max-image-pixels", type=int,
                     help="resize larger images down to this many pixels (default: the checkpoint's limit)")
    eng.add_argument("--max-images-per-request", type=int, default=20)
    eng.add_argument("--max-videos-per-request", type=int, default=1)
    eng.set_defaults(fn=cmd_engine)
    pan = sub.add_parser("panel", help="the control panel: start, watch and stop the engine from a browser")
    pan.add_argument("--host", default="127.0.0.1", help="0.0.0.0 opens the panel to the network (needs the UI token)")
    pan.add_argument("--port", type=int, default=8270)
    pan.add_argument("--home", help="settings, logs and history (default: $XDG_STATE_HOME/seventy)")
    pan.set_defaults(fn=cmd_panel)
    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
