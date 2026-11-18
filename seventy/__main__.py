"""Command line: `seventy <command>`."""

from __future__ import annotations

import argparse
import platform
import sys

from . import __version__


def cmd_info(args: argparse.Namespace) -> int:
    import torch

    from .ops import _lib

    print(f"seventy {__version__}")
    print(f"python {platform.python_version()}, torch {torch.__version__}, CUDA {torch.version.cuda}")
    if not torch.cuda.is_available():
        print("GPUs: none visible")
    else:
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {p.name}, {p.total_memory / 2**30:.1f} GiB, sm{p.major}{p.minor}")
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


def cmd_engine(args: argparse.Namespace) -> int:
    import json
    import os

    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    from .api.server import ServerConfig, run
    from .config import UnsupportedModel, load_config
    from .engine.engine import EngineConfig

    try:
        cfg = load_config(args.model)
    except UnsupportedModel as e:
        print("seventy cannot serve this model:\n" + "\n".join(f"  - {r}" for r in e.reasons))
        return 1
    tp = args.tensor_parallel or (len(args.gpus.split(",")) if args.gpus else 1)
    engine = EngineConfig(
        model=args.model, max_num_seqs=args.max_num_seqs,
        max_model_len=min(args.max_model_len or cfg.text.max_position_embeddings, cfg.text.max_position_embeddings),
        gpu_memory_utilization=args.gpu_memory_utilization, prefill_budget=args.prefill_budget,
        kv_dtype=args.kv_cache_dtype, tensor_parallel=tp, cuda_graphs=not args.no_cuda_graphs,
        draft_model=args.draft_model, prefix_cache=not args.no_prefix_cache,
    )
    server = ServerConfig(
        engine=engine, served_model_name=args.served_model_name or os.path.basename(os.path.normpath(args.model)),
        api_key=args.api_key or os.environ.get("SEVENTY_API_KEY") or None,
        default_chat_template_kwargs=json.loads(args.default_chat_template_kwargs or "{}"),
        max_tokens_default=args.max_tokens_default,
    )
    run(server, args.host, args.port)
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
    eng.add_argument("--model", required=True, help="checkpoint folder")
    eng.add_argument("--served-model-name")
    eng.add_argument("--host", default="127.0.0.1")
    eng.add_argument("--port", type=int, default=8000)
    eng.add_argument("--api-key", help="require this key (also SEVENTY_API_KEY)")
    eng.add_argument("--gpus", help="GPU ids to use, for example 0,1,2,3 (default: all visible)")
    eng.add_argument("--tensor-parallel", type=int, help="GPUs to split the model over (default: the number of --gpus)")
    eng.add_argument("--max-model-len", type=int)
    eng.add_argument("--max-num-seqs", type=int, default=8)
    eng.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    eng.add_argument("--prefill-budget", type=int, default=4096, help="prompt tokens per engine step")
    eng.add_argument("--kv-cache-dtype", choices=("int8", "fp16"), default="int8")
    eng.add_argument("--no-cuda-graphs", action="store_true")
    eng.add_argument("--draft-model", help="DFlash2 drafter checkpoint folder for speculative decoding")
    eng.add_argument("--no-prefix-cache", action="store_true")
    eng.add_argument("--default-chat-template-kwargs", help='JSON, for example {"enable_thinking": false}')
    eng.add_argument("--max-tokens-default", type=int)
    eng.set_defaults(fn=cmd_engine)
    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
