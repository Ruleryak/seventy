"""Model names on the command line: a checkpoint folder or a Hugging Face model ID. No PyTorch.

A name that is a folder on this machine is used as it is. Any other name of the form owner/name is a model ID:
huggingface_hub.snapshot_download gets it into the Hugging Face cache and gives its folder there. It uses the
settings of huggingface_hub (HF_HOME, HF_HUB_CACHE, HF_ENDPOINT, HF_TOKEN, HF_HUB_OFFLINE) and does not download
again the files that are already in the cache, the same as `hf download`.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

_PART = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class ModelNotFound(Exception):
    """The name is not a folder, or the Hub (or the cache) cannot give the model ID."""


def is_model_id(name: str) -> bool:
    """True when name has the form owner/name of a Hugging Face model ID."""
    parts = name.split("/")
    return (len(parts) == 2 and all(_PART.match(p) and ".." not in p and "--" not in p for p in parts)
            and len(name) <= 96)


def resolve(name: str, what: str = "model") -> str:
    """The checkpoint folder for name: the folder itself, or the cache folder of the model ID (downloaded first)."""
    folder = Path(name).expanduser()
    if folder.is_dir():
        return str(folder)
    if not is_model_id(name):
        raise ModelNotFound(f"the {what} {name!r} is not a folder, and it is not a model ID of the form owner/name")
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(name)
    except Exception as e:
        raise ModelNotFound(f"cannot get the {what} {name} from the Hugging Face Hub: {type(e).__name__}: {e}") from e


def served_name(name: str, folder: str) -> str:
    """The model name that the API shows: the name part of a model ID, else the folder name.

    A folder in the Hugging Face cache (models--owner--name/snapshots/<commit>) also gives the name part of its
    model ID, not the commit.
    """
    parts = Path(os.path.abspath(folder)).parts
    for i, part in enumerate(parts[:-1]):
        if part.startswith("models--") and parts[i + 1] == "snapshots" and "--" in part[len("models--"):]:
            return part[len("models--"):].split("--", 1)[1]
    if is_model_id(name) and not Path(name).expanduser().is_dir():
        return name.split("/", 1)[1]
    return os.path.basename(os.path.normpath(folder))
