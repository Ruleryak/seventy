"""Images in prompts: loading, resizing, prompt expansion and multimodal rotary positions, by the Qwen3.5 rules.

Resize (the checkpoint's preprocessor_config.json): both sides become multiples of patch_size x merge_size (32),
the area stays within [shortest_edge, longest_edge] pixels and the aspect ratio stays as close as possible. The
uint8 image is resized with antialiased bicubic interpolation on the CPU, the same operation the reference
processor's torchvision backend runs.

Prompt: the chat template writes one image pad token per image. Each pad becomes tokens(grid) pad tokens, one per
2 x 2 block of patches, which the vision tower's embeddings replace.

Positions (three axes, temporal / height / width): text tokens count up on all axes. The tokens of an image get
(P, P + row, P + col) over its merged grid, where P is the next text position; the text after the image continues
at P + max(rows, cols). Every token after the last image therefore has position index + delta, the same on all
axes.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import json
import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

MAX_IMAGE_BYTES = 64 * 1024 * 1024


class ImageError(ValueError):
    pass


@dataclass(frozen=True)
class ImageSettings:
    patch_size: int = 16
    merge_size: int = 2
    temporal_patch_size: int = 2
    min_pixels: int = 65536
    max_pixels: int = 16777216

    @staticmethod
    def from_model(path: str | Path) -> "ImageSettings":
        f = Path(path) / "preprocessor_config.json"
        if not f.is_file():
            return ImageSettings()
        raw = json.loads(f.read_text())
        size = raw.get("size") or {}
        return ImageSettings(
            patch_size=int(raw.get("patch_size", 16)),
            merge_size=int(raw.get("merge_size", 2)),
            temporal_patch_size=int(raw.get("temporal_patch_size", 2)),
            min_pixels=int(size.get("shortest_edge", raw.get("min_pixels", 65536))),
            max_pixels=int(size.get("longest_edge", raw.get("max_pixels", 16777216))),
        )

    @property
    def factor(self) -> int:
        return self.patch_size * self.merge_size


@dataclass
class ImageInput:
    """A resized image: uint8 pixels [H, W, 3], its grid (t, h, w) in patches and a digest of the pixels."""

    pixels: np.ndarray
    grid: tuple[int, int, int]
    digest: str

    def tokens(self, merge: int) -> int:
        t, h, w = self.grid
        return t * h * w // (merge * merge)


def smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    if min(height, width) <= 0:
        raise ImageError("the image is empty")
    if max(height, width) / min(height, width) > 200:
        raise ImageError("the image aspect ratio must be smaller than 200")
    h = round(height / factor) * factor
    w = round(width / factor) * factor
    if h * w > max_pixels:
        beta = math.sqrt(height * width / max_pixels)
        h = max(factor, math.floor(height / beta / factor) * factor)
        w = max(factor, math.floor(width / beta / factor) * factor)
    elif h * w < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h = math.ceil(height * beta / factor) * factor
        w = math.ceil(width * beta / factor) * factor
    return h, w


def _fetch(url: str) -> bytes:
    if url.startswith("data:"):
        head, _, data = url.partition(",")
        if ";base64" not in head:
            raise ImageError("data URLs must be base64")
        try:
            raw = base64.b64decode(data, validate=False)
        except (binascii.Error, ValueError) as e:
            raise ImageError(f"bad base64 image data: {e}") from e
    elif url.startswith(("http://", "https://")):
        req = urllib.request.Request(url, headers={"User-Agent": "seventy"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read(MAX_IMAGE_BYTES + 1)
        except OSError as e:
            raise ImageError(f"cannot fetch the image: {e}") from e
    else:
        raise ImageError("image URLs must be data: or http(s): URLs")
    if len(raw) > MAX_IMAGE_BYTES:
        raise ImageError(f"the image is larger than {MAX_IMAGE_BYTES // 2**20} MiB")
    return raw


def decode_image(raw: bytes) -> np.ndarray:
    """Encoded image bytes to uint8 RGB [H, W, 3]: EXIF orientation applied, transparency over white."""
    from PIL import Image, ImageOps

    try:
        im = Image.open(io.BytesIO(raw))
        im.load()
    except Exception as e:
        raise ImageError(f"cannot decode the image: {e}") from e
    im = ImageOps.exif_transpose(im)
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        base = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(base, im)
    return np.array(im.convert("RGB"), dtype=np.uint8)


def resize(pixels: np.ndarray, settings: ImageSettings) -> np.ndarray:
    height, width = pixels.shape[:2]
    h, w = smart_resize(height, width, settings.factor, settings.min_pixels, settings.max_pixels)
    if (h, w) == (height, width):
        return np.ascontiguousarray(pixels)
    t = torch.from_numpy(np.ascontiguousarray(pixels)).permute(2, 0, 1)[None]
    out = torch.nn.functional.interpolate(t, size=(h, w), mode="bicubic", antialias=True, align_corners=False)
    return out[0].permute(1, 2, 0).contiguous().numpy()


def prepare_image(pixels: np.ndarray, settings: ImageSettings) -> ImageInput:
    out = resize(pixels, settings)
    p = settings.patch_size
    digest = hashlib.sha256(np.asarray(out.shape, dtype=np.int64).tobytes() + out.tobytes()).hexdigest()
    return ImageInput(out, (1, out.shape[0] // p, out.shape[1] // p), digest)


def load_image(url: str, settings: ImageSettings) -> ImageInput:
    return prepare_image(decode_image(_fetch(url)), settings)


def image_urls(messages: list[dict]) -> list[str]:
    """Image URLs of chat messages in the order the chat template renders their pad tokens."""
    urls = []
    for m in messages:
        content = m.get("content") if isinstance(m, dict) else None
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            kind = item.get("type")
            if kind in ("video", "video_url", "input_video") or "video" in item or "video_url" in item:
                raise ImageError("video input is not supported")
            if kind in ("image_url", "image", "input_image") or "image_url" in item or "image" in item:
                ref = item.get("image_url", item.get("image"))
                url = ref.get("url") if isinstance(ref, dict) else ref
                if not isinstance(url, str) or not url:
                    raise ImageError("an image item has no URL")
                urls.append(url)
    return urls


@dataclass
class PromptImages:
    """The images of an expanded prompt: spans[i] = (offset, count) of image i's tokens."""

    images: list[ImageInput]
    spans: list[tuple[int, int]]
    positions: np.ndarray
    delta: int

    def cache_key(self, ids: list[int]) -> list[int]:
        """The prompt with every image's tokens replaced by a negative number derived from the image digest, so the
        prefix cache matches images by content."""
        key = list(ids)
        for img, (off, n) in zip(self.images, self.spans):
            tag = -1 - int(img.digest[:15], 16)
            key[off : off + n] = [tag] * n
        return key


def expand_prompt(ids: list[int], images: list[ImageInput], image_token: int, merge: int) -> tuple[list[int], PromptImages]:
    """Replace each image pad token by one pad per merged patch block; compute the rotary positions."""
    pads = [i for i, t in enumerate(ids) if t == image_token]
    if len(pads) != len(images):
        raise ImageError(f"the prompt has {len(pads)} image placeholders for {len(images)} images")
    out: list[int] = []
    spans = []
    last = 0
    for i, img in zip(pads, images):
        out += ids[last:i]
        n = img.tokens(merge)
        spans.append((len(out), n))
        out += [image_token] * n
        last = i + 1
    out += ids[last:]
    positions, delta = rope_positions(len(out), spans, [img.grid for img in images], merge)
    return out, PromptImages(images, spans, positions, delta)


def rope_positions(n: int, spans: list[tuple[int, int]], grids: list[tuple[int, int, int]],
                   merge: int) -> tuple[np.ndarray, int]:
    """Rotary positions [3, n] (int64) of a prompt with images at spans, and the delta of the tokens after it."""
    pos = np.empty((3, n), dtype=np.int64)
    cur = 0
    i = 0
    for (off, count), (t, h, w) in zip(spans, grids):
        pos[:, i:off] = cur + np.arange(off - i)
        cur += off - i
        lh, lw = h // merge, w // merge
        if t * lh * lw != count:
            raise ImageError("an image span does not match its grid")
        tt, hh, ww = np.meshgrid(np.arange(t), np.arange(lh), np.arange(lw), indexing="ij")
        pos[0, off : off + count] = cur + tt.reshape(-1)
        pos[1, off : off + count] = cur + hh.reshape(-1)
        pos[2, off : off + count] = cur + ww.reshape(-1)
        cur += max(lh, lw)
        i = off + count
    pos[:, i:] = cur + np.arange(n - i)
    delta = int(pos.max()) + 1 - n if n else 0
    return pos, delta
