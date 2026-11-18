"""Images and videos in prompts: loading, resizing, prompt expansion and multimodal rotary positions, by the
Qwen3.5 rules.

Resize (the checkpoint's preprocessor_config.json): both sides become multiples of patch_size x merge_size (32),
the area stays within [shortest_edge, longest_edge] pixels and the aspect ratio stays as close as possible. The
uint8 image is resized with antialiased bicubic interpolation on the CPU, the same operation the reference
processor's torchvision backend runs.

Videos (video_preprocessor_config.json): frames are decoded with OpenCV and sampled uniformly at fps (default 2)
frames per second, at least min_frames and at most max_frames; all frames share one resize, whose area limit
covers all frames together; an odd last frame is repeated. Each pair of frames is one group: the vision tower
encodes it like an image whose two temporal patch frames differ, and the prompt gets "<t seconds>" (the middle of
the pair, one decimal) before the group's vision start token.

Prompt: the chat template writes one image pad token per image, and vision start, video pad, vision end per
video. Each image pad becomes tokens(grid) pad tokens, one per 2 x 2 block of patches; a video's three tokens
become, per group, the timestamp tokens, vision start, the group's pad tokens and vision end. The vision tower's
embeddings replace the pad tokens.

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
MAX_VIDEO_BYTES = 512 * 1024 * 1024


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


@dataclass(frozen=True)
class VideoSettings:
    patch_size: int = 16
    merge_size: int = 2
    temporal_patch_size: int = 2
    min_pixels: int = 4096
    max_pixels: int = 25165824
    fps: float = 2.0
    min_frames: int = 4
    max_frames: int = 768

    @staticmethod
    def from_model(path: str | Path) -> "VideoSettings":
        f = Path(path) / "video_preprocessor_config.json"
        if not f.is_file():
            return VideoSettings()
        raw = json.loads(f.read_text())
        size = raw.get("size") or {}
        return VideoSettings(
            patch_size=int(raw.get("patch_size", 16)),
            merge_size=int(raw.get("merge_size", 2)),
            temporal_patch_size=int(raw.get("temporal_patch_size", 2)),
            min_pixels=int(size.get("shortest_edge", 4096)),
            max_pixels=int(size.get("longest_edge", 25165824)),
            fps=float(raw.get("fps", 2.0)),
            min_frames=int(raw.get("min_frames", 4)),
            max_frames=int(raw.get("max_frames", 768)),
        )

    @property
    def factor(self) -> int:
        return self.patch_size * self.merge_size


@dataclass
class ImageInput:
    """A resized image or video frame pair: uint8 pixels [H, W, 3] (image) or [2, H, W, 3] (the two frames of a
    video group), its grid (1, h, w) in patches and a digest of the pixels."""

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


def video_smart_resize(frames: int, height: int, width: int, temporal: int, factor: int, min_pixels: int,
                       max_pixels: int) -> tuple[int, int]:
    if frames < temporal:
        raise ImageError(f"the video has {frames} frames; at least {temporal} are needed")
    if height < factor or width < factor:
        scale = max(factor / height, factor / width)
        height, width = int(height * scale), int(width * scale)
    if max(height, width) / min(height, width) > 200:
        raise ImageError("the video aspect ratio must be smaller than 200")
    h = round(height / factor) * factor
    w = round(width / factor) * factor
    t = round(frames / temporal) * temporal
    if t * h * w > max_pixels:
        beta = math.sqrt(frames * height * width / max_pixels)
        h = max(factor, math.floor(height / beta / factor) * factor)
        w = max(factor, math.floor(width / beta / factor) * factor)
    elif t * h * w < min_pixels:
        beta = math.sqrt(min_pixels / (frames * height * width))
        h = math.ceil(height * beta / factor) * factor
        w = math.ceil(width * beta / factor) * factor
    return h, w


def _fetch(url: str, limit: int = MAX_IMAGE_BYTES) -> bytes:
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
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read(limit + 1)
        except OSError as e:
            raise ImageError(f"cannot fetch the image: {e}") from e
    else:
        raise ImageError("image and video URLs must be data: or http(s): URLs")
    if len(raw) > limit:
        raise ImageError(f"the file is larger than {limit // 2**20} MiB")
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


def _resize_to(pixels: np.ndarray, h: int, w: int) -> np.ndarray:
    """uint8 [..., H, W, 3] to [..., h, w, 3], antialiased bicubic (the reference processor's operation)."""
    if pixels.shape[-3:-1] == (h, w):
        return np.ascontiguousarray(pixels)
    lead = pixels.shape[:-3]
    t = torch.from_numpy(np.ascontiguousarray(pixels)).reshape(-1, *pixels.shape[-3:]).permute(0, 3, 1, 2)
    out = torch.nn.functional.interpolate(t, size=(h, w), mode="bicubic", antialias=True, align_corners=False)
    return out.permute(0, 2, 3, 1).contiguous().numpy().reshape(*lead, h, w, 3)


def resize(pixels: np.ndarray, settings: ImageSettings) -> np.ndarray:
    height, width = pixels.shape[:2]
    h, w = smart_resize(height, width, settings.factor, settings.min_pixels, settings.max_pixels)
    return _resize_to(pixels, h, w)


def prepare_image(pixels: np.ndarray, settings: ImageSettings) -> ImageInput:
    out = resize(pixels, settings)
    p = settings.patch_size
    digest = hashlib.sha256(np.asarray(out.shape, dtype=np.int64).tobytes() + out.tobytes()).hexdigest()
    return ImageInput(out, (1, out.shape[0] // p, out.shape[1] // p), digest)


def load_image(url: str, settings: ImageSettings) -> ImageInput:
    return prepare_image(decode_image(_fetch(url)), settings)


def sample_indices(total: int, fps: float, settings: VideoSettings) -> list[int]:
    """Frame indices sampled at settings.fps from a video of total frames at fps."""
    n = int(total / (fps if fps > 0 else 24.0) * settings.fps)
    n = min(max(n, settings.min_frames), settings.max_frames, total)
    return np.linspace(0, total - 1, n).round().astype(int).tolist()


def decode_video(raw: bytes, settings: VideoSettings) -> tuple[np.ndarray, list[int], float]:
    """Encoded video bytes to the sampled RGB frames [n, H, W, 3], their indices and the video's frame rate."""
    import cv2
    import cv2.videoio_registry as registry

    backend = next((b for b in registry.getStreamBufferedBackends() if registry.hasBackend(b)), None)
    if backend is None:
        raise ImageError("OpenCV has no stream-buffered video backend")
    stream = io.BytesIO(raw)
    cap = cv2.VideoCapture(stream, backend, [])
    try:
        if not cap.isOpened():
            raise ImageError("cannot decode the video")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if total <= 0:
            raise ImageError("the video has no frames")
        wanted = sample_indices(total, fps, settings)
        targets = set(wanted)
        frames, got = [], []
        for idx in range(max(wanted) + 1):
            if not cap.grab():
                continue
            if idx in targets:
                ok, frame = cap.retrieve()
                if ok and frame is not None and frame.size:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    for _ in range(wanted.count(idx)):
                        frames.append(rgb)
                        got.append(idx)
    finally:
        cap.release()
        stream.close()
    if len(frames) < 1:
        raise ImageError("no video frame could be decoded")
    return np.stack(frames), got, fps if fps > 0 else 24.0


@dataclass
class Media:
    """One image (one group) or one video (one group per pair of frames, with the pair's timestamp)."""

    kind: str
    groups: list[ImageInput]
    timestamps: list[float] | None = None


def prepare_video(frames: np.ndarray, indices: list[int], fps: float, settings: VideoSettings) -> Media:
    """Sampled frames [n, H, W, 3] to frame-pair groups; timestamps in seconds."""
    tp = settings.temporal_patch_size
    n, height, width = frames.shape[:3]
    h, w = video_smart_resize(max(n, tp), height, width, tp, settings.factor, settings.min_pixels, settings.max_pixels)
    frames = _resize_to(frames, h, w)
    idx = list(indices)
    if n % tp:
        pad = tp - n % tp
        frames = np.concatenate([frames, np.repeat(frames[-1:], pad, axis=0)])
        idx += [idx[-1]] * pad
    seconds = [i / fps for i in idx]
    p = settings.patch_size
    groups, stamps = [], []
    for g in range(0, len(frames), tp):
        pair = np.ascontiguousarray(frames[g : g + tp])
        digest = hashlib.sha256(np.asarray(pair.shape, dtype=np.int64).tobytes() + pair.tobytes()).hexdigest()
        groups.append(ImageInput(pair, (1, h // p, w // p), digest))
        stamps.append((seconds[g] + seconds[g + tp - 1]) / 2)
    return Media("video", groups, stamps)


def load_video(url: str, settings: VideoSettings) -> Media:
    frames, indices, fps = decode_video(_fetch(url, MAX_VIDEO_BYTES), settings)
    return prepare_video(frames, indices, fps, settings)


def media_urls(messages: list[dict]) -> list[tuple[str, str]]:
    """(kind, URL) of the images and videos of chat messages, in the order the chat template renders them."""
    out = []
    for m in messages:
        content = m.get("content") if isinstance(m, dict) else None
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            kind = item.get("type")
            if kind in ("video_url", "video", "input_video") or "video_url" in item or "video" in item:
                ref, what = item.get("video_url", item.get("video")), "video"
            elif kind in ("image_url", "image", "input_image") or "image_url" in item or "image" in item:
                ref, what = item.get("image_url", item.get("image")), "image"
            else:
                continue
            url = ref.get("url") if isinstance(ref, dict) else ref
            if not isinstance(url, str) or not url:
                raise ImageError(f"an {what} item has no URL")
            out.append((what, url))
    return out


def template_messages(messages: list[dict]) -> list[dict]:
    """Messages with each image or video content part replaced by {"type": "image"} or {"type": "video"}, the form
    the chat template expects (the media themselves go to the engine separately)."""
    out = []
    for m in messages:
        content = m.get("content") if isinstance(m, dict) else None
        if not isinstance(content, list):
            out.append(m)
            continue
        parts = []
        for item in content:
            if isinstance(item, dict):
                kind = item.get("type")
                if kind in ("video_url", "video", "input_video") or "video_url" in item or "video" in item:
                    item = {"type": "video"}
                elif kind in ("image_url", "image", "input_image") or "image_url" in item or "image" in item:
                    item = {"type": "image"}
            parts.append(item)
        out.append({**m, "content": parts})
    return out


def image_urls(messages: list[dict]) -> list[str]:
    """Image URLs of chat messages, in template order (videos raise ImageError)."""
    items = media_urls(messages)
    if any(kind == "video" for kind, _ in items):
        raise ImageError("video input is not supported here")
    return [url for _, url in items]


@dataclass(frozen=True)
class MediaTokens:
    image: int
    video: int
    vision_start: int
    vision_end: int


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
    """expand() for a prompt with images only."""
    return expand(ids, [Media("image", [img]) for img in images], MediaTokens(image_token, -1, -1, -1), merge)


def expand(ids: list[int], media: list[Media], tokens: MediaTokens, merge: int,
           encode=None) -> tuple[list[int], PromptImages]:
    """Expand the template's image and video placeholders (see the module notes); compute the rotary positions.
    encode: text to token ids, for video timestamps."""
    out: list[int] = []
    spans: list[tuple[int, int]] = []
    groups: list[ImageInput] = []
    it = iter(media)
    i = 0
    n = len(ids)
    while i < n:
        t = ids[i]
        if t == tokens.image:
            item = next(it, None)
            if item is None or item.kind != "image":
                raise ImageError("the prompt's image and video placeholders do not match the content items")
            img = item.groups[0]
            spans.append((len(out), img.tokens(merge)))
            groups.append(img)
            out += [tokens.image] * img.tokens(merge)
            i += 1
        elif t == tokens.vision_start and ids[i + 1 : i + 3] == [tokens.video, tokens.vision_end]:
            item = next(it, None)
            if item is None or item.kind != "video":
                raise ImageError("the prompt's image and video placeholders do not match the content items")
            if encode is None:
                raise ImageError("video timestamps need the tokenizer")
            for g, stamp in zip(item.groups, item.timestamps):
                out += encode(f"<{stamp:.1f} seconds>")
                out.append(tokens.vision_start)
                spans.append((len(out), g.tokens(merge)))
                groups.append(g)
                out += [tokens.video] * g.tokens(merge)
                out.append(tokens.vision_end)
            i += 3
        else:
            if t == tokens.video:
                raise ImageError("a video pad token outside a vision start / end pair")
            out.append(t)
            i += 1
    if next(it, None) is not None:
        raise ImageError("the prompt has fewer image and video placeholders than content items")
    positions, delta = rope_positions(len(out), spans, [g.grid for g in groups], merge)
    return out, PromptImages(groups, spans, positions, delta)


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
