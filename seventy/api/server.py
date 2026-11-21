"""OpenAI-compatible HTTP API over the engine process."""

from __future__ import annotations

import asyncio
import json
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from tokenizers import Tokenizer

from .. import __version__, multimodal
from ..config import load_config
from ..engine.engine import EngineConfig
from ..engine.sampling import SamplingParams
from .chat import ChatTemplate, TemplateError
from .core import EngineClient
from .detok import IncrementalDecoder
from .parsing import OutputParser
from .prefix_tokens import PrefixTokenizer


@dataclass
class ServerConfig:
    engine: EngineConfig
    served_model_name: str
    api_key: str | None = None
    default_chat_template_kwargs: dict = field(default_factory=dict)
    max_tokens_default: int | None = None
    max_image_pixels: int | None = None
    max_images_per_request: int = 20
    max_videos_per_request: int = 1


def _generation_defaults(model_path: Path) -> dict:
    path = model_path / "generation_config.json"
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text())
    return {k: raw[k] for k in ("temperature", "top_p", "top_k", "min_p", "repetition_penalty") if k in raw}


def _stop_list(stop) -> list[str]:
    if stop is None:
        return []
    return [stop] if isinstance(stop, str) else [s for s in stop if s]


class StopFilter:
    """Holds back text that could be the start of a stop string; reports when a stop string appears."""

    def __init__(self, stops: list[str]):
        self.stops = stops
        self.held = ""
        self.hit = False

    def feed(self, text: str, final: bool = False) -> str:
        if not self.stops:
            return text
        self.held += text
        for s in self.stops:
            i = self.held.find(s)
            if i >= 0:
                out, self.held, self.hit = self.held[:i], "", True
                return out
        keep = 0 if final else max(len(s) - 1 for s in self.stops)
        keep = min(keep, len(self.held))
        out, self.held = self.held[: len(self.held) - keep], self.held[len(self.held) - keep :]
        return out


def create_app(config: ServerConfig) -> FastAPI:
    model_path = Path(config.engine.model)
    tokenizer = Tokenizer.from_file(str(model_path / "tokenizer.json"))
    prompt_tokens = PrefixTokenizer(tokenizer)
    template = ChatTemplate(model_path)
    defaults = _generation_defaults(model_path)
    client = EngineClient(config.engine)
    max_len = config.engine.max_model_len
    model_cfg = load_config(model_path)
    images_cfg = multimodal.ImageSettings.from_model(model_path)
    videos_cfg = multimodal.VideoSettings.from_model(model_path)
    media_tokens = multimodal.MediaTokens(model_cfg.image_token_id, model_cfg.video_token_id,
                                          model_cfg.vision_start_token_id, model_cfg.vision_end_token_id) \
        if model_cfg.image_token_id is not None else None
    if config.max_image_pixels:
        import dataclasses

        images_cfg = dataclasses.replace(images_cfg, max_pixels=max(config.max_image_pixels, images_cfg.factor ** 2),
                                         min_pixels=min(images_cfg.min_pixels, config.max_image_pixels))

    @asynccontextmanager
    async def lifespan(app):
        await client.start()
        yield
        await client.shutdown()

    app = FastAPI(lifespan=lifespan, title="seventy")
    app.state.client = client

    def check_key(request: Request) -> None:
        if config.api_key and not secrets.compare_digest(request.headers.get("authorization", "").encode(),
                                                         f"Bearer {config.api_key}".encode()):
            raise HTTPException(status_code=401, detail="invalid API key")

    def error(status: int, message: str, kind: str = "invalid_request_error") -> JSONResponse:
        return JSONResponse(status_code=status, content={"error": {"message": message, "type": kind, "code": status}})

    def sampling(body: dict, prompt_len: int) -> SamplingParams:
        g = {**defaults}
        for k in ("temperature", "top_p", "top_k", "min_p", "repetition_penalty"):
            if body.get(k) is not None:
                g[k] = body[k]
        limit = body.get("max_completion_tokens") or body.get("max_tokens") or config.max_tokens_default
        room = max_len - prompt_len
        max_tokens = min(int(limit), room) if limit else room
        n_lp = None
        if body.get("logprobs") is True:
            n_lp = int(body.get("top_logprobs") or 0)
        elif isinstance(body.get("logprobs"), int) and not isinstance(body.get("logprobs"), bool):
            n_lp = int(body["logprobs"])
        p = SamplingParams(
            max_tokens=max(1, max_tokens), temperature=float(g.get("temperature", 1.0)),
            top_p=float(g.get("top_p", 1.0)), top_k=int(g.get("top_k") or 0), min_p=float(g.get("min_p") or 0.0),
            seed=body.get("seed"), stop_token_ids=tuple(body.get("stop_token_ids") or ()),
            ignore_eos=bool(body.get("ignore_eos", False)), logprobs=n_lp,
            prompt_logprobs=body.get("prompt_logprobs"),
            presence_penalty=float(body.get("presence_penalty") or 0.0),
            frequency_penalty=float(body.get("frequency_penalty") or 0.0),
            repetition_penalty=float(g.get("repetition_penalty") or 1.0), min_tokens=int(body.get("min_tokens") or 0),
        )
        p.check()
        return p

    def chat_prompt(body: dict) -> tuple[list[int], bool, list | None, multimodal.PromptImages | None]:
        """Token ids of a chat request (image tokens expanded), whether thinking is on, the tools, the images.
        Fetches, decodes and resizes images: run it in a thread."""
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list")
        urls = multimodal.media_urls(messages)
        if urls and (not client.info.get("vision") or media_tokens is None):
            raise ValueError("this model does not take images or videos")
        n_img = sum(kind == "image" for kind, _ in urls)
        n_vid = len(urls) - n_img
        if n_img > config.max_images_per_request:
            raise ValueError(f"the request has {n_img} images; the limit is {config.max_images_per_request}")
        if n_vid > config.max_videos_per_request:
            raise ValueError(f"the request has {n_vid} videos; the limit is {config.max_videos_per_request}")
        kwargs = {**config.default_chat_template_kwargs, **(body.get("chat_template_kwargs") or {})}
        if body.get("reasoning_effort"):
            kwargs.setdefault("reasoning_effort", body["reasoning_effort"])
        tools = body.get("tools") if body.get("tool_choice") != "none" else None
        text = template.render(multimodal.template_messages(messages) if urls else messages, tools=tools,
                               add_generation_prompt=True, **kwargs)
        thinking = kwargs.get("enable_thinking", True) is not False
        ids = prompt_tokens.encode(text)
        if not urls:
            return ids, thinking, tools, None
        media = [multimodal.Media("image", [multimodal.load_image(u, images_cfg)]) if kind == "image"
                 else multimodal.load_video(u, videos_cfg) for kind, u in urls]
        ids, info = multimodal.expand(ids, media, media_tokens, images_cfg.merge_size,
                                      lambda text: tokenizer.encode(text, add_special_tokens=False).ids)
        return ids, thinking, tools, info

    def usage(prompt: int, completion: int) -> dict:
        return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}

    def logprob_entry(lp) -> dict:
        tok = tokenizer.decode([lp.token], skip_special_tokens=False)
        return {"token": tok, "logprob": lp.logprob, "bytes": list(tok.encode()),
                "top_logprobs": [{"token": tokenizer.decode([t], skip_special_tokens=False), "logprob": v,
                                  "bytes": list(tokenizer.decode([t], skip_special_tokens=False).encode())}
                                 for t, v in lp.top]}

    @app.get("/health")
    async def health():
        if client.dead:
            return error(503, client.dead, "server_error")
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models(request: Request):
        check_key(request)
        return {"object": "list", "data": [{"id": config.served_model_name, "object": "model", "created": 0,
                                            "owned_by": "seventy", "max_model_len": max_len}]}

    @app.post("/tokenize")
    async def tokenize(request: Request):
        check_key(request)
        body = await request.json()
        try:
            if "messages" in body:
                ids, _, _, _ = await asyncio.to_thread(chat_prompt, body)
            else:
                ids = tokenizer.encode(body.get("prompt", ""), add_special_tokens=False).ids
        except (ValueError, TemplateError) as e:
            return error(400, str(e))
        return {"tokens": ids, "count": len(ids), "max_model_len": max_len}

    @app.post("/detokenize")
    async def detokenize(request: Request):
        check_key(request)
        body = await request.json()
        return {"prompt": tokenizer.decode(body.get("tokens", []), skip_special_tokens=False)}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        arrival = time.time()
        check_key(request)
        body = await request.json()
        if int(body.get("n") or 1) != 1:
            return error(400, "only n = 1 is supported")
        try:
            prompt, thinking, tools, images = await asyncio.to_thread(chat_prompt, body)
            params = sampling(body, len(prompt))
        except (ValueError, TemplateError) as e:
            return error(400, str(e))
        if len(prompt) >= max_len:
            return error(400, f"the prompt has {len(prompt)} tokens; the limit is {max_len - 1}")
        rid = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        stops = StopFilter(_stop_list(body.get("stop")))
        want_lp = params.logprobs is not None

        async def run():
            dec = IncrementalDecoder(tokenizer)
            parser = OutputParser(reasoning=thinking, tools=tools)
            out_tokens = 0
            async for out in client.generate(prompt, params, rid, images=images, arrival=arrival):
                out_tokens = out.output_tokens
                text = dec.push(out.new_tokens)
                if out.finished:
                    text += dec.flush()
                d = parser.feed(text, final=out.finished)
                content = stops.feed(d.content, final=out.finished)
                reason = out.finish_reason
                if stops.hit and not out.finished:
                    await client.abort(rid)
                    reason, out.finished = "stop", True
                yield d.reasoning, content, d.tool_calls, out.new_logprobs, out.finished, reason, out.prompt_tokens, out_tokens
                if out.finished:
                    return

        if not body.get("stream"):
            reasoning, content, calls, lps, reason, ptok, ctok = "", "", [], [], None, len(prompt), 0
            try:
                async for r, c, tc, lp, done, why, pt, ct in run():
                    reasoning, content, calls, lps = reasoning + r, content + c, calls + tc, lps + lp
                    reason, ptok, ctok = why, pt or ptok, ct
            except ValueError as e:
                return error(400, str(e))
            if calls and reason == "stop":
                reason = "tool_calls"
            message = {"role": "assistant", "content": content if (content or not calls) else None}
            if thinking:
                message["reasoning_content"] = reasoning
            if calls:
                message["tool_calls"] = [{"id": c.id, "type": "function",
                                          "function": {"name": c.name, "arguments": c.arguments}} for c in calls]
            choice = {"index": 0, "message": message, "finish_reason": reason,
                      "logprobs": {"content": [logprob_entry(x) for x in lps]} if want_lp else None}
            return {"id": rid, "object": "chat.completion", "created": created, "model": config.served_model_name,
                    "choices": [choice], "usage": usage(ptok, ctok)}

        include_usage = bool((body.get("stream_options") or {}).get("include_usage"))

        async def events():
            def chunk(delta: dict, reason=None, lp=None):
                ch = {"index": 0, "delta": delta, "finish_reason": reason}
                if want_lp:
                    ch["logprobs"] = {"content": [logprob_entry(x) for x in (lp or [])]}
                return "data: " + json.dumps({"id": rid, "object": "chat.completion.chunk", "created": created,
                                              "model": config.served_model_name, "choices": [ch]},
                                             ensure_ascii=False) + "\n\n"

            yield chunk({"role": "assistant", "content": ""})
            n_calls, ptok, ctok = 0, len(prompt), 0
            try:
                async for r, c, tc, lp, done, why, pt, ct in run():
                    ptok, ctok = pt or ptok, ct
                    if r:
                        yield chunk({"reasoning_content": r}, lp=lp)
                        lp = None
                    if c:
                        yield chunk({"content": c}, lp=lp)
                        lp = None
                    for call in tc:
                        yield chunk({"tool_calls": [{"index": n_calls, "id": call.id, "type": "function",
                                                     "function": {"name": call.name, "arguments": call.arguments}}]})
                        n_calls += 1
                    if done:
                        reason = "tool_calls" if (n_calls and why == "stop") else why
                        yield chunk({}, reason=reason)
            except (ValueError, RuntimeError) as e:
                yield "data: " + json.dumps({"error": {"message": str(e), "type": "server_error"}}) + "\n\n"
            if include_usage:
                yield "data: " + json.dumps({"id": rid, "object": "chat.completion.chunk", "created": created,
                                             "model": config.served_model_name, "choices": [],
                                             "usage": usage(ptok, ctok)}) + "\n\n"
            yield "data: [DONE]\n\n"

        async def guarded():
            try:
                async for e in events():
                    if await request.is_disconnected():
                        await client.abort(rid)
                        return
                    yield e
            except asyncio.CancelledError:
                await client.abort(rid)
                raise

        return StreamingResponse(guarded(), media_type="text/event-stream")

    @app.post("/v1/completions")
    async def completions(request: Request):
        arrival = time.time()
        check_key(request)
        body = await request.json()
        raw = body.get("prompt", "")
        if isinstance(raw, list) and raw and isinstance(raw[0], int):
            prompt = list(raw)
        elif isinstance(raw, list):
            if len(raw) != 1:
                return error(400, "only one prompt per request is supported")
            prompt = tokenizer.encode(raw[0], add_special_tokens=False).ids
        else:
            prompt = tokenizer.encode(raw, add_special_tokens=False).ids
        try:
            params = sampling(body, len(prompt))
        except ValueError as e:
            return error(400, str(e))
        if not prompt or len(prompt) >= max_len:
            return error(400, f"the prompt must have 1 to {max_len - 1} tokens")
        rid = f"cmpl-{uuid.uuid4().hex}"
        stops = StopFilter(_stop_list(body.get("stop")))
        dec = IncrementalDecoder(tokenizer)
        text, tokens, lps, reason, prompt_lps, ctok = "", [], [], None, None, 0
        try:
            async for out in client.generate(prompt, params, rid, arrival=arrival):
                piece = dec.push(out.new_tokens) + (dec.flush() if out.finished else "")
                text += stops.feed(piece, final=out.finished)
                tokens += out.new_tokens
                lps += out.new_logprobs
                reason, ctok = out.finish_reason, out.output_tokens
                if out.finished:
                    prompt_lps = out.prompt_logprobs
                if stops.hit and not out.finished:
                    await client.abort(rid)
                    reason = "stop"
                    break
        except ValueError as e:
            return error(400, str(e))
        if body.get("echo"):
            text = tokenizer.decode(prompt, skip_special_tokens=False) + text
        choice = {"index": 0, "text": text, "finish_reason": reason, "logprobs": None}
        if params.logprobs is not None:
            toks = [tokenizer.decode([x.token], skip_special_tokens=False) for x in lps]
            offsets, pos = [], 0
            for t in toks:
                offsets.append(pos)
                pos += len(t)
            choice["logprobs"] = {"tokens": toks, "token_logprobs": [x.logprob for x in lps],
                                  "top_logprobs": [{tokenizer.decode([t], skip_special_tokens=False): v
                                                    for t, v in x.top} for x in lps], "text_offset": offsets}
        if params.prompt_logprobs is not None and prompt_lps:
            entries = [None]
            for i, x in enumerate(prompt_lps[1:], start=1):
                if x is None:
                    entries.append(None)
                    continue
                e = {str(x.token): {"logprob": x.logprob, "rank": x.rank,
                                    "decoded_token": tokenizer.decode([x.token], skip_special_tokens=False)}}
                for r, (t, v) in enumerate(x.top, start=1):
                    e.setdefault(str(t), {"logprob": v, "rank": r,
                                          "decoded_token": tokenizer.decode([t], skip_special_tokens=False)})
                entries.append(e)
            choice["prompt_logprobs"] = entries
        return {"id": rid, "object": "text_completion", "created": int(time.time()),
                "model": config.served_model_name, "choices": [choice], "usage": usage(len(prompt), ctok)}

    @app.get("/metrics")
    async def metrics():
        s = client.stats or {}
        used = s.get("kv_blocks_total", 0) - s.get("kv_blocks_free", 0)
        lines = []

        def m(name, kind, value, help_text):
            lines.extend([f"# HELP seventy:{name} {help_text}", f"# TYPE seventy:{name} {kind}",
                          f"seventy:{name} {value}"])

        m("num_requests_running", "gauge", s.get("running", 0), "Sequences in the running batch.")
        m("num_requests_waiting", "gauge", s.get("waiting", 0), "Requests waiting for a slot.")
        m("kv_cache_usage", "gauge", (used / s["kv_blocks_total"]) if s.get("kv_blocks_total") else 0,
          "Fraction of KV cache blocks in use.")
        m("kv_cache_tokens", "gauge", s.get("kv_blocks_total", 0) * s.get("block_size", 0), "KV cache capacity.")
        m("prompt_tokens_total", "counter", s.get("prompt_tokens", 0), "Prompt tokens computed.")
        m("generation_tokens_total", "counter", s.get("generated_tokens", 0), "Tokens generated.")
        m("requests_total", "counter", s.get("requests", 0), "Requests accepted.")
        m("requests_finished_total", "counter", s.get("finished", 0), "Requests finished.")
        m("requests_aborted_total", "counter", s.get("aborted", 0), "Requests aborted.")
        m("preemptions_total", "counter", s.get("preemptions", 0), "Sequences preempted.")
        m("engine_steps_total", "counter", s.get("steps", 0), "Engine iterations.")
        m("step_time_ms", "gauge", round(s.get("step_ms", 0.0), 3), "Smoothed engine iteration time.")
        m("prefix_cache_hits_total", "counter", s.get("cache_hits", 0), "Requests that reused a prefix-cache entry.")
        m("prefix_cache_hit_tokens_total", "counter", s.get("cache_hit_tokens", 0),
          "Prompt tokens reused from the prefix cache (not computed).")
        m("spec_verify_steps_total", "counter", s.get("verify_steps", 0), "Speculative verify rows (one per sequence).")
        m("spec_draft_tokens_total", "counter", s.get("drafted_tokens", 0), "Draft tokens proposed.")
        m("spec_accepted_tokens_total", "counter", s.get("accepted_tokens", 0), "Draft tokens accepted.")
        m("vision_encodes_total", "counter", s.get("images", 0), "Images and video frame pairs encoded.")
        m("uptime_seconds", "gauge", round(s.get("uptime_s", 0.0), 1), "Seconds since the engine started.")
        lines.extend(client.ttft.lines("seventy:time_to_first_token_seconds",
                                       "Time from the HTTP request to its first token."))
        lines.extend(client.latency.lines("seventy:request_latency_seconds",
                                          "Time from the HTTP request to its last token."))
        return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")

    @app.get("/info")
    async def info(request: Request):
        check_key(request)
        return {"served_model_name": config.served_model_name, "model": str(model_path), "version": __version__,
                **client.info}

    return app


def run(config: ServerConfig, host: str, port: int) -> None:
    import uvicorn

    uvicorn.run(create_app(config), host=host, port=port, log_level="warning", timeout_graceful_shutdown=10)
