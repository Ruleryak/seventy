"""The model's chat template (chat_template.jinja in the checkpoint), rendered the way the model was trained."""

from __future__ import annotations

import json
from pathlib import Path

import jinja2


class TemplateError(ValueError):
    pass


def _raise(message: str):
    raise TemplateError(message)


def _tojson(value, indent=None, separators=None, sort_keys=False, ensure_ascii=False):
    return json.dumps(value, indent=indent, separators=separators, sort_keys=sort_keys, ensure_ascii=ensure_ascii)


def template_messages(messages: list[dict]) -> list[dict]:
    """Messages as the chat template takes them. OpenAI clients send the arguments of earlier tool calls as JSON
    text; the template iterates over them as a mapping, so they are parsed (an empty value becomes {}). The
    request's own objects are not changed."""
    out = []
    for m in messages:
        calls = m.get("tool_calls") if isinstance(m, dict) else None
        if not isinstance(calls, list) or not calls:
            out.append(m)
            continue
        fixed = []
        for call in calls:
            fn = call.get("function") if isinstance(call, dict) else None
            if not isinstance(fn, dict):
                fixed.append(call)
                continue
            args = fn.get("arguments")
            if isinstance(args, str):
                if args.strip():
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError as e:
                        raise TemplateError(f"tool call {fn.get('name')!r}: the arguments are not valid JSON ({e})")
                else:
                    args = {}
            elif args is None:
                args = {}
            if not isinstance(args, dict):
                raise TemplateError(f"tool call {fn.get('name')!r}: the arguments must be a JSON object")
            fixed.append({**call, "function": {**fn, "arguments": args}})
        out.append({**m, "tool_calls": fixed})
    return out


class ChatTemplate:
    def __init__(self, model_path: str | Path):
        path = Path(model_path)
        source = None
        if (path / "chat_template.jinja").is_file():
            source = (path / "chat_template.jinja").read_text()
        elif (path / "tokenizer_config.json").is_file():
            source = json.loads((path / "tokenizer_config.json").read_text()).get("chat_template")
        if not source:
            raise FileNotFoundError(f"no chat template in {path}")
        env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
        env.filters["tojson"] = _tojson
        env.globals["raise_exception"] = _raise
        self.template = env.from_string(source)

    def render(self, messages: list[dict], tools: list[dict] | None = None, add_generation_prompt: bool = True,
               **kwargs) -> str:
        try:
            return self.template.render(messages=template_messages(messages), tools=tools or None,
                                        add_generation_prompt=add_generation_prompt, **kwargs)
        except TemplateError:
            raise
        except jinja2.TemplateError as e:
            raise TemplateError(f"the chat template failed: {e}") from e
