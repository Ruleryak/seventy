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
            return self.template.render(messages=messages, tools=tools or None,
                                        add_generation_prompt=add_generation_prompt, **kwargs)
        except TemplateError:
            raise
        except jinja2.TemplateError as e:
            raise TemplateError(f"the chat template failed: {e}") from e
