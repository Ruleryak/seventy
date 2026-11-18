"""Splits the model's output text into reasoning, content and tool calls, as it streams.

Reasoning: with thinking on, the prompt ends with "<think>\\n", so everything until "</think>" is reasoning.
Tool calls (the format in the model's chat template):

    <tool_call>
    <function=NAME>
    <parameter=ARG>
    VALUE
    </parameter>
    </function>
    </tool_call>

Values of string parameters stay text; others are parsed as JSON when the tool's schema says they are not
strings (or when no schema is known and the value is valid JSON).
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field

THINK_END = "</think>"
CALL_START = "<tool_call>"
CALL_END = "</tool_call>"
_FUNCTION = re.compile(r"<function=([^>\n]+)>\s*(.*?)\s*(?:</function>|$)", re.S)
_PARAM = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.S)


@dataclass
class ToolCall:
    name: str
    arguments: str
    id: str = field(default_factory=lambda: f"call_{uuid.uuid4().hex[:24]}")


@dataclass
class Delta:
    reasoning: str = ""
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


def _schemas(tools: list[dict] | None) -> dict[str, dict]:
    out = {}
    for t in tools or []:
        fn = t.get("function", t)
        if isinstance(fn, dict) and fn.get("name"):
            out[fn["name"]] = (fn.get("parameters") or {}).get("properties") or {}
    return out


def _convert(value: str, schema: dict | None):
    kind = (schema or {}).get("type")
    if kind == "string" or (isinstance(kind, list) and "string" in kind and len(kind) == 1):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return value


def parse_tool_call(block: str, schemas: dict[str, dict]) -> ToolCall | None:
    m = _FUNCTION.search(block)
    if not m:
        return None
    name = m.group(1).strip()
    props = schemas.get(name, {})
    args = {p.strip(): _convert(v, props.get(p.strip())) for p, v in _PARAM.findall(m.group(2))}
    return ToolCall(name=name, arguments=json.dumps(args, ensure_ascii=False))


class OutputParser:
    def __init__(self, reasoning: bool, tools: list[dict] | None):
        self.in_reasoning = reasoning
        self.tools_enabled = bool(tools)
        self.schemas = _schemas(tools)
        self.buffer = ""
        self.in_call = False
        self.strip_newlines = False
        self.calls: list[ToolCall] = []

    def _hold(self, text: str, marker: str) -> int:
        """Length of the end of `text` that could be the start of `marker` (held back until it is known)."""
        for n in range(min(len(marker) - 1, len(text)), 0, -1):
            if marker.startswith(text[-n:]):
                return n
        return 0

    def feed(self, text: str, final: bool = False) -> Delta:
        self.buffer += text
        d = Delta()
        while self.buffer:
            if self.in_reasoning:
                i = self.buffer.find(THINK_END)
                if i >= 0:
                    d.reasoning += self.buffer[:i]
                    self.buffer = self.buffer[i + len(THINK_END) :]
                    self.in_reasoning = False
                    self.strip_newlines = True
                    continue
                keep = 0 if final else self._hold(self.buffer, THINK_END)
                d.reasoning += self.buffer[: len(self.buffer) - keep]
                self.buffer = self.buffer[len(self.buffer) - keep :]
                break
            if self.strip_newlines:
                self.buffer = self.buffer.lstrip("\n")
                if not self.buffer:
                    break
                self.strip_newlines = False
            if self.in_call:
                i = self.buffer.find(CALL_END)
                if i < 0 and not final:
                    break
                block = self.buffer[: i if i >= 0 else len(self.buffer)]
                call = parse_tool_call(block, self.schemas)
                if call is not None:
                    d.tool_calls.append(call)
                    self.calls.append(call)
                else:
                    d.content += CALL_START + block + (CALL_END if i >= 0 else "")
                self.buffer = self.buffer[i + len(CALL_END) :] if i >= 0 else ""
                self.in_call = False
                continue
            if self.tools_enabled:
                i = self.buffer.find(CALL_START)
                if i >= 0:
                    d.content += self.buffer[:i]
                    self.buffer = self.buffer[i + len(CALL_START) :]
                    self.in_call = True
                    continue
                keep = 0 if final else self._hold(self.buffer, CALL_START)
            else:
                keep = 0
            d.content += self.buffer[: len(self.buffer) - keep]
            self.buffer = self.buffer[len(self.buffer) - keep :]
            break
        if self.calls and d.content.strip() == "":
            d.content = ""
        return d
