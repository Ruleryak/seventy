import json

from seventy.api.parsing import OutputParser

TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {
    "city": {"type": "string"}, "days": {"type": "integer"}, "units": {"type": "string"}}}}}]


def run(parser, pieces):
    r, c, calls = "", "", []
    for i, p in enumerate(pieces):
        d = parser.feed(p, final=(i == len(pieces) - 1))
        r, c, calls = r + d.reasoning, c + d.content, calls + d.tool_calls
    return r, c, calls


def chars(text):
    return list(text)


def test_reasoning_split_streamed_by_characters():
    r, c, _ = run(OutputParser(reasoning=True, tools=None), chars("Let me think.\n</think>\n\nThe answer is 4."))
    assert r == "Let me think.\n" and c == "The answer is 4."


def test_no_reasoning_passes_content():
    r, c, _ = run(OutputParser(reasoning=False, tools=None), ["Hello ", "<tool_call>", " world"])
    assert r == "" and c == "Hello <tool_call> world"


def test_tool_call_with_schema_types():
    text = ("ok</think>\n\nI will check.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n"
            "<parameter=days>\n3\n</parameter>\n<parameter=units>\n123\n</parameter>\n</function>\n</tool_call>")
    r, c, calls = run(OutputParser(reasoning=True, tools=TOOLS), chars(text))
    assert r == "ok"
    assert c.strip() == "I will check."
    assert len(calls) == 1 and calls[0].name == "get_weather"
    assert json.loads(calls[0].arguments) == {"city": "Paris", "days": 3, "units": "123"}


def test_two_tool_calls_and_multiline_value():
    text = ("<tool_call>\n<function=a>\n<parameter=x>\nline one\nline two\n</parameter>\n</function>\n</tool_call>\n"
            "<tool_call>\n<function=b>\n<parameter=y>\n{\"k\": [1, 2]}\n</parameter>\n</function>\n</tool_call>")
    _, c, calls = run(OutputParser(reasoning=False, tools=[{"function": {"name": "a"}}, {"function": {"name": "b"}}]),
                      [text[:30], text[30:95], text[95:]])
    assert c == ""
    assert [x.name for x in calls] == ["a", "b"]
    assert json.loads(calls[0].arguments) == {"x": "line one\nline two"}
    assert json.loads(calls[1].arguments) == {"y": {"k": [1, 2]}}


def test_unterminated_reasoning_is_reasoning():
    r, c, _ = run(OutputParser(reasoning=True, tools=None), ["still thinking </thi"])
    assert r == "still thinking </thi" and c == ""
