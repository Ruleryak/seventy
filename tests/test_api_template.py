import pytest

from seventy.api.chat import ChatTemplate

pytestmark = pytest.mark.model


def test_template_matches_transformers(model_path):
    tr = pytest.importorskip("transformers")
    tok = tr.AutoTokenizer.from_pretrained(str(model_path))
    ours = ChatTemplate(model_path)
    tools = [{"type": "function", "function": {"name": "f", "description": "d <&>", "parameters": {
        "type": "object", "properties": {"x": {"type": "string"}}}}}]
    cases = [
        dict(messages=[{"role": "user", "content": "hi"}]),
        dict(messages=[{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}],
             enable_thinking=False),
        dict(messages=[{"role": "user", "content": "w?"},
                       {"role": "assistant", "content": "", "tool_calls": [
                           {"type": "function", "function": {"name": "f", "arguments": {"x": "é"}}}]},
                       {"role": "tool", "content": "sunny"}], tools=tools),
    ]
    for case in cases:
        kwargs = {k: v for k, v in case.items() if k not in ("messages", "tools")}
        want = tok.apply_chat_template(case["messages"], tools=case.get("tools"), tokenize=False,
                                       add_generation_prompt=True, **kwargs)
        assert ours.render(case["messages"], tools=case.get("tools"), **kwargs) == want


def test_openai_tool_call_arguments_render_as_mappings(model_path):
    ours = ChatTemplate(model_path)
    tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {}}}}]

    def history(args):
        return [{"role": "user", "content": "w?"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "f", "arguments": args}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "sunny"}]

    as_text = history('{"x": "é", "n": [1, 2]}')
    assert ours.render(as_text, tools=tools) == ours.render(history({"x": "é", "n": [1, 2]}), tools=tools)
    assert as_text[1]["tool_calls"][0]["function"]["arguments"] == '{"x": "é", "n": [1, 2]}'
    assert ours.render(history(""), tools=tools) == ours.render(history({}), tools=tools)
