import pytest

from seventy.api.chat import TemplateError, template_messages


def call(args):
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "f", "arguments": args}}]}


def test_tool_call_arguments_are_parsed_without_changing_the_request():
    msgs = [{"role": "user", "content": "hi"}, call('{"a": 1}'), call(""), call(None), call({"b": 2})]
    out = template_messages(msgs)
    assert [m["tool_calls"][0]["function"]["arguments"] for m in out[1:]] == [{"a": 1}, {}, {}, {"b": 2}]
    assert out[0] is msgs[0]
    assert msgs[1]["tool_calls"][0]["function"]["arguments"] == '{"a": 1}'


@pytest.mark.parametrize("bad", ['{"a": ', "[1, 2]", '"text"'])
def test_bad_tool_call_arguments_are_a_request_error(bad):
    with pytest.raises(TemplateError):
        template_messages([call(bad)])
