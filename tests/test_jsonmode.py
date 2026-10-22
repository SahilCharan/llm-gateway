import pytest

from llm_gateway import Gateway, JSONValidationError, chat_json, parse_json_reply
from llm_gateway.testing import FakeProvider

from conftest import make_config


def test_parses_plain_and_fenced_json():
    assert parse_json_reply('{"a": 1}', ["a"]) == {"a": 1}
    assert parse_json_reply('```json\n{"a": 1, "b": 2}\n```', ["a", "b"]) == {"a": 1, "b": 2}


@pytest.mark.parametrize(
    "text,reason",
    [
        ("", "empty"),
        ("   ", "empty"),
        ("{}", "empty object"),
        ('{"": "x"}', "blank key"),
        ('{"  ": "x", "a": 1}', "blank key"),
        ("[1, 2]", "expected a JSON object"),
        ("Sure! Here is the JSON: {", "not valid JSON"),
        ('{"a": 1}', "missing required keys: b"),
    ],
)
def test_rejects_bad_replies(text, reason):
    with pytest.raises(JSONValidationError, match=reason):
        parse_json_reply(text, ["a", "b"] if "missing" in reason else [])


def _gw(provider):
    return Gateway(make_config(qc=["a:m"]), {"a": provider}, sleep=lambda s: None)


def test_valid_first_reply_needs_no_repair():
    p = FakeProvider(['{"verdict": "pass", "score": 0.9}'])
    out = chat_json(_gw(p), "qc", "grade this", ["verdict", "score"])
    assert out.data["verdict"] == "pass" and not out.repaired
    assert p.calls[0][1].json_mode


def test_one_repair_reprompt_fixes_missing_key():
    p = FakeProvider(['{"verdict": "pass"}', '{"verdict": "pass", "score": 1}'])
    out = chat_json(_gw(p), "qc", "grade this", ["verdict", "score"])
    assert out.data == {"verdict": "pass", "score": 1} and out.repaired
    repair_msgs = p.calls[1][1].messages
    assert repair_msgs[-2] == {"role": "assistant", "content": '{"verdict": "pass"}'}
    assert "missing required keys: score" in repair_msgs[-1]["content"]


def test_repair_is_bounded():
    p = FakeProvider(["not json", "still not json", '{"ok": 1}'])
    with pytest.raises(JSONValidationError, match="after 1 repair"):
        chat_json(_gw(p), "qc", "x", ["ok"])
    assert len(p.calls) == 2  # original + exactly one repair, never a third call
