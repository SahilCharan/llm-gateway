"""Real HTTP adapters against a local http.server stub (127.0.0.1 only)."""

import pytest

from llm_gateway import (
    ChatRequest,
    ErrorKind,
    Gateway,
    GeminiProvider,
    OpenAICompatibleProvider,
    PermanentRequestError,
    ProviderError,
)

from conftest import make_config

REQ = ChatRequest([{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}])


def completion(text="hello", **usage):
    return {"choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, **usage}}


def test_openai_compatible_success_parses_usage_and_sends_auth(stub):
    stub.reply(200, completion(prompt_tokens_details={"cached_tokens": 4}))
    p = OpenAICompatibleProvider(stub.url + "/v1", "sk-test")
    res = p.chat("deepseek-chat", ChatRequest(REQ.messages, temperature=0, json_mode=True), 5)
    assert res.text == "hello"
    assert (res.usage.input_tokens, res.usage.output_tokens, res.usage.cached_input_tokens) == (10, 5, 4)
    sent = stub.requests[0]
    assert sent["path"] == "/v1/chat/completions"
    assert sent["headers"]["authorization"] == "Bearer sk-test"
    assert sent["body"]["response_format"] == {"type": "json_object"}
    assert sent["body"]["model"] == "deepseek-chat" and sent["body"]["temperature"] == 0


def test_deepseek_cache_hit_field_is_understood(stub):
    stub.reply(200, completion(prompt_cache_hit_tokens=7))
    res = OpenAICompatibleProvider(stub.url, "k").chat("m", REQ, 5)
    assert res.usage.cached_input_tokens == 7


@pytest.mark.parametrize(
    "status,body,kind",
    [
        (429, {"error": "slow down"}, ErrorKind.RETRYABLE),
        (503, {"error": "overloaded"}, ErrorKind.RETRYABLE),
        (400, {"error": "invalid messages"}, ErrorKind.PERMANENT),
        (422, {"error": "unprocessable"}, ErrorKind.PERMANENT),
        (402, {"error": "Insufficient Balance"}, ErrorKind.FAILOVER),
        (401, {"error": "invalid api key"}, ErrorKind.FAILOVER),
    ],
)
def test_http_errors_are_classified(stub, status, body, kind):
    stub.reply(status, body, headers={"Retry-After": "3"})
    with pytest.raises(ProviderError) as exc:
        OpenAICompatibleProvider(stub.url, "k").chat("m", REQ, 5)
    assert exc.value.kind is kind and exc.value.status == status
    assert exc.value.retry_after == 3.0


def test_error_inside_200_body_is_classified(stub):
    stub.reply(200, {"error": {"code": 402, "message": "Insufficient credits"}})
    with pytest.raises(ProviderError) as exc:
        OpenAICompatibleProvider(stub.url, "k").chat("m", REQ, 5)
    assert exc.value.kind is ErrorKind.FAILOVER


def test_read_timeout_is_retryable(stub):
    stub.reply(200, completion(), delay=1.0)
    with pytest.raises(ProviderError) as exc:
        OpenAICompatibleProvider(stub.url, "k").chat("m", REQ, 0.2)
    assert exc.value.kind is ErrorKind.RETRYABLE


def test_missing_api_key_fails_over_without_a_request(stub):
    with pytest.raises(ProviderError) as exc:
        OpenAICompatibleProvider(stub.url, None).chat("m", REQ, 5)
    assert exc.value.kind is ErrorKind.FAILOVER and stub.requests == []


def test_gateway_over_http_fails_over_after_two_503s(stub):
    stub.reply(503, {"error": "down"})
    stub.reply(503, {"error": "still down"})
    stub.reply(200, completion("from backup"))
    providers = {"primary": OpenAICompatibleProvider(stub.url, "k"),
                 "backup": OpenAICompatibleProvider(stub.url, "k")}
    gw = Gateway(make_config(fast=["primary:p-model", "backup:b-model"]), providers, sleep=lambda s: None)
    res = gw.chat("fast", "hi")
    assert res.text == "from backup"
    assert [r["body"]["model"] for r in stub.requests] == ["p-model", "p-model", "b-model"]


def test_gateway_over_http_400_is_sent_once(stub):
    stub.reply(400, {"error": "bad request"})
    providers = {"a": OpenAICompatibleProvider(stub.url, "k"), "b": OpenAICompatibleProvider(stub.url, "k")}
    gw = Gateway(make_config(fast=["a:m", "b:m"]), providers, sleep=lambda s: None)
    with pytest.raises(PermanentRequestError):
        gw.chat("fast", "hi")
    assert len(stub.requests) == 1


def test_gemini_adapter_maps_request_and_response(stub):
    stub.reply(200, {
        "candidates": [{"content": {"parts": [{"text": "{\"a\": "}, {"text": "1}"}]}}],
        "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 3, "cachedContentTokenCount": 2},
    })
    req = ChatRequest(REQ.messages + [{"role": "assistant", "content": "prev"}], max_tokens=50, json_mode=True)
    res = GeminiProvider(stub.url, "g-key").chat("gemini-flash", req, 5)
    assert res.text == '{"a": 1}' and res.usage.cached_input_tokens == 2
    sent = stub.requests[0]
    assert sent["path"] == "/models/gemini-flash:generateContent"
    assert sent["headers"]["x-goog-api-key"] == "g-key"
    assert sent["body"]["systemInstruction"] == {"parts": [{"text": "sys"}]}
    assert [c["role"] for c in sent["body"]["contents"]] == ["user", "model"]
    assert sent["body"]["generationConfig"] == {"maxOutputTokens": 50, "responseMimeType": "application/json"}
