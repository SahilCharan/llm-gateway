"""Provider adapters. Stdlib only (urllib); no vendor SDKs.

OpenAICompatibleProvider covers DeepSeek, OpenRouter, OpenAI and most hosts
that expose POST {base_url}/chat/completions. GeminiProvider is a minimal
text-only adapter for the native generateContent API.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol

from .errors import ErrorKind, ProviderError, classify_http


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0  # subset of input_tokens served from the provider's cache


@dataclass
class ChatRequest:
    messages: list[dict[str, str]]
    temperature: float | None = None
    max_tokens: int | None = None
    json_mode: bool = False


@dataclass
class ChatResponse:
    text: str
    usage: Usage = field(default_factory=Usage)


class Provider(Protocol):
    def chat(self, model: str, request: ChatRequest, timeout_s: float) -> ChatResponse: ...


def _retry_after(value: str | None) -> float | None:
    try:
        return max(0.0, float(value)) if value else None
    except ValueError:  # HTTP-date form; not worth parsing here
        return None


def post_json(url: str, payload: dict, headers: dict[str, str], timeout_s: float) -> dict:
    """POST JSON and return the decoded body, mapping every failure to a ProviderError."""
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read()
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", "replace")[:2000]
        raise ProviderError(
            classify_http(exc.code, text),
            f"HTTP {exc.code}: {text[:300]}",
            status=exc.code,
            retry_after=_retry_after(exc.headers.get("Retry-After")),
        ) from exc
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        raise ProviderError(ErrorKind.RETRYABLE, f"network error: {exc}") from exc
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise ProviderError(ErrorKind.RETRYABLE, "provider returned a non-JSON body") from exc


class OpenAICompatibleProvider:
    def __init__(self, base_url: str, api_key: str | None, *, headers: dict | None = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.headers = dict(headers or {})

    def chat(self, model: str, request: ChatRequest, timeout_s: float) -> ChatResponse:
        if not self.api_key:
            raise ProviderError(ErrorKind.FAILOVER, "no API key configured for this provider")
        payload: dict[str, Any] = {"model": model, "messages": request.messages}
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.json_mode:
            payload["response_format"] = {"type": "json_object"}
        data = post_json(
            f"{self.base_url}/chat/completions",
            payload,
            {"Authorization": f"Bearer {self.api_key}", **self.headers},
            timeout_s,
        )
        if isinstance(data.get("error"), dict):  # some hosts report errors inside a 200
            err = data["error"]
            status = err.get("code") if isinstance(err.get("code"), int) else 500
            raise ProviderError(classify_http(status, json.dumps(err)), str(err)[:300], status=status)
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(ErrorKind.RETRYABLE, f"unexpected response: {str(data)[:200]}") from exc
        u = data.get("usage") or {}
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or u.get(
            "prompt_cache_hit_tokens", 0  # DeepSeek's field name
        )
        return ChatResponse(
            text,
            Usage(int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0)), int(cached or 0)),
        )


class GeminiProvider:
    """Minimal text-only adapter for Gemini's generateContent endpoint."""

    def __init__(self, base_url: str, api_key: str | None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def chat(self, model: str, request: ChatRequest, timeout_s: float) -> ChatResponse:
        if not self.api_key:
            raise ProviderError(ErrorKind.FAILOVER, "no API key configured for this provider")
        system = [m["content"] for m in request.messages if m["role"] == "system"]
        payload: dict[str, Any] = {
            "contents": [
                {"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
                for m in request.messages
                if m["role"] != "system"
            ]
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system)}]}
        gen: dict[str, Any] = {}
        if request.temperature is not None:
            gen["temperature"] = request.temperature
        if request.max_tokens is not None:
            gen["maxOutputTokens"] = request.max_tokens
        if request.json_mode:
            gen["responseMimeType"] = "application/json"
        if gen:
            payload["generationConfig"] = gen
        data = post_json(
            f"{self.base_url}/models/{model}:generateContent",
            payload,
            {"x-goog-api-key": self.api_key},
            timeout_s,
        )
        try:
            parts = data["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(ErrorKind.RETRYABLE, f"unexpected response: {str(data)[:200]}") from exc
        meta = data.get("usageMetadata") or {}
        return ChatResponse(
            "".join(p.get("text", "") for p in parts),
            Usage(
                int(meta.get("promptTokenCount", 0)),
                int(meta.get("candidatesTokenCount", 0)),
                int(meta.get("cachedContentTokenCount", 0)),
            ),
        )
