"""Test doubles: a scripted FakeProvider and a FakeClock. Used by the tests and the example."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from .errors import ErrorKind, ProviderError, classify_http
from .providers import ChatRequest, ChatResponse, Usage


class FakeClock:
    """Deterministic monotonic clock; pass `clock` and `clock.sleep` to Gateway."""

    def __init__(self, start: float = 0.0):
        self.now = start
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def http_error(status: int, message: str = "", retry_after: float | None = None) -> ProviderError:
    """A ProviderError classified exactly as a real HTTP response would be."""
    return ProviderError(
        classify_http(status, message), f"HTTP {status}: {message}", status=status, retry_after=retry_after
    )


def timeout_error() -> ProviderError:
    return ProviderError(ErrorKind.RETRYABLE, "network error: timed out")


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class FakeProvider:
    """Replays a script of outcomes, one per call, then falls back to `default`.

    Each outcome is a str (reply text), a ChatResponse, an exception (raised),
    or a callable(model, request) returning one of those. If `latency_s` exceeds
    the timeout the gateway passes in, the call consumes the timeout and raises
    a retryable timeout error, like a real socket would.
    """

    def __init__(
        self,
        script: Iterable[Any] = (),
        *,
        default: Any = "ok",
        clock: FakeClock | None = None,
        latency_s: float | Callable[[], float] = 0.0,
    ):
        self.script = list(script)
        self.default = default
        self.clock = clock
        self.latency_s = latency_s
        self.calls: list[tuple[str, ChatRequest, float]] = []

    def chat(self, model: str, request: ChatRequest, timeout_s: float) -> ChatResponse:
        self.calls.append((model, request, timeout_s))
        latency = self.latency_s() if callable(self.latency_s) else self.latency_s
        if self.clock is not None:
            self.clock.advance(min(latency, timeout_s))
        if latency > timeout_s:
            raise timeout_error()
        item = self.script.pop(0) if self.script else self.default
        if callable(item) and not isinstance(item, BaseException):
            item = item(model, request)
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, ChatResponse):
            return item
        text = str(item)
        prompt = " ".join(m["content"] for m in request.messages)
        return ChatResponse(text, Usage(estimate_tokens(prompt), estimate_tokens(text)))
