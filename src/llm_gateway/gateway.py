"""The router: walk a job's route chain under a deadline, applying the failure rules."""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from .config import GatewayConfig, Route
from .errors import (
    AllRoutesFailed,
    DeadlineExceeded,
    ErrorKind,
    PermanentRequestError,
    ProviderError,
)
from .ledger import CallRecord, Ledger, PriceTable
from .providers import ChatRequest, GeminiProvider, OpenAICompatibleProvider, Provider, Usage

Messages = str | list[dict[str, str]]


@dataclass(frozen=True)
class Attempt:
    route: Route
    outcome: str  # "ok" or an ErrorKind value
    latency_s: float
    error: str | None = None
    status: int | None = None


@dataclass
class GatewayResult:
    job: str
    text: str
    route: Route
    usage: Usage
    usd: float | None
    latency_s: float  # whole call, including retries and failovers
    attempts: list[Attempt] = field(default_factory=list)


class Gateway:
    def __init__(
        self,
        config: GatewayConfig,
        providers: Mapping[str, Provider],
        *,
        ledger: Ledger | None = None,
        prices: PriceTable | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.providers = dict(providers)
        self.ledger = ledger
        self.prices = prices if prices is not None else PriceTable(config.prices)
        self.clock = clock
        self.sleep = sleep

    def chat(
        self,
        job: str,
        messages: Messages,
        *,
        system: str | None = None,
        json_mode: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
        deadline_s: float | None = None,
    ) -> GatewayResult:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        if system:
            messages = [{"role": "system", "content": system}, *messages]
        request = ChatRequest(list(messages), temperature, max_tokens, json_mode)
        routes = self.config.routes_for(job)
        budget = self.config.deadline_s if deadline_s is None else deadline_s
        start = self.clock()
        deadline = start + budget
        attempts: list[Attempt] = []

        for route in routes:
            provider = self.providers.get(route.provider)
            if provider is None:
                attempts.append(Attempt(route, ErrorKind.FAILOVER.value, 0.0, "provider not registered"))
                continue
            for attempt_no in range(self.config.retries_per_route + 1):
                remaining = deadline - self.clock()
                if remaining <= 0:
                    raise DeadlineExceeded(
                        f"job {job!r}: {budget:g}s deadline exhausted after {len(attempts)} attempt(s)",
                        attempts,
                    )
                t0 = self.clock()
                try:
                    resp = provider.chat(route.model, request, min(route.timeout_s, remaining))
                except ProviderError as err:
                    latency = self.clock() - t0
                    attempts.append(Attempt(route, err.kind.value, latency, str(err), err.status))
                    self._record(job, route, err.kind.value, latency, Usage(), 0.0, str(err))
                    if err.kind is ErrorKind.PERMANENT:
                        # Do not fail over: the next provider would reject (and maybe bill) it too.
                        raise PermanentRequestError(
                            f"job {job!r}: {route} rejected the request: {err}", attempts
                        ) from err
                    if err.kind is ErrorKind.FAILOVER or attempt_no == self.config.retries_per_route:
                        break
                    pause = self._backoff(err, deadline)
                    if pause is None:  # waiting would cost more than moving on
                        break
                    self.sleep(pause)
                    continue
                latency = self.clock() - t0
                usd = self.prices.cost(route.key, resp.usage)
                attempts.append(Attempt(route, "ok", latency))
                self._record(job, route, "ok", latency, resp.usage, usd, None)
                return GatewayResult(
                    job, resp.text, route, resp.usage, usd, self.clock() - start, attempts
                )

        summary = "; ".join(f"{a.route} {a.outcome}: {(a.error or '')[:80]}" for a in attempts)
        raise AllRoutesFailed(f"job {job!r}: all routes failed ({summary})", attempts)

    def _backoff(self, err: ProviderError, deadline: float) -> float | None:
        pause = err.retry_after if err.retry_after is not None else self.config.backoff_s
        if pause > self.config.max_backoff_s or self.clock() + pause >= deadline:
            return None
        return pause

    def _record(self, job, route, outcome, latency_s, usage, usd, error) -> None:
        if self.ledger is not None:
            self.ledger.record(
                CallRecord(job, route.key, outcome, latency_s * 1000, usage, usd, error)
            )


def build_gateway(
    config: GatewayConfig,
    *,
    ledger: Ledger | None = None,
    env: Mapping[str, str] | None = None,
) -> Gateway:
    """Instantiate real HTTP providers from config; API keys come from env vars."""
    env = os.environ if env is None else env
    providers: dict[str, Provider] = {}
    for name, spec in config.providers.items():
        key = env.get(spec.api_key_env)
        if spec.kind == "gemini":
            providers[name] = GeminiProvider(spec.base_url, key)
        else:
            providers[name] = OpenAICompatibleProvider(spec.base_url, key)
    return Gateway(config, providers, ledger=ledger)
