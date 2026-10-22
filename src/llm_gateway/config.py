"""Job -> route-chain configuration, from a dict, a TOML file and/or env vars.

Env overrides (applied last, so ops can change routing without a deploy):

    LLM_ROUTE_<JOB>=provider:model,provider:model   ordered chain for <job>
    LLM_GATEWAY_LEGACY=1                            rollback: send every job to...
    LLM_GATEWAY_LEGACY_ROUTE=provider:model         ...this single route
    LLM_GATEWAY_LEDGER=path.sqlite3                 cost ledger location
"""

from __future__ import annotations

import dataclasses
import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import ConfigError

DEFAULT_TIMEOUT_S = 30.0


@dataclass(frozen=True)
class Route:
    provider: str
    model: str
    timeout_s: float = DEFAULT_TIMEOUT_S

    @property
    def key(self) -> str:
        return f"{self.provider}:{self.model}"

    def __str__(self) -> str:
        return self.key

    @classmethod
    def parse(cls, spec: str, timeout_s: float = DEFAULT_TIMEOUT_S) -> Route:
        # Split on the first colon only: OpenRouter ids may contain ':' (".../model:free").
        provider, sep, model = spec.strip().partition(":")
        if not sep or not provider.strip() or not model.strip():
            raise ConfigError(f"route must look like 'provider:model', got {spec!r}")
        return cls(provider.strip().lower(), model.strip(), float(timeout_s))


@dataclass(frozen=True)
class ProviderSpec:
    kind: str  # "openai" (any OpenAI-compatible chat-completions host) or "gemini"
    base_url: str
    api_key_env: str


DEFAULT_PROVIDERS: dict[str, ProviderSpec] = {
    "deepseek": ProviderSpec("openai", "https://api.deepseek.com", "DEEPSEEK_API_KEY"),
    "openrouter": ProviderSpec("openai", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "openai": ProviderSpec("openai", "https://api.openai.com/v1", "OPENAI_API_KEY"),
    "gemini": ProviderSpec(
        "gemini", "https://generativelanguage.googleapis.com/v1beta", "GEMINI_API_KEY"
    ),
}


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "on")


@dataclass
class GatewayConfig:
    jobs: dict[str, list[Route]] = field(default_factory=dict)
    providers: dict[str, ProviderSpec] = field(default_factory=lambda: dict(DEFAULT_PROVIDERS))
    prices: dict[str, Mapping[str, float]] = field(default_factory=dict)
    legacy: bool = False
    legacy_route: Route | None = None
    deadline_s: float = 60.0
    default_timeout_s: float = DEFAULT_TIMEOUT_S
    retries_per_route: int = 1
    backoff_s: float = 0.5
    max_backoff_s: float = 5.0
    ledger_path: str = "llm_gateway_ledger.sqlite3"

    def routes_for(self, job: str) -> list[Route]:
        if self.legacy:
            if self.legacy_route is None:
                raise ConfigError("legacy mode is on but no legacy_route is configured")
            return [self.legacy_route]
        routes = self.jobs.get(job.lower()) or self.jobs.get("default")
        if not routes:
            raise ConfigError(
                f"no routes for job {job!r}: add it under [jobs] or set LLM_ROUTE_{job.upper()}"
            )
        return list(routes)

    def validate(self) -> GatewayConfig:
        if self.legacy and self.legacy_route is None:
            raise ConfigError("LLM_GATEWAY_LEGACY is on but no legacy route is configured")
        if self.deadline_s <= 0 or self.retries_per_route < 0:
            raise ConfigError("deadline_s must be > 0 and retries_per_route >= 0")
        return self

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GatewayConfig:
        gw = dict(data.get("gateway", {}))
        timeout = float(gw.get("timeout_s", DEFAULT_TIMEOUT_S))

        def route(entry: Any) -> Route:
            if isinstance(entry, str):
                return Route.parse(entry, timeout)
            return Route.parse(entry["route"], entry.get("timeout_s", timeout))

        jobs = {
            name.lower(): [route(r) for r in routes]
            for name, routes in data.get("jobs", {}).items()
        }
        providers = dict(DEFAULT_PROVIDERS)
        for name, spec in data.get("providers", {}).items():
            base = providers.get(name.lower())
            if base is None and "base_url" not in spec:
                raise ConfigError(f"provider {name!r} is not built in and has no base_url")
            providers[name.lower()] = ProviderSpec(
                kind=spec.get("kind", base.kind if base else "openai"),
                base_url=spec.get("base_url", base.base_url if base else ""),
                api_key_env=spec.get(
                    "api_key_env", base.api_key_env if base else f"{name.upper()}_API_KEY"
                ),
            )
        legacy_route = gw.get("legacy_route")
        return cls(
            jobs=jobs,
            providers=providers,
            prices=dict(data.get("prices", {})),
            legacy=_truthy(gw.get("legacy", False)),
            legacy_route=Route.parse(legacy_route, timeout) if legacy_route else None,
            deadline_s=float(gw.get("deadline_s", 60.0)),
            default_timeout_s=timeout,
            retries_per_route=int(gw.get("retries_per_route", 1)),
            backoff_s=float(gw.get("backoff_s", 0.5)),
            max_backoff_s=float(gw.get("max_backoff_s", 5.0)),
            ledger_path=str(gw.get("ledger_path", "llm_gateway_ledger.sqlite3")),
        ).validate()

    @classmethod
    def from_toml(cls, path: str | Path) -> GatewayConfig:
        with open(path, "rb") as fh:
            return cls.from_dict(tomllib.load(fh))

    def with_env(self, env: Mapping[str, str] | None = None) -> GatewayConfig:
        env = os.environ if env is None else env
        cfg = dataclasses.replace(self, jobs=dict(self.jobs))
        for name, value in env.items():
            if name.startswith("LLM_ROUTE_") and value.strip():
                job = name.removeprefix("LLM_ROUTE_").lower()
                cfg.jobs[job] = [
                    Route.parse(s, cfg.default_timeout_s) for s in value.split(",") if s.strip()
                ]
        if env.get("LLM_GATEWAY_LEGACY_ROUTE", "").strip():
            cfg.legacy_route = Route.parse(env["LLM_GATEWAY_LEGACY_ROUTE"], cfg.default_timeout_s)
        if "LLM_GATEWAY_LEGACY" in env:
            cfg.legacy = _truthy(env["LLM_GATEWAY_LEGACY"])
        if env.get("LLM_GATEWAY_LEDGER"):
            cfg.ledger_path = env["LLM_GATEWAY_LEDGER"]
        return cfg.validate()


def load_config(path: str | Path | None = None, env: Mapping[str, str] | None = None) -> GatewayConfig:
    """TOML file (explicit, $LLM_GATEWAY_CONFIG, or ./llm-gateway.toml) + env overrides."""
    env = os.environ if env is None else env
    path = path or env.get("LLM_GATEWAY_CONFIG")
    if path is None and Path("llm-gateway.toml").exists():
        path = "llm-gateway.toml"
    base = GatewayConfig.from_toml(path) if path else GatewayConfig()
    return base.with_env(env)
