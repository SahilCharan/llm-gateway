"""Simulate a day's traffic through the gateway: 3 fake providers, injected failures.

No network and no API keys: providers are FakeProviders and time is a FakeClock,
so the run is deterministic and instant. Run it with:

    python examples/simulate_traffic.py
"""

from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # run without installing

from llm_gateway import (  # noqa: E402
    AllRoutesFailed,
    DeadlineExceeded,
    Gateway,
    GatewayConfig,
    Ledger,
    PermanentRequestError,
    format_report,
)
from llm_gateway.testing import FakeClock, FakeProvider, http_error  # noqa: E402

CONFIG = {
    "gateway": {"deadline_s": 20, "timeout_s": 8},
    "jobs": {
        "vision": ["alpha:vision-lite", "beta:vision-std", "gamma:vision-max"],
        "qc": ["alpha:text-lite", "gamma:text-max"],
        "fast": ["beta:text-mini", "alpha:text-lite"],
    },
    # Made-up prices (USD per 1M tokens) for made-up models.
    "prices": {
        "alpha:vision-lite": {"input": 0.15, "output": 0.60, "cached_input": 0.02},
        "alpha:text-lite": {"input": 0.10, "output": 0.40},
        "beta:vision-std": {"input": 0.80, "output": 3.20},
        "beta:text-mini": {"input": 0.20, "output": 0.80},
        "gamma:vision-max": {"input": 2.50, "output": 10.00},
        "gamma:text-max": {"input": 2.00, "output": 8.00},
    },
}


def flaky(name: str, rng: random.Random, p_429: float, p_5xx: float, credit_calls: int | None = None):
    """Reply generator with injected 429s, 5xx and an optional credit cliff (402)."""
    served = 0

    def respond(model, request):
        nonlocal served
        if "MALFORMED" in request.messages[-1]["content"]:
            return http_error(400, "invalid request: unsupported parameter")
        if credit_calls is not None and served >= credit_calls:
            return http_error(402, "Insufficient Balance")
        r = rng.random()
        if r < p_429:
            return http_error(429, "rate limit exceeded")
        if r < p_429 + p_5xx:
            return http_error(503, "upstream overloaded")
        served += 1
        return f"{name}/{model}: " + "lorem ipsum " * rng.randint(5, 60)

    return respond


def main(calls: int = 400, seed: int = 7) -> None:
    rng = random.Random(seed)
    clock = FakeClock()
    providers = {
        # cheap, rate-limited, and runs out of prepaid credit after 220 successful calls
        "alpha": FakeProvider(default=flaky("alpha", rng, 0.08, 0.03, credit_calls=220),
                              clock=clock, latency_s=lambda: rng.uniform(0.6, 1.4)),
        # mid-priced, has a bad patch of 5xx and occasional 10s+ stalls (times out at 8s)
        "beta": FakeProvider(default=flaky("beta", rng, 0.02, 0.10),
                             clock=clock, latency_s=lambda: 12 if rng.random() < 0.03 else rng.uniform(0.8, 2.0)),
        # expensive and reliable
        "gamma": FakeProvider(default=flaky("gamma", rng, 0.0, 0.01),
                              clock=clock, latency_s=lambda: rng.uniform(1.5, 3.0)),
    }
    ledger = Ledger()  # in-memory; pass a path for a persistent ledger
    gateway = Gateway(GatewayConfig.from_dict(CONFIG), providers, ledger=ledger,
                      clock=clock, sleep=clock.sleep)

    outcomes: Counter[str] = Counter()
    served_by: Counter[str] = Counter()
    for i in range(calls):
        job = rng.choices(["vision", "qc", "fast"], weights=[5, 3, 2])[0]
        prompt = f"request {i}: " + "context " * rng.randint(50, 400)
        if rng.random() < 0.01:
            prompt += " MALFORMED"
        try:
            result = gateway.chat(job, prompt)
            outcomes["served"] += 1
            outcomes["served after failover"] += result.route != gateway.config.routes_for(job)[0]
            served_by[result.route.key] += 1
        except PermanentRequestError:
            outcomes["rejected as bad request (not failed over)"] += 1
        except AllRoutesFailed:
            outcomes["all routes failed"] += 1
        except DeadlineExceeded:
            outcomes["deadline exceeded"] += 1

    print(f"simulated {calls} calls (seed={seed})")
    for k, v in outcomes.items():
        print(f"  {k:<42} {v}")
    print()
    print(format_report(ledger.summary()))

    legacy = GatewayConfig.from_dict(CONFIG).with_env(
        {"LLM_GATEWAY_LEGACY": "1", "LLM_GATEWAY_LEGACY_ROUTE": "gamma:text-max"})
    print("\nrollback check with LLM_GATEWAY_LEGACY=1:")
    for job in CONFIG["jobs"]:
        print(f"  {job:<7} -> {', '.join(map(str, legacy.routes_for(job)))}")


if __name__ == "__main__":
    main()
