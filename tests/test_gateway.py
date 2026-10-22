"""Routing and failure rules, driven by FakeProvider + FakeClock (no network, no sleeping)."""

import pytest

from llm_gateway import (
    AllRoutesFailed,
    DeadlineExceeded,
    Gateway,
    Ledger,
    PermanentRequestError,
    Route,
)
from llm_gateway.testing import FakeProvider, http_error

from conftest import make_config


def gateway(clock, providers, ledger=None, **jobs):
    return Gateway(make_config(**jobs), providers, ledger=ledger, clock=clock, sleep=clock.sleep)


def test_first_route_wins_when_healthy(clock):
    a, b = FakeProvider(default="from a"), FakeProvider(default="from b")
    gw = gateway(clock, {"a": a, "b": b}, fast=["a:m1", "b:m2"])
    res = gw.chat("fast", "hi")
    assert (res.text, res.route.key) == ("from a", "a:m1")
    assert len(a.calls) == 1 and b.calls == []
    assert a.calls[0][0] == "m1"


def test_jobs_use_their_own_chains(clock):
    a, b = FakeProvider(default="a"), FakeProvider(default="b")
    gw = gateway(clock, {"a": a, "b": b}, fast=["a:m"], vision=["b:m"])
    assert gw.chat("vision", "x").text == "b"
    assert gw.chat("fast", "x").text == "a"


def test_429_is_retried_once_on_same_route(clock):
    a, b = FakeProvider([http_error(429), "recovered"]), FakeProvider()
    gw = gateway(clock, {"a": a, "b": b}, fast=["a:m", "b:m"])
    res = gw.chat("fast", "hi")
    assert res.text == "recovered" and res.route.key == "a:m"
    assert [x.outcome for x in res.attempts] == ["retryable", "ok"]
    assert clock.sleeps == [0.5] and b.calls == []


def test_retry_after_header_is_honoured_when_short(clock):
    a = FakeProvider([http_error(429, retry_after=2.0), "ok"])
    gateway(clock, {"a": a}, fast=["a:m"]).chat("fast", "hi")
    assert clock.sleeps == [2.0]


def test_long_retry_after_fails_over_instead_of_waiting(clock):
    a, b = FakeProvider([http_error(429, retry_after=120)]), FakeProvider(default="b")
    res = gateway(clock, {"a": a, "b": b}, fast=["a:m", "b:m"]).chat("fast", "hi")
    assert res.route.key == "b:m" and clock.sleeps == [] and len(a.calls) == 1


def test_5xx_retries_once_then_fails_over(clock):
    a, b = FakeProvider(default=http_error(503)), FakeProvider(default="from b")
    res = gateway(clock, {"a": a, "b": b}, fast=["a:m", "b:m"]).chat("fast", "hi")
    assert res.text == "from b"
    assert len(a.calls) == 2  # one try + exactly one retry
    assert [x.outcome for x in res.attempts] == ["retryable", "retryable", "ok"]


def test_400_does_not_fail_over(clock):
    a, b = FakeProvider([http_error(400, "max_tokens too large")]), FakeProvider()
    gw = gateway(clock, {"a": a, "b": b}, fast=["a:m", "b:m"])
    with pytest.raises(PermanentRequestError) as exc:
        gw.chat("fast", "hi")
    assert b.calls == [] and len(a.calls) == 1
    assert [x.outcome for x in exc.value.attempts] == ["permanent"]


@pytest.mark.parametrize(
    "err",
    [http_error(402, "Insufficient Balance"), http_error(429, '{"code": "insufficient_quota"}'),
     http_error(401, "bad key"), http_error(404, "model not found")],
)
def test_route_unusable_errors_fail_over_without_retry(clock, err):
    a, b = FakeProvider([err]), FakeProvider(default="from b")
    res = gateway(clock, {"a": a, "b": b}, fast=["a:m", "b:m"]).chat("fast", "hi")
    assert res.text == "from b" and len(a.calls) == 1 and clock.sleeps == []
    assert res.attempts[0].outcome == "failover"


def test_unregistered_provider_is_skipped(clock):
    b = FakeProvider(default="b")
    res = gateway(clock, {"b": b}, fast=["missing:m", "b:m"]).chat("fast", "hi")
    assert res.route.key == "b:m" and res.attempts[0].error == "provider not registered"


def test_all_routes_failed_lists_every_attempt(clock):
    a, b = FakeProvider(default=http_error(500)), FakeProvider(default=http_error(402))
    with pytest.raises(AllRoutesFailed) as exc:
        gateway(clock, {"a": a, "b": b}, fast=["a:m", "b:m"]).chat("fast", "hi")
    assert [(x.route.key, x.outcome) for x in exc.value.attempts] == [
        ("a:m", "retryable"), ("a:m", "retryable"), ("b:m", "failover")]


def test_deadline_spans_retries_and_caps_per_call_timeout(clock):
    slow = FakeProvider(clock=clock, latency_s=100)  # always hits its timeout
    backup = FakeProvider(default="never reached")
    cfg = make_config(fast=[])
    cfg.jobs["fast"] = [Route("slow", "m", timeout_s=30), Route("backup", "m")]
    cfg.deadline_s = 50
    gw = Gateway(cfg, {"slow": slow, "backup": backup}, clock=clock, sleep=clock.sleep)
    with pytest.raises(DeadlineExceeded):
        gw.chat("fast", "hi")
    # first call gets the route timeout; the retry only gets what's left (50 - 30 - 0.5 backoff)
    assert [c[2] for c in slow.calls] == [30, 19.5]
    assert backup.calls == [] and clock.now == pytest.approx(50)


def test_per_call_deadline_override(clock):
    a = FakeProvider()
    gateway(clock, {"a": a}, fast=["a:m"]).chat("fast", "hi", deadline_s=0.25)
    assert a.calls[0][2] == 0.25  # min(route timeout 30s, remaining budget)


def test_every_attempt_is_written_to_the_ledger(clock):
    ledger = Ledger()
    a, b = FakeProvider(default=http_error(502)), FakeProvider(default="ok")
    gateway(clock, {"a": a, "b": b}, ledger, fast=["a:m", "b:m"]).chat("fast", "hi")
    rows = {r["route"]: r for r in ledger.summary()}
    assert rows["a:m"]["attempts"] == 2 and rows["a:m"]["retryable"] == 2
    assert rows["b:m"]["ok"] == 1


def test_system_prompt_and_json_mode_reach_provider(clock):
    a = FakeProvider()
    gateway(clock, {"a": a}, fast=["a:m"]).chat("fast", "q", system="be brief", json_mode=True)
    req = a.calls[0][1]
    assert req.json_mode and req.messages[0] == {"role": "system", "content": "be brief"}
