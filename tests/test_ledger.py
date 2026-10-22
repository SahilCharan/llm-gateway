import pytest

from llm_gateway import CallRecord, ChatResponse, Gateway, Ledger, Price, PriceTable, Usage, format_report
from llm_gateway.cli import main
from llm_gateway.testing import FakeProvider, http_error

from conftest import make_config


def test_cost_math_with_cached_input():
    table = PriceTable({"p:m": {"input": 0.30, "output": 1.20, "cached_input": 0.03}})
    usd = table.cost("p:m", Usage(input_tokens=1_000, output_tokens=500, cached_input_tokens=400))
    # 600 uncached * 0.30 + 400 cached * 0.03 + 500 out * 1.20, per 1M tokens
    assert usd == pytest.approx((600 * 0.30 + 400 * 0.03 + 500 * 1.20) / 1e6)
    assert usd == pytest.approx(0.000792)


def test_cached_rate_defaults_to_input_rate_and_unknown_is_none():
    table = PriceTable({"p:m": Price(input=1.0, output=2.0)})
    assert table.cost("p:m", Usage(1_000_000, 0, 500_000)) == pytest.approx(1.0)
    assert table.cost("other:m", Usage(10, 10)) is None


def test_summary_aggregates_per_job_and_route():
    ledger = Ledger()
    ledger.record(CallRecord("vision", "a:m", "ok", 100, Usage(100, 10), 0.001))
    ledger.record(CallRecord("vision", "a:m", "ok", 300, Usage(200, 20), 0.002))
    ledger.record(CallRecord("vision", "a:m", "retryable", 50, error="HTTP 503"))
    ledger.record(CallRecord("vision", "b:m", "ok", 10, Usage(5, 5), None))
    ledger.record(CallRecord("qc", "a:m", "permanent", 20, error="HTTP 400"))
    rows = {(r["job"], r["route"]): r for r in ledger.summary()}
    va = rows[("vision", "a:m")]
    assert (va["attempts"], va["ok"], va["retryable"]) == (3, 2, 1)
    assert va["usd"] == pytest.approx(0.003) and va["avg_ok_ms"] == pytest.approx(200)
    assert va["input_tokens"] == 300
    assert rows[("vision", "b:m")]["unpriced"] == 1
    assert rows[("qc", "a:m")]["permanent"] == 1 and rows[("qc", "a:m")]["usd"] == 0

    text = format_report(ledger.summary())
    assert "TOTAL" in text and "0.003000" in text and "no price" in text


def test_gateway_costs_each_successful_call():
    prices = PriceTable({"a:m": Price(input=2.0, output=8.0)})
    ok = FakeProvider([ChatResponse("x", Usage(1000, 250))])
    gw = Gateway(make_config(fast=["a:m"]), {"a": ok}, prices=prices, ledger=Ledger())
    res = gw.chat("fast", "hi")
    assert res.usd == pytest.approx((1000 * 2.0 + 250 * 8.0) / 1e6)
    assert gw.ledger.summary()[0]["usd"] == pytest.approx(res.usd)


def test_failed_attempts_cost_nothing():
    prices = PriceTable({"a:m": Price(1.0, 1.0)})
    gw = Gateway(make_config(fast=["a:m", "b:m"]),
                 {"a": FakeProvider(default=http_error(402)), "b": FakeProvider()},
                 prices=prices, ledger=Ledger())
    gw.chat("fast", "hi")
    by_route = {r["route"]: r for r in gw.ledger.summary()}
    assert by_route["a:m"]["usd"] == 0 and by_route["a:m"]["failover"] == 1


def test_report_cli_reads_ledger_file(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = str(tmp_path / "ledger.sqlite3")
    ledger = Ledger(db)
    ledger.record(CallRecord("fast", "deepseek:deepseek-chat", "ok", 420, Usage(1200, 300), 0.000456))
    ledger.close()
    assert main(["report", "--ledger", db]) == 0
    out = capsys.readouterr().out
    assert "deepseek:deepseek-chat" in out and "0.000456" in out and "1/1" in out


def test_routes_cli_shows_rollback(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LLM_ROUTE_FAST", "deepseek:deepseek-chat,openrouter:x/y")
    monkeypatch.setenv("LLM_GATEWAY_LEGACY", "1")
    monkeypatch.setenv("LLM_GATEWAY_LEGACY_ROUTE", "openai:legacy-model")
    assert main(["routes"]) == 0
    out = capsys.readouterr().out
    assert "LEGACY ROLLBACK ACTIVE" in out and "fast         openai:legacy-model" in out
