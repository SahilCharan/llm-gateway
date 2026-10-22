"""Per-call cost accounting: an editable price table and a SQLite ledger.

Every attempt is recorded, including failures, so the report shows not only
what each job/route cost but how often each route failed. Failed attempts
are logged at $0: providers generally do not bill rejected or errored
requests (a request that timed out client-side may still have been billed
upstream; that cost is invisible to any client).
"""

from __future__ import annotations

import sqlite3
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass

from .providers import Usage


@dataclass(frozen=True)
class Price:
    """USD per 1M tokens. cached_input defaults to the input price if unknown."""

    input: float
    output: float
    cached_input: float | None = None


class PriceTable:
    """Prices keyed by route key ("provider:model"). Edit freely; it is just data."""

    def __init__(self, prices: Mapping[str, Price | Mapping[str, float]] | None = None):
        self._prices: dict[str, Price] = {}
        for key, value in (prices or {}).items():
            self.set(key, value if isinstance(value, Price) else Price(**value))

    def set(self, route_key: str, price: Price) -> None:
        self._prices[route_key] = price

    def get(self, route_key: str) -> Price | None:
        return self._prices.get(route_key)

    def cost(self, route_key: str, usage: Usage) -> float | None:
        """USD for one call, or None if the route has no price (reported as 'unpriced')."""
        price = self._prices.get(route_key)
        if price is None:
            return None
        cached = min(usage.cached_input_tokens, usage.input_tokens)
        cached_rate = price.input if price.cached_input is None else price.cached_input
        return (
            (usage.input_tokens - cached) * price.input
            + cached * cached_rate
            + usage.output_tokens * price.output
        ) / 1_000_000


@dataclass(frozen=True)
class CallRecord:
    job: str
    route: str
    outcome: str  # "ok" | "retryable" | "failover" | "permanent"
    latency_ms: float
    usage: Usage = Usage()
    usd: float | None = 0.0
    error: str | None = None
    ts: float = 0.0


_SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    id INTEGER PRIMARY KEY,
    ts REAL NOT NULL,
    job TEXT NOT NULL,
    route TEXT NOT NULL,
    outcome TEXT NOT NULL,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cached_input_tokens INTEGER NOT NULL DEFAULT 0,
    latency_ms REAL NOT NULL,
    usd REAL,
    error TEXT
);
CREATE INDEX IF NOT EXISTS calls_job_route ON calls (job, route);
"""

_SUMMARY_SQL = """
SELECT job, route,
       COUNT(*)                                          AS attempts,
       SUM(outcome = 'ok')                               AS ok,
       SUM(outcome = 'retryable')                        AS retryable,
       SUM(outcome = 'failover')                         AS failover,
       SUM(outcome = 'permanent')                        AS permanent,
       SUM(input_tokens)                                 AS input_tokens,
       SUM(output_tokens)                                AS output_tokens,
       COALESCE(SUM(CASE WHEN outcome = 'ok' THEN usd END), 0) AS usd,
       SUM(outcome = 'ok' AND usd IS NULL)               AS unpriced,
       AVG(CASE WHEN outcome = 'ok' THEN latency_ms END) AS avg_ok_ms
FROM calls WHERE ts >= ?
GROUP BY job, route
ORDER BY job, usd DESC, route
"""


class Ledger:
    def __init__(self, path: str = ":memory:"):
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)

    def record(self, rec: CallRecord) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO calls (ts, job, route, outcome, input_tokens, output_tokens,"
                " cached_input_tokens, latency_ms, usd, error) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    rec.ts or time.time(), rec.job, rec.route, rec.outcome,
                    rec.usage.input_tokens, rec.usage.output_tokens, rec.usage.cached_input_tokens,
                    rec.latency_ms, rec.usd, rec.error,
                ),
            )

    def summary(self, since_ts: float = 0.0) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._conn.execute(_SUMMARY_SQL, (since_ts,))]

    def close(self) -> None:
        self._conn.close()


_COLUMNS = [
    ("job", "job", "<"), ("route", "route", "<"), ("ok/att", None, ">"),
    ("retry", "retryable", ">"), ("failover", "failover", ">"), ("perm", "permanent", ">"),
    ("in_tok", "input_tokens", ">"), ("out_tok", "output_tokens", ">"),
    ("usd", None, ">"), ("usd/ok", None, ">"), ("avg_ms", None, ">"),
]


def format_report(rows: list[dict]) -> str:
    if not rows:
        return "no calls recorded"

    def cells(r: dict) -> list[str]:
        ok = r["ok"] or 0
        usd = r["usd"] or 0.0
        out = []
        for title, key, _ in _COLUMNS:
            if title == "ok/att":
                out.append(f"{ok}/{r['attempts']}")
            elif title == "usd":
                out.append(f"{usd:.6f}" + ("*" if r.get("unpriced") else ""))
            elif title == "usd/ok":
                out.append(f"{usd / ok:.6f}" if ok else "-")
            elif title == "avg_ms":
                out.append(f"{r['avg_ok_ms']:.0f}" if r.get("avg_ok_ms") is not None else "-")
            elif title in ("job", "route"):
                out.append(str(r[key]))
            else:
                out.append(str(r[key] or 0))
        return out

    total: dict = {"job": "TOTAL", "route": ""}
    for key in ("attempts", "ok", "retryable", "failover", "permanent",
                "input_tokens", "output_tokens", "usd", "unpriced"):
        total[key] = sum(r[key] or 0 for r in rows)
    timed = [(r["avg_ok_ms"], r["ok"]) for r in rows if r["avg_ok_ms"] is not None]
    total["avg_ok_ms"] = sum(a * n for a, n in timed) / sum(n for _, n in timed) if timed else None
    body = [cells(r) for r in rows]
    table = [[c[0] for c in _COLUMNS], *body, cells(total)]
    widths = [max(len(row[i]) for row in table) for i in range(len(_COLUMNS))]

    def line(row: list[str]) -> str:
        return "  ".join(f"{v:{a}{w}}" for v, (_, _, a), w in zip(row, _COLUMNS, widths)).rstrip()

    rule = "-" * len(line(table[0]))
    lines = [line(table[0]), rule, *map(line, body), rule, line(table[-1])]
    if total["unpriced"]:
        lines.append("* some successful calls had no price in the price table")
    return "\n".join(lines)
