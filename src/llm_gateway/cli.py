"""Command line: `llm-gateway ask | report | routes`."""

from __future__ import annotations

import argparse
import json
import sys
import time

from .config import load_config
from .errors import ConfigError, GatewayError
from .gateway import build_gateway
from .jsonmode import JSONValidationError, chat_json
from .ledger import Ledger, format_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="llm-gateway", description=__doc__)
    parser.add_argument("--config", help="TOML config (default: $LLM_GATEWAY_CONFIG or ./llm-gateway.toml)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    ask = sub.add_parser("ask", help="send one prompt through a job's route chain")
    ask.add_argument("--job", default="default")
    ask.add_argument("--system", help="optional system prompt")
    ask.add_argument("--json-keys", help="comma-separated required keys; turns on JSON mode")
    ask.add_argument("prompt")

    report = sub.add_parser("report", help="summarize spend per job/route from the ledger")
    report.add_argument("--ledger", help="ledger path (default: from config)")
    report.add_argument("--since-hours", type=float, help="only include recent calls")

    sub.add_parser("routes", help="print the resolved route chain per job")
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
    except (ConfigError, OSError) as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    if args.cmd == "routes":
        if config.legacy:
            print(f"LEGACY ROLLBACK ACTIVE: every job -> {config.legacy_route}")
        for job in sorted(config.jobs):
            print(f"{job:<12} " + " -> ".join(map(str, config.routes_for(job))))
        return 0

    if args.cmd == "report":
        since = time.time() - args.since_hours * 3600 if args.since_hours else 0.0
        ledger = Ledger(args.ledger or config.ledger_path)
        print(format_report(ledger.summary(since)))
        return 0

    gateway = build_gateway(config, ledger=Ledger(config.ledger_path))
    try:
        if args.json_keys is not None:
            keys = [k.strip() for k in args.json_keys.split(",") if k.strip()]
            out = chat_json(gateway, args.job, args.prompt, keys, system=args.system)
            result, text = out.results[-1], json.dumps(out.data, indent=2)
        else:
            result = gateway.chat(args.job, args.prompt, system=args.system)
            text = result.text
    except (GatewayError, JSONValidationError, ConfigError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(text)
    usd = "unpriced" if result.usd is None else f"${result.usd:.6f}"
    print(
        f"[{result.route} | {result.usage.input_tokens}+{result.usage.output_tokens} tok | "
        f"{usd} | {result.latency_s * 1000:.0f} ms | {len(result.attempts)} attempt(s)]",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
