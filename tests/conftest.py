"""Shared fixtures, including a scripted local HTTP stub (127.0.0.1 only, no internet)."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from llm_gateway import GatewayConfig, Route
from llm_gateway.testing import FakeClock


class StubServer:
    """Replies to POSTs from a queue of (status, json_body, headers, delay_s)."""

    def __init__(self) -> None:
        self.replies: list[tuple[int, dict, dict, float]] = []
        self.requests: list[dict] = []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", 0))
                stub.requests.append(
                    {"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()},
                     "body": json.loads(self.rfile.read(length) or b"{}")}
                )
                status, body, headers, delay = stub.replies.pop(0) if stub.replies else (500, {}, {}, 0)
                time.sleep(delay)
                payload = json.dumps(body).encode()
                try:
                    self.send_response(status)
                    for k, v in headers.items():
                        self.send_header(k, v)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass  # client gave up (timeout test)

            def log_message(self, *args) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, args=(0.05,), daemon=True)

    def reply(self, status: int, body: dict | None = None, headers: dict | None = None, delay: float = 0.0):
        self.replies.append((status, body or {}, headers or {}, delay))


@pytest.fixture
def stub():
    server = StubServer()
    server.thread.start()
    yield server
    server.server.shutdown()
    server.server.server_close()


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


def make_config(**jobs: list[str]) -> GatewayConfig:
    return GatewayConfig(jobs={j: [Route.parse(r) for r in routes] for j, routes in jobs.items()})
