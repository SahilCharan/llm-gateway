"""JSON-mode helper: request JSON, validate it, and allow one bounded repair re-prompt."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass

from .gateway import Gateway, GatewayResult, Messages

_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE)


class JSONValidationError(ValueError):
    pass


def parse_json_reply(text: str, required_keys: Sequence[str] = ()) -> dict:
    """Parse a model reply into a dict, or raise JSONValidationError explaining why not."""
    match = _FENCE.match(text or "")
    body = (match.group(1) if match else text or "").strip()
    if not body:
        raise JSONValidationError("reply was empty")
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise JSONValidationError(f"reply was not valid JSON ({exc.msg})") from None
    if not isinstance(data, dict):
        raise JSONValidationError(f"expected a JSON object, got {type(data).__name__}")
    if not data:
        raise JSONValidationError("reply was an empty object")
    if any(not str(k).strip() for k in data):
        raise JSONValidationError("reply contained a blank key")
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise JSONValidationError(f"missing required keys: {', '.join(missing)}")
    return data


@dataclass
class JSONResult:
    data: dict
    results: list[GatewayResult]  # one per model call (2 if a repair was needed)

    @property
    def repaired(self) -> bool:
        return len(self.results) > 1


def chat_json(
    gateway: Gateway,
    job: str,
    messages: Messages,
    required_keys: Sequence[str] = (),
    *,
    max_repairs: int = 1,
    **kwargs,
) -> JSONResult:
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    messages = list(messages)
    results: list[GatewayResult] = []
    for attempt in range(max_repairs + 1):
        result = gateway.chat(job, messages, json_mode=True, **kwargs)
        results.append(result)
        try:
            return JSONResult(parse_json_reply(result.text, required_keys), results)
        except JSONValidationError as exc:
            if attempt == max_repairs:
                raise JSONValidationError(f"{exc} (after {attempt} repair attempt(s))") from None
            keys = ", ".join(required_keys) or "at least one non-blank key"
            messages = [
                *messages,
                {"role": "assistant", "content": result.text},
                {
                    "role": "user",
                    "content": f"Your previous reply was rejected: {exc}. Reply again with ONLY a "
                    f"JSON object (no prose, no code fences) containing: {keys}.",
                },
            ]
    raise AssertionError("unreachable")
