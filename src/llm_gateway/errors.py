"""Error taxonomy.

Every provider failure is mapped to exactly one ErrorKind, and the gateway's
behaviour follows from the kind alone:

* RETRYABLE - transient (429, 5xx, timeouts, network). Retry once on the same
  route, then fail over to the next route.
* FAILOVER  - this route cannot serve us (out of credit, bad key, unknown
  model). Move to the next route immediately; retrying would not help.
* PERMANENT - the request itself is bad (400, 413, 422, ...). Raise. Sending
  the same bad request to the next provider would only get it billed twice.
"""

from __future__ import annotations

import enum


class ErrorKind(str, enum.Enum):
    RETRYABLE = "retryable"
    FAILOVER = "failover"
    PERMANENT = "permanent"


class ProviderError(Exception):
    def __init__(
        self,
        kind: ErrorKind,
        message: str,
        *,
        status: int | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.status = status
        self.retry_after = retry_after


class ConfigError(ValueError):
    """Invalid or incomplete gateway configuration."""


class GatewayError(Exception):
    """Raised by Gateway.chat; `attempts` lists every route attempt made."""

    def __init__(self, message: str, attempts: list | None = None) -> None:
        super().__init__(message)
        self.attempts = list(attempts or [])


class PermanentRequestError(GatewayError):
    """A provider rejected the request itself; it was not sent anywhere else."""


class AllRoutesFailed(GatewayError):
    """Every route in the job's chain failed with retryable/failover errors."""


class DeadlineExceeded(GatewayError):
    """The whole-call deadline ran out before any route succeeded."""


# Out-of-credit responses are not outages: they mean "use someone else".
# Some providers signal this with 402, others with 429 or 400 plus a message.
_BALANCE_MARKERS = (
    "insufficient balance",
    "insufficient_quota",
    "insufficient credits",
    "credit balance",
    "payment required",
)


def classify_http(status: int, body: str = "") -> ErrorKind:
    text = body.lower()
    if status == 402 or any(marker in text for marker in _BALANCE_MARKERS):
        return ErrorKind.FAILOVER
    if status in (408, 429) or status >= 500:
        return ErrorKind.RETRYABLE
    if status in (401, 403, 404):  # bad key / no access / unknown model on this route
        return ErrorKind.FAILOVER
    return ErrorKind.PERMANENT
