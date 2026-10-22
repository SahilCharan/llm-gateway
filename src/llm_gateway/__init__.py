"""llm-gateway: job-routed, cost-accounted, failover-safe LLM calls."""

from .config import GatewayConfig, ProviderSpec, Route, load_config
from .errors import (
    AllRoutesFailed,
    ConfigError,
    DeadlineExceeded,
    ErrorKind,
    GatewayError,
    PermanentRequestError,
    ProviderError,
    classify_http,
)
from .gateway import Attempt, Gateway, GatewayResult, build_gateway
from .jsonmode import JSONResult, JSONValidationError, chat_json, parse_json_reply
from .ledger import CallRecord, Ledger, Price, PriceTable, format_report
from .providers import (
    ChatRequest,
    ChatResponse,
    GeminiProvider,
    OpenAICompatibleProvider,
    Provider,
    Usage,
)

__version__ = "0.1.0"

__all__ = [
    "AllRoutesFailed", "Attempt", "CallRecord", "ChatRequest", "ChatResponse", "ConfigError",
    "DeadlineExceeded", "ErrorKind", "Gateway", "GatewayConfig", "GatewayError", "GatewayResult",
    "GeminiProvider", "JSONResult", "JSONValidationError", "Ledger", "OpenAICompatibleProvider",
    "PermanentRequestError", "Price", "PriceTable", "Provider", "ProviderError", "ProviderSpec",
    "Route", "Usage", "build_gateway", "chat_json", "classify_http", "format_report",
    "load_config", "parse_json_reply",
]
