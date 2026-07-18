"""
Custom exceptions for domain-specific failure modes.

Why not just raise fastapi.HTTPException everywhere? Because the code
that *detects* a failure (e.g. the LLM service discovering Ollama is
unreachable) usually isn't the code that knows how to talk HTTP. By
raising a plain-Python exception with a status_code attached, the
service layer stays framework-agnostic, and main.py's exception
handlers are the single place that translates "what went wrong" into
"what HTTP response to send". See main.py for the handlers that catch
these and build the JSON error body.
"""


class AppException(Exception):
    """
    Base class for every custom exception in this app.

    - message: human-readable explanation, becomes the "message" field
      in the JSON error response.
    - status_code: HTTP status to respond with. Each subclass sets a
      sensible class-level default so callers don't have to repeat it,
      but it can still be overridden per-instance if needed.
    - detail: optional extra context (e.g. the raw Ollama error text)
      that isn't meant for end users but is useful for debugging.
    """

    status_code: int = 500

    def __init__(self, message: str, detail: str | None = None) -> None:
        self.message = message
        self.detail = detail
        super().__init__(message)


class OllamaConnectionError(AppException):
    """Raised when the Ollama server can't be reached at all."""

    status_code = 503


class ModelNotFoundError(AppException):
    """Raised when the requested model isn't available in Ollama."""

    status_code = 404


class LLMTimeoutError(AppException):
    """Raised when a model call exceeds settings.request_timeout."""

    status_code = 504


class OutputParsingError(AppException):
    """Raised when the LLM's response can't be parsed into the expected shape."""

    status_code = 500


class TokenLimitExceeded(AppException):
    """Raised when a prompt exceeds the target model's context/token limit."""

    status_code = 400


class CircuitBreakerOpenError(AppException):
    """
    Raised by OllamaService when the circuit breaker (see
    OllamaCircuitBreaker in app/services/llm_service.py) has tripped
    OPEN after repeated Ollama connection failures, and the cooldown
    period hasn't elapsed yet. Raising this immediately -- instead of
    attempting the network call and waiting for it to time out -- is the
    entire point of the circuit breaker pattern: fail fast instead of
    burning 30+ seconds per request on a backend already known to be down.
    """

    status_code = 503


# ---------------------------------------------------------------------------
# Degradation messages
# ---------------------------------------------------------------------------
# Friendlier, user-facing copy substituted in for the raw technical
# exception message in specific known failure modes (see main.py's
# app_exception_handler and dependencies.py's check_rate_limit) -- a
# caller-facing app shouldn't have to show "OllamaConnectionError: Cannot
# connect to Ollama at http://localhost:11434" when "the AI model server
# is temporarily unavailable" says the same thing without leaking
# internal wiring. The raw technical message still goes into the
# response's `detail` field for anyone who does want it (debugging,
# support tickets), it's just no longer the headline `message`.
DEGRADATION_MESSAGES: dict[str, dict[str, str | None]] = {
    "ollama_down": {
        "response": (
            "I'm currently unable to process your request because the "
            "AI model server is temporarily unavailable. Please try "
            "again in a few moments."
        ),
        "suggestion": "You can check the server status at /health",
    },
    "rate_limited": {
        "response": "You've sent too many requests. Please wait a moment before trying again.",
        "suggestion": "Rate limit resets in {reset_seconds} seconds",
    },
    "model_overloaded": {
        "response": (
            "The requested model is currently busy. Your request has "
            "been routed to an alternative model."
        ),
        "suggestion": None,
    },
}
