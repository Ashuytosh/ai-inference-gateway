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
