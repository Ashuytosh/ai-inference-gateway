"""
HTTP middleware -- code that wraps *every* request/response, regardless
of which route handled it.
"""

import time
import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

logger = structlog.get_logger()


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Logs one structured line per request and tags the response with a
    unique request ID.

    The request ID matters beyond logging: if a user reports "my request
    failed", they can give us the X-Request-ID header value and we can
    grep straight to the matching log line instead of guessing which of
    thousands of "POST /api/chat" lines was theirs.
    """

    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())

        # time.perf_counter() is monotonic and unaffected by system
        # clock adjustments, which makes it the right tool for measuring
        # elapsed duration (as opposed to time.time(), which is meant
        # for wall-clock timestamps).
        start_time = time.perf_counter()

        response = await call_next(request)

        latency_ms = (time.perf_counter() - start_time) * 1000

        # Exposing the ID on the response lets a client correlate their
        # own request with our server-side logs.
        response.headers["X-Request-ID"] = request_id

        logger.info(
            "request_completed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            latency_ms=round(latency_ms, 2),
            request_id=request_id,
        )

        return response
