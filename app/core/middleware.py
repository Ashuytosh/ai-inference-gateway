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

    Phase 6 addition: this middleware also binds request_id/method/path/
    client_ip into structlog's contextvars *before* the request is
    dispatched to its route handler. structlog.contextvars.bind_contextvars
    stashes those values in a context-local (per-async-task) store that
    every other structlog logger.info()/warning()/error() call picks up
    automatically via the merge_contextvars processor (see
    app/core/logging_config.py) -- so a log line emitted deep inside
    router_service.py's classify_smart(), or llm_service.py's chat(),
    ends up carrying the same request_id as this middleware's own
    request_completed line, without those modules needing to know an
    HTTP request (or this middleware) exists at all. That's what makes
    "grep one request_id, see the whole request's story" actually work.
    """

    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())

        # Reset first -- contextvars persist across `await` boundaries
        # within the same task, but under some ASGI server configurations
        # tasks can be reused; clearing defensively guarantees a stale
        # request_id from a previous request can never leak into this
        # one's log lines.
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            client_ip=request.client.host if request.client else "unknown",
        )

        # time.perf_counter() is monotonic and unaffected by system
        # clock adjustments, which makes it the right tool for measuring
        # elapsed duration (as opposed to time.time(), which is meant
        # for wall-clock timestamps).
        start_time = time.perf_counter()

        # Content-Length rather than actually reading request.body(): the
        # spec this middleware is based on explicitly warns that reading
        # the body in middleware can interfere with FastAPI's own body
        # parsing downstream. The header gives us the byte count for
        # free, with zero risk of consuming a stream FastAPI still needs
        # to read itself (and it's simply absent/0 for GET requests,
        # which is the correct answer for them anyway).
        body_size = int(request.headers.get("content-length") or 0)

        try:
            response = await call_next(request)
        except Exception as exc:
            # In practice, most application errors (AppException
            # subclasses, HTTPException) never reach this branch --
            # FastAPI's own exception-handler middleware sits *inside*
            # this middleware in the Starlette stack (user-added
            # middlewares wrap the router + its exception handling), so
            # call_next() has usually already converted those into a
            # normal Response by the time it returns. This branch mainly
            # guards against something failing outside that handling
            # entirely (e.g. a bug in another middleware) -- still worth
            # logging and re-raising rather than silently swallowing.
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                "request_failed",
                error=str(exc),
                error_type=type(exc).__name__,
                latency_ms=round(latency_ms, 2),
                body_size_bytes=body_size,
            )
            raise

        latency_ms = (time.perf_counter() - start_time) * 1000

        # Exposing these on the response lets a client correlate their
        # own request with our server-side logs, and see server-measured
        # latency without needing to time the round-trip themselves.
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Latency-MS"] = str(round(latency_ms, 2))

        logger.info(
            "request_completed",
            status_code=response.status_code,
            latency_ms=round(latency_ms, 2),
            body_size_bytes=body_size,
        )

        return response
