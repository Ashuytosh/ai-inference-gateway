"""
structlog configuration -- called exactly once, at process startup,
before any other code runs (see app/main.py).

Why this needs its own explicit setup step: structlog.get_logger() (used
everywhere else in this codebase) returns a *lazy* proxy object -- it
doesn't actually decide how to format/route a log line until the first
time you call .info()/.warning()/etc. on it. Up through Phase 5 nothing
ever called structlog.configure(), so every log line in this app has
been running on structlog's built-in defaults (a reasonably formatted
but NOT structured-JSON console renderer) rather than anything we
actually chose. This module is what makes "structured logging" true in
practice, not just in theory.
"""

import logging

import structlog


def setup_logging(log_level: str = "INFO") -> None:
    """
    Configure structlog for production-grade structured JSON logging.

    Why structlog over the standard library's `logging` module directly?
    - Plain `logging` produces unstructured text: "INFO: request completed"
      -- a human can read that, but a log aggregator (Datadog, Grafana
      Loki, the ELK stack) can't reliably parse arbitrary free-form text
      into queryable fields.
    - structlog produces one JSON object per line: {"event":
      "request_completed", "method": "POST", "latency_ms": 234, ...} --
      every field is immediately filterable/aggregatable ("show me every
      request_completed event where latency_ms > 1000") without regex
      log-scraping.

    The processor list below runs top-to-bottom on every log call,
    each stage enriching or transforming the event dict before the next:

    1. contextvars.merge_contextvars -- pulls in whatever was bound via
       structlog.contextvars.bind_contextvars() (see middleware.py's
       request_id/method/path/client_ip binding) and merges it into this
       specific log call's fields. This is what makes request-scoped
       context show up automatically in logs from completely unrelated
       modules (llm_service, router_service, prompt_service,
       output_parser) without any of them needing to know an HTTP
       request is even involved.
    2. add_log_level -- stamps the "level" field (info/warning/error/...)
       onto the event dict, since JSONRenderer alone doesn't know to
       include it.
    3. StackInfoRenderer -- renders a stack trace into the event dict
       when a log call passes stack_info=True (mirrors stdlib logging's
       own stack_info support).
    4. set_exc_info -- when a log call passes exc_info=True (or an
       actual exception instance, as main.py's unhandled_exception_handler
       does), this ensures the exception gets picked up for formatting
       by the next stage.
    5. TimeStamper(fmt="iso") -- adds an ISO 8601 "timestamp" field. ISO
       format (not a raw float or a locale-dependent string) keeps
       timestamps sortable and unambiguous across timezones, same
       reasoning as ResponseMetadata.timestamp in app/models/responses.py.
    6. JSONRenderer -- the final stage: serializes the fully-assembled
       event dict to a single JSON string. This is deliberately always
       JSON (not swapped for a pretty ConsoleRenderer in dev) so log
       output is consistent regardless of environment -- matches the
       spec's "production JSON logging" goal; a developer who wants
       pretty colored output locally can pipe stdout through a JSON
       pretty-printer instead.

    wrapper_class=make_filtering_bound_logger(log_level) is what actually
    enforces the minimum level -- e.g. with log_level="INFO", any
    .debug() call anywhere in the app becomes a true no-op (skipped
    before even reaching the processor chain above), not just hidden by
    a renderer.

    logger_factory=PrintLoggerFactory() sends the final JSON string to
    stdout via a plain print() -- the simplest possible sink, and the
    right one here since this app just runs under `uvicorn` with logs
    going to the terminal/whatever redirects it (e.g. a container
    runtime's log driver), rather than writing to a file itself.

    cache_logger_on_first_use=True is a performance optimization: once a
    given logger name's processor chain has been resolved once, structlog
    reuses it instead of re-resolving on every single log call.
    """
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(log_level)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
