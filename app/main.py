"""
Application entry point: builds the FastAPI app, wires up middleware,
routers, exception handlers, static/template mounts, and lifecycle
logging. Run with:

    uvicorn app.main:app --reload --port 8000
"""

import time

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import settings
from app.core.dependencies import get_llm_service
from app.core.exceptions import (
    AppException,
    CircuitBreakerOpenError,
    DEGRADATION_MESSAGES,
    LLMTimeoutError,
    ModelNotFoundError,
    OllamaConnectionError,
    OutputParsingError,
    TokenLimitExceeded,
)
from app.core.logging_config import setup_logging
from app.core.middleware import RequestLoggingMiddleware
from app.models.responses import ErrorResponse
from app.routers import analytics, chat, health, models

# Configure structlog before literally anything else runs -- every other
# module in this app does `logger = structlog.get_logger()` at import
# time, and while that call itself is safe to make before configuration
# (it returns a lazy proxy), we want the *first log line ever emitted*
# to already use the real JSON configuration rather than structlog's
# unconfigured defaults. Calling this first, before even constructing
# the FastAPI app, guarantees that.
setup_logging(settings.log_level)

logger = structlog.get_logger()

app = FastAPI(
    title="AI Inference Gateway",
    description="Intelligent multi-model LLM gateway with smart routing",
    version=settings.app_version,
)

# Recorded at import time (== process start) so /health can compute
# uptime_seconds once that endpoint is wired to real data in Phase 2.
app_start_time = time.time()


# --- CORS -------------------------------------------------------------
# Wide open for now since this is a local-only gateway with a same-repo
# UI; if this ever serves untrusted browsers this should be locked down
# to specific origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Starlette applies middleware in reverse order of `add_middleware`
# calls, but there's only one custom middleware here so ordering
# relative to CORS doesn't matter yet.
app.add_middleware(RequestLoggingMiddleware)


# --- Routers ------------------------------------------------------------
app.include_router(health.router)
app.include_router(chat.router)
app.include_router(models.router)
app.include_router(analytics.router)


# --- Static files & templates -------------------------------------------
# Mounted/instantiated even though the full chat UI doesn't exist until
# Phase 7, so the wiring is in place ahead of time.
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# --- Exception handlers ---------------------------------------------------
# One handler per custom exception type would be repetitive since they
# all produce the same JSON shape -- so we register a single function
# for every AppException subclass. FastAPI dispatches to the most
# specific handler registered for the raised exception's type, so
# registering the same handler under each subclass (rather than just
# the base class) keeps things explicit and matches the spec's
# per-exception table.
async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
    """
    Substitutes friendlier copy for the two exception types that mean
    "Ollama itself is unreachable" -- OllamaConnectionError (a live
    connection attempt failed) and CircuitBreakerOpenError (we didn't
    even attempt one, because recent failures already told us it's
    down). A caller shouldn't have to parse "OllamaConnectionError:
    Cannot connect to Ollama at http://localhost:11434" to understand
    what happened; DEGRADATION_MESSAGES["ollama_down"] says the same
    thing in plain language. The original technical message is preserved
    in `detail` for anyone who does want it.
    """
    message = exc.message
    detail = exc.detail
    if isinstance(exc, (OllamaConnectionError, CircuitBreakerOpenError)):
        degradation = DEGRADATION_MESSAGES["ollama_down"]
        detail = exc.message if detail is None else f"{exc.message} ({detail})"
        message = degradation["response"]

    body = ErrorResponse(
        error=type(exc).__name__,
        message=message,
        status_code=exc.status_code,
        detail=detail,
    )
    return JSONResponse(status_code=exc.status_code, content=body.model_dump())


for exc_type in (
    OllamaConnectionError,
    ModelNotFoundError,
    LLMTimeoutError,
    OutputParsingError,
    TokenLimitExceeded,
    CircuitBreakerOpenError,
):
    app.exception_handler(exc_type)(app_exception_handler)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Catch-all for anything we didn't anticipate. Logs the real exception
    server-side but never leaks internal details (stack traces, etc.) to
    the client -- that's what `detail=None` here is protecting against.
    """
    logger.error("unhandled_exception", error=str(exc), exc_info=exc)
    body = ErrorResponse(
        error=type(exc).__name__,
        message="An unexpected error occurred",
        status_code=500,
        detail=None,
    )
    return JSONResponse(status_code=500, content=body.model_dump())


# --- Lifecycle hooks -------------------------------------------------------
@app.on_event("startup")
async def on_startup() -> None:
    """
    Runs once, before the server starts accepting requests. We use this
    to "warm up" the gateway: confirm Ollama is actually reachable, and
    if so, force the default model into VRAM ahead of time so the very
    first real chat request isn't the one paying the multi-second model
    load cost.
    """
    service = await get_llm_service()
    if await service.health_check():
        if settings.preload_model_on_startup:
            await service.preload_model(settings.default_model)
            logger.info(
                "AI Inference Gateway started",
                ollama_connected=True,
                preloaded_model=settings.default_model,
            )
        else:
            logger.info("AI Inference Gateway started", ollama_connected=True)
    else:
        # Deliberately not raising here -- a gateway that can't reach
        # Ollama yet should still boot and serve /health as "unhealthy"
        # rather than crash-looping (Ollama might just not be started
        # yet, or the operator is troubleshooting).
        logger.warning(
            "AI Inference Gateway started in degraded mode: Ollama not available",
            ollama_connected=False,
        )


@app.on_event("shutdown")
async def on_shutdown() -> None:
    service = await get_llm_service()
    await service.close()
    logger.info("AI Inference Gateway shutting down")


# --- Root route -------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def root(request: Request) -> HTMLResponse:
    """
    Serves the chat UI: a Jinja2-rendered, Tailwind-styled ChatGPT-style
    interface (templates/chat.html) that talks to this app's own JSON
    and streaming endpoints from the browser.
    """
    return templates.TemplateResponse(request, "chat.html")
