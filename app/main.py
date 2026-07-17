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
from app.core.exceptions import (
    AppException,
    LLMTimeoutError,
    ModelNotFoundError,
    OllamaConnectionError,
    OutputParsingError,
    TokenLimitExceeded,
)
from app.core.middleware import RequestLoggingMiddleware
from app.models.responses import ErrorResponse
from app.routers import chat, health, models

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
    body = ErrorResponse(
        error=type(exc).__name__,
        message=exc.message,
        status_code=exc.status_code,
        detail=exc.detail,
    )
    return JSONResponse(status_code=exc.status_code, content=body.model_dump())


for exc_type in (
    OllamaConnectionError,
    ModelNotFoundError,
    LLMTimeoutError,
    OutputParsingError,
    TokenLimitExceeded,
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
    logger.info("AI Inference Gateway started")


@app.on_event("shutdown")
async def on_shutdown() -> None:
    logger.info("AI Inference Gateway shutting down")


# --- Root route -------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def root() -> str:
    """
    Minimal placeholder landing page. The real ChatGPT-style UI (see
    CLAUDE.md) is built in Phase 7 using templates/chat.html.
    """
    return "AI Inference Gateway is running. API docs at /docs"
