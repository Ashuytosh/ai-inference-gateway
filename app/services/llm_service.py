"""
LLM service layer -- everything that talks to the local Ollama server.

This module has three pieces:
  1. OllamaService: an async HTTP client wrapper around Ollama's REST API
     (POST /api/chat, GET /api/tags, GET /api/ps). Routers never call
     httpx directly -- they go through this class so the HTTP details
     (retries, error mapping, streaming line parsing) live in one place.
  2. ResponseCache: a small in-memory LRU-ish cache so repeated
     low-temperature prompts don't re-hit the LLM (temperature >= a
     threshold is inherently non-deterministic, so it's never cached).
  3. with_retry: a decorator that adds exponential-backoff retries around
     the two methods that actually hit the network for generation
     (chat/chat_stream), since Ollama can be transiently slow to answer
     while a model is loading into VRAM.

Everything here is async because uvicorn runs one event loop handling
many concurrent requests -- a blocking HTTP call (e.g. the `requests`
library) would freeze every other in-flight request on the server while
it waited on Ollama. httpx.AsyncClient gives us non-blocking I/O instead.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass

import httpx
import structlog

from app.core.exceptions import (
    LLMTimeoutError,
    ModelNotFoundError,
    OllamaConnectionError,
    OutputParsingError,
)
from app.models.responses import ModelInfo

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Model capability mapping
# ---------------------------------------------------------------------------
# Ollama itself has no concept of "what is this model good at" -- that's
# domain knowledge we own. This map is what Phase 5's query router will
# eventually use to pick a model automatically; for now it just decorates
# ModelInfo so the /api/models endpoint is informative in /docs and the
# (future) UI dropdown.
CAPABILITY_MAP: dict[str, list[str]] = {
    "gemma3:4b": ["general", "simple", "fast"],
    "phi4-mini": ["logic", "math", "reasoning", "fast"],
    "qwen2.5:7b": ["general", "complex", "analysis", "detailed"],
    "qwen2.5-coder:7b": ["code", "technical", "debugging"],
    "mistral:7b": ["creative", "conversation", "writing"],
}


def _capabilities_for(model_name: str) -> list[str]:
    """
    Unknown models (custom Modelfiles, new pulls) still get a sane
    default. Also tries the name with a trailing ":latest" stripped --
    Ollama appends that tag automatically for models pulled without an
    explicit tag (e.g. "phi4-mini" -> "phi4-mini:latest"), but
    CAPABILITY_MAP is keyed by the untagged name for those entries.
    """
    if model_name in CAPABILITY_MAP:
        return CAPABILITY_MAP[model_name]
    if model_name.endswith(":latest"):
        base_name = model_name.removesuffix(":latest")
        if base_name in CAPABILITY_MAP:
            return CAPABILITY_MAP[base_name]
    return ["general"]


# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------
@dataclass
class OllamaResponse:
    """
    Normalized result of a single (non-streaming) chat call.

    We don't hand Ollama's raw JSON back to routers because its field
    names (prompt_eval_count, eval_count, total_duration in nanoseconds)
    are implementation details of Ollama specifically -- if we ever swap
    or add another backend, only this dataclass's construction needs to
    change, not every caller.
    """

    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_duration_ms: float


@dataclass
class CacheEntry:
    """A cached response plus the wall-clock time it was stored, for TTL checks."""

    response: OllamaResponse
    timestamp: float


# ---------------------------------------------------------------------------
# Response cache
# ---------------------------------------------------------------------------
class ResponseCache:
    """
    Tiny in-memory cache for deterministic (low-temperature) chat
    responses.

    Why cache at all? Because "What is 2+2?" at temperature 0.1 asked
    twice in a row is going to produce (near-)identical output both
    times, so the second call is wasted GPU time -- we can just replay
    the first answer instantly.

    Why not cache high-temperature responses? Because a high temperature
    is a deliberate request for *variety* -- serving a stale cached
    answer would silently defeat the whole point of that setting.

    This is deliberately not a general-purpose LRU (no linked list of
    access order) -- eviction here is FIFO by insertion, not by last
    access. That's enough for "keep memory bounded"; nothing here
    depends on true LRU semantics.
    """

    def __init__(self, max_size: int = 100, ttl_seconds: int = 600) -> None:
        self.cache: dict[str, CacheEntry] = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds

    def get(self, key: str) -> OllamaResponse | None:
        entry = self.cache.get(key)
        if entry is None:
            return None
        if time.time() - entry.timestamp > self.ttl_seconds:
            # Expired -- treat exactly like a miss, but also clean up the
            # stale entry so it doesn't sit around taking up space.
            del self.cache[key]
            return None
        return entry.response

    def set(self, key: str, response: OllamaResponse) -> None:
        if len(self.cache) >= self.max_size and key not in self.cache:
            # Plain dicts preserve insertion order (Python 3.7+), so the
            # first key yielded by iter() is the oldest entry -- that's
            # our FIFO eviction victim.
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        self.cache[key] = CacheEntry(response=response, timestamp=time.time())

    @staticmethod
    def make_key(
        prompt: str,
        model: str,
        temperature: float,
        system_prompt: str | None,
        threshold: float,
    ) -> str | None:
        """
        Returns None (meaning "do not cache this") when temperature is at
        or above the threshold -- callers should treat a None key as a
        signal to skip both cache lookup and cache write entirely.
        """
        if temperature >= threshold:
            return None
        raw = f"{model}|{temperature}|{system_prompt or ''}|{prompt}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def clear(self) -> None:
        self.cache.clear()


# ---------------------------------------------------------------------------
# Retry decorator
# ---------------------------------------------------------------------------
# Errors worth retrying: they're *transient* -- a slow model load, a
# dropped connection while Ollama swaps VRAM. ModelNotFoundError is
# deliberately excluded from RETRYABLE_EXCEPTIONS: asking for a model
# that doesn't exist will still not exist on attempt 2 or 3, so retrying
# it would just burn 7 seconds (1+2+4) before failing anyway.
#
# OllamaConnectionError/LLMTimeoutError are included alongside the raw
# httpx exceptions because _post_chat() (and chat_stream()'s own
# try/except) already translate httpx.ConnectError/httpx.ReadTimeout
# into our own exception types *before* this decorator ever sees them --
# so by the time an exception reaches here it's normally already the
# translated form, not the raw httpx one.
RETRYABLE_EXCEPTIONS = (
    LLMTimeoutError,
    OllamaConnectionError,
    httpx.ConnectError,
    httpx.ReadTimeout,
)
_BACKOFF_SECONDS = (1, 2, 4)


def with_retry(func: Callable[..., Awaitable]) -> Callable[..., Awaitable]:
    """
    Async decorator: retries the wrapped coroutine up to len(_BACKOFF_SECONDS)
    extra times (3 attempts total) on transient network/timeout errors,
    sleeping longer between each attempt (1s, then 2s, then 4s) so we
    don't hammer an Ollama instance that's still busy loading a model.
    """

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        last_exc: Exception | None = None
        for attempt, delay in enumerate((0, *_BACKOFF_SECONDS), start=1):
            if delay:
                logger.warning(
                    "ollama_retry",
                    function=func.__name__,
                    attempt=attempt,
                    delay_seconds=delay,
                    error=str(last_exc),
                )
                await asyncio.sleep(delay)
            try:
                return await func(*args, **kwargs)
            except RETRYABLE_EXCEPTIONS as exc:
                last_exc = exc
        # Exhausted all attempts -- surface the last failure. httpx errors
        # are translated to our own exception types by the caller before
        # this decorator ever sees them (see _translate_errors below), so
        # by the time we get here it's always one of our AppException
        # subclasses or a raw httpx error on the very last try.
        raise last_exc  # type: ignore[misc]

    return wrapper


def with_retry_stream(
    func: Callable[..., AsyncGenerator],
) -> Callable[..., AsyncGenerator]:
    """
    Retry variant for async *generator* functions (chat_stream).

    `with_retry` above works by `await`-ing the wrapped coroutine, which
    doesn't apply here -- calling an async generator function returns a
    generator object immediately, without running any code or making any
    network call until the first `__anext__()`. So retrying has to happen
    around iteration instead of around the call.

    Retries only apply if the connection fails before any token has been
    yielded yet -- once we've already streamed partial output to the
    client, silently restarting from scratch would duplicate/garble what
    they've already seen, so a mid-stream failure is instead let through
    to the caller as-is.
    """

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        last_exc: Exception | None = None
        for attempt, delay in enumerate((0, *_BACKOFF_SECONDS), start=1):
            if delay:
                logger.warning(
                    "ollama_retry",
                    function=func.__name__,
                    attempt=attempt,
                    delay_seconds=delay,
                    error=str(last_exc),
                )
                await asyncio.sleep(delay)
            started_yielding = False
            try:
                async for item in func(*args, **kwargs):
                    started_yielding = True
                    yield item
                return
            except RETRYABLE_EXCEPTIONS as exc:
                if started_yielding:
                    raise
                last_exc = exc
        raise last_exc  # type: ignore[misc]

    return wrapper


# ---------------------------------------------------------------------------
# OllamaService
# ---------------------------------------------------------------------------
class OllamaService:
    """
    Async client for the local Ollama server. One instance is created as
    a singleton (see app/core/dependencies.py) and shared across all
    requests, which lets it track which model is currently loaded in
    VRAM without re-querying Ollama on every single call.
    """

    def __init__(self, base_url: str, timeout: int) -> None:
        self.base_url = base_url
        self.timeout = timeout
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout)
        # Best-effort local cache of "what model did we last talk to" --
        # purely informational/optimization-oriented. get_loaded_models()
        # is the source of truth; this is just so future phases can avoid
        # redundant /api/ps calls if they only care about the last model.
        self._loaded_model: str | None = None

    # -- Health / discovery -------------------------------------------------
    async def health_check(self) -> bool:
        """Used by GET /health -- True only if Ollama answers at all."""
        try:
            response = await self._client.get("/api/tags")
            return response.status_code == 200
        except (httpx.ConnectError, httpx.HTTPError):
            return False

    async def list_models(self) -> list[ModelInfo]:
        try:
            response = await self._client.get("/api/tags")
            response.raise_for_status()
            data = response.json()
        except httpx.ConnectError as exc:
            raise OllamaConnectionError(
                f"Cannot connect to Ollama at {self.base_url}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise OutputParsingError("Invalid response from Ollama") from exc

        loaded_names = set(await self.get_loaded_models())

        models: list[ModelInfo] = []
        for entry in data.get("models", []):
            name = entry["name"]
            details = entry.get("details", {})
            size_bytes = entry.get("size")
            models.append(
                ModelInfo(
                    name=name,
                    size_gb=round(size_bytes / 1_000_000_000, 1)
                    if size_bytes is not None
                    else None,
                    parameter_count=details.get("parameter_size"),
                    quantization=details.get("quantization_level"),
                    capabilities=_capabilities_for(name),
                    loaded=name in loaded_names,
                )
            )
        return models

    async def get_loaded_models(self) -> list[str]:
        """
        Queries Ollama's own VRAM-resident-model list (GET /api/ps) --
        this is the actual source of truth for "loaded", as opposed to
        guessing from what we've personally sent requests for (Ollama
        also evicts models after `keep_alive` expires, which we wouldn't
        otherwise know about).
        """
        try:
            response = await self._client.get("/api/ps")
            response.raise_for_status()
            data = response.json()
        except httpx.ConnectError as exc:
            raise OllamaConnectionError(
                f"Cannot connect to Ollama at {self.base_url}"
            ) from exc
        return [entry["name"] for entry in data.get("models", [])]

    async def get_model_status(self, model_name: str) -> ModelInfo | None:
        for model in await self.list_models():
            if model.name == model_name:
                return model
        return None

    # -- Generation -----------------------------------------------------
    @with_retry
    async def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> OllamaResponse:
        """
        Blocking (non-streaming) chat completion. Ollama still buffers
        the whole generation server-side and returns one JSON object --
        "non-streaming" here just means we get one response body instead
        of many chunked lines.
        """
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                # Ollama's option name for max output tokens is
                # num_predict, not max_tokens -- OpenAI-style naming
                # doesn't apply here.
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        data = await self._post_chat(payload)
        self._loaded_model = model

        message = data.get("message", {})
        return OllamaResponse(
            text=message.get("content", ""),
            model=data.get("model", model),
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            # Ollama reports durations in nanoseconds; /1e6 -> milliseconds.
            total_duration_ms=data.get("total_duration", 0) / 1_000_000,
        )

    @with_retry_stream
    async def chat_stream(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> AsyncGenerator[str | dict, None]:
        """
        Streaming chat completion. Ollama sends newline-delimited JSON
        (NDJSON) -- one small JSON object per line, each carrying the
        next text delta, with `done: true` on the final line carrying
        aggregate stats instead of more text.

        Yields:
          - str tokens (text deltas) while generation is in progress
          - a single dict of final metadata as the last item, once done
        """
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        try:
            async with self._client.stream(
                "POST", "/api/chat", json=payload
            ) as response:
                if response.status_code == 404:
                    raise ModelNotFoundError(f"Model '{model}' not found")
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise OutputParsingError(
                            "Invalid response from Ollama"
                        ) from exc

                    if chunk.get("done"):
                        self._loaded_model = model
                        yield {
                            "prompt_tokens": chunk.get("prompt_eval_count", 0),
                            "completion_tokens": chunk.get("eval_count", 0),
                            "total_duration_ms": chunk.get("total_duration", 0)
                            / 1_000_000,
                        }
                    else:
                        yield chunk.get("message", {}).get("content", "")
        except httpx.ConnectError as exc:
            raise OllamaConnectionError(
                f"Cannot connect to Ollama at {self.base_url}"
            ) from exc
        except httpx.ReadTimeout as exc:
            raise LLMTimeoutError("Model took too long to respond") from exc

    async def preload_model(self, model_name: str) -> bool:
        """
        Forces Ollama to load a model into VRAM ahead of the first real
        request, by sending a throwaway 1-token generation. num_predict=1
        keeps this cheap (we don't care about the output, only the
        side-effect of loading weights), and keep_alive tells Ollama how
        long to keep it resident afterward so the *next* real request
        doesn't pay the load cost again.

        Returns False (rather than raising) on failure -- a failed
        preload shouldn't prevent the server from starting; it just means
        the first real chat request will be slower than usual.
        """
        from app.config import settings  # local import avoids a config<->service import cycle

        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "options": {"num_predict": 1},
            "keep_alive": settings.keep_alive,
        }
        try:
            await self._post_chat(payload)
        except Exception as exc:  # noqa: BLE001 -- preload failure is non-fatal by design
            logger.warning("model_preload_failed", model=model_name, error=str(exc))
            return False

        logger.info("model_preloaded", model=model_name, keep_alive=settings.keep_alive)
        self._loaded_model = model_name
        return True

    async def close(self) -> None:
        """Called on FastAPI shutdown to release the underlying connection pool."""
        await self._client.aclose()

    # -- Internal helpers -------------------------------------------------
    async def _post_chat(self, payload: dict) -> dict:
        """
        Shared POST /api/chat + error-translation logic used by both
        chat() and preload_model() (chat_stream() has its own streaming
        variant above since it can't buffer the whole response first).
        """
        try:
            response = await self._client.post("/api/chat", json=payload)
        except httpx.ConnectError as exc:
            raise OllamaConnectionError(
                f"Cannot connect to Ollama at {self.base_url}"
            ) from exc
        except httpx.ReadTimeout as exc:
            raise LLMTimeoutError("Model took too long to respond") from exc

        if response.status_code == 404:
            raise ModelNotFoundError(f"Model '{payload['model']}' not found")
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise OllamaConnectionError(
                f"Ollama returned an error: {response.status_code}"
            ) from exc

        try:
            return response.json()
        except json.JSONDecodeError as exc:
            raise OutputParsingError("Invalid response from Ollama") from exc
