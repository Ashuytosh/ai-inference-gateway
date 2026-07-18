"""
Chat endpoints -- the core of the gateway.

Both routes accept the same ChatRequest body. POST / returns a single
JSON response once generation is complete; POST /stream returns
Server-Sent Events (SSE) so a browser can render tokens as they arrive,
ChatGPT-style.

Model selection here is intentionally simple: request.model if the
caller specified one, else settings.default_model. Smart classification
that picks a model automatically based on the *content* of the prompt is
Phase 5's job (app/services/router_service.py) -- this phase just needs
real generation working end-to-end.
"""

import json
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.config import settings
from app.core.dependencies import get_llm_service, get_response_cache
from app.models.requests import ChatRequest
from app.models.responses import ChatResponse, ResponseMetadata
from app.services.llm_service import OllamaResponse, OllamaService, ResponseCache

router = APIRouter(prefix="/api/chat", tags=["Chat"])


def _resolve_model(request: ChatRequest) -> str:
    """None means "no preference" -- fall back to the configured default."""
    return request.model or settings.default_model


def _build_messages(request: ChatRequest) -> list[dict[str, str]]:
    """
    Ollama's /api/chat takes a messages array like OpenAI's chat format:
    an optional system message first (sets behavior/persona), then the
    user's actual prompt.
    """
    messages: list[dict[str, str]] = []
    if request.system_prompt:
        messages.append({"role": "system", "content": request.system_prompt})
    messages.append({"role": "user", "content": request.prompt})
    return messages


def _build_metadata(
    result: OllamaResponse,
    request: ChatRequest,
    latency_ms: float,
    cached: bool,
) -> ResponseMetadata:
    return ResponseMetadata(
        model_used=result.model,
        # Real classification is Phase 5 -- every query is "general" for now.
        query_type="general",
        latency_ms=round(latency_ms, 2),
        tokens_prompt=result.prompt_tokens,
        tokens_completion=result.completion_tokens,
        tokens_total=result.prompt_tokens + result.completion_tokens,
        temperature=request.temperature,
        # Real fallback (retry-with-different-model) is Phase 5 territory;
        # the retry logic in this phase only retries the *same* model.
        fallback_used=False,
        timestamp=datetime.now(timezone.utc).isoformat(),
        cached=cached,
    )


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    service: OllamaService = Depends(get_llm_service),
    cache: ResponseCache = Depends(get_response_cache),
) -> ChatResponse:
    """
    Non-streaming chat. `request` is validated by ChatRequest before this
    function body even runs -- e.g. an empty prompt or an out-of-range
    temperature never reaches this code, FastAPI returns a 422 first.
    """
    model = _resolve_model(request)
    cache_key = cache.make_key(
        prompt=request.prompt,
        model=model,
        temperature=request.temperature,
        system_prompt=request.system_prompt,
        threshold=settings.cache_temperature_threshold,
    )

    start_time = time.perf_counter()

    if cache_key is not None:
        cached_result = cache.get(cache_key)
        if cached_result is not None:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return ChatResponse(
                response=cached_result.text,
                metadata=_build_metadata(cached_result, request, latency_ms, cached=True),
            )

    result = await service.chat(
        model=model,
        messages=_build_messages(request),
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )
    latency_ms = (time.perf_counter() - start_time) * 1000

    if cache_key is not None:
        cache.set(cache_key, result)

    return ChatResponse(
        response=result.text,
        metadata=_build_metadata(result, request, latency_ms, cached=False),
    )


async def _sse_token_stream(
    request: ChatRequest, service: OllamaService, model: str
):
    """
    Generator yielding Server-Sent Events.

    SSE format is `data: <json>\n\n` per event -- the blank line is what
    tells the browser's EventSource (or our own fetch-based reader) that
    one event has ended. We yield partial tokens with done=false, then a
    final empty-token event with done=true carrying the full metadata,
    so the client knows exactly when to stop appending text and instead
    render the metadata badges.

    Streamed responses are never cached (see ResponseCache docstring --
    caching is about replaying a *complete* answer instantly; a stream is
    already "instant enough" token-by-token, and buffering a whole
    streamed reply just to maybe cache it would add latency to every
    streaming request for a benefit that mainly matters for blocking
    calls).
    """
    start_time = time.perf_counter()
    prompt_tokens = 0
    completion_tokens = 0

    async for item in service.chat_stream(
        model=model,
        messages=_build_messages(request),
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    ):
        if isinstance(item, dict):
            # Final chunk: aggregate stats instead of more text.
            prompt_tokens = item["prompt_tokens"]
            completion_tokens = item["completion_tokens"]
        else:
            yield f"data: {json.dumps({'token': item, 'done': False})}\n\n"

    latency_ms = (time.perf_counter() - start_time) * 1000
    metadata = ResponseMetadata(
        model_used=model,
        query_type="general",
        latency_ms=round(latency_ms, 2),
        tokens_prompt=prompt_tokens,
        tokens_completion=completion_tokens,
        tokens_total=prompt_tokens + completion_tokens,
        temperature=request.temperature,
        fallback_used=False,
        timestamp=datetime.now(timezone.utc).isoformat(),
        cached=False,
    )
    final_event = {"token": "", "done": True, "metadata": metadata.model_dump()}
    yield f"data: {json.dumps(final_event)}\n\n"


@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    service: OllamaService = Depends(get_llm_service),
) -> StreamingResponse:
    """Streaming chat via Server-Sent Events."""
    model = _resolve_model(request)
    return StreamingResponse(
        _sse_token_stream(request, service, model),
        media_type="text/event-stream",
    )
