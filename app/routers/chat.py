"""
Chat endpoints -- the core of the gateway.

Both routes accept the same ChatRequest body. POST / returns a single
JSON response once generation is complete; POST /stream returns
Server-Sent Events (SSE) so a browser can render tokens as they arrive,
ChatGPT-style.

As of Phase 3, every request goes through PromptService: the raw prompt
is classified into a QueryType (a temporary keyword/length heuristic
here -- proper classification is Phase 5's router_service.py), which
picks a tailored system prompt, a prompting strategy (direct /
chain-of-thought / creative-enhancement / technical-precision), and a
recommended temperature. Model selection is still simple: request.model
if the caller specified one, else settings.default_model -- picking the
best model for a query's content is also Phase 5's job.
"""

import json
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

import structlog

from app.config import settings
from app.core.dependencies import get_llm_service, get_prompt_service, get_response_cache
from app.models.enums import QueryType
from app.models.requests import ChatRequest
from app.models.responses import ChatResponse, ResponseMetadata
from app.services.llm_service import OllamaResponse, OllamaService, ResponseCache
from app.services.prompt_service import STRATEGY_NAMES, PromptService

logger = structlog.get_logger()

router = APIRouter(prefix="/api/chat", tags=["Chat"])


# Keyword lists for the inline query-type heuristic below. Real
# classification (blending heuristics with an LLM fallback for
# ambiguous cases) is Phase 5 -- this is intentionally simple.
_TECHNICAL_KEYWORDS = (
    "def", "class", "function", "error", "bug", "code", "debug",
    "api", "database", "sql", "python", "javascript",
)
_CREATIVE_KEYWORDS = (
    "write", "story", "poem", "create", "imagine", "design", "brainstorm",
)


def _classify_query_type(prompt: str) -> QueryType:
    """
    Temporary heuristic classifier: length first, then keyword matching.

    The spec this heuristic comes from describes the SIMPLE rule as
    "length < 50 chars AND no technical keywords" checked before the
    keyword rules -- but its own verification examples contradict that:
    "What is Python?" (15 chars, contains the technical keyword "python")
    is expected to classify as SIMPLE, which the literal AND-condition
    would fail. Every other example prompt is >= 50 chars. So the rule
    that actually satisfies every example is simpler: short prompts are
    always SIMPLE regardless of keyword content, and keyword matching
    only kicks in for longer prompts.
    """
    if len(prompt) < 50:
        return QueryType.SIMPLE

    lowered = prompt.lower()
    if any(keyword in lowered for keyword in _TECHNICAL_KEYWORDS):
        return QueryType.TECHNICAL
    if any(keyword in lowered for keyword in _CREATIVE_KEYWORDS):
        return QueryType.CREATIVE
    return QueryType.COMPLEX


def _resolve_model(request: ChatRequest) -> str:
    """None means "no preference" -- fall back to the configured default."""
    return request.model or settings.default_model


def _resolve_temperature(
    request: ChatRequest, query_type: QueryType, prompt_service: PromptService
) -> float:
    """
    If the caller left temperature at the application default, we know
    they didn't express a real preference -- so we're free to substitute
    a temperature better suited to this query type (e.g. low for
    technical precision, high for creative writing). An explicit,
    non-default value is an intentional user choice and always wins.
    """
    if request.temperature != settings.default_temperature:
        return request.temperature

    recommended = prompt_service.get_recommended_temperature(query_type)
    logger.info(
        "temperature_overridden",
        query_type=query_type.value,
        original=request.temperature,
        recommended=recommended,
    )
    return recommended


def _build_metadata(
    result: OllamaResponse,
    query_type: QueryType,
    prompt_strategy: str | None,
    temperature: float,
    latency_ms: float,
    cached: bool,
) -> ResponseMetadata:
    return ResponseMetadata(
        model_used=result.model,
        query_type=query_type.value,
        latency_ms=round(latency_ms, 2),
        tokens_prompt=result.prompt_tokens,
        tokens_completion=result.completion_tokens,
        tokens_total=result.prompt_tokens + result.completion_tokens,
        temperature=temperature,
        # Real fallback (retry-with-different-model) is Phase 5 territory;
        # the retry logic in Phase 2 only retries the *same* model.
        fallback_used=False,
        timestamp=datetime.now(timezone.utc).isoformat(),
        cached=cached,
        prompt_strategy=prompt_strategy,
    )


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    service: OllamaService = Depends(get_llm_service),
    cache: ResponseCache = Depends(get_response_cache),
    prompt_service: PromptService = Depends(get_prompt_service),
) -> ChatResponse:
    """
    Non-streaming chat. `request` is validated by ChatRequest before this
    function body even runs -- e.g. an empty prompt or an out-of-range
    temperature never reaches this code, FastAPI returns a 422 first.
    """
    model = _resolve_model(request)
    query_type = _classify_query_type(request.prompt)
    temperature = _resolve_temperature(request, query_type, prompt_service)

    # A custom system_prompt means the caller took explicit control of
    # the model's behavior -- PromptService skips strategy enhancement
    # in that case (see build_messages), so there's no strategy to report.
    prompt_strategy = None if request.system_prompt else STRATEGY_NAMES[query_type]

    messages = prompt_service.build_messages(
        prompt=request.prompt,
        query_type=query_type,
        custom_system_prompt=request.system_prompt,
    )

    # The cache key must be derived from the *resolved* temperature, not
    # the raw request value -- otherwise a query bumped from the default
    # 0.7 to e.g. 0.2 (technical) could be cached/looked-up under a key
    # that doesn't match what was actually generated.
    cache_key = cache.make_key(
        prompt=request.prompt,
        model=model,
        temperature=temperature,
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
                metadata=_build_metadata(
                    cached_result, query_type, prompt_strategy, temperature, latency_ms, cached=True
                ),
            )

    result = await service.chat(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=request.max_tokens,
    )
    latency_ms = (time.perf_counter() - start_time) * 1000

    if cache_key is not None:
        cache.set(cache_key, result)

    return ChatResponse(
        response=result.text,
        metadata=_build_metadata(
            result, query_type, prompt_strategy, temperature, latency_ms, cached=False
        ),
    )


async def _sse_token_stream(
    request: ChatRequest,
    service: OllamaService,
    prompt_service: PromptService,
    model: str,
):
    """
    Generator yielding Server-Sent Events.

    SSE format is `data: <json>\n\n` per event -- the blank line is what
    tells the browser's EventSource (or our own fetch-based reader) that
    one event has ended. We yield partial tokens with done=false, then a
    final empty-token event with done=true carrying the full metadata,
    so the client knows exactly when to stop appending text and instead
    render the metadata badges.

    Streamed responses are never cached (see ResponseCache docstring in
    llm_service.py) -- caching is about replaying a *complete* answer
    instantly; a stream is already "instant enough" token-by-token.
    """
    query_type = _classify_query_type(request.prompt)
    temperature = _resolve_temperature(request, query_type, prompt_service)
    prompt_strategy = None if request.system_prompt else STRATEGY_NAMES[query_type]
    messages = prompt_service.build_messages(
        prompt=request.prompt,
        query_type=query_type,
        custom_system_prompt=request.system_prompt,
    )

    start_time = time.perf_counter()
    prompt_tokens = 0
    completion_tokens = 0
    model_used = model

    async for item in service.chat_stream(
        model=model,
        messages=messages,
        temperature=temperature,
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
        model_used=model_used,
        query_type=query_type.value,
        latency_ms=round(latency_ms, 2),
        tokens_prompt=prompt_tokens,
        tokens_completion=completion_tokens,
        tokens_total=prompt_tokens + completion_tokens,
        temperature=temperature,
        fallback_used=False,
        timestamp=datetime.now(timezone.utc).isoformat(),
        cached=False,
        prompt_strategy=prompt_strategy,
    )
    final_event = {"token": "", "done": True, "metadata": metadata.model_dump()}
    yield f"data: {json.dumps(final_event)}\n\n"


@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    service: OllamaService = Depends(get_llm_service),
    prompt_service: PromptService = Depends(get_prompt_service),
) -> StreamingResponse:
    """Streaming chat via Server-Sent Events."""
    model = _resolve_model(request)
    return StreamingResponse(
        _sse_token_stream(request, service, prompt_service, model),
        media_type="text/event-stream",
    )
