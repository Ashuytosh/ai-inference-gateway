"""
Chat endpoints -- the core of the gateway.

Both routes accept the same ChatRequest body. POST / returns a single
JSON response once generation is complete; POST /stream returns
Server-Sent Events (SSE) so a browser can render tokens as they arrive,
ChatGPT-style.

As of Phase 5, every request goes through the full smart-routing
pipeline instead of a hardcoded default model:

  1. RouterService.classifier.classify_smart() classifies the prompt into
     a QueryType using the two-tier heuristic/LLM classifier (see
     app/services/router_service.py) instead of the old inline keyword
     check.
  2. RouterService.router.route() picks the actual model to use --
     respecting a manual `request.model` override, or auto-selecting the
     best available model for the query type while avoiding an
     unnecessary VRAM swap when a "good enough" model is already loaded.
  3. PromptService still builds the tailored system prompt/strategy for
     that query_type (unchanged since Phase 3).
  4. If the chosen model's call fails with a transient/connection error,
     we walk the query type's fallback chain and retry -- this is what
     ResponseMetadata.fallback_used actually reports now, instead of
     being hardcoded False.
  5. A rough context-window check warns (and, if far enough over,
     truncates) before the prompt goes to Ollama at all.

As of Phase 4, POST / additionally supports structured output: if
request.output_format isn't TEXT, OutputParser (app/services/output_parser.py)
forces the model to emit a specific JSON shape, validates it, and retries
with a stricter prompt (up to 2 retries) on parse failure. Streaming
doesn't support structured output (see chat_stream) -- you can't validate
JSON until the full response has arrived, which defeats the point of
streaming, so that combination is rejected outright with a 400.
"""

import json
import time
from datetime import datetime, timezone

import structlog
from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import StreamingResponse

from app.config import settings
from app.core.analytics import RequestAnalytics
from app.core.dependencies import (
    RouterService,
    check_rate_limit,
    get_analytics_service,
    get_llm_service,
    get_output_parser,
    get_prompt_service,
    get_response_cache,
    get_router_service,
)
from app.core.exceptions import AppException, OutputParsingError
from app.models.enums import QueryType
from app.models.output_formats import OutputFormat
from app.models.requests import ChatRequest
from app.models.responses import ChatResponse, ResponseMetadata
from app.services.llm_service import OllamaResponse, OllamaService, ResponseCache
from app.services.output_parser import OutputParser
from app.services.prompt_service import STRATEGY_NAMES, PromptService

logger = structlog.get_logger()

router = APIRouter(prefix="/api/chat", tags=["Chat"])

# Structured output gets at most this many extra attempts (on top of the
# first try) before giving up and returning raw text with parsed=None.
MAX_PARSE_RETRIES = 2

# Default context window assumed for routing purposes -- matches what
# most of the locally-served 7B models support out of the box. This is a
# simplification (a Modelfile could configure a larger context), but it's
# the same value the spec calls for and is good enough for a warn/
# truncate safety check rather than an exact accounting.
MODEL_CONTEXT_LIMIT = 4096


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


def _unwrap_parsed(parsed_model, output_format: OutputFormat) -> dict:
    """
    Converts a validated structured-output model into the plain dict
    that goes into ChatResponse.parsed.

    OutputFormat.JSON is a special case: its model (CustomFormat) wraps
    the caller's actual data under a `data` field (see
    OutputParser.validate_output's docstring for why) -- so it gets
    unwrapped here to `.data` instead of `.model_dump()`'d directly,
    otherwise callers would see their requested shape double-wrapped as
    {"data": {...their shape...}} instead of just {...their shape...}.
    """
    if output_format == OutputFormat.JSON:
        return parsed_model.data
    return parsed_model.model_dump()


async def _classify_and_route(
    request: ChatRequest,
    service: OllamaService,
    router_service: RouterService,
) -> tuple[QueryType, float, str, str, list[str]]:
    """
    Shared classify -> route pipeline used by both the chat() and
    chat_stream() endpoints, so the two-tier classification and
    VRAM-aware routing logic exists in exactly one place.

    Returns (query_type, confidence, classification_method, model,
    loaded_models_before_this_call) -- the loaded-models snapshot is
    returned too since chat()/chat_stream() need it again for the
    "query_routed" log line (was_already_loaded / swap_required), and
    re-fetching it a second time would just be a wasted extra call to
    Ollama's /api/ps for information we already have.
    """
    query_type, confidence, method = await router_service.classifier.classify_smart(
        request.prompt, service
    )

    loaded = await service.get_loaded_models()
    available = [m.name for m in await service.list_models()]
    model = router_service.router.route(
        query_type, request.model, loaded, available, confidence
    )

    logger.info(
        "query_routed",
        query_preview=request.prompt[:50],
        query_type=query_type.value,
        confidence=confidence,
        classification_method=method,
        selected_model=model,
        was_already_loaded=model in loaded,
        swap_required=model not in loaded,
        fallback_used=False,
    )

    return query_type, confidence, method, model, loaded


def _check_context_window(system_prompt: str, user_prompt: str, service: OllamaService) -> tuple[str, int, int]:
    """
    Rough context-budget check (Optimization 5). Estimates token count
    for the combined system+user prompt; if we're already past 80% of
    the assumed context limit we log a warning (the response still goes
    through -- Ollama itself will error out if it's genuinely too large),
    and if we're past 95% we truncate the *user* prompt (never the
    system prompt, which is comparatively tiny and defines the model's
    behavior) so the request has a real chance of succeeding instead of
    reliably failing at the model.

    Returns (possibly-truncated user_prompt, estimated_tokens, context_limit).
    estimated_tokens is computed on the (possibly truncated) final prompt,
    so ResponseMetadata always reflects what was actually sent.
    """
    estimated_tokens = service.estimate_tokens(system_prompt + user_prompt)

    if estimated_tokens > MODEL_CONTEXT_LIMIT * 0.95:
        # Truncate to roughly 95% of the limit, converting the token
        # budget back to a character budget with the same ~4 chars/token
        # heuristic estimate_tokens uses, then trimming the user prompt
        # down to fit alongside the (untouched) system prompt.
        max_total_chars = int(MODEL_CONTEXT_LIMIT * 0.95 * 4)
        max_user_chars = max(0, max_total_chars - len(system_prompt))
        truncated_prompt = user_prompt[:max_user_chars]
        logger.warning(
            "context_window_truncated",
            estimated_tokens=estimated_tokens,
            context_limit=MODEL_CONTEXT_LIMIT,
            original_length=len(user_prompt),
            truncated_length=len(truncated_prompt),
        )
        user_prompt = truncated_prompt
        estimated_tokens = service.estimate_tokens(system_prompt + user_prompt)
    elif estimated_tokens > MODEL_CONTEXT_LIMIT * 0.8:
        logger.warning(
            "context_window_high",
            estimated_tokens=estimated_tokens,
            context_limit=MODEL_CONTEXT_LIMIT,
        )

    return user_prompt, estimated_tokens, MODEL_CONTEXT_LIMIT


async def _call_with_fallback(
    service: OllamaService,
    router_service: RouterService,
    model: str,
    query_type: QueryType,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
) -> tuple[OllamaResponse, str, bool]:
    """
    Calls the LLM with `model`; if that call fails with one of our
    AppException types (connection error, timeout, model-not-found --
    note OllamaService.chat already retries transient errors internally
    via @with_retry, so an exception reaching here means those retries
    were already exhausted), walks the query type's fallback chain and
    tries each candidate in turn.

    Returns (result, model_actually_used, fallback_used) so callers can
    build accurate metadata even when the originally-routed model wasn't
    the one that ultimately answered.
    """
    try:
        result = await service.chat(
            model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
        )
        return result, model, False
    except AppException as exc:
        logger.warning("model_call_failed", model=model, error=str(exc))

    available = [m.name for m in await service.list_models()]
    fallback_chain = router_service.router.get_fallback_chain(model, query_type, available)

    last_error: AppException | None = None
    for fallback_model in fallback_chain:
        try:
            result = await service.chat(
                model=fallback_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            logger.info(
                "fallback_succeeded", original_model=model, fallback_model=fallback_model
            )
            return result, fallback_model, True
        except AppException as exc:
            last_error = exc
            logger.warning(
                "fallback_failed", fallback_model=fallback_model, error=str(exc)
            )

    # Every fallback also failed (or there were none available) -- re-raise
    # so the global exception handler in main.py still produces the usual
    # structured JSON error response, rather than swallowing the failure.
    raise last_error if last_error is not None else RuntimeError(
        f"Model '{model}' failed and no fallback was available"
    )


def _build_metadata(
    result: OllamaResponse,
    query_type: QueryType,
    prompt_strategy: str | None,
    temperature: float,
    latency_ms: float,
    cached: bool,
    output_format: OutputFormat,
    parse_attempts: int,
    confidence: float,
    classification_method: str,
    fallback_used: bool,
    context_tokens_estimated: int | None,
    context_limit: int | None,
) -> ResponseMetadata:
    return ResponseMetadata(
        model_used=result.model,
        query_type=query_type.value,
        latency_ms=round(latency_ms, 2),
        tokens_prompt=result.prompt_tokens,
        tokens_completion=result.completion_tokens,
        tokens_total=result.prompt_tokens + result.completion_tokens,
        temperature=temperature,
        fallback_used=fallback_used,
        timestamp=datetime.now(timezone.utc).isoformat(),
        cached=cached,
        prompt_strategy=prompt_strategy,
        output_format=output_format.value,
        parse_attempts=parse_attempts,
        classification_confidence=confidence,
        classification_method=classification_method,
        context_tokens_estimated=context_tokens_estimated,
        context_limit=context_limit,
    )


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    response: Response,
    service: OllamaService = Depends(get_llm_service),
    cache: ResponseCache = Depends(get_response_cache),
    prompt_service: PromptService = Depends(get_prompt_service),
    output_parser: OutputParser = Depends(get_output_parser),
    router_service: RouterService = Depends(get_router_service),
    analytics: RequestAnalytics = Depends(get_analytics_service),
    rate_info: dict = Depends(check_rate_limit),
) -> ChatResponse:
    """
    Non-streaming chat. `request` is validated by ChatRequest before this
    function body even runs -- e.g. an empty prompt or an out-of-range
    temperature never reaches this code, FastAPI returns a 422 first.

    `rate_info` isn't used directly here -- depending on check_rate_limit
    is what actually enforces the limit (raising a 429 before this
    function body even starts) and stamps the X-RateLimit-* headers onto
    `response`; the return value only exists so a caller could inspect
    it if they wanted to, chat() itself doesn't need it.
    """
    query_type, confidence, method, model, _loaded = await _classify_and_route(
        request, service, router_service
    )
    temperature = _resolve_temperature(request, query_type, prompt_service)

    # A custom system_prompt means the caller took explicit control of
    # the model's behavior -- PromptService skips strategy enhancement
    # in that case (see build_messages), so there's no strategy to report.
    prompt_strategy = None if request.system_prompt else STRATEGY_NAMES[query_type]

    wants_structured_output = request.output_format != OutputFormat.TEXT
    format_instruction = ""
    prompt_for_llm = request.prompt
    if wants_structured_output:
        format_instruction = output_parser.build_format_instruction(
            request.output_format, request.schema_hint
        )
        # Appended before prompt engineering runs, so e.g. a technical
        # query's precision suffix still gets layered on after the
        # format instruction -- both instructions reach the model.
        prompt_for_llm = f"{request.prompt}\n\n{format_instruction}"

    system_prompt_preview = prompt_service.get_system_prompt(
        query_type, request.system_prompt
    )
    prompt_for_llm, context_tokens_estimated, context_limit = _check_context_window(
        system_prompt_preview, prompt_for_llm, service
    )

    messages = prompt_service.build_messages(
        prompt=prompt_for_llm,
        query_type=query_type,
        custom_system_prompt=request.system_prompt,
    )

    # Structured-output requests always hit the LLM fresh: ResponseCache's
    # key doesn't account for output_format, so caching them risks
    # returning a cached plain-text answer for a "sentiment" request (or
    # vice versa) that happens to share prompt/model/temperature.
    cache_key = None
    if not wants_structured_output:
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
            cache_hit_metadata = _build_metadata(
                cached_result, query_type, prompt_strategy, temperature,
                latency_ms, cached=True, output_format=request.output_format,
                parse_attempts=1, confidence=confidence,
                classification_method=method, fallback_used=False,
                context_tokens_estimated=context_tokens_estimated,
                context_limit=context_limit,
            )
            analytics.record_request(
                model=cache_hit_metadata.model_used,
                query_type=cache_hit_metadata.query_type,
                latency_ms=cache_hit_metadata.latency_ms,
                tokens_total=cache_hit_metadata.tokens_total,
                cached=cache_hit_metadata.cached,
                fallback_used=cache_hit_metadata.fallback_used,
            )
            return ChatResponse(
                response=cached_result.text,
                metadata=cache_hit_metadata,
            )

    result, model_used, fallback_used = await _call_with_fallback(
        service, router_service, model, query_type, messages, temperature, request.max_tokens
    )

    parsed: dict | None = None
    parse_attempts = 1

    if wants_structured_output:
        parsed_model = None
        last_error: OutputParsingError | None = None
        try:
            parsed_model = output_parser.parse(result.text, request.output_format)
        except OutputParsingError as exc:
            last_error = exc

        # Retry with an increasingly explicit prompt on parse failure.
        # Each retry is a standalone call: [same system message] + [retry
        # prompt] -- not routed back through prompt_service, since the
        # retry prompt is already a complete, self-contained instruction
        # and re-running query-type strategy enhancement on top of it
        # (e.g. appending a chain-of-thought suffix to a JSON-formatting
        # correction) would only work against the correction's intent.
        while parsed_model is None and parse_attempts <= MAX_PARSE_RETRIES:
            parse_attempts += 1
            retry_prompt = output_parser.build_retry_prompt(
                request.prompt, format_instruction, str(last_error)
            )
            retry_messages = [messages[0], {"role": "user", "content": retry_prompt}]
            result = await service.chat(
                model=model_used,
                messages=retry_messages,
                temperature=temperature,
                max_tokens=request.max_tokens,
            )
            try:
                parsed_model = output_parser.parse(result.text, request.output_format)
            except OutputParsingError as exc:
                last_error = exc

        if parsed_model is not None:
            parsed = _unwrap_parsed(parsed_model, request.output_format)
        else:
            logger.warning(
                "structured_output_parse_failed",
                output_format=request.output_format.value,
                attempts=parse_attempts,
                error=str(last_error),
            )

    latency_ms = (time.perf_counter() - start_time) * 1000

    if cache_key is not None:
        cache.set(cache_key, result)

    final_metadata = _build_metadata(
        result, query_type, prompt_strategy, temperature, latency_ms,
        cached=False, output_format=request.output_format,
        parse_attempts=parse_attempts, confidence=confidence,
        classification_method=method, fallback_used=fallback_used,
        context_tokens_estimated=context_tokens_estimated,
        context_limit=context_limit,
    )
    analytics.record_request(
        model=final_metadata.model_used,
        query_type=final_metadata.query_type,
        latency_ms=final_metadata.latency_ms,
        tokens_total=final_metadata.tokens_total,
        cached=final_metadata.cached,
        fallback_used=final_metadata.fallback_used,
    )

    return ChatResponse(
        response=result.text,
        parsed=parsed,
        metadata=final_metadata,
    )


async def _sse_token_stream(
    request: ChatRequest,
    service: OllamaService,
    prompt_service: PromptService,
    router_service: RouterService,
    analytics: RequestAnalytics,
    query_type: QueryType,
    confidence: float,
    classification_method: str,
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
    Likewise, mid-stream model-swap fallback isn't attempted here: once
    tokens have already been sent to the client, silently restarting on
    a different model would duplicate/garble what they've already seen
    (see with_retry_stream's docstring in llm_service.py for the same
    reasoning applied to transient-error retries) -- so a stream failure
    just propagates as-is rather than trying a fallback model mid-flight.

    Structured output is rejected before this generator is ever
    constructed (see chat_stream), so output_format is always TEXT here.
    """
    temperature = _resolve_temperature(request, query_type, prompt_service)
    prompt_strategy = None if request.system_prompt else STRATEGY_NAMES[query_type]

    system_prompt_preview = prompt_service.get_system_prompt(
        query_type, request.system_prompt
    )
    user_prompt, context_tokens_estimated, context_limit = _check_context_window(
        system_prompt_preview, request.prompt, service
    )

    messages = prompt_service.build_messages(
        prompt=user_prompt,
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
        output_format=OutputFormat.TEXT.value,
        parse_attempts=1,
        classification_confidence=confidence,
        classification_method=classification_method,
        context_tokens_estimated=context_tokens_estimated,
        context_limit=context_limit,
    )
    analytics.record_request(
        model=metadata.model_used,
        query_type=metadata.query_type,
        latency_ms=metadata.latency_ms,
        tokens_total=metadata.tokens_total,
        cached=metadata.cached,
        fallback_used=metadata.fallback_used,
    )

    final_event = {"token": "", "done": True, "metadata": metadata.model_dump()}
    yield f"data: {json.dumps(final_event)}\n\n"


@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    service: OllamaService = Depends(get_llm_service),
    prompt_service: PromptService = Depends(get_prompt_service),
    router_service: RouterService = Depends(get_router_service),
    analytics: RequestAnalytics = Depends(get_analytics_service),
    rate_info: dict = Depends(check_rate_limit),
) -> StreamingResponse:
    """
    Streaming chat via Server-Sent Events.

    Structured output isn't supported here: you can't validate JSON
    until the complete response has arrived, which means "stream it
    token by token" and "guarantee a validated shape" are fundamentally
    at odds. Rather than silently collect the whole stream server-side
    and only then emit it (technically possible, but it defeats the
    entire point of choosing the streaming endpoint), this rejects the
    combination outright -- the caller should use POST /api/chat instead
    when they need structured output.

    Rate-limit headers are passed explicitly via StreamingResponse's own
    `headers=` here rather than through the "inject a Response and
    mutate it" trick check_rate_limit uses for chat() -- this endpoint
    constructs and returns its own StreamingResponse instance, which
    becomes the real response FastAPI sends, so headers set on any
    separately-injected Response object would never make it onto the
    client's actual reply.
    """
    if request.output_format != OutputFormat.TEXT:
        raise HTTPException(
            status_code=400,
            detail=(
                "Structured output is not supported with streaming. "
                "Use the non-streaming endpoint for structured responses."
            ),
        )

    query_type, confidence, method, model, _loaded = await _classify_and_route(
        request, service, router_service
    )
    return StreamingResponse(
        _sse_token_stream(
            request, service, prompt_service, router_service, analytics,
            query_type, confidence, method, model,
        ),
        media_type="text/event-stream",
        headers={
            "X-RateLimit-Limit": str(rate_info["limit"]),
            "X-RateLimit-Remaining": str(rate_info["remaining"]),
            "X-RateLimit-Reset": str(rate_info["reset_seconds"]),
        },
    )
