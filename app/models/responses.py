"""
Pydantic models describing the shape of outgoing response bodies.

Declaring these explicitly (rather than returning raw dicts) means
FastAPI validates *our own* output too -- if a future change to the
service layer accidentally omits a field, we get a 500 with a clear
error instead of silently shipping a malformed response to the client.
"""

from pydantic import BaseModel, Field


class ResponseMetadata(BaseModel):
    """
    Everything about *how* a response was produced, separate from the
    response text itself. This is what gets rendered as small badges
    under each chat bubble in the UI (see CLAUDE.md design language).
    """

    model_used: str
    query_type: str  # one of: "simple", "complex", "creative", "technical"
    latency_ms: float
    tokens_prompt: int
    tokens_completion: int
    tokens_total: int
    temperature: float

    # True when the preferred/requested model failed and we fell back
    # to a different one (Phase 5 territory). Defaults False for now
    # since Phase 1 has no real routing yet.
    fallback_used: bool = False

    # ISO 8601 string rather than a datetime object -- keeps JSON
    # serialization simple and unambiguous across timezones.
    timestamp: str

    # True when this response was served from ResponseCache instead of
    # calling Ollama -- lets the UI/client see why latency_ms is
    # suspiciously low for what should be an LLM call.
    cached: bool = False

    # Which prompt-engineering strategy PromptService applied: "direct",
    # "chain-of-thought", "creative-enhancement", or "technical-precision".
    # None when the caller supplied their own system_prompt, since that
    # overrides our auto-strategy entirely (see PromptService.build_messages).
    prompt_strategy: str | None = None

    # Which OutputFormat was requested ("text", "sentiment", ...) --
    # None only in contexts that predate structured output entirely;
    # in practice this is always set now, defaulting to "text".
    output_format: str | None = None

    # How many LLM calls it took to get valid structured output: 1 means
    # the first attempt already parsed cleanly; up to 3 if OutputParser
    # had to retry with a stricter prompt (see chat.py's retry loop).
    # Always 1 for output_format="text", where no parsing happens.
    parse_attempts: int = 1

    # How sure QueryClassifier was about query_type (0.0-1.0). Low
    # confidence values are the signal that triggered an LLM-tier
    # classification instead of trusting the cheap heuristic outright
    # (see RouterService.classifier.classify_smart).
    classification_confidence: float = 0.0

    # Which classification tier actually produced query_type: "heuristic"
    # (fast keyword/length rules), "llm" (heuristic was uncertain, a
    # small model resolved it), or "heuristic-fallback" (heuristic was
    # uncertain but no LLM service was available to escalate to).
    classification_method: str | None = None

    # Rough token count of the full prompt sent to the model (system +
    # user message combined), from OllamaService.estimate_tokens's
    # chars-per-4 heuristic -- not exact, but enough to warn/truncate
    # before hitting the model's real context window.
    context_tokens_estimated: int | None = None

    # The context window size that context_tokens_estimated was compared
    # against (see chat.py's context check). None only in contexts that
    # skip the check entirely.
    context_limit: int | None = None


class ChatResponse(BaseModel):
    """Body returned by POST /api/chat."""

    response: str  # always present -- raw text, even when parsed is set
    # Structured data once output_format != TEXT and parsing succeeded;
    # None for plain text requests, and also None if every parse attempt
    # (initial + retries) failed -- callers should check this rather than
    # assume it's always populated when they asked for a structured format.
    parsed: dict | None = None
    metadata: ResponseMetadata


class ModelInfo(BaseModel):
    """Describes a single model known to Ollama."""

    name: str
    size_gb: float | None = None
    parameter_count: str | None = None
    quantization: str | None = None
    capabilities: list[str] = Field(default_factory=list)
    loaded: bool = False


class ModelsListResponse(BaseModel):
    """Body returned by GET /api/models."""

    models: list[ModelInfo]
    total: int


class CircuitBreakerStatus(BaseModel):
    """
    Snapshot of OllamaCircuitBreaker's state (see
    app/services/llm_service.py) for GET /health to expose. A kept-as-a-
    typed-model (rather than a raw dict) so this response field gets the
    same output validation as everything else in this file.
    """

    state: str  # "closed", "open", or "half_open"
    failure_count: int


class HealthResponse(BaseModel):
    """Body returned by GET /health."""

    status: str  # "healthy" or "unhealthy"
    ollama_connected: bool
    models_loaded: int
    uptime_seconds: float
    version: str
    # Actual names of models currently resident in VRAM (from Ollama's
    # GET /api/ps) -- lets a caller see *which* models are loaded, not
    # just the count.
    loaded_model_names: list[str] = Field(default_factory=list)
    # Whether the circuit breaker protecting Ollama calls is currently
    # letting requests through (closed), rejecting them immediately
    # (open), or testing recovery with one request (half_open).
    circuit_breaker: CircuitBreakerStatus


class ErrorResponse(BaseModel):
    """
    Structured shape used by the global exception handlers in main.py
    whenever an AppException (or an unexpected error) is caught.
    """

    error: str
    message: str
    status_code: int
    detail: str | None = None
