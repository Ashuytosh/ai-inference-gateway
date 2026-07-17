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


class ChatResponse(BaseModel):
    """Body returned by POST /api/chat."""

    response: str
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


class HealthResponse(BaseModel):
    """Body returned by GET /health."""

    status: str  # "healthy" or "unhealthy"
    ollama_connected: bool
    models_loaded: int
    uptime_seconds: float
    version: str


class ErrorResponse(BaseModel):
    """
    Structured shape used by the global exception handlers in main.py
    whenever an AppException (or an unexpected error) is caught.
    """

    error: str
    message: str
    status_code: int
    detail: str | None = None
