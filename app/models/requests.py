"""
Pydantic models describing the shape of incoming request bodies.

FastAPI uses these for automatic request validation *and* for the
auto-generated OpenAPI docs at /docs -- so the Field(...) constraints
below aren't just documentation, they're enforced on every request
before our route handler code ever runs.
"""

from pydantic import BaseModel, Field

from app.models.output_formats import OutputFormat


class ChatRequest(BaseModel):
    """Body for POST /api/chat and POST /api/chat/stream."""

    # min_length=1 rejects empty strings ("") outright; max_length keeps
    # a client from sending something absurd that would blow the model's
    # context window before we even get a chance to check.
    prompt: str = Field(min_length=1, max_length=10000)

    # None means "let the router (Phase 5) pick the best model for this
    # query" instead of forcing a specific one.
    model: str | None = Field(
        default=None, description="If None, auto-route to best model"
    )

    # ge/le bound this to Ollama's usual sane temperature range; values
    # outside it tend to produce garbage or are simply not supported.
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)

    # gt=0 (not ge=0) because a 0-token response is meaningless.
    max_tokens: int = Field(default=1024, gt=0, le=4096)

    # When True, the client should hit /api/chat/stream instead of
    # expecting a single JSON body back from /api/chat.
    stream: bool = Field(default=False)

    system_prompt: str | None = Field(default=None, max_length=2000)

    # TEXT (default) returns free-form text with parsed=None; any other
    # value asks OutputParser (app/services/output_parser.py) to force
    # and validate a specific JSON shape from the model.
    output_format: OutputFormat = Field(
        default=OutputFormat.TEXT,
        description="Desired output format. TEXT returns free-form text, others return structured JSON.",
    )

    # Only meaningful when output_format is JSON -- describes the custom
    # shape the caller wants (e.g. as a JSON-Schema-ish string), since
    # there's no named Pydantic model to fall back on for a caller-defined format.
    schema_hint: str | None = Field(
        default=None,
        max_length=1000,
        description="JSON schema hint when output_format is 'json'. Describes the shape you want.",
    )
