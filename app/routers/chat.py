"""
Chat endpoints -- the core of the gateway.

Both routes accept the same ChatRequest body. POST / returns a single
JSON response once generation is complete; POST /stream returns
Server-Sent Events (SSE) so a browser can render tokens as they arrive,
ChatGPT-style. Phase 2 will replace the placeholder bodies with real
Ollama calls; the request/response *shapes* defined here are meant to
stay stable across that change.
"""

import json
from datetime import datetime, timezone

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.models.requests import ChatRequest
from app.models.responses import ChatResponse, ResponseMetadata

router = APIRouter(prefix="/api/chat", tags=["Chat"])


def _placeholder_metadata() -> ResponseMetadata:
    """Shared stub metadata used by both routes until Phase 2 lands."""
    return ResponseMetadata(
        model_used="placeholder",
        query_type="simple",
        latency_ms=0.0,
        tokens_prompt=0,
        tokens_completion=0,
        tokens_total=0,
        temperature=0.7,
        fallback_used=False,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Non-streaming chat. `request` is validated by ChatRequest before this
    function body even runs -- e.g. an empty prompt or an out-of-range
    temperature never reaches this code, FastAPI returns a 422 first.
    """
    return ChatResponse(
        response="Placeholder: LLM integration coming in Phase 2",
        metadata=_placeholder_metadata(),
    )


async def _placeholder_token_stream():
    """
    Generator yielding Server-Sent Events.

    SSE format is `data: <json>\n\n` per event -- the blank line is what
    tells the browser's EventSource (or our own fetch-based reader) that
    one event has ended. We yield partial tokens with done=false, then a
    final empty-token event with done=true carrying the full metadata,
    so the client knows exactly when to stop appending text and instead
    render the metadata badges.
    """
    tokens = ["This ", "is ", "a ", "placeholder ", "stream."]
    for token in tokens:
        yield f"data: {json.dumps({'token': token, 'done': False})}\n\n"

    final_event = {
        "token": "",
        "done": True,
        "metadata": _placeholder_metadata().model_dump(),
    }
    yield f"data: {json.dumps(final_event)}\n\n"


@router.post("/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """Streaming chat via Server-Sent Events."""
    return StreamingResponse(
        _placeholder_token_stream(), media_type="text/event-stream"
    )
