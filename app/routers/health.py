"""Health check endpoint -- used by uptime monitors / load balancers."""

from fastapi import APIRouter

from app.config import settings
from app.models.responses import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Placeholder health check.

    Phase 2 will replace the hardcoded values here with a real ping to
    Ollama (ollama_connected), a count of loaded models, and actual
    process uptime tracked from app startup time.
    """
    return HealthResponse(
        status="healthy",
        ollama_connected=False,
        models_loaded=0,
        uptime_seconds=0.0,
        version=settings.app_version,
    )
