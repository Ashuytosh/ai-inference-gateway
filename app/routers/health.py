"""Health check endpoint -- used by uptime monitors / load balancers."""

import time

from fastapi import APIRouter, Depends

from app.config import settings
from app.core.dependencies import get_llm_service
from app.models.responses import CircuitBreakerStatus, HealthResponse
from app.services.llm_service import OllamaService

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(
    service: OllamaService = Depends(get_llm_service),
) -> HealthResponse:
    """
    Reports whether Ollama is actually reachable right now (not just
    whether it was reachable at startup) plus how many models are
    installed and how long this process has been running.

    Deliberately never raises -- an "unhealthy" JSON body with a 200
    status is more useful to a monitoring tool than a 5xx, since the
    caller still needs to read ollama_connected to know *why*.
    """
    # Imported here (not at module load time) to avoid a circular import:
    # app.main imports this router, so importing app_start_time from
    # app.main at the top of this file would try to import app.main
    # before it finishes defining itself.
    from app.main import app_start_time

    uptime_seconds = time.time() - app_start_time
    ollama_connected = await service.health_check()

    loaded_model_names: list[str] = []
    if ollama_connected:
        loaded_model_names = await service.get_loaded_models()

    breaker_status = service.get_circuit_breaker_status()

    return HealthResponse(
        status="healthy" if ollama_connected else "unhealthy",
        ollama_connected=ollama_connected,
        models_loaded=len(loaded_model_names),
        uptime_seconds=round(uptime_seconds, 1),
        version=settings.app_version,
        loaded_model_names=loaded_model_names,
        circuit_breaker=CircuitBreakerStatus(**breaker_status),
    )
