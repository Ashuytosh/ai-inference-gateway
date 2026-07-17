"""
Model discovery endpoints -- lets the UI/clients see what's available in
Ollama before (or without) sending a chat request.
"""

from fastapi import APIRouter

from app.core.exceptions import ModelNotFoundError
from app.models.responses import ModelInfo, ModelsListResponse

router = APIRouter(prefix="/api/models", tags=["Models"])


@router.get("", response_model=ModelsListResponse)
async def list_models() -> ModelsListResponse:
    """
    Placeholder model list. Phase 2 will call Ollama's /api/tags
    endpoint here and populate this with real ModelInfo entries.
    """
    return ModelsListResponse(models=[], total=0)


@router.get("/{model_name}/status", response_model=ModelInfo)
async def model_status(model_name: str) -> ModelInfo:
    """
    Placeholder model status lookup.

    No models are wired up to a real Ollama connection yet, so every
    lookup is "not found" -- this also lets us exercise the
    ModelNotFoundError -> 404 JSON error path end-to-end before Phase 2
    gives it real data to be right or wrong about.
    """
    raise ModelNotFoundError(
        message=f"Model '{model_name}' not found in Ollama",
    )
