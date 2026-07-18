"""
Model discovery endpoints -- lets the UI/clients see what's available in
Ollama before (or without) sending a chat request.
"""

from fastapi import APIRouter, Depends, Response

from app.core.dependencies import get_llm_service
from app.core.exceptions import ModelNotFoundError, OllamaConnectionError
from app.models.responses import ModelInfo, ModelsListResponse
from app.services.llm_service import OllamaService

router = APIRouter(prefix="/api/models", tags=["Models"])


@router.get("", response_model=ModelsListResponse)
async def list_models(
    response: Response,
    service: OllamaService = Depends(get_llm_service),
) -> ModelsListResponse:
    """
    Lists every model Ollama has pulled locally. If Ollama itself is
    unreachable, this still returns 200 with an empty list (plus a
    Warning header) rather than a 5xx -- a client just showing "no
    models available" degrades more gracefully than a hard error for
    what is, after all, just a listing endpoint.
    """
    try:
        models = await service.list_models()
    except OllamaConnectionError as exc:
        response.headers["Warning"] = f'199 - "{exc.message}"'
        return ModelsListResponse(models=[], total=0)

    return ModelsListResponse(models=models, total=len(models))


@router.get("/{model_name}/status", response_model=ModelInfo)
async def model_status(
    model_name: str,
    service: OllamaService = Depends(get_llm_service),
) -> ModelInfo:
    """
    Looks up a single model by exact name (e.g. "qwen2.5:7b"). Unlike
    list_models(), a connection failure here is allowed to propagate as
    a 503 (via OllamaConnectionError's handler in main.py) -- there's no
    sensible "empty" answer to give for "is this specific model
    available", so surfacing the real error is more honest.
    """
    model = await service.get_model_status(model_name)
    if model is None:
        available = [m.name for m in await service.list_models()]
        raise ModelNotFoundError(
            message=f"Model '{model_name}' not found in Ollama",
            detail=f"Available models: {', '.join(available) or 'none'}",
        )
    return model
