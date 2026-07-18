"""
FastAPI dependency-injection providers.

Routes ask for services via `Depends(get_llm_service)` rather than
importing/instantiating them directly. That indirection is what lets us
swap in a real implementation later (Phase 2/3/5) -- or a mock, in
tests -- without touching every route that uses it.

Phase 2 fills in get_llm_service and get_response_cache with real
singletons. get_router_service / get_prompt_service remain placeholders
until Phase 5 / Phase 3.
"""

from app.config import settings
from app.services.llm_service import OllamaService, ResponseCache

# Module-level singleton slots. FastAPI calls a dependency function once
# per request by default, which would create a brand new httpx client
# (and a cold, empty cache) on every single request -- these globals plus
# the "create once, then reuse" pattern below turn each provider into a
# true singleton shared across the whole app's lifetime instead.
_llm_service: OllamaService | None = None
_cache: ResponseCache | None = None


async def get_llm_service() -> OllamaService:
    global _llm_service
    if _llm_service is None:
        _llm_service = OllamaService(
            base_url=settings.ollama_base_url,
            timeout=settings.request_timeout,
        )
    return _llm_service


async def get_response_cache() -> ResponseCache:
    global _cache
    if _cache is None:
        _cache = ResponseCache(
            max_size=settings.cache_max_size,
            ttl_seconds=settings.cache_ttl_seconds,
        )
    return _cache


def get_router_service():
    """Returns router service instance. Placeholder for Phase 5."""
    return None


def get_prompt_service():
    """Returns prompt service instance. Placeholder for Phase 3."""
    return None
