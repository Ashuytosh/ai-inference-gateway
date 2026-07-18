"""
FastAPI dependency-injection providers.

Routes ask for services via `Depends(get_llm_service)` rather than
importing/instantiating them directly. That indirection is what lets us
swap in a real implementation later (Phase 2/3/5/6) -- or a mock, in
tests -- without touching every route that uses it.

Phase 2 fills in get_llm_service and get_response_cache with real
singletons. Phase 3 filled in get_prompt_service. Phase 5 filled in
get_router_service. Phase 6 adds get_analytics_service, get_rate_limiter,
and the check_rate_limit dependency itself.
"""

from fastapi import Depends, HTTPException, Request, Response

from app.config import settings
from app.core.analytics import RequestAnalytics
from app.core.exceptions import DEGRADATION_MESSAGES
from app.core.rate_limiter import RateLimiter
from app.services.llm_service import OllamaService, ResponseCache
from app.services.output_parser import OutputParser
from app.services.prompt_service import PromptService
from app.services.router_service import ModelRouter, QueryClassifier


class RouterService:
    """
    Thin wrapper bundling the two Phase 5 routing pieces into a single
    dependency -- chat.py needs both the classifier (what kind of query
    is this?) and the router (which model should handle it?), so rather
    than injecting two separate FastAPI dependencies it injects one
    RouterService and reaches `.classifier` / `.router` off it. Both
    pieces are stateless, so bundling them costs nothing.
    """

    def __init__(self) -> None:
        self.classifier = QueryClassifier()
        self.router = ModelRouter()


# Module-level singleton slots. FastAPI calls a dependency function once
# per request by default, which would create a brand new httpx client
# (and a cold, empty cache) on every single request -- these globals plus
# the "create once, then reuse" pattern below turn each provider into a
# true singleton shared across the whole app's lifetime instead.
_llm_service: OllamaService | None = None
_cache: ResponseCache | None = None
_prompt_service: PromptService | None = None
_output_parser: OutputParser | None = None
_router_service: RouterService | None = None
_analytics_service: RequestAnalytics | None = None
_rate_limiter: RateLimiter | None = None


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


async def get_router_service() -> RouterService:
    global _router_service
    if _router_service is None:
        _router_service = RouterService()
    return _router_service


async def get_prompt_service() -> PromptService:
    global _prompt_service
    if _prompt_service is None:
        _prompt_service = PromptService()
    return _prompt_service


async def get_output_parser() -> OutputParser:
    global _output_parser
    if _output_parser is None:
        _output_parser = OutputParser()
    return _output_parser


async def get_analytics_service() -> RequestAnalytics:
    global _analytics_service
    if _analytics_service is None:
        _analytics_service = RequestAnalytics()
    return _analytics_service


async def get_rate_limiter() -> RateLimiter:
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(
            max_requests=settings.rate_limit_requests,
            window_seconds=settings.rate_limit_window,
        )
    return _rate_limiter


async def check_rate_limit(
    request: Request,
    response: Response,
    limiter: RateLimiter = Depends(get_rate_limiter),
) -> dict:
    """
    FastAPI dependency applied to the chat endpoints only (see
    app/routers/chat.py) -- not health/models, which are cheap to serve
    and don't touch Ollama, so rate-limiting them would protect nothing
    while adding friction for monitoring tools that poll /health.

    Identifies a client by IP address (request.client.host) -- simple
    and sufficient for a local/single-tenant gateway; a public multi-
    tenant deployment would want a real API-key-based identity instead,
    since IPs can be shared (NAT) or spoofed.

    On success, writes the X-RateLimit-* headers directly onto the
    `response` object FastAPI injects here -- because this is the exact
    same Response instance the route handler's eventual return value
    gets serialized into, headers set on it here survive through to the
    client's actual HTTP response, even though this code runs inside a
    dependency rather than the route body itself.
    """
    client_id = request.client.host if request.client else "unknown"
    allowed, info = limiter.is_allowed(client_id)

    if not allowed:
        message = DEGRADATION_MESSAGES["rate_limited"]
        raise HTTPException(
            status_code=429,
            detail={
                "error": message["response"],
                "message": f"Max {info['limit']} requests per minute",
                "suggestion": message["suggestion"].format(
                    reset_seconds=info["reset_seconds"]
                ),
                "retry_after_seconds": info["reset_seconds"],
            },
        )

    response.headers["X-RateLimit-Limit"] = str(info["limit"])
    response.headers["X-RateLimit-Remaining"] = str(info["remaining"])
    response.headers["X-RateLimit-Reset"] = str(info["reset_seconds"])
    return info
