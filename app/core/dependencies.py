"""
FastAPI dependency-injection providers.

Routes ask for services via `Depends(get_llm_service)` rather than
importing/instantiating them directly. That indirection is what lets us
swap in a real implementation later (Phase 2/3/5) -- or a mock, in
tests -- without touching every route that uses it. For now, each
provider is a stub returning None; routers in this phase don't call
these yet, they're wired in ahead of time so later phases only need to
fill in the function bodies.
"""


def get_llm_service():
    """Returns LLM service instance. Placeholder for Phase 2."""
    return None


def get_router_service():
    """Returns router service instance. Placeholder for Phase 5."""
    return None


def get_prompt_service():
    """Returns prompt service instance. Placeholder for Phase 3."""
    return None
