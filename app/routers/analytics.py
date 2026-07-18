"""
Request analytics endpoints -- lets an operator (or a future admin UI)
see usage patterns without grepping through JSON log lines by hand.

See app/core/analytics.py for what's actually tracked and why.
"""

from fastapi import APIRouter, Depends

from app.core.analytics import RequestAnalytics
from app.core.dependencies import get_analytics_service

router = APIRouter(prefix="/api/analytics", tags=["Analytics"])


@router.get("")
async def get_analytics(
    analytics: RequestAnalytics = Depends(get_analytics_service),
) -> dict:
    """Rolled-up usage stats since the last reset (or process start)."""
    return analytics.get_stats()


@router.post("/reset")
async def reset_analytics(
    analytics: RequestAnalytics = Depends(get_analytics_service),
) -> dict:
    """
    Zeroes every counter -- useful for starting a fresh measurement
    window (e.g. before a load test) without restarting the whole
    server, which would also drop the response cache and force the
    default model to reload into VRAM.
    """
    analytics.reset()
    return {"message": "Analytics reset"}
