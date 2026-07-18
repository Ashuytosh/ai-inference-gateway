"""
Request analytics -- a simple in-memory tracker of usage patterns.

Why this matters for a gateway like this one: without it, you can only
answer "is the server up?" (see health.py). Analytics answers the more
interesting operational questions -- "which models actually get used in
practice?", "is the cache pulling its weight?", "how often does the
fallback chain kick in (a signal that a primary model is unreliable or
frequently unavailable)?", "what does real-world latency look like, not
just in the best case?" -- the kind of data you'd want before deciding
whether to retune MODEL_ROUTING_TABLE, adjust cache_temperature_threshold,
or pull a flaky model out of rotation.

This is intentionally a single in-process object, not backed by a
database or time-series store -- it resets on restart and only reflects
one process's view. That's a fine trade for a single-instance local
gateway; a multi-instance deployment would need to ship these metrics to
something like Prometheus instead of relying on process memory.
"""

from collections import defaultdict
from dataclasses import dataclass, field
import time


@dataclass
class AnalyticsSnapshot:
    """
    All the running totals/counters analytics tracks, bundled into one
    object so RequestAnalytics.reset() can throw the whole thing away
    and start over with a single assignment instead of resetting each
    field individually (and risking forgetting one).
    """

    total_requests: int = 0
    total_errors: int = 0
    requests_per_model: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    requests_per_query_type: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    cache_hits: int = 0
    cache_misses: int = 0
    fallback_count: int = 0
    avg_latency_ms: float = 0.0
    total_tokens_used: int = 0
    # Every individual request's latency, kept around so get_stats() can
    # compute percentiles (see RequestAnalytics.get_stats for why
    # percentiles matter beyond just the average).
    _latencies: list[float] = field(default_factory=list)


class RequestAnalytics:
    """
    Records one data point per completed chat request (see
    app/routers/chat.py's three call sites) and exposes rolled-up
    statistics via get_stats().
    """

    def __init__(self) -> None:
        self.data = AnalyticsSnapshot()
        self._start_time = time.time()

    def record_request(
        self,
        model: str,
        query_type: str,
        latency_ms: float,
        tokens_total: int,
        cached: bool,
        fallback_used: bool,
        error: bool = False,
    ) -> None:
        self.data.total_requests += 1
        self.data.requests_per_model[model] += 1
        self.data.requests_per_query_type[query_type] += 1
        self.data.total_tokens_used += tokens_total
        self.data._latencies.append(latency_ms)

        if cached:
            self.data.cache_hits += 1
        else:
            self.data.cache_misses += 1

        if fallback_used:
            self.data.fallback_count += 1

        if error:
            self.data.total_errors += 1

        # Recomputed on every call rather than kept as a running mean --
        # simpler to reason about and correct-by-construction, and at the
        # request volumes a single local gateway sees this is nowhere
        # near expensive enough to matter.
        self.data.avg_latency_ms = sum(self.data._latencies) / len(self.data._latencies)

    def get_stats(self) -> dict:
        """
        Rolls up the running counters into a single stats dict for
        GET /api/analytics.

        Why percentiles (p50/p95/p99) alongside the average: an average
        latency hides exactly the thing you usually care about most --
        the slow tail. If 95 requests take 200ms and 5 take 8 seconds,
        the average (~600ms) looks fine while 5% of your users are
        having a bad time. p95/p99 surface that tail directly. p50
        (median) is included alongside average since a skewed
        distribution can also make the average look worse than what a
        "typical" request actually experiences.
        """
        uptime = time.time() - self._start_time
        latencies = sorted(self.data._latencies) if self.data._latencies else [0]

        return {
            "total_requests": self.data.total_requests,
            "total_errors": self.data.total_errors,
            "error_rate": round(
                self.data.total_errors / max(self.data.total_requests, 1) * 100, 2
            ),
            "requests_per_model": dict(self.data.requests_per_model),
            "requests_per_query_type": dict(self.data.requests_per_query_type),
            "cache_hit_rate": round(
                self.data.cache_hits
                / max(self.data.cache_hits + self.data.cache_misses, 1)
                * 100,
                2,
            ),
            "fallback_count": self.data.fallback_count,
            "avg_latency_ms": round(self.data.avg_latency_ms, 2),
            "p50_latency_ms": round(latencies[len(latencies) // 2], 2),
            "p95_latency_ms": round(latencies[int(len(latencies) * 0.95)], 2),
            "p99_latency_ms": round(latencies[int(len(latencies) * 0.99)], 2),
            "total_tokens_used": self.data.total_tokens_used,
            "uptime_seconds": round(uptime, 1),
        }

    def reset(self) -> None:
        """Used by POST /api/analytics/reset -- e.g. after a load test,
        or when starting a fresh measurement window without restarting
        the whole server."""
        self.data = AnalyticsSnapshot()
        self._start_time = time.time()
