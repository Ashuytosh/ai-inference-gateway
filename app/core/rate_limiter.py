"""
Sliding-window rate limiter -- pure logic, no FastAPI/HTTP imports here
on purpose (mirrors app/services/router_service.py and prompt_service.py,
which are also framework-agnostic). The FastAPI-facing dependency
function that uses this class (check_rate_limit) lives in
app/core/dependencies.py instead, alongside every other Depends()
target -- keeping it here would create a circular import, since that
dependency function needs get_rate_limiter() from dependencies.py, which
in turn needs RateLimiter from this module.
"""

from collections import defaultdict
import time


class RateLimiter:
    """
    Per-client sliding-window request limiter.

    Why rate limit an AI gateway specifically?
    - Every chat request eventually costs real GPU time and VRAM on the
      Ollama box behind this gateway -- unlike a typical CRUD API where
      a request is cheap, here each one is expensive by design.
    - Ollama itself effectively serializes generation per model (it can't
      usefully process unlimited concurrent requests against one loaded
      model), so a client that fires requests faster than Ollama can
      drain them just piles up latency for everyone else.
    - Protects against runaway/buggy clients (e.g. an accidental retry
      loop) monopolizing the one shared backend.

    Why a *sliding* window instead of a fixed window (e.g. "reset the
    counter every 60 seconds on the clock")? A fixed window lets a client
    send its full quota right before a window boundary and its full quota
    again right after -- effectively 2x the intended rate for a brief
    burst around the boundary. A sliding window (only counting requests
    within the last `window_seconds`, measured from *now*, not from a
    fixed clock boundary) doesn't have that gap.

    Why in-memory rather than Redis/a shared store? This gateway runs as
    a single process serving one local Ollama instance -- there's no
    multi-instance deployment to coordinate across, so the added
    complexity of a distributed store would buy nothing here.
    """

    def __init__(self, max_requests: int = 30, window_seconds: int = 60) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        # One timestamp list per client identifier (IP address in
        # practice -- see check_rate_limit in dependencies.py). defaultdict
        # means a never-seen-before client gets a fresh empty list
        # automatically instead of needing an explicit "if client_id not
        # in self.requests" check on every call.
        self.requests: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, client_id: str) -> tuple[bool, dict]:
        """
        Returns (allowed, info) where info always carries limit/remaining/
        reset_seconds -- useful both for the X-RateLimit-* response
        headers on a successful request and for the 429 error body when
        the limit is hit.
        """
        now = time.time()
        window_start = now - self.window_seconds

        # Trim timestamps that have aged out of the window -- this is
        # what makes it "sliding" rather than a fixed bucket that only
        # clears on a schedule.
        self.requests[client_id] = [
            ts for ts in self.requests[client_id] if ts > window_start
        ]

        current_count = len(self.requests[client_id])
        remaining = self.max_requests - current_count

        if current_count >= self.max_requests:
            # The window won't have room again until the oldest request
            # currently counted against this client ages out.
            reset_time = self.requests[client_id][0] + self.window_seconds
            return False, {
                "limit": self.max_requests,
                "remaining": 0,
                "reset_seconds": round(reset_time - now, 1),
            }

        self.requests[client_id].append(now)
        return True, {
            "limit": self.max_requests,
            "remaining": remaining - 1,
            "reset_seconds": self.window_seconds,
        }
