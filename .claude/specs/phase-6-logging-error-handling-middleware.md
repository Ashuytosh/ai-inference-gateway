# Phase 6 — Logging, Error Handling & Middleware (Production Hardening)

## Goal
Harden the gateway for production readiness. After this phase, every
request has structured JSON logging with full traceability, errors are
handled gracefully at every level, rate limiting protects the server,
and a request analytics system tracks usage patterns. This phase
transforms "it works" into "it works reliably under real conditions."

IMPORTANT: Add detailed comments explaining every concept, pattern, and
"why" behind each implementation. Comments should teach the developer
what each piece does and why it exists. Treat the codebase as a learning
resource.

---

## 1. Structured Logging Overhaul

### Current State
The app uses structlog but minimally. Phase 6 upgrades logging to
production-grade with consistent structure across every component.

### Configure structlog properly (app/core/logging_config.py)

Create a new file that configures structlog once at app startup:

```python
import structlog
import logging
import sys

def setup_logging(log_level: str = "INFO"):
    """
    Configure structlog for production JSON logging.
    
    Why structlog over standard logging?
    - Standard logging produces unstructured text: "INFO: request completed"
    - structlog produces structured JSON: {"event": "request_completed", "method": "POST", "latency_ms": 234}
    - Structured logs can be queried, filtered, and analyzed by tools like
      Datadog, Grafana, ELK stack. Unstructured text can't.
    """
    
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer()  # production: JSON
            # For development, swap JSONRenderer with:
            # structlog.dev.ConsoleRenderer()  # pretty colored output
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(log_level)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
```

Call setup_logging(settings.log_level) in app/main.py startup BEFORE
any other code runs.

### Add contextual logging throughout the request lifecycle

Use structlog.contextvars to bind request-scoped variables that
automatically appear in every log line during that request:

In middleware.py, after generating request_id:
```python
structlog.contextvars.clear_contextvars()
structlog.contextvars.bind_contextvars(
    request_id=request_id,
    method=request.method,
    path=request.url.path,
)
```

Now every log line from llm_service, router_service, prompt_service,
and output_parser during that request automatically includes
request_id, method, and path — without any of those services needing
to know about HTTP requests.

---

## 2. Enhanced Request Logging Middleware

### Update app/core/middleware.py

Upgrade RequestLoggingMiddleware to capture more information:

```python
async def dispatch(self, request, call_next):
    request_id = str(uuid.uuid4())
    start_time = time.perf_counter()
    
    # Bind context for all downstream loggers
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        client_ip=request.client.host if request.client else "unknown",
    )
    
    # Track request body size for POST requests
    body_size = 0
    if request.method == "POST":
        body = await request.body()
        body_size = len(body)
    
    try:
        response = await call_next(request)
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Latency-MS"] = str(round(latency_ms, 2))
        
        logger.info(
            "request_completed",
            status_code=response.status_code,
            latency_ms=round(latency_ms, 2),
            body_size_bytes=body_size,
        )
        return response
        
    except Exception as exc:
        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.error(
            "request_failed",
            error=str(exc),
            error_type=type(exc).__name__,
            latency_ms=round(latency_ms, 2),
        )
        raise
```

NOTE: Reading request.body() in middleware can cause issues with
FastAPI's body parsing. Use a safer approach — either read
content-length header instead, or use a middleware pattern that
doesn't consume the body. Research the best approach for your
FastAPI version and implement accordingly.

---

## 3. Request Analytics Tracker

Create app/core/analytics.py — a simple in-memory analytics system
that tracks usage patterns.

### Class: RequestAnalytics

```python
from dataclasses import dataclass, field
from collections import defaultdict
import time

@dataclass
class AnalyticsSnapshot:
    total_requests: int = 0
    total_errors: int = 0
    requests_per_model: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    requests_per_query_type: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    cache_hits: int = 0
    cache_misses: int = 0
    fallback_count: int = 0
    avg_latency_ms: float = 0.0
    total_tokens_used: int = 0
    # Track latencies for percentile calculation
    _latencies: list[float] = field(default_factory=list)

class RequestAnalytics:
    def __init__(self):
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
    ):
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
        
        # Rolling average latency
        self.data.avg_latency_ms = (
            sum(self.data._latencies) / len(self.data._latencies)
        )
    
    def get_stats(self) -> dict:
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
                self.data.cache_hits / max(self.data.cache_hits + self.data.cache_misses, 1) * 100, 2
            ),
            "fallback_count": self.data.fallback_count,
            "avg_latency_ms": round(self.data.avg_latency_ms, 2),
            "p50_latency_ms": round(latencies[len(latencies) // 2], 2),
            "p95_latency_ms": round(latencies[int(len(latencies) * 0.95)], 2),
            "p99_latency_ms": round(latencies[int(len(latencies) * 0.99)], 2),
            "total_tokens_used": self.data.total_tokens_used,
            "uptime_seconds": round(uptime, 1),
        }
    
    def reset(self):
        self.data = AnalyticsSnapshot()
        self._start_time = time.time()
```

### Add analytics endpoint

Create a new router: app/routers/analytics.py

```python
router = APIRouter(prefix="/api/analytics", tags=["Analytics"])

@router.get("")
async def get_analytics(analytics = Depends(get_analytics)):
    return analytics.get_stats()

@router.post("/reset")
async def reset_analytics(analytics = Depends(get_analytics)):
    analytics.reset()
    return {"message": "Analytics reset"}
```

### Wire into chat router

After every successful chat response, call:
```python
analytics.record_request(
    model=metadata.model_used,
    query_type=metadata.query_type,
    latency_ms=metadata.latency_ms,
    tokens_total=metadata.tokens_total,
    cached=metadata.cached,
    fallback_used=metadata.fallback_used,
)
```

### Add dependency

In dependencies.py:
```python
_analytics: RequestAnalytics | None = None

async def get_analytics() -> RequestAnalytics:
    global _analytics
    if _analytics is None:
        _analytics = RequestAnalytics()
    return _analytics
```

---

## 4. Rate Limiting

Add simple in-memory rate limiting to prevent abuse and protect Ollama
from being overwhelmed.

### Create rate limiter in app/core/rate_limiter.py

```python
class RateLimiter:
    """
    Simple sliding-window rate limiter.
    
    Why rate limit an AI gateway?
    - LLM calls are expensive (GPU time, VRAM, power)
    - Without limits, one client can monopolize the server
    - Ollama can only handle one request at a time per model
    - Prevents accidental infinite loops from buggy clients
    """
    
    def __init__(self, max_requests: int = 30, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: dict[str, list[float]] = defaultdict(list)
    
    def is_allowed(self, client_id: str) -> tuple[bool, dict]:
        now = time.time()
        window_start = now - self.window_seconds
        
        # Remove expired timestamps
        self.requests[client_id] = [
            ts for ts in self.requests[client_id] if ts > window_start
        ]
        
        current_count = len(self.requests[client_id])
        remaining = self.max_requests - current_count
        
        if current_count >= self.max_requests:
            # Calculate when the oldest request in the window expires
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
```

### Apply rate limiting in middleware or as a dependency

Create a FastAPI dependency that checks rate limits:

```python
async def check_rate_limit(
    request: Request,
    limiter: RateLimiter = Depends(get_rate_limiter),
):
    client_id = request.client.host if request.client else "unknown"
    allowed, info = limiter.is_allowed(client_id)
    
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Rate limit exceeded",
                "message": f"Max {info['limit']} requests per minute",
                "retry_after_seconds": info["reset_seconds"],
            }
        )
    return info
```

Apply this dependency to the chat endpoints only (not health or models):

```python
@router.post("")
async def chat(
    request: ChatRequest,
    rate_info: dict = Depends(check_rate_limit),
    ...
):
```

Add rate limit headers to response:
```
X-RateLimit-Limit: 30
X-RateLimit-Remaining: 25
X-RateLimit-Reset: 60
```

### Configuration

Add to Settings in config.py:
```python
rate_limit_requests: int = 30
rate_limit_window: int = 60
```

Add to .env:
```
RATE_LIMIT_REQUESTS=30
RATE_LIMIT_WINDOW=60
```

---

## 5. Enhanced Error Recovery

### Add circuit breaker pattern for Ollama connection

When Ollama goes down, don't keep hammering it with requests.
Implement a simple circuit breaker:

```python
class OllamaCircuitBreaker:
    """
    Circuit breaker pattern: when Ollama fails repeatedly, stop trying
    for a cooldown period instead of wasting time on requests that will
    definitely fail.
    
    States:
    - CLOSED: normal operation, requests go through
    - OPEN: Ollama is down, reject requests immediately with 503
    - HALF_OPEN: cooldown expired, try one request to see if Ollama is back
    
    Why? Without this, if Ollama crashes, every single request waits
    30 seconds for timeout × 3 retries = 90 seconds before failing.
    With a circuit breaker, after 3 consecutive failures, subsequent
    requests fail instantly with a clear message: "Ollama is temporarily
    unavailable, retrying in X seconds."
    """
    
    def __init__(self, failure_threshold: int = 3, reset_timeout: int = 30):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.state = "CLOSED"   # CLOSED, OPEN, HALF_OPEN
        self.last_failure_time = 0.0
    
    def record_success(self):
        self.failure_count = 0
        self.state = "CLOSED"
    
    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
    
    def can_proceed(self) -> tuple[bool, str]:
        if self.state == "CLOSED":
            return True, "closed"
        
        if self.state == "OPEN":
            elapsed = time.time() - self.last_failure_time
            if elapsed >= self.reset_timeout:
                self.state = "HALF_OPEN"
                return True, "half_open"
            return False, f"open (retry in {round(self.reset_timeout - elapsed)}s)"
        
        # HALF_OPEN: let one request through to test
        return True, "half_open"
```

Integrate into OllamaService — before every LLM call, check
circuit_breaker.can_proceed(). After success, record_success().
After failure, record_failure().

Add circuit breaker state to /health endpoint response.

---

## 6. Graceful Degradation Responses

When the system is under stress or Ollama is down, provide helpful
degradation messages instead of raw errors:

```python
DEGRADATION_MESSAGES = {
    "ollama_down": {
        "response": "I'm currently unable to process your request because the AI model server is temporarily unavailable. Please try again in a few moments.",
        "suggestion": "You can check the server status at /health"
    },
    "rate_limited": {
        "response": "You've sent too many requests. Please wait a moment before trying again.",
        "suggestion": "Rate limit resets in {reset_seconds} seconds"
    },
    "model_overloaded": {
        "response": "The requested model is currently busy. Your request has been routed to an alternative model.",
        "suggestion": None
    }
}
```

---

## 7. Configuration Updates

Add to Settings in config.py:
```python
# Rate limiting
rate_limit_requests: int = 30
rate_limit_window: int = 60

# Circuit breaker
circuit_breaker_threshold: int = 3
circuit_breaker_reset: int = 30
```

Add to .env:
```
RATE_LIMIT_REQUESTS=30
RATE_LIMIT_WINDOW=60
CIRCUIT_BREAKER_THRESHOLD=3
CIRCUIT_BREAKER_RESET=30
```

---

## 8. Verification Checklist

After implementation, ALL of these must work:

### Structured logging outputs JSON:
Start the server and send a request. Terminal output should show
JSON-formatted log lines with timestamp, level, event, request_id,
method, path, and any bound context variables.

### Request ID traceability:
```bash
curl http://localhost:8000/api/chat -X POST \
  -H "Content-Type: application/json" \
  -d '{"prompt": "hello"}'
```
Response should have X-Request-ID and X-Latency-MS headers.
Same request_id should appear in all related terminal log lines
(classification, routing, LLM call, response).

### Analytics endpoint:
```bash
# Send a few requests first, then:
curl http://localhost:8000/api/analytics
```
Should return stats: total_requests, requests_per_model, cache_hit_rate,
avg_latency_ms, p50/p95/p99 latencies, total_tokens_used.

### Analytics reset:
```bash
curl -X POST http://localhost:8000/api/analytics/reset
curl http://localhost:8000/api/analytics
```
Should show zeroed stats after reset.

### Rate limiting:
Send 31+ requests rapidly within 60 seconds:
```bash
for i in $(seq 1 35); do
  curl -s -o /dev/null -w "%{http_code}\n" \
    -X POST http://localhost:8000/api/chat \
    -H "Content-Type: application/json" \
    -d '{"prompt": "hi"}'
done
```
First 30 should return 200, remaining should return 429 with
retry_after_seconds in the response.

### Rate limit headers present:
```bash
curl -v -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "hello"}'
```
Response should include X-RateLimit-Limit, X-RateLimit-Remaining,
X-RateLimit-Reset headers.

### Circuit breaker (test with Ollama stopped):
1. Stop Ollama
2. Send 4+ requests
3. First 3 should timeout normally (with retry)
4. Request 4+ should fail INSTANTLY with "Ollama temporarily unavailable"
5. Start Ollama again
6. Wait 30 seconds (circuit breaker reset)
7. Next request should succeed (half-open → closed)

### Health endpoint shows circuit breaker state:
```bash
curl http://localhost:8000/health
```
Should include circuit breaker state (closed/open/half_open).

### All Phase 1-5 features still work:
- Smart routing still classifies and routes correctly
- Structured output parsing still works
- Prompt engineering still applies strategies
- Caching still works
- Fallback chains still trigger on failure
- VRAM-aware routing still avoids unnecessary swaps

---

## 9. Files Modified in This Phase

| File                            | Action                              |
|--------------------------------|-------------------------------------|
| app/core/logging_config.py      | NEW — structlog configuration       |
| app/core/analytics.py           | NEW — request analytics tracker     |
| app/core/rate_limiter.py        | NEW — sliding window rate limiter   |
| app/core/middleware.py          | Enhanced logging, context binding   |
| app/routers/analytics.py        | NEW — analytics endpoints           |
| app/routers/chat.py             | Wire analytics, rate limiting       |
| app/routers/health.py           | Add circuit breaker state           |
| app/services/llm_service.py     | Add circuit breaker integration     |
| app/core/dependencies.py        | Add analytics, rate limiter deps    |
| app/config.py                   | Add rate limit, circuit breaker settings |
| app/main.py                     | Setup logging, include analytics router |
| .env                            | Add new configuration variables     |
| templates/*                     | NOT touched (Phase 7)               |
| static/*                        | NOT touched (Phase 7)               |