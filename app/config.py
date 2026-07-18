"""
Application configuration.

We use pydantic-settings instead of raw os.environ / python-dotenv calls
scattered through the code. This gives us:
  - Type validation on every setting (a bad int in .env fails fast at
    startup instead of causing a weird bug three requests later)
  - One documented source of truth for every tunable value
  - Easy overriding in tests (Settings(default_model="foo")) without
    touching the real .env file
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central app configuration, populated from environment variables /
    a .env file. Field names map to UPPER_SNAKE_CASE env vars by default
    (e.g. `ollama_base_url` <- OLLAMA_BASE_URL) -- pydantic-settings does
    this case-insensitive matching for us.
    """

    app_name: str = "AI Inference Gateway"
    app_version: str = "1.0.0"

    # Where the local Ollama server is listening. Used by the LLM
    # service in Phase 2 to make httpx calls.
    ollama_base_url: str = "http://localhost:11434"

    # Fallback model used when the query router (Phase 5) hasn't picked
    # a specific one, or when no model is specified in the request. Must
    # be an exact Ollama tag (e.g. "gemma3:4b"), not a bare family name,
    # or lookups against locally pulled models will never match.
    default_model: str = "gemma3:4b"
    default_temperature: float = 0.7
    default_max_tokens: int = 1024

    log_level: str = "INFO"

    # How long (seconds) we'll wait on an Ollama call before raising
    # LLMTimeoutError (Phase 2).
    request_timeout: int = 30

    # --- Phase 2: response cache (see ResponseCache in llm_service.py) ---
    cache_max_size: int = 100
    cache_ttl_seconds: int = 600
    # Only responses generated below this temperature get cached -- high
    # temperature is intentionally non-deterministic, so caching it would
    # just serve stale/wrong-flavored answers for what should be a fresh
    # roll each time.
    cache_temperature_threshold: float = 0.3

    # --- Phase 2: startup warm-up ---
    preload_model_on_startup: bool = True

    # How long Ollama keeps a model resident in VRAM after last use.
    # Reuses the existing OLLAMA_KEEP_ALIVE env var (rather than adding a
    # near-duplicate KEEP_ALIVE var) since that value already means the
    # same thing wherever it's set.
    keep_alive: str = Field(default="30m", alias="OLLAMA_KEEP_ALIVE")

    # --- Phase 6: rate limiting (see app/core/rate_limiter.py) ---
    # Max requests a single client (identified by IP) may make within
    # rate_limit_window seconds before getting a 429.
    rate_limit_requests: int = 30
    rate_limit_window: int = 60

    # --- Phase 6: circuit breaker (see OllamaCircuitBreaker in
    # app/services/llm_service.py) ---
    # Consecutive Ollama connection/timeout failures before the breaker
    # trips OPEN and starts failing fast instead of attempting real calls.
    circuit_breaker_threshold: int = 3
    # How long (seconds) the breaker stays OPEN before allowing one
    # HALF_OPEN test request through to check if Ollama has recovered.
    circuit_breaker_reset: int = 30

    # Tells pydantic-settings to also read from a .env file in the repo
    # root, in addition to real environment variables (which always win).
    # populate_by_name lets fields still be set by their Python name
    # (e.g. Settings(keep_alive=...) in tests) even though keep_alive
    # also has an explicit env alias.
    model_config = SettingsConfigDict(
        env_file=".env", populate_by_name=True, extra="ignore"
    )


# A single shared instance, imported everywhere else in the app
# (`from app.config import settings`). Avoids re-parsing .env repeatedly
# and gives every module the same view of configuration.
settings = Settings()
