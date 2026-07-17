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
    # a specific one, or when no model is specified in the request.
    default_model: str = "gemma3"
    default_temperature: float = 0.7
    default_max_tokens: int = 1024

    log_level: str = "INFO"

    # How long (seconds) we'll wait on an Ollama call before raising
    # LLMTimeoutError (Phase 2).
    request_timeout: int = 30

    # Tells pydantic-settings to also read from a .env file in the repo
    # root, in addition to real environment variables (which always win).
    model_config = SettingsConfigDict(env_file=".env")


# A single shared instance, imported everywhere else in the app
# (`from app.config import settings`). Avoids re-parsing .env repeatedly
# and gives every module the same view of configuration.
settings = Settings()
