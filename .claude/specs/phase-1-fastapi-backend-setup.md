# Phase 1 — FastAPI Backend Setup

## Goal
Set up the complete FastAPI backend skeleton with working endpoints,
Pydantic models, configuration, middleware, error handling, and
dependency injection. After this phase, the server runs, all routes
return placeholder responses, and the architecture is production-ready
for adding Ollama integration in Phase 2.

---

## 1. Configuration (app/config.py)

Use pydantic-settings to load environment variables from .env file.

```python
class Settings(BaseSettings):
    app_name: str = "AI Inference Gateway"
    app_version: str = "1.0.0"
    ollama_base_url: str = "http://localhost:11434"
    default_model: str = "gemma3"
    default_temperature: float = 0.7
    default_max_tokens: int = 1024
    log_level: str = "INFO"
    request_timeout: int = 30

    model_config = SettingsConfigDict(env_file=".env")
```

Create a single settings instance that the whole app imports.

---

## 2. Custom Exceptions (app/core/exceptions.py)

Define these custom exception classes inheriting from base Exception:

| Exception             | HTTP Code | Purpose                        |
|-----------------------|-----------|--------------------------------|
| OllamaConnectionError | 503       | Ollama server unreachable      |
| ModelNotFoundError     | 404       | Requested model not available  |
| LLMTimeoutError        | 504       | Model response took too long   |
| OutputParsingError     | 500       | LLM returned unparseable data  |
| TokenLimitExceeded     | 400       | Prompt exceeds model limit     |

Each exception should have:
- message: str
- status_code: int (class-level default)
- detail: str or None (optional extra info)

Register global exception handlers in main.py that catch these
and return structured JSON:
```json
{
    "error": "ModelNotFoundError",
    "message": "Model 'xyz' not found in Ollama",
    "status_code": 404,
    "detail": null
}
```

---

## 3. Pydantic Request Models (app/models/requests.py)

### ChatRequest
- prompt: str with Field(min_length=1, max_length=10000)
- model: str or None with Field(default=None, description="If None, auto-route to best model")
- temperature: float with Field(default=0.7, ge=0.0, le=2.0)
- max_tokens: int with Field(default=1024, gt=0, le=4096)
- stream: bool with Field(default=False)
- system_prompt: str or None with Field(default=None, max_length=2000)

---

## 4. Pydantic Response Models (app/models/responses.py)

### ResponseMetadata
- model_used: str
- query_type: str (one of "simple", "complex", "creative", "technical")
- latency_ms: float
- tokens_prompt: int
- tokens_completion: int
- tokens_total: int
- temperature: float
- fallback_used: bool with default False
- timestamp: str (ISO format datetime string)

### ChatResponse
- response: str
- metadata: ResponseMetadata

### ModelInfo
- name: str
- size_gb: float or None with default None
- parameter_count: str or None with default None
- quantization: str or None with default None
- capabilities: list[str] with default empty list
- loaded: bool with default False

### ModelsListResponse
- models: list[ModelInfo]
- total: int

### HealthResponse
- status: str (either "healthy" or "unhealthy")
- ollama_connected: bool
- models_loaded: int
- uptime_seconds: float
- version: str

### ErrorResponse
- error: str
- message: str
- status_code: int
- detail: str or None with default None

---

## 5. Routers

### Health Router (app/routers/health.py)
- Path prefix: none (root level)
- Tag: "Health"

GET /health
- Returns: HealthResponse
- For now return placeholder with status "healthy", ollama_connected false,
  models_loaded 0, uptime_seconds 0.0, version from settings

### Chat Router (app/routers/chat.py)
- Path prefix: /api/chat
- Tag: "Chat"

POST /
- Input: ChatRequest (request body)
- Returns: ChatResponse
- For now return placeholder response with text "Placeholder: LLM integration coming in Phase 2"
  and hardcoded metadata with model_used "placeholder", query_type "simple",
  latency_ms 0.0, all token counts 0, temperature 0.7, fallback_used false,
  timestamp as current ISO time

POST /stream
- Input: ChatRequest (request body)
- Returns: StreamingResponse with media_type "text/event-stream"
- For now yield 5 placeholder SSE tokens in this format:
  data: {"token": "This ", "done": false}
  data: {"token": "is ", "done": false}
  data: {"token": "a ", "done": false}
  data: {"token": "placeholder ", "done": false}
  data: {"token": "stream.", "done": false}
  data: {"token": "", "done": true, "metadata": {full metadata object}}

### Models Router (app/routers/models.py)
- Path prefix: /api/models
- Tag: "Models"

GET /
- Returns: ModelsListResponse
- For now return empty models list with total 0

GET /{model_name}/status
- Path param: model_name (str)
- Returns: ModelInfo
- For now raise ModelNotFoundError since no models connected yet

---

## 6. Middleware (app/core/middleware.py)

### RequestLoggingMiddleware
- Runs on EVERY request
- Logs: method, path, status_code, latency_ms
- Generates a unique request_id (UUID4) per request
- Adds request_id to response headers as "X-Request-ID"
- Uses structlog for JSON formatted logging

### CORS Middleware
- Add in main.py
- allow_origins: ["*"]
- allow_methods: ["*"]
- allow_headers: ["*"]

---

## 7. Dependencies (app/core/dependencies.py)

Create placeholder dependency functions that will be filled in later phases:

- get_llm_service() returns None with docstring "Returns LLM service instance. Placeholder for Phase 2."
- get_router_service() returns None with docstring "Returns router service instance. Placeholder for Phase 5."
- get_prompt_service() returns None with docstring "Returns prompt service instance. Placeholder for Phase 3."

---

## 8. App Entry Point (app/main.py)

### FastAPI App Setup
- title: "AI Inference Gateway"
- description: "Intelligent multi-model LLM gateway with smart routing"
- version: from settings

### Startup
- Log "AI Inference Gateway started" with structlog
- Record startup time (for uptime calculation in /health)

### Shutdown
- Log "AI Inference Gateway shutting down"

### Include
- All three routers (health, chat, models)
- CORS middleware
- RequestLoggingMiddleware

### Exception Handlers
- Register handlers for all custom exceptions
- Register a catch-all handler for unexpected Exception

### Static Files and Templates
- Mount /static directory for CSS/JS/assets
- Set up Jinja2Templates pointing to templates/ directory

### Root Route
GET /
- Serve a simple placeholder HTML page (not the full chat UI yet, that is Phase 7)
- Just return: "AI Inference Gateway is running. API docs at /docs"

---

## 9. Verification Checklist

After implementation, ALL of these must work:

Server starts without errors:
uvicorn app.main:app --reload --port 8000

Health check returns valid JSON:
curl http://localhost:8000/health

Chat endpoint accepts request and returns placeholder:
curl -X POST http://localhost:8000/api/chat -H "Content-Type: application/json" -d '{"prompt": "hello"}'

Stream endpoint returns SSE events:
curl -X POST http://localhost:8000/api/chat/stream -H "Content-Type: application/json" -d '{"prompt": "hello"}'

Models endpoint returns empty list:
curl http://localhost:8000/api/models

Model status returns 404 error properly:
curl http://localhost:8000/api/models/gemma/status

Validation rejects bad input (empty prompt):
curl -X POST http://localhost:8000/api/chat -H "Content-Type: application/json" -d '{"prompt": ""}'

Validation rejects bad temperature:
curl -X POST http://localhost:8000/api/chat -H "Content-Type: application/json" -d '{"prompt": "hello", "temperature": 5.0}'

Auto-generated docs page loads at http://localhost:8000/docs in browser

Request logging appears in terminal with structlog format showing method, path, status_code, latency_ms, request_id

---

## 10. Files Modified in This Phase

| File                        | Action                   |
|-----------------------------|--------------------------|
| app/main.py                 | Full implementation      |
| app/config.py               | Full implementation      |
| app/models/requests.py      | Full implementation      |
| app/models/responses.py     | Full implementation      |
| app/routers/health.py       | Full implementation      |
| app/routers/chat.py         | Full implementation      |
| app/routers/models.py       | Full implementation      |
| app/core/exceptions.py      | Full implementation      |
| app/core/middleware.py       | Full implementation      |
| app/core/dependencies.py    | Placeholder implementation |
| app/services/*              | NOT touched (Phase 2+)   |
| templates/*                 | NOT touched (Phase 7)    |
| static/*                    | NOT touched (Phase 7)    |

## 
- IMPORTANT: Add detailed comments explaining every concept, pattern, and "why" behind each implementation. Comments should teach the developer what each piece does and why it exists. Treat the codebase as a learning resource.