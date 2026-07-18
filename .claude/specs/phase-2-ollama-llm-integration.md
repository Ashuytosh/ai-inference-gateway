# Phase 2 — Ollama LLM Integration

## Goal
Connect the FastAPI backend to Ollama running locally. After this phase,
the gateway can list real models, check their status, send prompts to
any model, receive responses, stream tokens in real-time, handle errors
gracefully, and preload the default model on startup. All placeholder
responses from Phase 1 get replaced with real LLM calls.

IMPORTANT: Add detailed comments explaining every concept, pattern, and
"why" behind each implementation. Comments should teach the developer
what each piece does and why it exists. Treat the codebase as a learning
resource.

---

## 1. Understanding Ollama's API

Ollama runs a local HTTP server at http://localhost:11434 with these
endpoints that we will use:

### POST /api/chat
Send a chat completion request. This is the main endpoint for
generating responses.

Request body:
```json
{
    "model": "qwen2.5:7b",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain RAG in simple terms"}
    ],
    "stream": false,
    "options": {
        "temperature": 0.7,
        "num_predict": 1024
    }
}
```

Response (non-streaming):
```json
{
    "model": "qwen2.5:7b",
    "message": {
        "role": "assistant",
        "content": "RAG stands for..."
    },
    "done": true,
    "total_duration": 5000000000,
    "prompt_eval_count": 45,
    "eval_count": 180
}
```

Response (streaming): Returns newline-delimited JSON chunks:
```
{"model":"qwen2.5:7b","message":{"role":"assistant","content":"RAG"},"done":false}
{"model":"qwen2.5:7b","message":{"role":"assistant","content":" stands"},"done":false}
{"model":"qwen2.5:7b","message":{"role":"assistant","content":" for"},"done":false}
...
{"model":"qwen2.5:7b","message":{"role":"assistant","content":""},"done":true,"total_duration":5000000000,"prompt_eval_count":45,"eval_count":180}
```

Note: Ollama uses "num_predict" not "max_tokens" for limiting output tokens.
Note: total_duration is in nanoseconds. Divide by 1_000_000 to get milliseconds.
Note: prompt_eval_count = tokens in the prompt, eval_count = tokens generated.

### GET /api/tags
List all locally available models.

Response:
```json
{
    "models": [
        {
            "name": "qwen2.5:7b",
            "size": 4700000000,
            "details": {
                "parameter_size": "7B",
                "quantization_level": "Q4_K_M",
                "family": "qwen2"
            }
        }
    ]
}
```

Note: size is in bytes. Divide by 1_000_000_000 to get GB.

### POST /api/show
Get detailed info about a specific model.

Request body:
```json
{
    "name": "qwen2.5:7b"
}
```

### GET /api/ps
List currently loaded/running models in VRAM.

Response:
```json
{
    "models": [
        {
            "name": "qwen2.5:7b",
            "size": 4700000000,
            "size_vram": 4700000000,
            "expires_at": "2026-07-18T15:30:00Z"
        }
    ]
}
```

This is critical for VRAM-aware optimization — tells us which models
are currently loaded so we can avoid unnecessary swaps.

---

## 2. LLM Service Implementation (app/services/llm_service.py)

### Class: OllamaService

This is the core service that communicates with Ollama. All communication
happens via httpx.AsyncClient for non-blocking async HTTP calls.

#### Constructor
```
__init__(self, base_url: str, timeout: int)
```
- Create httpx.AsyncClient with base_url and timeout
- Initialize a variable to track the currently loaded model name (for VRAM optimization)
- Store the base_url and timeout as instance variables

#### Method: health_check() -> bool
- Send GET request to {base_url}/api/tags
- If response status is 200, return True
- If connection fails (httpx.ConnectError), return False
- This is used by the /health endpoint

#### Method: list_models() -> list[ModelInfo]
- Send GET request to {base_url}/api/tags
- Parse the response JSON
- For each model in the response, create a ModelInfo object:
  - name: from model["name"]
  - size_gb: from model["size"] / 1_000_000_000, rounded to 1 decimal
  - parameter_count: from model["details"]["parameter_size"] if available
  - quantization: from model["details"]["quantization_level"] if available
  - capabilities: assign based on model name (see capability mapping below)
  - loaded: check against currently loaded models via get_loaded_models()
- Return list of ModelInfo objects
- On connection error, raise OllamaConnectionError

#### Method: get_loaded_models() -> list[str]
- Send GET request to {base_url}/api/ps
- Parse response and return list of model names currently in VRAM
- This is the VRAM tracking optimization — we know exactly what is loaded

#### Method: get_model_status(model_name: str) -> ModelInfo or None
- First call list_models() to check if model exists
- Find the matching model by name
- If not found, return None
- If found, return the ModelInfo with loaded status checked against get_loaded_models()

#### Method: chat(model, messages, temperature, max_tokens) -> OllamaResponse
- Build the request body for POST /api/chat with stream=false
- messages should be a list of dicts with "role" and "content" keys
- Map max_tokens to num_predict in the options
- Send POST request to {base_url}/api/chat
- Parse the response and return an OllamaResponse dataclass containing:
  - text: the generated text from message.content
  - model: model name used
  - prompt_tokens: from prompt_eval_count
  - completion_tokens: from eval_count
  - total_duration_ms: from total_duration / 1_000_000
- Update the currently loaded model tracker
- On timeout, raise LLMTimeoutError
- On connection error, raise OllamaConnectionError
- On model not found (404 or model error), raise ModelNotFoundError

#### Method: chat_stream(model, messages, temperature, max_tokens) -> AsyncGenerator
- Build same request body but with stream=true
- Send POST request with stream=True on httpx (this returns chunks)
- Use async iteration over response:
  - For each line in the response stream
  - Parse each line as JSON
  - If done is false, yield the token text from message.content
  - If done is true, yield a final object/dict with metadata:
    - prompt_tokens, completion_tokens, total_duration_ms
- Update the currently loaded model tracker
- Handle same errors as chat()

#### Method: preload_model(model_name: str) -> bool
- Send a chat request with an empty user message to force Ollama to load the model
- Use messages: [{"role": "user", "content": "hi"}] with num_predict: 1
- This forces the model into VRAM without generating a real response
- Set keep_alive to "30m" in the request body
- Return True if successful, False if failed
- Log the preload action with structlog

#### Method: close()
- Close the httpx.AsyncClient
- Called during FastAPI shutdown

### Model Capability Mapping
Define a dictionary that maps model names to their capabilities:
```
CAPABILITY_MAP = {
    "gemma3:4b": ["general", "simple", "fast"],
    "phi4-mini": ["logic", "math", "reasoning", "fast"],
    "qwen2.5:7b": ["general", "complex", "analysis", "detailed"],
    "qwen2.5-coder:7b": ["code", "technical", "debugging"],
    "mistral:7b": ["creative", "conversation", "writing"]
}
```
If a model name is not in the map, default to ["general"].

### OllamaResponse Dataclass
Create a simple dataclass or Pydantic model:
```
class OllamaResponse:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_duration_ms: float
```

---

## 3. Retry Logic with Exponential Backoff

Wrap the chat() and chat_stream() methods with retry logic.
Create a retry decorator or implement retry inline:

- Max attempts: 3
- Backoff: 1 second, then 2 seconds, then 4 seconds
- Retry on: LLMTimeoutError, httpx.ConnectError, httpx.ReadTimeout
- Do NOT retry on: ModelNotFoundError (retrying won't help)
- Log each retry attempt with structlog

This should be implemented as an async decorator similar to the
retry pattern shown in Module 1 of the learning material.

---

## 4. Response Caching (LRU Cache)

Implement a simple in-memory cache for repeated queries.

### Class: ResponseCache

#### Properties
- cache: dict mapping cache_key -> CacheEntry
- max_size: int = 100
- ttl_seconds: int = 600 (10 minutes)

#### CacheEntry dataclass
- response: OllamaResponse
- timestamp: float (time.time() when cached)

#### Method: get(key: str) -> OllamaResponse or None
- Check if key exists in cache
- If exists, check if TTL has expired
- If expired, delete the entry and return None
- If valid, return the cached response

#### Method: set(key: str, response: OllamaResponse)
- If cache is at max_size, remove the oldest entry
- Store the response with current timestamp

#### Method: make_key(prompt, model, temperature, system_prompt) -> str
- Create a hash of all parameters combined
- Use hashlib.sha256 for consistent hashing
- Only cache when temperature < 0.3 (low temperature = deterministic)
- Return None if temperature >= 0.3 (signal to skip caching)

#### Method: clear()
- Empty the entire cache

---

## 5. Update Dependencies (app/core/dependencies.py)

Replace the placeholder get_llm_service() with a real implementation:

### Singleton Pattern
Create a module-level variable to hold the OllamaService instance.
Use a function that creates it once and returns the same instance:

```
_llm_service: OllamaService | None = None

async def get_llm_service() -> OllamaService:
    global _llm_service
    if _llm_service is None:
        _llm_service = OllamaService(
            base_url=settings.ollama_base_url,
            timeout=settings.request_timeout
        )
    return _llm_service
```

Also create a get_response_cache() dependency:
```
_cache: ResponseCache | None = None

async def get_response_cache() -> ResponseCache:
    global _cache
    if _cache is None:
        _cache = ResponseCache()
    return _cache
```

---

## 6. Update App Entry Point (app/main.py)

### Startup Event
- Get the LLM service instance
- Call health_check() to verify Ollama is running
- If healthy, call preload_model() with the default model from settings
- Log the startup status: "Ollama connected, preloaded {model}" or
  "Warning: Ollama not available, running in degraded mode"
- Store the startup timestamp for uptime calculation

### Shutdown Event
- Get the LLM service instance
- Call close() to clean up httpx client
- Log "AI Inference Gateway shutting down"

---

## 7. Update Health Router (app/routers/health.py)

Replace placeholder with real implementation:

### GET /health
- Inject OllamaService via Depends(get_llm_service)
- Call health_check() on the service
- If healthy, call list_models() to get model count
- Call get_loaded_models() to see what is in VRAM
- Calculate uptime from startup timestamp
- Return HealthResponse with real data:
  ```json
  {
      "status": "healthy",
      "ollama_connected": true,
      "models_loaded": 2,
      "uptime_seconds": 3600.5,
      "version": "1.0.0"
  }
  ```
- If Ollama is not connected, return status "unhealthy" with
  ollama_connected false, don't crash

---

## 8. Update Models Router (app/routers/models.py)

Replace placeholders with real implementations:

### GET /api/models
- Inject OllamaService via Depends
- Call list_models() on the service
- Return ModelsListResponse with real model data
- On OllamaConnectionError, return empty list with a warning header

### GET /api/models/{model_name}/status
- Inject OllamaService via Depends
- Call get_model_status(model_name) on the service
- If model found, return ModelInfo
- If not found, raise ModelNotFoundError with helpful message
  listing available models

---

## 9. Update Chat Router (app/routers/chat.py)

Replace placeholders with real LLM calls:

### POST /api/chat
- Inject OllamaService and ResponseCache via Depends
- Determine which model to use:
  - If request.model is set, use that model
  - If request.model is None, use settings.default_model for now
    (smart routing comes in Phase 5)
- Build messages list:
  - If request.system_prompt is set, add system message first
  - Add user message with request.prompt
- Check cache first:
  - Generate cache key from request parameters
  - If cache hit, return cached response immediately with metadata
    showing cached=true (add this field to ResponseMetadata)
- Record start time for latency measurement
- Call llm_service.chat() with the parameters
- Calculate latency_ms from start time
- Build ChatResponse with real metadata:
  - model_used: from OllamaResponse
  - query_type: "general" for now (classification comes in Phase 5)
  - latency_ms: calculated
  - tokens_prompt: from OllamaResponse
  - tokens_completion: from OllamaResponse
  - tokens_total: sum of both
  - temperature: from request
  - fallback_used: false for now (fallback comes in Phase 5)
  - timestamp: current ISO time
- Cache the response if applicable
- Return ChatResponse

### POST /api/chat/stream
- Inject OllamaService via Depends
- Same model selection logic as /api/chat
- Build messages list same way
- Create an async generator function that:
  - Records start time
  - Calls llm_service.chat_stream()
  - For each yielded token, format as SSE event:
    data: {"token": "the_token", "done": false}
  - For the final metadata object, format as:
    data: {"token": "", "done": true, "metadata": {full metadata}}
  - Each SSE line must end with two newlines: "data: ...\n\n"
- Return StreamingResponse wrapping the generator
  with media_type="text/event-stream"

---

## 10. Add ResponseMetadata Update

Update app/models/responses.py to add cached field:

### ResponseMetadata (updated)
Add this field:
- cached: bool = False

This shows in the response when a cached result was returned
instead of a fresh LLM call.

---

## 11. Error Handling in All Updated Files

Every Ollama call must be wrapped in try/except:

- httpx.ConnectError → raise OllamaConnectionError("Cannot connect to Ollama at {base_url}")
- httpx.ReadTimeout → raise LLMTimeoutError("Model took too long to respond")
- httpx.HTTPStatusError with 404 → raise ModelNotFoundError("Model '{name}' not found")
- json.JSONDecodeError → raise OutputParsingError("Invalid response from Ollama")
- Any unexpected error → log with structlog, raise generic HTTPException 500

Never let an unhandled exception crash the server.

---

## 12. Configuration Update (app/config.py)

Add these new settings to the Settings class:

- cache_max_size: int = 100
- cache_ttl_seconds: int = 600
- cache_temperature_threshold: float = 0.3
- preload_model_on_startup: bool = True
- keep_alive: str = "30m"

Update .env file with matching variables:
```
CACHE_MAX_SIZE=100
CACHE_TTL_SECONDS=600
CACHE_TEMPERATURE_THRESHOLD=0.3
PRELOAD_MODEL_ON_STARTUP=true
KEEP_ALIVE=30m
```

---

## 13. Verification Checklist

After implementation, ALL of these must work:

### Prerequisites
Make sure Ollama is running:
```bash
ollama serve
```

### Tests

Server starts and preloads default model:
```bash
uvicorn app.main:app --reload --port 8000
```
Terminal should show log: "Ollama connected, preloaded qwen2.5:7b" or similar

Health check shows real Ollama status:
```bash
curl http://localhost:8000/health
```
Should return ollama_connected: true, models_loaded: actual count

List real models from Ollama:
```bash
curl http://localhost:8000/api/models
```
Should return all 5 installed models with real sizes and details

Get specific model status:
```bash
curl http://localhost:8000/api/models/qwen2.5:7b/status
```
Should return real model info with loaded status

Non-existent model returns proper error:
```bash
curl http://localhost:8000/api/models/nonexistent/status
```
Should return 404 with ModelNotFoundError

Send a real chat request:
```bash
curl -X POST http://localhost:8000/api/chat -H "Content-Type: application/json" -d '{"prompt": "What is Python in one sentence?"}'
```
Should return real LLM response with full metadata (latency, tokens, model)

Send a chat request with specific model:
```bash
curl -X POST http://localhost:8000/api/chat -H "Content-Type: application/json" -d '{"prompt": "Write a haiku about coding", "model": "mistral:7b", "temperature": 0.9}'
```
Should use mistral:7b specifically

Test streaming:
```bash
curl -N -X POST http://localhost:8000/api/chat/stream -H "Content-Type: application/json" -d '{"prompt": "Count from 1 to 5"}'
```
Should see tokens appearing one by one as SSE events

Test caching (send same low-temperature request twice):
```bash
curl -X POST http://localhost:8000/api/chat -H "Content-Type: application/json" -d '{"prompt": "What is 2+2?", "temperature": 0.1}'
```
Second request should return much faster with cached: true in metadata

Test error when Ollama is stopped:
Stop Ollama, then:
```bash
curl http://localhost:8000/health
```
Should return status: "unhealthy", ollama_connected: false

```bash
curl -X POST http://localhost:8000/api/chat -H "Content-Type: application/json" -d '{"prompt": "hello"}'
```
Should return 503 OllamaConnectionError, not crash

API docs still work:
Open http://localhost:8000/docs — all endpoints should show updated schemas

---

## 14. Files Modified in This Phase

| File                        | Action                              |
|-----------------------------|-------------------------------------|
| app/services/llm_service.py | Full implementation (OllamaService) |
| app/core/dependencies.py    | Real implementations replacing placeholders |
| app/routers/health.py       | Real Ollama health check            |
| app/routers/models.py       | Real model listing and status       |
| app/routers/chat.py         | Real LLM calls replacing placeholders |
| app/models/responses.py     | Add cached field to ResponseMetadata |
| app/config.py               | Add cache and preload settings      |
| app/main.py                 | Add startup preload and shutdown cleanup |
| .env                        | Add new configuration variables     |
| app/services/router_service.py | NOT touched (Phase 5)            |
| app/services/prompt_service.py | NOT touched (Phase 3)            |
| templates/*                 | NOT touched (Phase 7)               |
| static/*                    | NOT touched (Phase 7)               |