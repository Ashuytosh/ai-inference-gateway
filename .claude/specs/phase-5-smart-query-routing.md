# Phase 5 — Smart Query Routing

## Goal
Build the intelligence layer of your gateway — a QueryClassifier that
accurately categorizes incoming queries and a ModelRouter that picks
the optimal model based on classification, VRAM state, and fallback
chains. After this phase, Auto mode actually works intelligently,
manual mode respects user choice, and the system gracefully handles
model failures with automatic fallback. This is the phase that makes
your gateway "smart" instead of just functional.

IMPORTANT: Add detailed comments explaining every concept, pattern, and
"why" behind each implementation. Comments should teach the developer
what each piece does and why it exists. Treat the codebase as a learning
resource.

---

## 1. Core Concept: Why Smart Routing Matters

Without routing, every query goes to the same model. A simple "hi"
uses the same 7B model that a complex analysis question uses. This
means simple queries are slower than they need to be (waiting for a
big model) and complex queries might go to a model that's not great
at reasoning.

Smart routing means:
- "hi" → gemma3:4b (fast, 1 second)
- "explain quantum computing" → qwen2.5:7b (thorough, 3 seconds)
- "write a function to sort a list" → qwen2.5-coder:7b (code expert)
- "write a poem about rain" → mistral:7b (creative)
- "solve this logic puzzle" → phi4-mini (reasoning expert)

Same gateway, five different models, each optimal for the task.

---

## 2. Router Service Implementation (app/services/router_service.py)

This file contains two classes: QueryClassifier and ModelRouter.

### Class: QueryClassifier

This replaces the simple inline heuristic from Phase 3's chat.py
with a proper two-tier classification system (Optimization 6).

#### Tier 1: Heuristic Classifier (fast, handles 80% of queries)

Keyword-based classification that runs in under 1ms:

```python
TECHNICAL_KEYWORDS = {
    "code", "function", "class", "def", "error", "bug", "debug",
    "api", "database", "sql", "python", "javascript", "typescript",
    "java", "rust", "golang", "html", "css", "react", "node",
    "docker", "kubernetes", "git", "compile", "runtime", "syntax",
    "algorithm", "data structure", "binary", "array", "list",
    "hashmap", "stack", "queue", "tree", "graph", "sort", "search",
    "regex", "http", "rest", "graphql", "json", "xml", "yaml",
    "variable", "loop", "recursion", "inheritance", "polymorphism",
    "async", "await", "promise", "callback", "middleware", "endpoint",
    "deploy", "server", "client", "frontend", "backend", "fullstack",
    "framework", "library", "package", "module", "import", "export",
    "test", "unittest", "pytest", "exception", "try", "catch"
}

CREATIVE_KEYWORDS = {
    "write", "story", "poem", "creative", "imagine", "fiction",
    "narrative", "character", "plot", "scene", "dialogue",
    "brainstorm", "ideate", "design", "sketch", "draft",
    "song", "lyrics", "script", "novel", "essay", "blog",
    "metaphor", "describe", "paint", "compose", "invent",
    "fantasy", "adventure", "romance", "mystery", "thriller"
}

COMPLEX_INDICATORS = {
    "compare", "contrast", "analyze", "evaluate", "explain why",
    "pros and cons", "trade-offs", "tradeoffs", "advantages",
    "disadvantages", "implications", "consequences", "impact",
    "relationship between", "difference between", "how does",
    "why does", "what causes", "step by step", "in detail",
    "comprehensive", "thorough", "elaborate", "deep dive"
}
```

#### Method: classify(query: str) -> tuple[QueryType, float]

Returns both the classification AND a confidence score (0.0 to 1.0):

```
Algorithm:
1. Lowercase the query, split into words
2. Count keyword matches for each category
3. Apply these rules in order:

Rule 1 — Short simple queries:
   If query length < 30 chars AND no keyword matches → SIMPLE (confidence 0.95)
   If query is a greeting (hi, hello, hey, thanks, etc.) → SIMPLE (confidence 0.99)

Rule 2 — Technical detection:
   Count technical keyword matches
   If matches >= 2 → TECHNICAL (confidence = min(0.95, 0.6 + matches * 0.1))
   If matches == 1 AND query mentions code/error/debug → TECHNICAL (confidence 0.7)

Rule 3 — Creative detection:
   Count creative keyword matches
   If matches >= 1 AND query starts with creative verb (write, create, compose, imagine)
     → CREATIVE (confidence = min(0.95, 0.7 + matches * 0.1))

Rule 4 — Complex detection:
   Count complex indicator matches (these are multi-word phrases, check with "in")
   If matches >= 1 → COMPLEX (confidence = min(0.9, 0.6 + matches * 0.15))
   If query length > 100 chars AND no other category matched → COMPLEX (confidence 0.6)

Rule 5 — Default:
   If query length < 80 chars → SIMPLE (confidence 0.5)
   Else → COMPLEX (confidence 0.4)

Return (query_type, confidence)
```

The confidence score matters — when confidence is below a threshold
(e.g., 0.5), the router can make smarter decisions (like preferring
the currently loaded model to avoid a swap).

#### Tier 2: LLM-Based Classifier (slow, used for uncertain cases)

Only called when heuristic confidence is below 0.5:

```python
async def classify_with_llm(self, query: str, llm_service: OllamaService) -> QueryType:
    classification_prompt = f"""Classify this user query into exactly one category.
    
Categories:
- simple: basic questions, greetings, short factual queries
- complex: multi-part analysis, comparisons, detailed explanations
- creative: writing, storytelling, brainstorming, artistic content
- technical: programming, debugging, code review, technical concepts

Query: "{query}"

Respond with ONLY one word: simple, complex, creative, or technical"""

    result = await llm_service.chat(
        model="gemma3:4b",  # always use smallest model for classification
        messages=[{"role": "user", "content": classification_prompt}],
        temperature=0.1,      # low temp for consistency
        max_tokens=10,         # only need one word
    )
    
    response_text = result.text.strip().lower()
    # Map response to QueryType, default to COMPLEX if unrecognizable
```

This uses gemma3:4b (smallest, fastest) with low temperature and
max_tokens=10 since we only need one word back. Costs about 0.5-1
second but gives much better classification for ambiguous queries.

#### Method: classify_smart(query: str, llm_service: OllamaService | None) -> tuple[QueryType, float, str]

Main entry point combining both tiers. Returns (query_type, confidence, method):

```
1. Run heuristic classifier → get (type, confidence)
2. If confidence >= 0.5 → return (type, confidence, "heuristic")
3. If confidence < 0.5 AND llm_service is available:
   a. Run LLM classifier → get type
   b. Return (type, 0.8, "llm")  # LLM classifier assumed 0.8 confidence
4. If confidence < 0.5 AND no llm_service:
   a. Return (type, confidence, "heuristic-fallback")
```

---

### Class: ModelRouter

Picks the best model based on query type, VRAM state, and user preference.

#### Model-to-QueryType mapping:

```python
MODEL_ROUTING_TABLE = {
    QueryType.SIMPLE: {
        "primary": "gemma3:4b",
        "fallbacks": ["phi4-mini:latest", "qwen2.5:7b"]
    },
    QueryType.COMPLEX: {
        "primary": "qwen2.5:7b",
        "fallbacks": ["mistral:7b", "phi4-mini:latest"]
    },
    QueryType.CREATIVE: {
        "primary": "mistral:7b",
        "fallbacks": ["qwen2.5:7b", "gemma3:4b"]
    },
    QueryType.TECHNICAL: {
        "primary": "qwen2.5-coder:7b",
        "fallbacks": ["qwen2.5:7b", "phi4-mini:latest"]
    }
}
```

#### Method: route(query_type: QueryType, preferred_model: str | None, loaded_models: list[str], available_models: list[str]) -> str

The core routing logic:

```
1. If preferred_model is set (Manual mode):
   - If it exists in available_models → return it
   - If not → raise ModelNotFoundError

2. Auto mode — smart selection:
   a. Get the primary model for this query type from routing table
   b. If primary model is already loaded in VRAM → return it (no swap needed)
   c. If primary model is available but not loaded:
      - Check if any fallback model IS loaded AND is reasonably good for this task
      - If a loaded fallback exists → return it (avoid swap, Optimization 2)
      - If no loaded fallback → return primary (accept the swap cost)
   d. If primary model is not available at all:
      - Try fallbacks in order
      - Return first available fallback
   e. If nothing is available → raise ModelNotFoundError
```

The key insight: avoiding a model swap (4-7 seconds) is worth using
a slightly less optimal model. A "good enough" answer in 2 seconds
beats the "perfect" answer in 9 seconds for most queries.

#### Method: get_fallback_chain(model: str, query_type: QueryType) -> list[str]

Returns ordered list of fallback models for a given primary model:

```
1. Get fallbacks from MODEL_ROUTING_TABLE for this query_type
2. Filter to only available models
3. Return the filtered list
```

#### Method: should_swap_model(current_model: str, target_model: str, query_type: QueryType, confidence: float) -> bool

Decides whether swapping models is worth the latency cost:

```
Rules:
- If current_model == target_model → False (already loaded)
- If current_model is in the fallbacks for this query_type AND
  confidence < 0.8 → False (current model is acceptable, don't swap
  for a low-confidence classification)
- If target_model is the primary for this query_type AND
  confidence >= 0.8 → True (high confidence, swap is worth it)
- If current_model is gemma3:4b or phi4-mini (small models that
  coexist with 7B) → True (no swap cost, both fit in VRAM)
- Default → True (swap to optimal model)
```

---

## 3. Update Chat Router (app/routers/chat.py)

Major refactor to integrate the full routing pipeline:

### Remove the inline heuristic classifier from Phase 3
The simple keyword check in chat.py gets replaced by the proper
QueryClassifier in router_service.py.

### New flow for POST /api/chat:

```
1. Inject RouterService via Depends(get_router_service)
2. Classify the query:
   classifier.classify_smart(request.prompt, llm_service)
   → get (query_type, confidence, classification_method)
3. Route to best model:
   If request.model is set → use it (Manual mode)
   If request.model is None:
     loaded = await llm_service.get_loaded_models()
     available = [m.name for m in await llm_service.list_models()]
     model = router.route(query_type, None, loaded, available)
4. Build prompt using prompt_service with the classified query_type
5. Call LLM with selected model
6. If LLM call fails:
   a. Get fallback chain from router
   b. Try each fallback in order
   c. If a fallback succeeds, set fallback_used=True in metadata
   d. If all fail, raise the original error
7. Return response with full metadata
```

### Update metadata to include routing info:

Add these to ResponseMetadata in responses.py:
```python
classification_confidence: float = 0.0
classification_method: str | None = None   # "heuristic", "llm", or "heuristic-fallback"
```

### Same changes for POST /api/chat/stream:
Apply the same routing and fallback logic to the streaming endpoint.

---

## 4. Context Window Management (Optimization 5)

Add context awareness to the routing pipeline:

### Method in LLMService: estimate_tokens(text: str) -> int
Simple heuristic: roughly 1 token per 4 characters.
```python
def estimate_tokens(self, text: str) -> int:
    return len(text) // 4
```

### Add context check before LLM call in chat router:
```python
total_prompt_text = system_prompt + enhanced_prompt
estimated_tokens = llm_service.estimate_tokens(total_prompt_text)
model_context_limit = 4096  # default for most 7B models

if estimated_tokens > model_context_limit * 0.8:
    # Warn in metadata but don't block
    # Truncate if over 95% of limit
```

Add to ResponseMetadata:
```python
context_tokens_estimated: int | None = None
context_limit: int | None = None
```

---

## 5. Update Dependencies (app/core/dependencies.py)

Replace the placeholder get_router_service() with real implementation:

```python
_router_service: RouterService | None = None

async def get_router_service() -> RouterService:
    global _router_service
    if _router_service is None:
        _router_service = RouterService()
    return _router_service
```

RouterService is a thin wrapper that holds both QueryClassifier and
ModelRouter instances:

```python
class RouterService:
    def __init__(self):
        self.classifier = QueryClassifier()
        self.router = ModelRouter()
```

Or you can keep them as separate dependencies — either approach works.
The key is that chat.py can access both classification and routing.

---

## 6. VRAM-Aware Optimization Integration

The ModelRouter.route() method already considers loaded_models, but
also add VRAM tracking to OllamaService:

### Update OllamaService (app/services/llm_service.py):

Add a method that reports VRAM optimization savings:

```python
async def get_vram_status(self) -> dict:
    loaded = await self.get_loaded_models()
    return {
        "loaded_models": loaded,
        "last_used_model": self._loaded_model,
        "can_coexist": [m for m in loaded if m in SMALL_MODELS],
    }
```

Where SMALL_MODELS = {"gemma3:4b", "phi4-mini:latest"} — models
small enough to coexist with a 7B in 8GB VRAM.

### Add VRAM info to /health endpoint:

Update HealthResponse to include:
```python
loaded_model_names: list[str] = Field(default_factory=list)
```

Update health router to populate this from get_loaded_models().

---

## 7. Logging for Routing Decisions

Log every routing decision with structlog:

```python
logger.info(
    "query_routed",
    query_preview=query[:50],
    query_type=query_type.value,
    confidence=confidence,
    classification_method=method,
    selected_model=model,
    was_already_loaded=model in loaded_models,
    swap_required=model not in loaded_models,
    fallback_used=False,
)
```

This creates a full audit trail. In interviews you can say:
"I log every routing decision with classification confidence, selected
model, and whether a VRAM swap was needed. This lets me analyze
routing accuracy and optimize the keyword sets over time."

---

## 8. Verification Checklist

After implementation, ALL of these must work:

### Auto mode — simple query routes to fast model:
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "hi"}'
```
Should use gemma3:4b, classification_method: "heuristic",
query_type: "simple", confidence high (>0.9)

### Auto mode — technical query routes to code model:
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a Python function to implement merge sort with type hints and error handling"}'
```
Should use qwen2.5-coder:7b, query_type: "technical"

### Auto mode — creative query routes to creative model:
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a short story about a time traveler who accidentally brings a smartphone to medieval England"}'
```
Should use mistral:7b, query_type: "creative"

### Auto mode — complex query routes to reasoning model:
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Compare and contrast SQL and NoSQL databases in terms of scalability, consistency, use cases, and when to choose each"}'
```
Should use qwen2.5:7b, query_type: "complex"

### Manual mode — respects user choice:
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "hello", "model": "mistral:7b"}'
```
Should use mistral:7b even though "hello" would normally route to gemma3:4b

### Manual mode — nonexistent model returns error:
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "hello", "model": "nonexistent:1b"}'
```
Should return 404 ModelNotFoundError with available models listed

### Fallback works when primary model fails:
To test this, temporarily rename or remove a model from the routing
table's primary slot and verify the fallback is used. Check that
fallback_used: true appears in metadata.

### VRAM optimization — consecutive same-type queries avoid swap:
```bash
# Send two technical queries back to back
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is a binary tree?"}'

curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "How do hash maps work internally?"}'
```
Second request should be faster if same model was already loaded.
Check logs for "swap_required: false" on second request.

### Health endpoint shows loaded models:
```bash
curl http://localhost:8000/health
```
Should show loaded_model_names with actual model names.

### Context token estimation in metadata:
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain everything about machine learning"}'
```
Metadata should include context_tokens_estimated and context_limit.

### Streaming still works with routing:
```bash
curl -N -X POST http://localhost:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a poem about the ocean"}'
```
Should stream from mistral:7b (creative query).

### All Phase 1-4 features still work:
- Structured output parsing still works
- Prompt engineering strategies still apply
- Caching still works for low-temperature queries
- Error handling still returns structured JSON

---

## 9. Files Modified in This Phase

| File                            | Action                              |
|--------------------------------|-------------------------------------|
| app/services/router_service.py  | Full implementation (Classifier + Router) |
| app/routers/chat.py             | Major refactor — full routing pipeline |
| app/routers/health.py           | Add loaded model names              |
| app/models/responses.py         | Add routing metadata fields         |
| app/core/dependencies.py        | Real get_router_service()           |
| app/services/llm_service.py     | Add estimate_tokens, get_vram_status |
| templates/*                     | NOT touched (Phase 7)               |
| static/*                        | NOT touched (Phase 7)               |