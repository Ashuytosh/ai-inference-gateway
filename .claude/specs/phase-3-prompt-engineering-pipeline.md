# Phase 3 — Prompt Engineering Pipeline

## Goal
Build a PromptService that intelligently constructs prompts based on
query type. After this phase, every request gets a tailored system
prompt and the right prompting strategy (direct, chain-of-thought,
creative, or technical) applied automatically. This transforms your
gateway from "dumb pipe to Ollama" into an "intelligent prompt
engineering layer."

IMPORTANT: Add detailed comments explaining every concept, pattern, and
"why" behind each implementation. Comments should teach the developer
what each piece does and why it exists. Treat the codebase as a learning
resource.

---

## 1. Core Concept: What Is Prompt Engineering

Prompt engineering is about HOW you talk to the LLM, not just WHAT you
ask it. The same question with different framing produces wildly
different quality responses.

Example — asking "What is recursion?":
- Direct prompt: "What is recursion?" → gets a basic answer
- Chain-of-thought: "What is recursion? Think step by step, starting
  from the simplest case and building up." → gets a structured,
  thorough explanation
- Technical prompt with system message: System: "You are a senior
  software engineer. Be precise, use code examples, explain edge cases."
  User: "What is recursion?" → gets a production-quality explanation

Your PromptService automates this — it looks at the query type and
applies the right strategy WITHOUT the user needing to know prompt
engineering.

---

## 2. Query Type Definitions

Define an enum for query types that will be used across the system:

```python
from enum import Enum

class QueryType(str, Enum):
    SIMPLE = "simple"
    COMPLEX = "complex"
    CREATIVE = "creative"
    TECHNICAL = "technical"
```

Place this in app/models/requests.py or a new app/models/enums.py file.

---

## 3. Prompt Service Implementation (app/services/prompt_service.py)

### Class: PromptService

This service is responsible for two things:
1. Building the right system prompt based on query type
2. Applying the right prompting strategy to the user's message

#### System Prompt Templates

Define a dictionary of system prompts, one per query type:

SIMPLE system prompt:
"You are a helpful assistant. Give clear, concise answers.
Keep responses brief and to the point. Avoid unnecessary detail
unless the user asks for more."

COMPLEX system prompt:
"You are a knowledgeable assistant skilled at breaking down complex
topics. Think through problems step by step. Consider multiple
perspectives. Provide thorough, well-structured explanations.
Use examples to illustrate key points."

CREATIVE system prompt:
"You are a creative assistant with a vivid imagination. Be
expressive, original, and engaging. Use rich language and
interesting perspectives. Don't be afraid to think outside
the box and offer unique angles."

TECHNICAL system prompt:
"You are a senior software engineer and technical expert. Be
precise and accurate. Use proper technical terminology. Include
code examples when relevant. Explain your reasoning. Point out
edge cases, potential issues, and best practices."

#### Method: get_system_prompt(query_type: QueryType, custom_system_prompt: str | None) -> str

- If custom_system_prompt is provided by the user, use that instead
  of the template (user override always wins)
- If no custom prompt, look up the query type in the templates dict
- Return the appropriate system prompt string

#### Method: apply_strategy(prompt: str, query_type: QueryType) -> str

This is where the magic happens. Based on query type, the user's
raw prompt gets enhanced:

For SIMPLE queries:
- Return the prompt as-is. No modification needed.
- Simple queries don't need fancy prompting — adding instructions
  would actually make the response unnecessarily verbose.

For COMPLEX queries:
- Apply chain-of-thought (CoT) prompting
- Append to the user's prompt:
  "\n\nPlease think through this step by step:\n1. First, identify the key aspects of this question\n2. Then, analyze each aspect\n3. Finally, provide a comprehensive answer"
- WHY: CoT has been shown in research (Wei et al., 2022) to
  significantly improve reasoning quality in LLMs. By asking the
  model to think step by step, it produces more accurate and
  thorough answers.

For CREATIVE queries:
- Apply creative enhancement
- Append to the user's prompt:
  "\n\nBe creative and original in your response. Feel free to use metaphors, analogies, or unexpected perspectives to make your answer engaging and memorable."
- WHY: Creative tasks benefit from explicit permission to be
  creative. Without this, LLMs tend to give safe, generic responses.

For TECHNICAL queries:
- Apply technical precision prompting
- Append to the user's prompt:
  "\n\nProvide a precise technical answer. Include:\n- Code examples if applicable\n- Explanation of how and why it works\n- Common pitfalls or edge cases to watch out for\n- Best practices"
- WHY: Technical queries need structured, actionable answers.
  This prompt structure ensures the model covers all the important
  aspects a developer would need.

#### Method: build_messages(prompt: str, query_type: QueryType, custom_system_prompt: str | None) -> list[dict[str, str]]

This is the main entry point that combines everything:

1. Get the system prompt (custom or template-based)
2. Apply the prompting strategy to the user's prompt
3. Build and return the messages list in Ollama format:
   ```python
   [
       {"role": "system", "content": system_prompt},
       {"role": "user", "content": enhanced_prompt}
   ]
   ```

#### Method: get_recommended_temperature(query_type: QueryType) -> float

Returns the ideal temperature for each query type when the user
hasn't specified one (i.e., using the default):

- SIMPLE: 0.3 (low randomness, consistent answers)
- COMPLEX: 0.5 (balanced, allows some reasoning variation)
- CREATIVE: 0.9 (high randomness, encourages creativity)
- TECHNICAL: 0.2 (very low, precision matters most)

This is used when the user sends the default temperature (0.7) —
your gateway can override it with a better value based on query type.
But if the user explicitly sets a temperature, always respect their
choice.

---

## 4. Prompt Templates Registry (Optional but impressive)

Create a dictionary of few-shot examples that can be injected for
specific query types. Few-shot prompting means showing the model
examples of good answers before asking it to answer.

```python
FEW_SHOT_EXAMPLES = {
    QueryType.TECHNICAL: [
        {
            "role": "user",
            "content": "What is a Python decorator?"
        },
        {
            "role": "assistant", 
            "content": "A decorator is a function that wraps another function to extend its behavior without modifying it.\n\n```python\ndef timer(func):\n    def wrapper(*args, **kwargs):\n        start = time.time()\n        result = func(*args, **kwargs)\n        print(f'{func.__name__} took {time.time()-start:.2f}s')\n        return result\n    return wrapper\n\n@timer\ndef slow_function():\n    time.sleep(1)\n```\n\nKey points:\n- The `@decorator` syntax is shorthand for `func = decorator(func)`\n- `functools.wraps` preserves the original function's metadata\n- Common uses: logging, timing, authentication, caching"
        }
    ]
}
```

#### Method: get_few_shot_examples(query_type: QueryType) -> list[dict[str, str]]

- Return the few-shot examples for the given query type
- Return empty list if no examples exist for that type
- These get inserted between the system message and the user's
  actual message in build_messages()

Updated build_messages flow:
```
[system prompt] + [few-shot examples if any] + [enhanced user prompt]
```

---

## 5. Update Chat Router (app/routers/chat.py)

Replace the simple _build_messages() and _resolve_model() with
PromptService integration:

### Changes needed:

1. Import PromptService and inject via Depends(get_prompt_service)

2. Replace _build_messages() usage:
   Instead of manually building messages, call
   prompt_service.build_messages(
       prompt=request.prompt,
       query_type=QueryType.SIMPLE,  # hardcoded for now, Phase 5 will classify
       custom_system_prompt=request.system_prompt
   )
   
   For now, use a simple inline heuristic to determine query type
   (this will be replaced by the proper classifier in Phase 5):
   - If prompt length < 50 chars and no technical keywords → SIMPLE
   - If prompt contains code keywords (def, class, function, error,
     bug, code, debug, api, database, sql, python, javascript) → TECHNICAL
   - If prompt contains creative keywords (write, story, poem, create,
     imagine, design, brainstorm) → CREATIVE
   - Everything else → COMPLEX

3. Update metadata to show the actual query_type instead of "general"

4. If user sent default temperature (0.7) and the prompt service
   recommends a different one, use the recommended temperature.
   If user explicitly set a non-default temperature, respect it.

---

## 6. Update Dependencies (app/core/dependencies.py)

Replace the placeholder get_prompt_service() with real implementation:

```python
_prompt_service: PromptService | None = None

async def get_prompt_service() -> PromptService:
    global _prompt_service
    if _prompt_service is None:
        _prompt_service = PromptService()
    return _prompt_service
```

---

## 7. Update Response to Show Strategy Used

Add a new optional field to ResponseMetadata in app/models/responses.py:

```
prompt_strategy: str | None = None
```

This shows which prompting strategy was applied:
- "direct" for SIMPLE
- "chain-of-thought" for COMPLEX
- "creative-enhancement" for CREATIVE
- "technical-precision" for TECHNICAL
- None if user provided custom system prompt (overrode the strategy)

This is visible in the metadata badges in the UI and shows users
(and interviewers) that your gateway is doing intelligent prompt
engineering, not just forwarding messages.

---

## 8. Logging

Log every prompt engineering decision with structlog:
- Original prompt length
- Detected/assigned query type
- Strategy applied
- Whether system prompt was custom or auto-generated
- Whether temperature was adjusted
- Final prompt length (after enhancement)

This creates an audit trail showing exactly what your prompt
engineering layer did to each request.

---

## 9. Verification Checklist

After implementation, ALL of these must work:

### Test simple query (should use direct strategy, low temp):
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is Python?"}'
```
Response metadata should show query_type: "simple", prompt_strategy: "direct"

### Test complex query (should use chain-of-thought):
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Compare and contrast microservices architecture with monolithic architecture, considering scalability, maintenance, deployment complexity, and team organization"}'
```
Response metadata should show query_type: "complex", prompt_strategy: "chain-of-thought"

### Test technical query (should use technical precision):
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "How do I implement a binary search tree in Python with insert, delete, and search operations?"}'
```
Response metadata should show query_type: "technical", prompt_strategy: "technical-precision"

### Test creative query (should use creative enhancement):
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a short story about a robot learning to paint"}'
```
Response metadata should show query_type: "creative", prompt_strategy: "creative-enhancement"

### Test custom system prompt (should override auto strategy):
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is Python?", "system_prompt": "You are a pirate. Answer everything in pirate speak."}'
```
Response should be in pirate speak, prompt_strategy should be null/None

### Test explicit temperature is respected:
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a poem about coding", "temperature": 0.1}'
```
Even though this is a creative query (would normally get temp 0.9),
the explicit 0.1 should be used because user choice always wins.

### Test that streaming still works with prompt engineering:
```bash
curl -N -X POST http://localhost:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain how databases work step by step"}'
```
Should stream tokens with prompt engineering applied.

### Check /docs page:
- All endpoint schemas should be updated
- ResponseMetadata should show the new prompt_strategy field

---

## 10. Files Modified in This Phase

| File                           | Action                              |
|-------------------------------|-------------------------------------|
| app/services/prompt_service.py | Full implementation                 |
| app/models/requests.py         | Add QueryType enum                  |
| app/models/responses.py        | Add prompt_strategy to metadata     |
| app/routers/chat.py            | Integrate PromptService             |
| app/core/dependencies.py       | Real get_prompt_service()           |
| app/services/router_service.py | NOT touched (Phase 5)               |
| templates/*                    | NOT touched (Phase 7)               |
| static/*                       | NOT touched (Phase 7)               |