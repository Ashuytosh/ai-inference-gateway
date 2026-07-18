# Phase 4 — Structured Output Parsing

## Goal
Add an output parsing layer that can force LLMs to return structured
data (JSON, specific formats) instead of free-form text. After this
phase, your gateway can request structured responses from LLMs, validate
them against Pydantic models, and retry with a stricter prompt if the
LLM returns malformed output. This is critical for production AI — you
can't build reliable systems on unpredictable text output.

IMPORTANT: Add detailed comments explaining every concept, pattern, and
"why" behind each implementation. Comments should teach the developer
what each piece does and why it exists. Treat the codebase as a learning
resource.

---

## 1. Core Concept: Why Structured Output Matters

LLMs return free-form text by default. If you ask "analyze this
sentiment", you might get:
- "The sentiment is positive" (one time)
- "I'd say this is a positive message because..." (another time)
- "Positive 😊" (yet another time)

All correct, but your frontend can't reliably parse any of these.
You need:
```json
{"sentiment": "positive", "confidence": 0.92, "reasoning": "..."}
```

Structured output parsing solves this by:
1. Instructing the LLM to return JSON in a specific format
2. Parsing the LLM's response and extracting JSON
3. Validating the JSON against a Pydantic model
4. Retrying with a stricter prompt if parsing fails

---

## 2. Output Format Definitions

Create a new file: app/models/output_formats.py

Define Pydantic models for common structured output formats that
users can request:

### SentimentAnalysis
```python
class SentimentAnalysis(BaseModel):
    sentiment: str  # "positive", "negative", "neutral", "mixed"
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
```

### Summary
```python
class Summary(BaseModel):
    title: str
    summary: str
    key_points: list[str]
    word_count: int
```

### CodeReview
```python
class CodeReview(BaseModel):
    language: str
    issues: list[str]
    suggestions: list[str]
    quality_score: int = Field(ge=1, le=10)
    explanation: str
```

### QuestionAnswer
```python
class QuestionAnswer(BaseModel):
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    sources_or_reasoning: str
    follow_up_questions: list[str] = Field(default_factory=list)
```

### CustomFormat
```python
class CustomFormat(BaseModel):
    """For when the user provides their own JSON schema."""
    data: dict  # accepts any valid JSON object
```

### OutputFormat enum
```python
class OutputFormat(str, Enum):
    TEXT = "text"               # default, no parsing
    SENTIMENT = "sentiment"
    SUMMARY = "summary"
    CODE_REVIEW = "code_review"
    QA = "qa"
    JSON = "json"              # custom JSON, user provides schema hint
```

---

## 3. Output Parser Service (app/services/output_parser.py)

Create a new file for the output parsing logic.

### Class: OutputParser

#### Method: build_format_instruction(output_format: OutputFormat, schema_hint: str | None) -> str

Builds the instruction text that gets appended to the prompt telling
the LLM exactly what JSON shape to return.

For SENTIMENT:
```
Respond ONLY with a JSON object in this exact format, no other text:
{
    "sentiment": "positive|negative|neutral|mixed",
    "confidence": 0.0 to 1.0,
    "reasoning": "brief explanation"
}
```

For SUMMARY:
```
Respond ONLY with a JSON object in this exact format, no other text:
{
    "title": "short title",
    "summary": "concise summary",
    "key_points": ["point 1", "point 2", "point 3"],
    "word_count": number
}
```

For CODE_REVIEW:
```
Respond ONLY with a JSON object in this exact format, no other text:
{
    "language": "programming language",
    "issues": ["issue 1", "issue 2"],
    "suggestions": ["suggestion 1", "suggestion 2"],
    "quality_score": 1 to 10,
    "explanation": "overall assessment"
}
```

For QA:
```
Respond ONLY with a JSON object in this exact format, no other text:
{
    "answer": "your answer",
    "confidence": 0.0 to 1.0,
    "sources_or_reasoning": "how you arrived at this answer",
    "follow_up_questions": ["question 1", "question 2"]
}
```

For JSON (custom):
- If schema_hint is provided, use it:
  "Respond ONLY with a JSON object matching this schema: {schema_hint}"
- If no schema_hint:
  "Respond ONLY with a valid JSON object, no other text."

For TEXT:
- Return empty string (no format instruction needed)

#### Method: extract_json(raw_text: str) -> dict

LLMs don't always return clean JSON. They often wrap it in markdown
code blocks or add explanatory text before/after. This method handles
all common cases:

1. Try parsing raw_text directly as JSON (cleanest case)
2. If that fails, look for JSON inside markdown code blocks:
   - Find text between ```json and ``` 
   - Find text between ``` and ```
3. If that fails, look for JSON between { and } (find first { and
   last } and try parsing everything between)
4. If all attempts fail, raise OutputParsingError

```python
def extract_json(self, raw_text: str) -> dict:
    import re
    
    # Attempt 1: direct parse
    text = raw_text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Attempt 2: extract from markdown code blocks
    patterns = [
        r'```json\s*(.*?)\s*```',
        r'```\s*(.*?)\s*```',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
    
    # Attempt 3: find first { to last }
    first_brace = text.find('{')
    last_brace = text.rfind('}')
    if first_brace != -1 and last_brace != -1:
        try:
            return json.loads(text[first_brace:last_brace + 1])
        except json.JSONDecodeError:
            pass
    
    raise OutputParsingError(
        message="Could not extract valid JSON from LLM response",
        detail=f"Raw response: {text[:200]}"
    )
```

#### Method: validate_output(data: dict, output_format: OutputFormat) -> BaseModel

Takes the extracted JSON dict and validates it against the correct
Pydantic model:

```python
FORMAT_MODEL_MAP = {
    OutputFormat.SENTIMENT: SentimentAnalysis,
    OutputFormat.SUMMARY: Summary,
    OutputFormat.CODE_REVIEW: CodeReview,
    OutputFormat.QA: QuestionAnswer,
    OutputFormat.JSON: CustomFormat,
}
```

- Look up the Pydantic model for the given format
- Try to create an instance: model_class(**data) or
  model_class.model_validate(data)
- If validation fails (missing field, wrong type), raise
  OutputParsingError with details about which field failed
- If TEXT format, this method should not be called

#### Method: parse(raw_text: str, output_format: OutputFormat) -> BaseModel

Main entry point combining extract + validate:

```python
def parse(self, raw_text: str, output_format: OutputFormat) -> BaseModel:
    if output_format == OutputFormat.TEXT:
        return None  # no parsing needed
    data = self.extract_json(raw_text)
    return self.validate_output(data, output_format)
```

#### Method: build_retry_prompt(original_prompt: str, format_instruction: str, error_message: str) -> str

When parsing fails, build a stricter retry prompt:

```
"Your previous response was not valid JSON. Error: {error_message}

Please try again. {format_instruction}

IMPORTANT: Return ONLY the JSON object. Do not include any text,
explanation, or markdown formatting before or after the JSON.

Original request: {original_prompt}"
```

---

## 4. Update ChatRequest (app/models/requests.py)

Add two new optional fields:

```python
output_format: OutputFormat = Field(
    default=OutputFormat.TEXT,
    description="Desired output format. TEXT returns free-form text, others return structured JSON."
)
schema_hint: str | None = Field(
    default=None,
    max_length=1000,
    description="JSON schema hint when output_format is 'json'. Describes the shape you want."
)
```

---

## 5. Update ChatResponse (app/models/responses.py)

Add a new optional field to ChatResponse:

```python
class ChatResponse(BaseModel):
    response: str                       # always present (raw text or JSON string)
    parsed: dict | None = None          # structured data if output_format != TEXT
    metadata: ResponseMetadata
```

When output_format is TEXT, parsed is None and response is the text.
When output_format is structured, response is the raw text AND parsed
contains the validated structured data as a dict.

Also add to ResponseMetadata:
```python
output_format: str | None = None       # which format was requested
parse_attempts: int = 1                # how many tries to get valid JSON
```

---

## 6. Update Chat Router (app/routers/chat.py)

### Changes to POST /api/chat:

1. Inject OutputParser via Depends (or create inline since it's stateless)

2. If request.output_format is not TEXT:
   a. Get format instruction from output_parser.build_format_instruction()
   b. Append the format instruction to the user's prompt before sending
      to prompt_service.build_messages()
   c. After getting the LLM response, try to parse it:
      - Call output_parser.parse(result.text, request.output_format)
      - If parsing succeeds, include the parsed data in the response
      - If parsing fails (OutputParsingError):
        * Build a retry prompt with build_retry_prompt()
        * Call the LLM again with the retry prompt (max 2 retries)
        * If retry succeeds, use that response
        * If all retries fail, return the raw text with parsed=None
          and a warning in the metadata

3. The retry flow:
```
First attempt → LLM returns text → try to parse
    ↓ (parsing fails)
Retry 1 → stricter prompt → LLM returns text → try to parse
    ↓ (parsing fails again)
Retry 2 → even stricter prompt → LLM returns text → try to parse
    ↓ (still fails)
Give up → return raw text, parsed=None, note in metadata
```

4. Update metadata with:
   - output_format: request.output_format value
   - parse_attempts: number of attempts made (1 = first try worked)

### Changes to POST /api/chat/stream:

Streaming with structured output is tricky — you can't validate JSON
until you have the complete response. Two approaches:

Option A (simpler, implement this): If output_format is not TEXT,
internally collect the full streamed response, parse it, and then
stream the validated JSON back. Yes this defeats streaming purpose,
but structured output needs the full response to validate.

Option B: If output_format is not TEXT, reject the request with a
clear error message: "Structured output is not supported with
streaming. Use the non-streaming endpoint for structured responses."

Implement Option B — it's honest and simpler. Streaming is for
free-form text responses. Structured output is for the regular
endpoint. This is how most production APIs handle it.

---

## 7. Update Dependencies (app/core/dependencies.py)

Add output parser dependency:

```python
_output_parser: OutputParser | None = None

async def get_output_parser() -> OutputParser:
    global _output_parser
    if _output_parser is None:
        _output_parser = OutputParser()
    return _output_parser
```

---

## 8. Verification Checklist

After implementation, ALL of these must work:

### Regular text response (default, unchanged behavior):
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is Python?"}'
```
Should work exactly as before. parsed should be null.

### Sentiment analysis:
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Analyze the sentiment of: I absolutely love this product, it changed my life!", "output_format": "sentiment"}'
```
Should return parsed JSON with sentiment, confidence, reasoning fields.

### Summary format:
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Summarize: Python is a high-level programming language known for its readability. It was created by Guido van Rossum and first released in 1991. Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.", "output_format": "summary"}'
```
Should return parsed JSON with title, summary, key_points, word_count.

### Code review format:
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Review this code:\ndef add(a,b):\n  return a+b\ndef multiply(a,b):\n  return a*b", "output_format": "code_review"}'
```
Should return parsed JSON with language, issues, suggestions, quality_score.

### QA format:
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is the capital of France?", "output_format": "qa"}'
```
Should return parsed JSON with answer, confidence, sources_or_reasoning.

### Custom JSON with schema hint:
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "List 3 programming languages with their main use case", "output_format": "json", "schema_hint": "{\"languages\": [{\"name\": \"string\", \"use_case\": \"string\"}]}"}'
```
Should return parsed JSON matching the schema hint.

### Streaming with structured output should be rejected:
```bash
curl -X POST http://localhost:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Analyze sentiment of: great day!", "output_format": "sentiment"}'
```
Should return a clear error: structured output not supported with streaming.

### Streaming with text format should still work:
```bash
curl -N -X POST http://localhost:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Count from 1 to 5"}'
```
Should stream normally.

### Check /docs:
- ChatRequest should show new output_format and schema_hint fields
- ChatResponse should show new parsed field
- OutputFormat enum values should be visible in the schema

---

## 9. Files Modified in This Phase

| File                            | Action                           |
|--------------------------------|----------------------------------|
| app/services/output_parser.py   | NEW — full implementation        |
| app/models/output_formats.py    | NEW — structured output models   |
| app/models/requests.py          | Add output_format, schema_hint   |
| app/models/responses.py         | Add parsed field, parse_attempts |
| app/routers/chat.py             | Integrate output parsing + retry |
| app/core/dependencies.py        | Add get_output_parser()          |
| app/services/router_service.py  | NOT touched (Phase 5)            |
| templates/*                     | NOT touched (Phase 7)            |
| static/*                        | NOT touched (Phase 7)            |