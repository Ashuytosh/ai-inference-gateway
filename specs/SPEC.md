# AI Inference Gateway — Product Specification

## Overview
A production-grade FastAPI backend that acts as an intelligent gateway between users and multiple local LLMs via Ollama. It classifies queries, routes to optimal models, applies prompt engineering strategies, streams responses, and logs everything with full metadata.

## Tech Stack
- Backend: Python 3.12, FastAPI, Uvicorn
- LLM Runtime: Ollama (local models — gemma3, qwen3, llama3.2)
- Data Validation: Pydantic v2
- Frontend: Jinja2 + Tailwind CSS (CDN) + Vanilla JS
- Logging: structlog
- Environment: python-dotenv

## Architecture Flow
User Request → FastAPI → Pydantic Validation → Query Classifier → Model Router → Prompt Engineer → Ollama LLM Call (streaming) → Output Parser → Response Logger → Streamed Response to User

## API Endpoints

### POST /api/chat
- Input: ChatRequest (prompt, model?, temperature?, max_tokens?, stream?)
- Output: ChatResponse (response, metadata with model_used, latency_ms, tokens_prompt, tokens_completion, query_type)

### POST /api/chat/stream
- Same input as /api/chat
- Returns Server-Sent Events stream with token-by-token response

### GET /api/models
- Returns list of available Ollama models with status

### GET /api/models/{model_name}/status
- Returns detailed status of specific model

### GET /health
- Returns server health, Ollama connection status, loaded models count

### GET /
- Serves the chat UI (Jinja2 template)

## Pydantic Models

### ChatRequest
- prompt: str (min_length=1, max_length=10000)
- model: str | None = None (if None, auto-route)
- temperature: float = 0.7 (ge=0.0, le=2.0)
- max_tokens: int = 1024 (gt=0, le=4096)
- stream: bool = False
- system_prompt: str | None = None

### ChatResponse
- response: str
- metadata: ResponseMetadata

### ResponseMetadata
- model_used: str
- query_type: str (simple/complex/creative/technical)
- latency_ms: float
- tokens_prompt: int
- tokens_completion: int
- tokens_total: int
- temperature: float
- fallback_used: bool
- timestamp: str (ISO format)

## Core Services

### LLM Service (llm_service.py)
- connect to Ollama at localhost:11434
- async chat completion (regular + streaming)
- model listing and status checking
- timeout handling (30s default)
- retry logic (3 attempts with exponential backoff)

### Router Service (router_service.py)
- classify query into: simple, complex, creative, technical
- route to best model based on classification
- fallback chain if primary model fails
- model capability registry

### Prompt Service (prompt_service.py)
- build system prompts based on query type
- apply prompting strategies (direct, chain-of-thought, creative, technical)
- prompt template management

## Frontend
- Dark theme ChatGPT-like interface
- Jinja2 + Tailwind CSS (CDN) + Vanilla JS
- Sidebar: model selector, temperature slider
- Chat area: streaming message bubbles
- Metadata badges below responses
- Mobile responsive

## Error Handling
- Ollama not running → clear error + health check failure
- Model not found → suggest available models
- LLM timeout → retry then fallback model
- All errors return structured JSON

## Logging
- structlog JSON logging
- Every request: timestamp, endpoint, model, latency, tokens, status
