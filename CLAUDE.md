
# AI Inference Gateway — Claude Code Instructions

## Project Context
Production-grade FastAPI backend serving as an intelligent multi-model gateway to local Ollama LLMs. Features smart query classification, automatic model routing, prompt engineering pipeline, streaming responses, and full request metadata tracking.

This project follows specs-driven development. ALWAYS read specs/SPEC.md before implementing anything.

## Theme & Design Language
- Dark theme (gray-900/950 backgrounds)
- Accent color: Indigo-500/Blue-500 gradient
- Font: Inter (Google Fonts CDN)
- Rounded corners (rounded-xl on cards, rounded-2xl on main containers)
- Subtle glass morphism effects (backdrop-blur, bg-opacity)
- Smooth transitions and animations
- ChatGPT-style chat interface with streaming text
- Metadata shown as small badges below AI responses
- Professional, clean, minimal — NOT colorful or playful
- Mobile responsive with Tailwind breakpoints

## Tech Stack
- Python 3.12+ with modern type hints (str | None, not Optional[str])
- FastAPI with async everywhere
- Pydantic v2 (model_dump, model_validate, Field)
- Ollama for local LLM inference via httpx
- Jinja2 templates served from FastAPI
- Tailwind CSS via CDN for styling
- Vanilla JavaScript for streaming and interactivity
- structlog for JSON logging
- python-dotenv for environment variables

## Architecture

- User (Browser) → Jinja2 Chat UI → JavaScript fetch/SSE
- ↓
- FastAPI Server (uvicorn)
- ↓
- Request Validation (Pydantic models in app/models/)
- ↓
- Query Classifier (app/services/router_service.py)
- ↓
- Model Router → selects best Ollama model
- ↓
- Prompt Engineer (app/services/prompt_service.py)
- ↓
- Ollama LLM Call via httpx (app/services/llm_service.py)
- ↓
- Response with full metadata → streamed back to UI


