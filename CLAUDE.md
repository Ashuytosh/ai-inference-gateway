# AI Inference Gateway — Claude Code Instructions

## Project Context
Production-grade FastAPI backend serving as intelligent gateway to local Ollama LLMs. Specs-driven development — always refer to specs/SPEC.md before implementing.

## Rules
- Always follow specs/SPEC.md for architecture and data models
- Use Python 3.12+ features (type hints with | syntax)
- All services must be async
- All Pydantic models use v2 syntax (model_dump, model_validate)
- Error handling: never crash, always return structured error response
- Every function needs docstring explaining what it does
- Use structlog for logging, not print statements

## Code Style
- Type hints on every function parameter and return type
- Pydantic models for ALL request/response data
- Async/await for all I/O operations
- Dependency injection via FastAPI Depends()
- Custom exceptions in app/core/exceptions.py

## Git Workflow
- Commit after each module completion
- Commit message format: "feat(module-N): description"
- Push daily
