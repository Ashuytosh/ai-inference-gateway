# Screenshots

This directory holds the screenshots referenced by the README. They aren't
committed automatically (see `.gitignore`), so capture and drop them in here
manually before sharing the repo.

Start the app first (`uvicorn app.main:app --reload --port 8000`, with
Ollama running) and pull at least a couple of models so the UI has real
data to show.

## 1. `chat-interface.png`
Full chat window mid-conversation: user message, streamed assistant
response, and the metadata badges (model, latency, tokens, query type)
visible underneath the response.

## 2. `model-selection.png`
Sidebar with the model dropdown open, showing all 5 models plus the
"Auto (Smart Route)" option, each with its loaded/not-loaded indicator.

## 3. `structured-output.png`
A chat turn using the "Sentiment Analysis" output format, showing the
parsed JSON rendered in the response.

## 4. `analytics.png`
The JSON returned by `GET /api/analytics` (a browser screenshot of the
raw response is fine).

## 5. `api-docs.png`
The Swagger UI at `GET /docs`, showing the full list of endpoints.

Save each file under this directory using the exact filename above so the
README's image links resolve correctly.
