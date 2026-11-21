# Foldr

Foldr is an origami simulator and an AI agent that folds paper. Watch curated folds in 3D, fold a sheet yourself step by step, or replay and run an AI agent that plans and performs the folds.

Live demo: _link coming soon_ (placeholder: `https://YOURNAME.github.io/foldr/`)

## Features

- Watch examples (crane, cup and more) as a 3D animation with step-by-step pictures.
- Fold yourself: pick from the folds the simulator offers, or draw your own, and see layers stack up. Physics is checked after every step.
- Open your own fold-step JSON files.
- Replay recorded AI runs, or run the agent live (needs the optional backend).
- Export `.fold`, step JSON, timeline JSON, step pictures and crease pattern.
- The simulator runs in your browser (Python via Pyodide); no server is needed except for the AI.

## Quick start

Frontend only (no AI):

```
cd frontend
npm ci
npm run dev
```

Backend with AI:

```
cd backend
cp config.example.toml foldr.toml
cp .env.example .env        # then add a provider key to .env
uv run python -m foldr_backend
```

Then start the frontend with `VITE_API_URL=http://localhost:8000 npm run dev`.

## AI is off by default

The public build calls no model. The backend only enables AI when `[ai] enabled = true` in `foldr.toml` (or `FOLDR_AI_ENABLED=1`). Spend caps (`per_run_usd`, `daily_usd`), per-IP cooldown and concurrency limits are in the same file. Keys live only in environment variables or `backend/.env`, never in the config.

## Switching models and providers

Edit `backend/foldr.toml` (start from `backend/config.example.toml`):

- OpenAI-compatible servers (OpenAI, OpenRouter, others): a `[providers.X]` table with `kind = "openai"`, `base_url` and `api_key_env`.
- Anthropic: `kind = "anthropic"` with `api_key_env = "ANTHROPIC_API_KEY"`.
- Ollama / LM Studio: local `base_url`, `api_key_env = ""` (no key, free).
- Add `[[models]]` entries to choose what shows in the UI. An offline "demo" model replays a cup fold, which is handy for testing setup.

## Deploy

- Frontend: GitHub Pages through `.github/workflows/pages.yml`. In the repo go to Settings, Pages, Source, and pick "GitHub Actions". Push to main to publish.
- Backend (optional): Render, using `render.yaml` (Docker). AI stays off until you set `FOLDR_AI_ENABLED=1`, a provider key and `FOLDR_CORS_ORIGINS` in the dashboard. Then build the frontend with `VITE_API_URL` pointing at it.

## Repo layout

- `origami/` Python simulator, physics checker, agent and tests (uv project)
- `backend/` FastAPI service for AI runs
- `frontend/` Vite + React + TypeScript app
- `examples/` hand-written fold models and curated runs
- `tools/` build scripts (`build_examples.py`, Pyodide smoke test)
- `docs/` step format and event schema

## Tests

```
cd origami && uv run pytest -q
cd backend && uv run pytest -q
cd frontend && npm run typecheck && npm test
cd frontend && npm run build
```

## Model JSON format

Fold steps are plain JSON; see [docs/STEP_FORMAT.md](docs/STEP_FORMAT.md). The AI event stream is described by `docs/events.schema.json`.
