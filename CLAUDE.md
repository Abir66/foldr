# foldr

Public web demo: origami simulator + AI agent that folds paper. Watch folds in 3D, fold yourself, replay/run the AI.

Layout: `origami/` (Python package, uv project, tests), `backend/` (FastAPI, AI runs), `frontend/` (Vite + React + TS, simulator runs in Pyodide), `examples/` (models, curated runs), `tools/`, `docs/` (STEP_FORMAT.md, events.schema.json).

Commands:
- Origami tests: `cd origami && uv run pytest -q` (cwd must be `origami/`; `origami/models` symlinks to `examples/models`)
- Backend tests: `cd backend && uv run pytest -q`; run: `cd backend && uv run python -m foldr_backend` (config: `foldr.toml`, keys: `.env`)
- Frontend: `cd frontend && npm ci && npm run dev`; `npm run typecheck`; `npm test`; `npm run build` (builds wheel + examples)
- Pyodide check: `node tools/pyodide_smoke.mjs`

Generated, git-ignored: `frontend/public/examples`, `frontend/public/py`, `dist`, `runs/`.
Never commit `.env`, `foldr.toml`, `runs/` or API keys.
