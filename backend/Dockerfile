# Build from the repo root:  docker build -f backend/Dockerfile -t foldr-backend .
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy MPLBACKEND=Agg ORIGAMI_MENU_WORKERS=0 \
    FOLDR_MODELS_DIR=/app/examples/models HOST=0.0.0.0
WORKDIR /app
COPY origami/pyproject.toml origami/uv.lock* origami/
COPY origami/src origami/src
COPY backend/pyproject.toml backend/uv.lock backend/
RUN uv sync --project backend --frozen --no-dev
COPY backend/foldr_backend backend/foldr_backend
COPY backend/config.example.toml backend/
COPY examples/models examples/models
WORKDIR /app/backend
# Config: mount a file and set FOLDR_CONFIG, or set the FOLDR_* env vars; with neither, AI is off.
CMD ["sh", "-c", "exec uv run --frozen --no-dev python -m foldr_backend --port ${PORT:-8000}"]
