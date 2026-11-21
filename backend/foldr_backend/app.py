"""The FastAPI app: GET /api/health, /api/config, /api/runs/{id}[/events]; POST /api/runs, /api/runs/{id}/stop,
/api/timeline, /api/export."""

from __future__ import annotations

import asyncio
import json
import threading
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.concurrency import run_in_threadpool

from .config import LOCAL_ORIGINS, Settings, load_settings
from .runs import ApiError, Run, RunManager

VERSION = "0.1.0"
MAX_BODY = 4_000_000
HEARTBEAT_S = 15.0
LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd.strip():
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "?"


def _key_channel_ok(request: Request) -> bool:
    if request.headers.get("x-forwarded-proto", request.url.scheme).split(",")[0].strip() == "https":
        return True
    return request.url.hostname in LOCAL_HOSTS


async def _json_body(request: Request) -> dict:
    if int(request.headers.get("content-length") or 0) > MAX_BODY:
        raise ApiError(413, "request too large")
    raw = await request.body()
    if len(raw) > MAX_BODY:
        raise ApiError(413, "request too large")
    try:
        body = json.loads(raw or b"{}")
    except ValueError:
        raise ApiError(422, "body is not valid JSON") from None
    if not isinstance(body, dict):
        raise ApiError(422, "body must be a JSON object")
    return body


def create_app(settings: Settings | None = None, client_factory=None) -> FastAPI:
    s = settings or load_settings()
    kw = {"client_factory": client_factory} if client_factory else {}
    mgr = RunManager(s, **kw)
    app = FastAPI(title="foldr backend", version=VERSION, docs_url=None, redoc_url=None)
    app.state.settings, app.state.manager = s, mgr
    app.add_middleware(CORSMiddleware, allow_origins=list(dict.fromkeys(LOCAL_ORIGINS + s.cors_origins)),
                       allow_methods=["GET", "POST"], allow_headers=["Content-Type", "Last-Event-ID"])
    tool_slots = threading.BoundedSemaphore(2)

    @app.exception_handler(ApiError)
    async def _api_error(_: Request, e: ApiError):
        return JSONResponse({"detail": e.detail}, status_code=e.status)

    @app.get("/api/health")
    def health():
        return {"ok": True, "version": VERSION}

    @app.get("/api/config")
    def config():
        models, providers = [], []
        if s.enabled:
            for m in s.models:
                owner_ok = s.owner_problem(m) is None
                user_ok = s.user_key_ok(m)
                if owner_ok or user_ok:
                    models.append({"id": m.key, "label": m.label, "provider": m.provider, "images": m.images,
                                   "reasoning": m.reasoning, "default": m.default, "owner_key": owner_ok,
                                   "user_key": user_ok, "price": m.price.__dict__ if m.price else None})
            used = {m["provider"] for m in models}
            providers = [{"id": pid, "kind": p["kind"], "user_key": s.allow_user_keys and bool(p.get("api_key_env"))}
                         for pid, p in s.providers.items() if pid in used]
        if models and not any(m["default"] for m in models):
            models[0]["default"] = True
        return {"ai": {"enabled": s.enabled, "allow_user_keys": s.enabled and s.allow_user_keys,
                       "decision_mode": False, "max_turns": s.max_turns, "per_run_usd": s.per_run_usd,
                       "daily_left_usd": round(mgr.daily_left(), 4) if s.enabled else 0.0},
                "models": models, "providers": providers,
                "examples": sorted(p.stem for p in s.models_dir.glob("*.json"))}

    @app.post("/api/runs", status_code=202)
    async def start_run(request: Request):
        body = await _json_body(request)
        run = await run_in_threadpool(mgr.start, body, ip=_client_ip(request),
                                      key_channel_ok=_key_channel_ok(request))
        return {"run_id": run.id}

    def _run(run_id: str) -> Run:
        run = mgr.get(run_id)
        if run is None:
            raise ApiError(404, "unknown run")
        return run

    @app.post("/api/runs/{run_id}/stop")
    def stop_run(run_id: str):
        run = _run(run_id)
        mgr.stop(run)
        return {"status": run.status}

    @app.get("/api/runs/{run_id}")
    def run_info(run_id: str):
        run = _run(run_id)
        lines = [ln for _, ln in list(run.lines)]
        last = json.loads(lines[-1]) if run.status == "finished" and lines else {}
        summary = {k: last.get(k) for k in ("stop_reason", "score", "totals", "error")} if last else None
        text = '{"status":%s,"summary":%s,"events":[%s]}' % (json.dumps(run.status), json.dumps(summary),
                                                             ",".join(lines))
        return Response(text, media_type="application/json")

    @app.get("/api/runs/{run_id}/events")
    async def run_events(run_id: str, request: Request):
        run = _run(run_id)
        try:
            after = int(request.headers.get("last-event-id", "-1"))
        except ValueError:
            after = -1

        async def stream():
            i, last_sent = 0, time.monotonic()
            while True:
                lines = run.lines
                while i < len(lines):
                    seq, line = lines[i]
                    i += 1
                    if seq > after:
                        yield f"id: {seq}\ndata: {line}\n\n"
                        last_sent = time.monotonic()
                if run.status == "finished" and i >= len(run.lines):
                    return
                if time.monotonic() - last_sent > HEARTBEAT_S:
                    yield ": keep-alive\n\n"
                    last_sent = time.monotonic()
                if await request.is_disconnected():
                    return
                await asyncio.sleep(0.05)

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    async def _tool(method: str, request: Request):
        body = await _json_body(request)
        from origami.web import bridge

        if not tool_slots.acquire(blocking=False):
            raise ApiError(429, "busy; try again in a moment")

        def work():
            try:
                return bridge.handle(method, body)
            finally:
                tool_slots.release()

        try:
            out = await run_in_threadpool(work)
        except (ValueError, KeyError, TypeError) as e:
            raise ApiError(422, f"{type(e).__name__}: {e}"[:300]) from None
        return Response(json.dumps(out, allow_nan=False), media_type="application/json")

    @app.post("/api/timeline")
    async def timeline(request: Request):
        return await _tool("timeline", request)

    @app.post("/api/export")
    async def export(request: Request):
        return await _tool("export", request)

    return app


def app_from_env() -> FastAPI:
    return create_app()
