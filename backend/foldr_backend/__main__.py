"""uv run python -m foldr_backend [--host H] [--port P]  (or: uvicorn foldr_backend.app:app_from_env --factory)"""

import argparse
import os

import uvicorn

from .config import ConfigError, load_settings


def main() -> None:
    ap = argparse.ArgumentParser(description="foldr backend")
    ap.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    args = ap.parse_args()
    try:
        s = load_settings()
    except ConfigError as e:
        raise SystemExit(f"config error: {e}") from None
    print(f"config: {s.source}")
    if s.enabled:
        usable = [m.key for m in s.models if s.owner_problem(m) is None]
        print(f"AI on: {len(usable)} model(s) usable with the server's keys: {', '.join(usable) or 'none'}"
              f"; user keys {'allowed' if s.allow_user_keys else 'not allowed'}; "
              f"caps ${s.per_run_usd:g}/run, ${s.daily_usd:g}/day")
    else:
        print("AI is OFF: no endpoint will call a model")
    uvicorn.run("foldr_backend.app:app_from_env", factory=True, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
