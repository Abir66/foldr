"""Shared JSON-over-HTTP POST with retries (stdlib urllib).

Retries: 408 / 409 / 429 / 5xx (incl. Anthropic 529 overloaded) and network errors, exponential backoff,
`Retry-After` honoured (capped at 120s). No retry: 402 / spend cap (NoCredit), daily limits (DailyLimit),
other 4xx (ProviderError). Error text is clipped and the API key is blanked out of it.
The transport is injectable (tests pass a fake; no network).
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from .errors import DailyLimit, NoCredit, ProviderError, RateLimited, redact

RETRY_CODES = {408, 409, 429}


def urllib_transport(url: str, data: bytes, headers: dict, timeout: float):
    """-> (status, headers, body bytes). HTTP errors are returned, network errors raised."""
    req = urllib.request.Request(url, data=data, method="POST", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, dict(r.headers.items()), r.read()
    except urllib.error.HTTPError as e:
        try:
            body = e.read() if e.fp else str(e.msg).encode()
        except Exception:  # noqa: BLE001
            body = str(e.msg).encode()
        return e.code, dict(e.headers.items()) if e.headers else {}, body


NETWORK_ERRORS = (urllib.error.URLError, TimeoutError, ConnectionError, OSError)


def _retry_after(headers: dict) -> float | None:
    for k, v in (headers or {}).items():
        if k.lower() == "retry-after":
            try:
                return float(v)
            except (TypeError, ValueError):
                return None
    return None


class HTTPClientBase:
    """Holds the key (never shown), timing, retry policy and request counter for one provider client."""

    provider_name = "http"

    def __init__(self, model: str, *, api_key: str | None = None, timeout: float = 300, max_retries: int = 4,
                 min_interval: float = 0.0, log=print, transport=None, sleep=time.sleep, base_delay: float = 8.0):
        self.model = model
        self._key = api_key or None
        self.timeout = timeout
        self.max_retries = max_retries
        self.min_interval = min_interval
        self.log = log
        self._transport = transport or urllib_transport
        self._sleep = sleep
        self.base_delay = base_delay
        self._last = 0.0
        self.calls = 0  # HTTP requests sent (incl. retries)

    def __repr__(self) -> str:  # never show the key
        return f"{type(self).__name__}(model={self.model!r})"

    __str__ = __repr__

    def _secrets(self):
        return (self._key,) if self._key else ()

    def _err(self, cls, msg: str):
        return cls(redact(msg, self._secrets(), clip=400))

    def _classify(self, code: int, text: str):
        """-> exception to raise now, or None if the status is retryable."""
        low = text.lower()
        if code == 402:
            return self._err(NoCredit, f"HTTP 402: {text}")
        if "enforced_spend_limit_reached" in low or "spend limit" in low or "spend cap" in low:
            return self._err(NoCredit, f"HTTP {code}: {text}")
        if "per-day" in low or "per day" in low or "daily" in low:
            return self._err(DailyLimit, f"HTTP {code}: {text}")
        if code in RETRY_CODES or code >= 500:
            return None
        return self._err(ProviderError, f"HTTP {code}: {text}")

    def _body_error(self, out: dict):
        """Provider-specific: an error carried in a 200 body -> (code, text) or None."""
        return None

    def _post(self, url: str, body: dict, headers: dict) -> dict:
        data = json.dumps(body).encode()
        delay = self.base_delay
        last = "too many retries"
        for attempt in range(self.max_retries + 1):
            wait = self.min_interval - (time.time() - self._last)
            if wait > 0:
                self._sleep(wait)
            self._last = time.time()
            self.calls += 1
            try:
                status, rh, raw = self._transport(url, data, headers, self.timeout)
            except NETWORK_ERRORS as e:
                last = f"network error: {type(e).__name__}"
                if attempt == self.max_retries:
                    break
                self.log(f"  [api] {last}, retry {attempt + 1}/{self.max_retries} in {delay:.0f}s")
                self._sleep(delay)
                delay *= 2
                continue
            text = raw.decode(errors="replace") if isinstance(raw, (bytes, bytearray)) else str(raw)
            out = None
            if status < 300:
                try:
                    out = json.loads(text)
                except ValueError:
                    raise self._err(ProviderError, f"bad JSON from provider: {text[:200]}") from None
                be = self._body_error(out)
                if be is None:
                    return out
                status, text = be
            exc = self._classify(status, text[:800])
            if exc is not None:
                raise exc
            last = f"HTTP {status}: {text[:200]}"
            if attempt == self.max_retries:
                cls = RateLimited if status == 429 else ProviderError
                raise self._err(cls, f"HTTP {status} after {attempt + 1} tries: {text[:200]}")
            ra = _retry_after(rh)
            pause = min(ra if ra is not None else delay, 120.0)
            self.log(f"  [api] HTTP {status}, retry {attempt + 1}/{self.max_retries} in {pause:.0f}s: "
                     f"{redact(text, self._secrets(), 160)}")
            self._sleep(pause)
            delay *= 2
        raise self._err(ProviderError, f"{last} (after {self.max_retries + 1} tries)")
