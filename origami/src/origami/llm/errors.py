"""Provider errors and their run stop reasons:

    BudgetExceeded -> "budget", NoCredit -> "no_credit", DailyLimit -> "daily_limit", other ProviderError -> "api_error"
"""

from __future__ import annotations


class ProviderError(RuntimeError):
    """Base class for errors from an LLM provider (HTTP, network, bad response)."""


APIError = ProviderError


class RateLimited(ProviderError):
    """HTTP 429 that did not clear after all retries."""


class DailyLimit(ProviderError):
    """Provider says no more requests today (or no credits): stop the run, don't retry."""


class NoCredit(DailyLimit):
    """HTTP 402 / spend cap reached: no credit left."""


class BudgetExceeded(RuntimeError):
    """Our own cost cap (ledger / backend budget hook) was reached before a call. Not a provider error."""


def stop_reason_for(e: BaseException) -> str:
    if isinstance(e, BudgetExceeded):
        return "budget"
    if isinstance(e, NoCredit):
        return "no_credit"
    if isinstance(e, DailyLimit):
        return "daily_limit"
    return "api_error"


def redact(text: str, secrets=(), clip: int = 300) -> str:
    """Clip provider text and blank out any secret (API key) that a provider echoed back."""
    text = str(text)
    for s in secrets:
        if s and len(s) >= 4:
            text = text.replace(s, "***")
    return text[:clip]
