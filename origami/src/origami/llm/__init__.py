"""LLM provider layer. Messages are OpenAI chat format inside the package; adapters convert at the edge.

    from origami.llm import make_client, ScriptedClient, ChatResult
"""

from .anthropic import AnthropicClient
from .config import (DEFAULT_PROVIDERS, KNOWN_PRICES, PROVIDER_KINDS, client_from_cli, make_client,
                     register_provider, resolve_api_key)
from .errors import (APIError, BudgetExceeded, DailyLimit, NoCredit, ProviderError, RateLimited, redact,
                     stop_reason_for)
from .openai_compat import OpenAICompatClient, OpenRouterClient
from .scripted import ScriptedClient, reply, tool_call
from .types import UNSET, ChatResult, LLMClient, Price, Usage, as_result, compute_cost

__all__ = [
    "AnthropicClient", "OpenAICompatClient", "OpenRouterClient", "ScriptedClient", "LLMClient", "ChatResult",
    "Usage", "Price", "UNSET", "as_result", "compute_cost", "make_client", "client_from_cli", "register_provider",
    "resolve_api_key", "DEFAULT_PROVIDERS", "KNOWN_PRICES", "PROVIDER_KINDS", "ProviderError", "APIError",
    "RateLimited", "DailyLimit", "NoCredit", "BudgetExceeded", "redact", "stop_reason_for", "reply", "tool_call",
]
