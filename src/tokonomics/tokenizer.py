"""Token counting for various LLM providers.

Uses tiktoken when available, falls back to a simple word-based estimator.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from tokonomics.models import get_model

# Mapping of model families to tiktoken encoding names
_ENCODING_MAP: dict[str, str] = {
    # GPT-4.1 / o3 / o4 series use o200k_base
    "gpt-4.1": "o200k_base",
    "o3": "o200k_base",
    "o4": "o200k_base",
    "o1": "o200k_base",
    # GPT-4o
    "gpt-4o": "o200k_base",
    # Older GPT-4 and GPT-3.5
    "gpt-4-turbo": "cl100k_base",
    "gpt-4": "cl100k_base",
    "gpt-3.5": "cl100k_base",
    # Embeddings
    "text-embedding-3": "cl100k_base",
    "text-embedding-ada": "cl100k_base",
}

# Per-message overhead tokens for chat models (OpenAI-style)
_CHAT_OVERHEAD_PER_MESSAGE = 3
_CHAT_OVERHEAD_REPLY = 3

_tiktoken_cache: dict[str, Any] = {}


def _get_tiktoken_encoding(model: str) -> Any:
    """Get the tiktoken encoding for a model, or None if unavailable."""
    try:
        import tiktoken
    except ImportError:
        return None

    # Find the best encoding name
    encoding_name: Optional[str] = None
    model_lower = model.lower()
    for prefix, enc in _ENCODING_MAP.items():
        if model_lower.startswith(prefix):
            encoding_name = enc
            break

    if encoding_name is None:
        # Default to o200k_base for unknown models (most modern models use BPE
        # with a vocabulary this size or similar)
        encoding_name = "o200k_base"

    if encoding_name not in _tiktoken_cache:
        _tiktoken_cache[encoding_name] = tiktoken.get_encoding(encoding_name)
    return _tiktoken_cache[encoding_name]


def _estimate_tokens_fallback(text: str) -> int:
    """Rough token estimate when tiktoken is not available.

    Uses the rule of thumb: ~0.75 words per token for English.
    """
    words = len(re.findall(r"\S+", text))
    # Slightly pessimistic: round up
    return max(1, int(words / 0.75) + 1)


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count the number of tokens in *text* for *model*.

    Uses tiktoken for accurate counting when available, otherwise falls back
    to a word-based estimate.
    """
    if not text:
        return 0

    encoding = _get_tiktoken_encoding(model)
    if encoding is not None:
        return len(encoding.encode(text))
    return _estimate_tokens_fallback(text)


def count_message_tokens(
    messages: List[Dict[str, str]],
    model: str = "gpt-4o",
) -> int:
    """Count tokens for a list of chat messages (OpenAI format).

    Each message should have ``role`` and ``content`` keys.
    """
    total = 0
    for message in messages:
        total += _CHAT_OVERHEAD_PER_MESSAGE
        for value in message.values():
            total += count_tokens(str(value), model)
    total += _CHAT_OVERHEAD_REPLY  # priming for assistant reply
    return total


def fits_context(text: str, model: str = "gpt-4o") -> bool:
    """Check whether *text* fits within the model's context window."""
    pricing = get_model(model)
    tokens = count_tokens(text, model)
    return tokens <= pricing.context_window
