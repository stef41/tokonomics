"""Core data types for tokonomics."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any

if sys.version_info >= (3, 11):
    from enum import StrEnum as _StrEnumBase
else:
    class _StrEnumBase(str, Enum):
        pass


class Provider(_StrEnumBase):
    """Supported LLM API providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    MISTRAL = "mistral"
    COHERE = "cohere"
    DEEPSEEK = "deepseek"
    XAI = "xai"


@dataclass(frozen=True)
class ModelPricing:
    """Pricing information for a single model.

    All costs are in USD per million tokens unless otherwise noted.
    """

    model_id: str
    provider: Provider
    input_per_million: Decimal
    output_per_million: Decimal
    context_window: int
    max_output_tokens: int | None = None

    # Cached input pricing (some providers offer discounts for prompt caching)
    cached_input_per_million: Decimal | None = None

    # Batch API pricing
    batch_input_per_million: Decimal | None = None
    batch_output_per_million: Decimal | None = None

    # Reasoning / thinking token pricing (o-series, DeepSeek-R1, etc.)
    thinking_output_per_million: Decimal | None = None

    # Model aliases (e.g. "gpt-4o" -> "gpt-4o-2024-08-06")
    aliases: tuple[str, ...] = ()

    @property
    def input_per_token(self) -> Decimal:
        return self.input_per_million / Decimal("1000000")

    @property
    def output_per_token(self) -> Decimal:
        return self.output_per_million / Decimal("1000000")

    def cost(
        self,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
        thinking_tokens: int = 0,
    ) -> Decimal:
        """Calculate total cost for the given token counts."""
        regular_input = input_tokens - cached_tokens
        if regular_input < 0:
            regular_input = 0

        input_cost = Decimal(str(regular_input)) * self.input_per_token
        if cached_tokens > 0 and self.cached_input_per_million is not None:
            cached_rate = self.cached_input_per_million / Decimal("1000000")
            input_cost += Decimal(str(cached_tokens)) * cached_rate
        elif cached_tokens > 0:
            input_cost += Decimal(str(cached_tokens)) * self.input_per_token

        regular_output = output_tokens - thinking_tokens
        if regular_output < 0:
            regular_output = 0

        output_cost = Decimal(str(regular_output)) * self.output_per_token
        if thinking_tokens > 0 and self.thinking_output_per_million is not None:
            thinking_rate = self.thinking_output_per_million / Decimal("1000000")
            output_cost += Decimal(str(thinking_tokens)) * thinking_rate
        elif thinking_tokens > 0:
            output_cost += Decimal(str(thinking_tokens)) * self.output_per_token

        return input_cost + output_cost


@dataclass
class UsageRecord:
    """Records token usage and cost for a single LLM API call."""

    model: str
    provider: Provider
    input_tokens: int
    output_tokens: int
    input_cost: Decimal
    output_cost: Decimal
    total_cost: Decimal
    cached_tokens: int = 0
    thinking_tokens: int = 0
    timestamp: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CostEstimate:
    """Estimated cost for sending a text to a model."""

    model: str
    provider: Provider
    estimated_input_tokens: int
    estimated_input_cost: Decimal
    context_window: int
    max_output_tokens: int | None = None
    estimated_max_output_cost: Decimal | None = None


class TokonomicsError(Exception):
    """Base exception for tokonomics."""


class ModelNotFoundError(TokonomicsError):
    """Raised when a model ID is not found in the registry."""


class BudgetExceededError(TokonomicsError):
    """Raised when a budget limit is exceeded."""
