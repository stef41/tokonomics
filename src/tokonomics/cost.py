"""Cost calculation engine."""

from __future__ import annotations

from decimal import Decimal

from tokonomics._types import CostEstimate, UsageRecord
from tokonomics.models import get_model
from tokonomics.tokenizer import count_tokens


def estimate_cost(text: str, model: str = "gpt-4o") -> CostEstimate:
    """Estimate the cost of sending *text* as input to *model*.

    Returns a :class:`CostEstimate` with the token count, input cost, and the
    maximum output cost (if the model has a defined ``max_output_tokens``).
    """
    pricing = get_model(model)
    tokens = count_tokens(text, model)
    input_cost = Decimal(str(tokens)) * pricing.input_per_token
    max_output_cost = None
    if pricing.max_output_tokens is not None:
        max_output_cost = (
            Decimal(str(pricing.max_output_tokens)) * pricing.output_per_token
        )
    return CostEstimate(
        model=pricing.model_id,
        provider=pricing.provider,
        estimated_input_tokens=tokens,
        estimated_input_cost=input_cost,
        context_window=pricing.context_window,
        max_output_tokens=pricing.max_output_tokens,
        estimated_max_output_cost=max_output_cost,
    )


def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
    thinking_tokens: int = 0,
) -> UsageRecord:
    """Calculate the exact cost given known token counts.

    Returns a :class:`UsageRecord` with detailed cost breakdown.
    """
    pricing = get_model(model)

    # Input cost
    regular_input = max(0, input_tokens - cached_tokens)
    input_cost = Decimal(str(regular_input)) * pricing.input_per_token
    if cached_tokens > 0 and pricing.cached_input_per_million is not None:
        cached_rate = pricing.cached_input_per_million / Decimal("1000000")
        input_cost += Decimal(str(cached_tokens)) * cached_rate
    elif cached_tokens > 0:
        input_cost += Decimal(str(cached_tokens)) * pricing.input_per_token

    # Output cost
    regular_output = max(0, output_tokens - thinking_tokens)
    output_cost = Decimal(str(regular_output)) * pricing.output_per_token
    if thinking_tokens > 0 and pricing.thinking_output_per_million is not None:
        thinking_rate = pricing.thinking_output_per_million / Decimal("1000000")
        output_cost += Decimal(str(thinking_tokens)) * thinking_rate
    elif thinking_tokens > 0:
        output_cost += Decimal(str(thinking_tokens)) * pricing.output_per_token

    total = input_cost + output_cost

    return UsageRecord(
        model=pricing.model_id,
        provider=pricing.provider,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_cost=input_cost,
        output_cost=output_cost,
        total_cost=total,
        cached_tokens=cached_tokens,
        thinking_tokens=thinking_tokens,
    )


def cost_per_token(model: str) -> tuple[Decimal, Decimal]:
    """Return ``(input_cost_per_token, output_cost_per_token)`` for *model*."""
    pricing = get_model(model)
    return pricing.input_per_token, pricing.output_per_token
