"""Cross-provider cost comparison utilities."""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, List, Optional

from tokonomics._types import ModelPricing, Provider
from tokonomics.models import list_models, get_model
from tokonomics.tokenizer import count_tokens


def compare_models(
    text: str,
    models: Optional[List[str]] = None,
    output_tokens: int = 500,
) -> List[Dict[str, Any]]:
    """Compare cost of sending *text* to different models.

    Parameters
    ----------
    text:
        The input text (or a representative sample).
    models:
        Specific model IDs to compare.  ``None`` compares all.
    output_tokens:
        Assumed number of output tokens for cost estimation.

    Returns
    -------
    list of dicts
        Sorted cheapest-first, each containing ``model``, ``provider``,
        ``input_tokens``, ``input_cost``, ``output_cost``, ``total_cost``,
        and ``context_window``.
    """
    if models is not None:
        pricing_list = [get_model(m) for m in models]
    else:
        pricing_list = list_models()

    results: list[dict[str, Any]] = []
    for pricing in pricing_list:
        # Skip embedding models (no output cost)
        if pricing.output_per_million == 0 and "embed" in pricing.model_id.lower():
            continue

        input_tok = count_tokens(text, pricing.model_id)

        # Skip if the text doesn't fit
        if input_tok > pricing.context_window:
            continue

        input_cost = Decimal(str(input_tok)) * pricing.input_per_token
        output_cost = Decimal(str(output_tokens)) * pricing.output_per_token
        total = input_cost + output_cost

        results.append(
            {
                "model": pricing.model_id,
                "provider": pricing.provider.value,
                "input_tokens": input_tok,
                "output_tokens": output_tokens,
                "input_cost": input_cost,
                "output_cost": output_cost,
                "total_cost": total,
                "context_window": pricing.context_window,
            }
        )

    results.sort(key=lambda r: r["total_cost"])
    return results


def cheapest_model(
    text: str,
    providers: Optional[List[Provider]] = None,
    min_context_window: int = 0,
    output_tokens: int = 500,
) -> ModelPricing:
    """Find the cheapest model that fits the given text.

    Parameters
    ----------
    text:
        The input text.
    providers:
        Restrict to these providers.  ``None`` considers all.
    min_context_window:
        Only include models with at least this context window size.
    output_tokens:
        Assumed output length in tokens.

    Returns
    -------
    ModelPricing
        The cheapest matching model.

    Raises
    ------
    ValueError
        If no model matches the constraints.
    """
    candidates = list_models()
    if providers:
        provider_set = set(providers)
        candidates = [m for m in candidates if m.provider in provider_set]
    if min_context_window > 0:
        candidates = [m for m in candidates if m.context_window >= min_context_window]

    # Filter out embedding-only models
    candidates = [
        m for m in candidates
        if not (m.output_per_million == 0 and "embed" in m.model_id.lower())
    ]

    best: Optional[ModelPricing] = None
    best_cost: Optional[Decimal] = None

    for model in candidates:
        input_tok = count_tokens(text, model.model_id)
        if input_tok > model.context_window:
            continue

        total = (
            Decimal(str(input_tok)) * model.input_per_token
            + Decimal(str(output_tokens)) * model.output_per_token
        )
        if best_cost is None or total < best_cost:
            best = model
            best_cost = total

    if best is None:
        raise ValueError("No model matches the given constraints.")
    return best


def format_comparison(results: List[Dict[str, Any]], top_n: int = 0) -> str:
    """Format comparison results as a readable table.

    Parameters
    ----------
    results:
        Output from :func:`compare_models`.
    top_n:
        Only show the top N cheapest results.  0 means all.
    """
    if not results:
        return "No models to compare."

    items = results[:top_n] if top_n > 0 else results

    lines = [
        f"{'Model':<30} {'Provider':<12} {'Input Tok':>10} {'Input $':>10} "
        f"{'Output $':>10} {'Total $':>10}",
        "-" * 84,
    ]
    for r in items:
        lines.append(
            f"{r['model']:<30} {r['provider']:<12} {r['input_tokens']:>10,} "
            f"${r['input_cost']:>9.6f} ${r['output_cost']:>9.6f} "
            f"${r['total_cost']:>9.6f}"
        )
    return "\n".join(lines)
