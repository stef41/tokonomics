"""Tests for cost calculation."""

import pytest
from decimal import Decimal

from tokonomics.cost import estimate_cost, calculate_cost, cost_per_token
from tokonomics._types import Provider


class TestEstimateCost:
    def test_basic_estimate(self, sample_text):
        est = estimate_cost(sample_text, "gpt-4o")
        assert est.model == "gpt-4o"
        assert est.provider == Provider.OPENAI
        assert est.estimated_input_tokens > 0
        assert est.estimated_input_cost > Decimal("0")

    def test_has_max_output_cost(self, sample_text):
        est = estimate_cost(sample_text, "gpt-4o")
        assert est.estimated_max_output_cost is not None
        assert est.estimated_max_output_cost > Decimal("0")

    def test_embedding_model_no_output_cost(self, sample_text):
        est = estimate_cost(sample_text, "text-embedding-3-large")
        # Embedding models have no max_output_tokens, so max output cost is None
        assert est.estimated_max_output_cost is None

    @pytest.mark.parametrize(
        "model", ["gpt-4o", "claude-sonnet-4-20250514", "gemini-2.5-pro", "deepseek-chat"]
    )
    def test_various_models(self, sample_text, model):
        est = estimate_cost(sample_text, model)
        assert est.estimated_input_tokens > 0
        assert est.estimated_input_cost >= Decimal("0")


class TestCalculateCost:
    def test_basic(self):
        usage = calculate_cost("gpt-4o", input_tokens=1000, output_tokens=500)
        assert usage.model == "gpt-4o"
        assert usage.input_tokens == 1000
        assert usage.output_tokens == 500
        # $2.50/M input + $10.00/M output
        expected_input = Decimal("1000") * Decimal("2.50") / Decimal("1000000")
        expected_output = Decimal("500") * Decimal("10.00") / Decimal("1000000")
        assert usage.input_cost == expected_input
        assert usage.output_cost == expected_output
        assert usage.total_cost == expected_input + expected_output

    def test_with_cached_tokens(self):
        usage_no_cache = calculate_cost("gpt-4o", input_tokens=1000, output_tokens=0)
        usage_cache = calculate_cost(
            "gpt-4o", input_tokens=1000, output_tokens=0, cached_tokens=500
        )
        assert usage_cache.total_cost < usage_no_cache.total_cost
        assert usage_cache.cached_tokens == 500

    def test_with_thinking_tokens(self):
        usage = calculate_cost(
            "o3", input_tokens=100, output_tokens=1000, thinking_tokens=800
        )
        assert usage.thinking_tokens == 800
        assert usage.total_cost > Decimal("0")

    def test_zero_tokens(self):
        usage = calculate_cost("gpt-4o", input_tokens=0, output_tokens=0)
        assert usage.total_cost == Decimal("0")

    def test_large_token_counts(self):
        usage = calculate_cost("gpt-4o", input_tokens=1_000_000, output_tokens=500_000)
        # Input: $2.50, Output: $5.00
        assert usage.input_cost == Decimal("2.50")
        assert usage.output_cost == Decimal("5.00")
        assert usage.total_cost == Decimal("7.50")

    def test_cached_exceeding_input(self):
        # cached_tokens > input_tokens shouldn't cause negative costs
        usage = calculate_cost(
            "gpt-4o", input_tokens=100, output_tokens=0, cached_tokens=200
        )
        assert usage.total_cost >= Decimal("0")

    @pytest.mark.parametrize(
        "model,input_rate,output_rate",
        [
            ("gpt-4o", Decimal("2.50"), Decimal("10.00")),
            ("gpt-4o-mini", Decimal("0.15"), Decimal("0.60")),
            ("gpt-4.1", Decimal("2.00"), Decimal("8.00")),
            ("claude-sonnet-4-20250514", Decimal("3.00"), Decimal("15.00")),
            ("deepseek-chat", Decimal("0.14"), Decimal("0.28")),
        ],
    )
    def test_exact_rates(self, model, input_rate, output_rate):
        usage = calculate_cost(model, input_tokens=1_000_000, output_tokens=1_000_000)
        assert usage.input_cost == input_rate
        assert usage.output_cost == output_rate


class TestCostPerToken:
    def test_returns_tuple(self):
        inp, out = cost_per_token("gpt-4o")
        assert isinstance(inp, Decimal)
        assert isinstance(out, Decimal)
        assert inp > 0
        assert out > 0

    def test_matches_model(self):
        inp, out = cost_per_token("gpt-4o")
        assert inp == Decimal("2.50") / Decimal("1000000")
        assert out == Decimal("10.00") / Decimal("1000000")
