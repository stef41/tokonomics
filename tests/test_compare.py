"""Tests for the comparison module."""

from decimal import Decimal

import pytest

from tokonomics.compare import compare_models, cheapest_model, format_comparison
from tokonomics._types import Provider


class TestCompareModels:
    def test_compare_all(self, sample_text):
        results = compare_models(sample_text)
        assert len(results) > 5
        # Sorted by total cost
        for i in range(len(results) - 1):
            assert results[i]["total_cost"] <= results[i + 1]["total_cost"]

    def test_compare_specific_models(self, sample_text):
        results = compare_models(
            sample_text,
            models=["gpt-4o", "claude-sonnet-4-20250514"],
            output_tokens=100,
        )
        assert len(results) == 2
        model_ids = {r["model"] for r in results}
        assert "gpt-4o" in model_ids
        assert "claude-sonnet-4-20250514" in model_ids

    def test_result_structure(self, sample_text):
        results = compare_models(sample_text, models=["gpt-4o"])
        r = results[0]
        assert "model" in r
        assert "provider" in r
        assert "input_tokens" in r
        assert "output_tokens" in r
        assert "input_cost" in r
        assert "output_cost" in r
        assert "total_cost" in r
        assert "context_window" in r

    def test_costs_are_decimal(self, sample_text):
        results = compare_models(sample_text, models=["gpt-4o"])
        r = results[0]
        assert isinstance(r["input_cost"], Decimal)
        assert isinstance(r["output_cost"], Decimal)
        assert isinstance(r["total_cost"], Decimal)

    def test_output_tokens_param(self, sample_text):
        r1 = compare_models(sample_text, models=["gpt-4o"], output_tokens=100)
        r2 = compare_models(sample_text, models=["gpt-4o"], output_tokens=1000)
        assert r1[0]["total_cost"] < r2[0]["total_cost"]


class TestCheapestModel:
    def test_finds_cheapest(self, sample_text):
        model = cheapest_model(sample_text)
        assert model is not None
        # The cheapest should be one of the very affordable models
        assert model.input_per_million <= Decimal("5")

    def test_filter_by_provider(self, sample_text):
        model = cheapest_model(sample_text, providers=[Provider.OPENAI])
        assert model.provider == Provider.OPENAI

    def test_filter_by_context_window(self, sample_text):
        model = cheapest_model(sample_text, min_context_window=200000)
        assert model.context_window >= 200000

    def test_no_match_raises(self):
        # Unrealistically large context requirement
        with pytest.raises(ValueError, match="No model matches"):
            cheapest_model("hi", min_context_window=999_999_999)


class TestFormatComparison:
    def test_format_empty(self):
        result = format_comparison([])
        assert "No models" in result

    def test_format_results(self, sample_text):
        results = compare_models(sample_text, models=["gpt-4o", "deepseek-chat"])
        formatted = format_comparison(results)
        assert "gpt-4o" in formatted
        assert "deepseek-chat" in formatted
        assert "$" in formatted

    def test_top_n(self, sample_text):
        results = compare_models(sample_text)
        formatted = format_comparison(results, top_n=3)
        lines = [l for l in formatted.split("\n") if l.strip() and not l.startswith("-")]
        # header + 3 data rows
        assert len(lines) <= 4
