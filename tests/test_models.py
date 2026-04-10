"""Tests for the model registry."""

import pytest
from decimal import Decimal

from tokonomics.models import (
    get_model,
    list_models,
    find_models,
    MODEL_REGISTRY,
)
from tokonomics._types import ModelNotFoundError, ModelPricing, Provider


class TestGetModel:
    def test_get_by_canonical_id(self):
        model = get_model("gpt-4o")
        assert model.model_id == "gpt-4o"
        assert model.provider == Provider.OPENAI

    def test_get_by_alias(self):
        model = get_model("gpt-4o-2024-08-06")
        assert model.model_id == "gpt-4o"

    def test_get_anthropic_model(self):
        model = get_model("claude-sonnet-4-20250514")
        assert model.provider == Provider.ANTHROPIC
        assert model.input_per_million == Decimal("3.00")

    def test_get_anthropic_alias(self):
        model = get_model("claude-sonnet-4")
        assert model.model_id == "claude-sonnet-4-20250514"

    def test_get_google_model(self):
        model = get_model("gemini-2.5-pro")
        assert model.provider == Provider.GOOGLE

    def test_get_deepseek_model(self):
        model = get_model("deepseek-chat")
        assert model.provider == Provider.DEEPSEEK

    def test_get_deepseek_alias(self):
        model = get_model("deepseek-v3")
        assert model.model_id == "deepseek-chat"

    def test_get_xai_model(self):
        model = get_model("grok-3")
        assert model.provider == Provider.XAI

    def test_case_insensitive(self):
        model = get_model("GPT-4o")
        assert model.model_id == "gpt-4o"

    def test_not_found_raises(self):
        with pytest.raises(ModelNotFoundError, match="not-a-real-model"):
            get_model("not-a-real-model")


class TestListModels:
    def test_list_all(self):
        models = list_models()
        assert len(models) > 20

    def test_list_by_provider(self):
        openai_models = list_models(Provider.OPENAI)
        assert all(m.provider == Provider.OPENAI for m in openai_models)
        assert len(openai_models) >= 5

    def test_list_anthropic(self):
        models = list_models(Provider.ANTHROPIC)
        assert all(m.provider == Provider.ANTHROPIC for m in models)
        assert len(models) >= 3

    def test_sorted_output(self):
        models = list_models()
        for i in range(len(models) - 1):
            assert (models[i].provider.value, models[i].model_id) <= (
                models[i + 1].provider.value,
                models[i + 1].model_id,
            )


class TestFindModels:
    def test_find_gpt(self):
        results = find_models("gpt")
        assert len(results) >= 3
        assert all("gpt" in m.model_id.lower() for m in results)

    def test_find_claude(self):
        results = find_models("claude")
        assert len(results) >= 3

    def test_find_by_alias(self):
        results = find_models("deepseek-v3")
        assert len(results) >= 1

    def test_find_no_matches(self):
        results = find_models("nonexistentxyz")
        assert results == []

    def test_find_case_insensitive(self):
        results = find_models("GPT")
        assert len(results) >= 3


class TestModelPricing:
    def test_per_token_rates(self):
        model = get_model("gpt-4o")
        # $2.50 per million = $0.0000025 per token
        assert model.input_per_token == Decimal("2.50") / Decimal("1000000")
        assert model.output_per_token == Decimal("10.00") / Decimal("1000000")

    def test_cost_simple(self):
        model = get_model("gpt-4o")
        cost = model.cost(input_tokens=1000, output_tokens=500)
        expected_input = Decimal("1000") * Decimal("2.50") / Decimal("1000000")
        expected_output = Decimal("500") * Decimal("10.00") / Decimal("1000000")
        assert cost == expected_input + expected_output

    def test_cost_with_caching(self):
        model = get_model("gpt-4o")
        # 1000 input, 500 cached
        cost_cached = model.cost(
            input_tokens=1000, output_tokens=0, cached_tokens=500
        )
        cost_no_cache = model.cost(
            input_tokens=1000, output_tokens=0, cached_tokens=0
        )
        assert cost_cached < cost_no_cache

    def test_cost_with_thinking(self):
        model = get_model("o3")
        assert model.thinking_output_per_million is not None
        cost = model.cost(
            input_tokens=100, output_tokens=1000, thinking_tokens=500
        )
        assert cost > Decimal("0")

    def test_all_models_have_valid_pricing(self):
        for model_id, model in MODEL_REGISTRY.items():
            assert model.input_per_million >= 0, f"{model_id} has negative input price"
            assert model.output_per_million >= 0, f"{model_id} has negative output price"
            assert model.context_window > 0, f"{model_id} has invalid context window"

    @pytest.mark.parametrize(
        "model_id",
        ["gpt-4o", "claude-sonnet-4-20250514", "gemini-2.5-pro", "deepseek-chat"],
    )
    def test_cached_input_cheaper(self, model_id):
        model = get_model(model_id)
        if model.cached_input_per_million is not None:
            assert model.cached_input_per_million < model.input_per_million

    def test_frozen_dataclass(self):
        model = get_model("gpt-4o")
        with pytest.raises(AttributeError):
            model.input_per_million = Decimal("999")  # type: ignore[misc]
