"""Model registry with pricing data for all major LLM providers.

Prices are in USD per million tokens, sourced from official provider pricing
pages. Last verified: April 2026.
"""

from __future__ import annotations

from decimal import Decimal

from tokonomics._types import ModelNotFoundError, ModelPricing, Provider

D = Decimal

# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------
_OPENAI_MODELS: list[ModelPricing] = [
    # GPT-4.1 family (released early 2025)
    ModelPricing(
        model_id="gpt-4.1",
        provider=Provider.OPENAI,
        input_per_million=D("2.00"),
        output_per_million=D("8.00"),
        context_window=1047576,
        max_output_tokens=32768,
        cached_input_per_million=D("0.50"),
        aliases=("gpt-4.1-2025-04-14",),
    ),
    ModelPricing(
        model_id="gpt-4.1-mini",
        provider=Provider.OPENAI,
        input_per_million=D("0.40"),
        output_per_million=D("1.60"),
        context_window=1047576,
        max_output_tokens=32768,
        cached_input_per_million=D("0.10"),
        aliases=("gpt-4.1-mini-2025-04-14",),
    ),
    ModelPricing(
        model_id="gpt-4.1-nano",
        provider=Provider.OPENAI,
        input_per_million=D("0.10"),
        output_per_million=D("0.40"),
        context_window=1047576,
        max_output_tokens=32768,
        cached_input_per_million=D("0.025"),
        aliases=("gpt-4.1-nano-2025-04-14",),
    ),
    # GPT-4o family
    ModelPricing(
        model_id="gpt-4o",
        provider=Provider.OPENAI,
        input_per_million=D("2.50"),
        output_per_million=D("10.00"),
        context_window=128000,
        max_output_tokens=16384,
        cached_input_per_million=D("1.25"),
        aliases=("gpt-4o-2024-08-06", "gpt-4o-2024-11-20"),
    ),
    ModelPricing(
        model_id="gpt-4o-mini",
        provider=Provider.OPENAI,
        input_per_million=D("0.15"),
        output_per_million=D("0.60"),
        context_window=128000,
        max_output_tokens=16384,
        cached_input_per_million=D("0.075"),
        aliases=("gpt-4o-mini-2024-07-18",),
    ),
    # o-series reasoning models
    ModelPricing(
        model_id="o3",
        provider=Provider.OPENAI,
        input_per_million=D("10.00"),
        output_per_million=D("40.00"),
        context_window=200000,
        max_output_tokens=100000,
        cached_input_per_million=D("2.50"),
        thinking_output_per_million=D("40.00"),
        aliases=("o3-2025-04-16",),
    ),
    ModelPricing(
        model_id="o3-pro",
        provider=Provider.OPENAI,
        input_per_million=D("20.00"),
        output_per_million=D("80.00"),
        context_window=200000,
        max_output_tokens=100000,
        cached_input_per_million=D("5.00"),
        thinking_output_per_million=D("80.00"),
    ),
    ModelPricing(
        model_id="o3-mini",
        provider=Provider.OPENAI,
        input_per_million=D("1.10"),
        output_per_million=D("4.40"),
        context_window=200000,
        max_output_tokens=100000,
        cached_input_per_million=D("0.55"),
        thinking_output_per_million=D("4.40"),
        aliases=("o3-mini-2025-01-31",),
    ),
    ModelPricing(
        model_id="o4-mini",
        provider=Provider.OPENAI,
        input_per_million=D("1.10"),
        output_per_million=D("4.40"),
        context_window=200000,
        max_output_tokens=100000,
        cached_input_per_million=D("0.275"),
        thinking_output_per_million=D("4.40"),
        aliases=("o4-mini-2025-04-16",),
    ),
    ModelPricing(
        model_id="o1",
        provider=Provider.OPENAI,
        input_per_million=D("15.00"),
        output_per_million=D("60.00"),
        context_window=200000,
        max_output_tokens=100000,
        cached_input_per_million=D("7.50"),
        thinking_output_per_million=D("60.00"),
        aliases=("o1-2024-12-17",),
    ),
    ModelPricing(
        model_id="o1-mini",
        provider=Provider.OPENAI,
        input_per_million=D("1.10"),
        output_per_million=D("4.40"),
        context_window=128000,
        max_output_tokens=65536,
        cached_input_per_million=D("0.55"),
        thinking_output_per_million=D("4.40"),
        aliases=("o1-mini-2024-09-12",),
    ),
    # Legacy GPT-4 (still available)
    ModelPricing(
        model_id="gpt-4-turbo",
        provider=Provider.OPENAI,
        input_per_million=D("10.00"),
        output_per_million=D("30.00"),
        context_window=128000,
        max_output_tokens=4096,
        aliases=("gpt-4-turbo-2024-04-09",),
    ),
    ModelPricing(
        model_id="gpt-4",
        provider=Provider.OPENAI,
        input_per_million=D("30.00"),
        output_per_million=D("60.00"),
        context_window=8192,
        max_output_tokens=8192,
    ),
    # Embeddings
    ModelPricing(
        model_id="text-embedding-3-large",
        provider=Provider.OPENAI,
        input_per_million=D("0.13"),
        output_per_million=D("0"),
        context_window=8191,
    ),
    ModelPricing(
        model_id="text-embedding-3-small",
        provider=Provider.OPENAI,
        input_per_million=D("0.02"),
        output_per_million=D("0"),
        context_window=8191,
    ),
]

# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------
_ANTHROPIC_MODELS: list[ModelPricing] = [
    ModelPricing(
        model_id="claude-sonnet-4-20250514",
        provider=Provider.ANTHROPIC,
        input_per_million=D("3.00"),
        output_per_million=D("15.00"),
        context_window=200000,
        max_output_tokens=16000,
        cached_input_per_million=D("0.30"),
        aliases=("claude-sonnet-4",),
    ),
    ModelPricing(
        model_id="claude-opus-4-20250514",
        provider=Provider.ANTHROPIC,
        input_per_million=D("15.00"),
        output_per_million=D("75.00"),
        context_window=200000,
        max_output_tokens=32000,
        cached_input_per_million=D("1.50"),
        thinking_output_per_million=D("75.00"),
        aliases=("claude-opus-4",),
    ),
    ModelPricing(
        model_id="claude-3.7-sonnet",
        provider=Provider.ANTHROPIC,
        input_per_million=D("3.00"),
        output_per_million=D("15.00"),
        context_window=200000,
        max_output_tokens=16000,
        cached_input_per_million=D("0.30"),
        thinking_output_per_million=D("15.00"),
        aliases=("claude-3-7-sonnet-20250219",),
    ),
    ModelPricing(
        model_id="claude-3.5-sonnet",
        provider=Provider.ANTHROPIC,
        input_per_million=D("3.00"),
        output_per_million=D("15.00"),
        context_window=200000,
        max_output_tokens=8192,
        cached_input_per_million=D("0.30"),
        aliases=(
            "claude-3-5-sonnet-20241022",
            "claude-3-5-sonnet-20240620",
            "claude-3-5-sonnet-latest",
        ),
    ),
    ModelPricing(
        model_id="claude-3.5-haiku",
        provider=Provider.ANTHROPIC,
        input_per_million=D("0.80"),
        output_per_million=D("4.00"),
        context_window=200000,
        max_output_tokens=8192,
        cached_input_per_million=D("0.08"),
        aliases=("claude-3-5-haiku-20241022",),
    ),
    ModelPricing(
        model_id="claude-3-opus",
        provider=Provider.ANTHROPIC,
        input_per_million=D("15.00"),
        output_per_million=D("75.00"),
        context_window=200000,
        max_output_tokens=4096,
        cached_input_per_million=D("1.50"),
        aliases=("claude-3-opus-20240229",),
    ),
    ModelPricing(
        model_id="claude-3-haiku",
        provider=Provider.ANTHROPIC,
        input_per_million=D("0.25"),
        output_per_million=D("1.25"),
        context_window=200000,
        max_output_tokens=4096,
        cached_input_per_million=D("0.03"),
        aliases=("claude-3-haiku-20240307",),
    ),
]

# ---------------------------------------------------------------------------
# Google (Gemini)
# ---------------------------------------------------------------------------
_GOOGLE_MODELS: list[ModelPricing] = [
    ModelPricing(
        model_id="gemini-2.5-pro",
        provider=Provider.GOOGLE,
        input_per_million=D("1.25"),
        output_per_million=D("10.00"),
        context_window=1048576,
        max_output_tokens=65536,
        cached_input_per_million=D("0.3125"),
        thinking_output_per_million=D("10.00"),
        aliases=("gemini-2.5-pro-preview-05-06",),
    ),
    ModelPricing(
        model_id="gemini-2.5-flash",
        provider=Provider.GOOGLE,
        input_per_million=D("0.15"),
        output_per_million=D("0.60"),
        context_window=1048576,
        max_output_tokens=65536,
        cached_input_per_million=D("0.0375"),
        thinking_output_per_million=D("3.50"),
        aliases=("gemini-2.5-flash-preview-04-17",),
    ),
    ModelPricing(
        model_id="gemini-2.0-flash",
        provider=Provider.GOOGLE,
        input_per_million=D("0.10"),
        output_per_million=D("0.40"),
        context_window=1048576,
        max_output_tokens=8192,
        cached_input_per_million=D("0.025"),
        aliases=("gemini-2.0-flash-001",),
    ),
    ModelPricing(
        model_id="gemini-1.5-pro",
        provider=Provider.GOOGLE,
        input_per_million=D("1.25"),
        output_per_million=D("5.00"),
        context_window=2097152,
        max_output_tokens=8192,
        cached_input_per_million=D("0.3125"),
        aliases=("gemini-1.5-pro-002", "gemini-1.5-pro-latest"),
    ),
    ModelPricing(
        model_id="gemini-1.5-flash",
        provider=Provider.GOOGLE,
        input_per_million=D("0.075"),
        output_per_million=D("0.30"),
        context_window=1048576,
        max_output_tokens=8192,
        cached_input_per_million=D("0.01875"),
        aliases=("gemini-1.5-flash-002", "gemini-1.5-flash-latest"),
    ),
]

# ---------------------------------------------------------------------------
# Mistral
# ---------------------------------------------------------------------------
_MISTRAL_MODELS: list[ModelPricing] = [
    ModelPricing(
        model_id="mistral-large-latest",
        provider=Provider.MISTRAL,
        input_per_million=D("2.00"),
        output_per_million=D("6.00"),
        context_window=128000,
        max_output_tokens=8192,
        aliases=("mistral-large",),
    ),
    ModelPricing(
        model_id="mistral-small-latest",
        provider=Provider.MISTRAL,
        input_per_million=D("0.10"),
        output_per_million=D("0.30"),
        context_window=32000,
        max_output_tokens=8192,
        aliases=("mistral-small",),
    ),
    ModelPricing(
        model_id="codestral-latest",
        provider=Provider.MISTRAL,
        input_per_million=D("0.30"),
        output_per_million=D("0.90"),
        context_window=256000,
        max_output_tokens=8192,
        aliases=("codestral",),
    ),
    ModelPricing(
        model_id="pixtral-large-latest",
        provider=Provider.MISTRAL,
        input_per_million=D("2.00"),
        output_per_million=D("6.00"),
        context_window=128000,
        max_output_tokens=8192,
        aliases=("pixtral-large",),
    ),
    ModelPricing(
        model_id="mistral-embed",
        provider=Provider.MISTRAL,
        input_per_million=D("0.10"),
        output_per_million=D("0"),
        context_window=8192,
    ),
]

# ---------------------------------------------------------------------------
# DeepSeek
# ---------------------------------------------------------------------------
_DEEPSEEK_MODELS: list[ModelPricing] = [
    ModelPricing(
        model_id="deepseek-chat",
        provider=Provider.DEEPSEEK,
        input_per_million=D("0.14"),
        output_per_million=D("0.28"),
        context_window=65536,
        max_output_tokens=8192,
        cached_input_per_million=D("0.014"),
        aliases=("deepseek-v3",),
    ),
    ModelPricing(
        model_id="deepseek-reasoner",
        provider=Provider.DEEPSEEK,
        input_per_million=D("0.55"),
        output_per_million=D("2.19"),
        context_window=65536,
        max_output_tokens=8192,
        cached_input_per_million=D("0.14"),
        thinking_output_per_million=D("2.19"),
        aliases=("deepseek-r1",),
    ),
]

# ---------------------------------------------------------------------------
# xAI (Grok)
# ---------------------------------------------------------------------------
_XAI_MODELS: list[ModelPricing] = [
    ModelPricing(
        model_id="grok-2",
        provider=Provider.XAI,
        input_per_million=D("2.00"),
        output_per_million=D("10.00"),
        context_window=131072,
        max_output_tokens=8192,
        aliases=("grok-2-1212",),
    ),
    ModelPricing(
        model_id="grok-3",
        provider=Provider.XAI,
        input_per_million=D("3.00"),
        output_per_million=D("15.00"),
        context_window=131072,
        max_output_tokens=16384,
        aliases=("grok-3-beta",),
    ),
    ModelPricing(
        model_id="grok-3-mini",
        provider=Provider.XAI,
        input_per_million=D("0.30"),
        output_per_million=D("0.50"),
        context_window=131072,
        max_output_tokens=16384,
        thinking_output_per_million=D("0.50"),
        aliases=("grok-3-mini-beta",),
    ),
]

# ---------------------------------------------------------------------------
# Cohere
# ---------------------------------------------------------------------------
_COHERE_MODELS: list[ModelPricing] = [
    ModelPricing(
        model_id="command-r-plus",
        provider=Provider.COHERE,
        input_per_million=D("2.50"),
        output_per_million=D("10.00"),
        context_window=128000,
        max_output_tokens=4096,
        aliases=("command-r-plus-08-2024",),
    ),
    ModelPricing(
        model_id="command-r",
        provider=Provider.COHERE,
        input_per_million=D("0.15"),
        output_per_million=D("0.60"),
        context_window=128000,
        max_output_tokens=4096,
        aliases=("command-r-08-2024",),
    ),
    ModelPricing(
        model_id="embed-english-v3.0",
        provider=Provider.COHERE,
        input_per_million=D("0.10"),
        output_per_million=D("0"),
        context_window=512,
    ),
    ModelPricing(
        model_id="embed-multilingual-v3.0",
        provider=Provider.COHERE,
        input_per_million=D("0.10"),
        output_per_million=D("0"),
        context_window=512,
    ),
]

# ---------------------------------------------------------------------------
# Build the unified registry
# ---------------------------------------------------------------------------
_ALL_MODELS: list[ModelPricing] = (
    _OPENAI_MODELS
    + _ANTHROPIC_MODELS
    + _GOOGLE_MODELS
    + _MISTRAL_MODELS
    + _DEEPSEEK_MODELS
    + _XAI_MODELS
    + _COHERE_MODELS
)

MODEL_REGISTRY: dict[str, ModelPricing] = {}
_ALIAS_MAP: dict[str, str] = {}

for _model in _ALL_MODELS:
    MODEL_REGISTRY[_model.model_id] = _model
    for _alias in _model.aliases:
        _ALIAS_MAP[_alias] = _model.model_id


def _resolve(model_id: str) -> str:
    """Resolve an alias to the canonical model ID."""
    return _ALIAS_MAP.get(model_id, model_id)


def get_model(model_id: str) -> ModelPricing:
    """Look up a model by its ID or alias.

    Raises ``ModelNotFoundError`` if the model is not in the registry.
    """
    canonical = _resolve(model_id)
    try:
        return MODEL_REGISTRY[canonical]
    except KeyError:
        # Try case-insensitive match
        lower = canonical.lower()
        for key, model in MODEL_REGISTRY.items():
            if key.lower() == lower:
                return model
        raise ModelNotFoundError(
            f"Model '{model_id}' not found. Use list_models() to see available models."
        ) from None


def list_models(provider: Provider | None = None) -> list[ModelPricing]:
    """List all models, optionally filtered by provider."""
    models = list(MODEL_REGISTRY.values())
    if provider is not None:
        models = [m for m in models if m.provider == provider]
    return sorted(models, key=lambda m: (m.provider.value, m.model_id))


def find_models(query: str) -> list[ModelPricing]:
    """Find models whose ID contains the query string (case-insensitive)."""
    q = query.lower()
    results: list[ModelPricing] = []
    seen: set[str] = set()
    for model_id, model in MODEL_REGISTRY.items():
        if q in model_id.lower() and model_id not in seen:
            results.append(model)
            seen.add(model_id)
    for alias, canonical in _ALIAS_MAP.items():
        if q in alias.lower() and canonical not in seen:
            results.append(MODEL_REGISTRY[canonical])
            seen.add(canonical)
    return sorted(results, key=lambda m: (m.provider.value, m.model_id))
