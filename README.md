# tokonomics

**Know what your LLM calls actually cost.**

tokonomics is a lightweight Python library for token counting and cost management across every major LLM provider. It replaces guesswork with exact numbers — so you can track spending, set budgets, and compare providers before committing to a model.

```python
from tokonomics import estimate_cost, calculate_cost, compare_models

# How much will this prompt cost?
est = estimate_cost("Explain quantum computing in simple terms", model="gpt-4o")
print(f"Input: {est.estimated_input_tokens} tokens, ${est.estimated_input_cost:.6f}")

# Track exact costs from API responses
usage = calculate_cost("claude-sonnet-4-20250514", input_tokens=1500, output_tokens=800)
print(f"Total: ${usage.total_cost:.6f}")

# Which model is cheapest for this prompt?
results = compare_models("Your long prompt here...", output_tokens=2000)
for r in results[:5]:
    print(f"  {r['model']:<30} ${r['total_cost']:.6f}")
```

## Why tokonomics?

Every team using LLM APIs has the same question: *"how much is this costing us?"*

Existing solutions are either unmaintained (tokencost hasn't been updated in 7 months), locked behind a signup wall, or buried inside massive frameworks. tokonomics is none of those things. It's a focused library that does one job well.

**What you get:**

- Accurate pricing for 40+ models across OpenAI, Anthropic, Google, Mistral, DeepSeek, xAI, and Cohere
- Proper handling of cached input tokens, thinking/reasoning tokens, and batch pricing
- Token counting via tiktoken (with a fallback estimator when tiktoken isn't installed)
- Cost tracking with decorators and context managers
- Budget management with threshold alerts
- Cross-provider comparison to find the cheapest model for any input
- A CLI for quick estimates without writing code

## Install

```bash
pip install tokonomics
```

For token counting with tiktoken (recommended):

```bash
pip install tokonomics[tiktoken]
```

For the CLI:

```bash
pip install tokonomics[cli]
```

Everything:

```bash
pip install tokonomics[all]
```

## Usage

### Count tokens

```python
from tokonomics import count_tokens, count_message_tokens

count_tokens("Hello, world!", model="gpt-4o")
# 4

count_message_tokens(
    [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is 2+2?"},
    ],
    model="gpt-4o",
)
# 18
```

### Estimate cost before calling the API

```python
from tokonomics import estimate_cost

est = estimate_cost("Your prompt text here", model="claude-sonnet-4-20250514")
print(est.estimated_input_tokens)     # 5
print(est.estimated_input_cost)       # Decimal('0.000015')
print(est.estimated_max_output_cost)  # Decimal('0.240000')
```

### Calculate exact cost from API response

```python
from tokonomics import calculate_cost

# After getting token counts from the API response:
usage = calculate_cost(
    model="gpt-4o",
    input_tokens=1500,
    output_tokens=800,
    cached_tokens=500,    # prompt caching discount
)
print(usage.total_cost)    # Decimal('0.010875')
print(usage.input_cost)    # Decimal('0.002875')
print(usage.output_cost)   # Decimal('0.008000')
```

### Track costs across multiple calls

```python
from tokonomics import CostTracker

with CostTracker() as tracker:
    # After each API call, record the usage:
    tracker.record("gpt-4o", input_tokens=500, output_tokens=200)
    tracker.record("gpt-4o", input_tokens=300, output_tokens=150)
    tracker.record("claude-sonnet-4-20250514", input_tokens=1000, output_tokens=500)

print(tracker.total_cost)        # Decimal('0.013325')
print(tracker.by_model())        # {'claude-sonnet-4-20250514': ..., 'gpt-4o': ...}
print(tracker.by_provider())     # {'anthropic': ..., 'openai': ...}
print(tracker.summary())
```

### Use the decorator for automatic tracking

```python
from tokonomics import track_cost

@track_cost(model="gpt-4o")
def ask_gpt(prompt: str) -> dict:
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    )
    return {
        "text": response.choices[0].message.content,
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
    }

result = ask_gpt("What is Python?")
print(ask_gpt.get_last_usage().total_cost)
print(ask_gpt.get_total_cost())
```

### Set budgets

```python
from tokonomics import Budget, BudgetExceededError

budget = Budget(limit=5.00, period="daily")
budget.on_threshold(0.8, lambda b: print(f"Warning: {b.utilization:.0%} of daily budget used"))

# In your API call loop:
try:
    usage = calculate_cost("gpt-4o", input_tokens=1000, output_tokens=500)
    budget.record(usage.total_cost)
except BudgetExceededError:
    print("Daily budget exceeded, switching to cheaper model")
```

### Compare providers

```python
from tokonomics import compare_models, cheapest_model, format_comparison

# Compare all models
results = compare_models("Your prompt text", output_tokens=1000)
print(format_comparison(results, top_n=10))

# Find the absolute cheapest
model = cheapest_model("Your prompt text", min_context_window=128000)
print(f"Use {model.model_id}: ${model.input_per_million}/M input")
```

### Look up model details

```python
from tokonomics import get_model, list_models, find_models, Provider

model = get_model("gpt-4o")
print(model.input_per_million)           # Decimal('2.50')
print(model.cached_input_per_million)    # Decimal('1.25')
print(model.context_window)              # 128000

# List all Anthropic models
for m in list_models(Provider.ANTHROPIC):
    print(f"{m.model_id}: ${m.input_per_million}/M in, ${m.output_per_million}/M out")

# Search models
for m in find_models("claude"):
    print(m.model_id)
```

## CLI

```bash
# Estimate cost for a prompt
tokonomics estimate "What is the meaning of life?" -m gpt-4o

# Estimate from a file
tokonomics estimate @prompt.txt -m claude-sonnet-4-20250514

# Compare costs across all providers
tokonomics compare "Your prompt" -n 10

# Filter by provider
tokonomics compare "Your prompt" -p openai,anthropic

# List all models
tokonomics models

# List models for a specific provider
tokonomics models -p google

# Get detailed pricing
tokonomics price gpt-4.1

# Find the cheapest model
tokonomics cheapest "Your prompt" -c 128000
```

## Supported Models

| Provider | Models | Cached Pricing | Thinking Tokens |
|----------|--------|---------------|-----------------|
| OpenAI | GPT-4.1, GPT-4o, o1, o3, o4-mini, embeddings | Yes | Yes (o-series) |
| Anthropic | Claude Opus 4, Sonnet 4, 3.7/3.5 Sonnet, 3.5 Haiku | Yes | Yes (Opus 4) |
| Google | Gemini 2.5 Pro/Flash, 2.0 Flash, 1.5 Pro/Flash | Yes | Yes (2.5 series) |
| Mistral | Large, Small, Codestral, Pixtral | No | No |
| DeepSeek | Chat (V3), Reasoner (R1) | Yes | Yes (Reasoner) |
| xAI | Grok-2, Grok-3, Grok-3 Mini | No | Yes (Mini) |
| Cohere | Command R+, Command R, embeddings | No | No |

Prices are verified against official provider pricing pages. If you notice a discrepancy, please [open an issue](https://github.com/zbhatti/tokonomics/issues).

## Updating Prices

Model pricing changes frequently. When prices change:

1. Update the relevant entries in `src/tokonomics/models.py`
2. Run the tests to verify consistency
3. Submit a PR

We aim to update prices within 48 hours of provider announcements.

## How It Works

- **Token counting** uses [tiktoken](https://github.com/openai/tiktoken) for accurate BPE tokenization. For non-OpenAI models, tiktoken's `o200k_base` encoding provides a reasonable approximation. If tiktoken isn't installed, a word-based heuristic kicks in.
- **Pricing** is stored as `Decimal` values to avoid floating-point rounding issues. $2.50 per million tokens is exactly $0.0000025 per token, not $0.0000024999999999.
- **The tracker** is thread-safe and uses monotonic timestamps for period-based budgets.

## Contributing

Contributions are welcome — especially pricing updates, new provider support, and bug fixes.

```bash
git clone https://github.com/zbhatti/tokonomics.git
cd tokonomics
pip install -e ".[all]"
pip install pytest ruff mypy
pytest
```

## License

Apache 2.0
