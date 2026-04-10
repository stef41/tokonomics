"""Compare costs across models and providers.

Demonstrates how to compare pricing for a given prompt across
multiple models, find the cheapest option, and format results
for display.
"""

from tokonomics import compare_models, cheapest_model, format_comparison

PROMPT = (
    "Translate the following technical documentation from English to "
    "French, preserving all code examples and formatting. The document "
    "covers advanced Python async patterns including asyncio, "
    "structured concurrency, and error handling strategies."
)


def main():
    # Compare a specific set of models
    models = ["gpt-4o", "gpt-4o-mini", "claude-3.5-sonnet", "claude-3-haiku"]
    results = compare_models(PROMPT, models=models, output_tokens=1000)

    print("=== Cost comparison (1000 output tokens) ===\n")
    for r in results:
        print(f"  {r['model']:30s} {r['provider']:10s} "
              f"${r['total_cost']:.6f}")

    print()

    # Find the absolute cheapest model
    best = cheapest_model(PROMPT, output_tokens=1000)
    print(f"Cheapest overall: {best.model_id} ({best.provider.value})")
    print(f"  Input rate:  ${best.input_per_million}/M tokens")
    print(f"  Output rate: ${best.output_per_million}/M tokens")

    print()

    # Use format_comparison for a quick readable table
    table = format_comparison(PROMPT, models=models, output_tokens=1000)
    print("=== Formatted comparison ===\n")
    print(table)


if __name__ == "__main__":
    main()
