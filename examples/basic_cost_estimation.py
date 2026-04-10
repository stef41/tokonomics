"""Basic cost estimation with tokonomics.

Demonstrates how to estimate costs before making API calls,
calculate exact costs from known token counts, and count tokens
for a given model.
"""

from tokonomics import count_tokens, estimate_cost, calculate_cost

SAMPLE_PROMPT = (
    "You are an expert Python developer. Review the following code "
    "and suggest improvements for readability, performance, and "
    "maintainability. Explain each suggestion with a brief rationale."
)


def main():
    # Count tokens for different models
    for model in ["gpt-4o", "gpt-4o-mini", "claude-3.5-sonnet"]:
        tokens = count_tokens(SAMPLE_PROMPT, model)
        print(f"{model}: {tokens} tokens")

    print()

    # Estimate cost before sending a request
    estimate = estimate_cost(SAMPLE_PROMPT, model="gpt-4o")
    print(f"Model: {estimate.model}")
    print(f"Input tokens: {estimate.estimated_input_tokens}")
    print(f"Input cost: ${estimate.estimated_input_cost:.6f}")
    print(f"Max output cost: ${estimate.estimated_max_output_cost:.6f}")
    print(f"Context window: {estimate.context_window:,}")

    print()

    # Calculate exact cost from known usage
    usage = calculate_cost("gpt-4o", input_tokens=1500, output_tokens=800)
    print(f"Exact cost for 1500 in / 800 out:")
    print(f"  Input:  ${usage.input_cost:.6f}")
    print(f"  Output: ${usage.output_cost:.6f}")
    print(f"  Total:  ${usage.total_cost:.6f}")

    # Cost with cached tokens
    cached = calculate_cost("gpt-4o", input_tokens=1500, output_tokens=800, cached_tokens=500)
    print(f"\nWith 500 cached input tokens: ${cached.total_cost:.6f}")
    savings = usage.total_cost - cached.total_cost
    print(f"Savings from caching: ${savings:.6f}")


if __name__ == "__main__":
    main()
