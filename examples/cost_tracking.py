"""Cost tracking with CostTracker and the track_cost decorator.

Demonstrates how to accumulate costs across multiple API calls,
inspect per-model breakdowns, and use the decorator for automatic
tracking.
"""

from tokonomics import CostTracker, track_cost, get_global_tracker


def main():
    # Manual tracking with a context manager
    with CostTracker() as tracker:
        tracker.record("gpt-4o", input_tokens=500, output_tokens=200)
        tracker.record("gpt-4o", input_tokens=1200, output_tokens=600)
        tracker.record("gpt-4o-mini", input_tokens=3000, output_tokens=1500)
        tracker.record("claude-3.5-sonnet", input_tokens=800, output_tokens=400)

    print("=== Manual tracking ===\n")
    print(tracker.summary())

    print(f"\nCost by model:")
    for model, cost in tracker.by_model().items():
        print(f"  {model}: ${cost:.6f}")

    print(f"\nCost by provider:")
    for provider, cost in tracker.by_provider().items():
        print(f"  {provider}: ${cost:.6f}")

    # Using the @track_cost decorator
    @track_cost(model="gpt-4o")
    def fake_chat(prompt):
        """Simulate an API call returning token usage."""
        return {
            "text": f"Response to: {prompt[:30]}...",
            "input_tokens": len(prompt.split()) * 2,
            "output_tokens": 150,
        }

    print("\n=== Decorator tracking ===\n")
    fake_chat("Explain quantum computing in simple terms")
    fake_chat("Write a haiku about Python programming")
    fake_chat("Summarize the history of machine learning")

    print(f"Calls tracked: {len(get_global_tracker().records)}")
    print(f"Total spend:   ${get_global_tracker().total_cost:.6f}")


if __name__ == "__main__":
    main()
