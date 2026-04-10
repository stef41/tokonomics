"""Budget management with threshold alerts.

Demonstrates how to set spending limits, track usage against a budget,
receive alerts at configurable thresholds, and handle budget overruns.
"""

from decimal import Decimal

from tokonomics import Budget, BudgetExceededError, calculate_cost

def alert_80(budget):
    print(f"  ⚠ 80% of ${budget.limit} budget used! "
          f"${budget.remaining:.4f} remaining")

def alert_95(budget):
    print(f"  🚨 95% used — only ${budget.remaining:.4f} left!")


def main():
    # Create a $1.00 daily budget
    budget = Budget(limit=1.00, period="daily")
    budget.on_threshold(0.80, alert_80)
    budget.on_threshold(0.95, alert_95)

    print(f"Budget: ${budget.limit} ({budget.period})")
    print(f"Remaining: ${budget.remaining}")
    print()

    # Simulate a series of API calls
    calls = [
        ("gpt-4o", 2000, 500),
        ("gpt-4o-mini", 5000, 1000),
        ("gpt-4o", 3000, 1500),
        ("gpt-4o", 8000, 2000),
        ("gpt-4o", 15000, 5000),
    ]

    for model, inp, out in calls:
        usage = calculate_cost(model, input_tokens=inp, output_tokens=out)
        print(f"Call: {model} ({inp} in / {out} out) = ${usage.total_cost:.6f}")

        if not budget.check(usage.total_cost):
            print(f"  ✗ Would exceed budget — skipping")
            continue

        try:
            budget.record(usage.total_cost)
            print(f"  ✓ Recorded. Used: ${budget.used:.6f} "
                  f"({budget.utilization:.0%})")
        except BudgetExceededError as e:
            print(f"  ✗ {e}")

    print(f"\nFinal: ${budget.used:.6f} of ${budget.limit} used")


if __name__ == "__main__":
    main()
