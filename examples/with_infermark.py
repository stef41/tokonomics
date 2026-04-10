#!/usr/bin/env python3
"""Integration: infermark + tokonomics — benchmark an endpoint, then cost the run.

Flow: Configure a benchmark → run it with infermark → feed the token counts to
tokonomics to estimate what the benchmark actually cost and compare models.

Install: pip install infermark tokonomics
"""

try:
    from infermark import (
        BenchmarkConfig, BenchmarkMode, run_benchmark, format_report_text,
    )
except ImportError:
    raise SystemExit("pip install infermark  # required for this example")

try:
    from tokonomics import (
        estimate_cost, get_model, compare_models, format_comparison,
        CostTracker, count_tokens,
    )
except ImportError:
    raise SystemExit("pip install tokonomics  # required for this example")


def main() -> None:
    model_name = "gpt-4o-mini"

    # ── 1. Run an inference benchmark with infermark ─────────────────
    print("=" * 60)
    print("STEP 1: Benchmark endpoint with infermark")
    print("=" * 60)
    config = BenchmarkConfig(
        base_url="http://localhost:8000/v1",
        model=model_name,
        prompts=["Explain quantum computing in one paragraph.",
                 "Write a Python function to compute Fibonacci numbers.",
                 "What are the benefits of renewable energy?"],
        max_tokens=256,
        mode=BenchmarkMode.THROUGHPUT,
        concurrency=2,
        num_requests=6,
    )
    print(f"  Model: {config.model}")
    print(f"  Requests: {config.num_requests}, concurrency: {config.concurrency}")
    report = run_benchmark(config)
    print(f"\n{format_report_text(report)}")

    # ── 2. Extract token totals from the benchmark report ────────────
    print("=" * 60)
    print("STEP 2: Calculate cost with tokonomics")
    print("=" * 60)
    total_input_tokens = sum(r.prompt_tokens or 0 for r in report.results)
    total_output_tokens = sum(r.completion_tokens or 0 for r in report.results)
    print(f"  Total input tokens:  {total_input_tokens:,}")
    print(f"  Total output tokens: {total_output_tokens:,}")

    cost = estimate_cost(
        model=model_name,
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
    )
    print(f"  Estimated cost: ${cost.total_cost:.6f}")
    print(f"    Input:  ${cost.input_cost:.6f}")
    print(f"    Output: ${cost.output_cost:.6f}")

    # ── 3. Compare: what would this same workload cost elsewhere? ────
    print("\n" + "=" * 60)
    print("STEP 3: Compare across models with tokonomics")
    print("=" * 60)
    alternatives = ["gpt-4o", "gpt-4o-mini", "claude-3.5-sonnet", "claude-3-haiku"]
    comparison = compare_models(
        models=alternatives,
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
    )
    print(format_comparison(comparison))
    print("\nBenchmark + cost analysis complete.")


if __name__ == "__main__":
    main()
