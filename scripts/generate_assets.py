"""Generate SVG assets for README."""
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def gen_cost_comparison():
    console = Console(record=True, width=95)
    table = Table(title="Cost Comparison — 10K tokens input, 2K tokens output")
    table.add_column("Provider / Model", style="cyan")
    table.add_column("Input Cost", justify="right")
    table.add_column("Output Cost", justify="right")
    table.add_column("Total", justify="right", style="bold")
    table.add_column("vs Cheapest", justify="right")

    rows = [
        ("GPT-4o", "$0.0250", "$0.0200", "$0.0450", Text("1.0x", style="green bold")),
        ("GPT-4o mini", "$0.0015", "$0.0012", "$0.0027", Text("0.06x", style="green bold")),
        ("Claude 3.5 Sonnet", "$0.0300", "$0.0300", "$0.0600", Text("1.3x", style="yellow")),
        ("Claude 3 Haiku", "$0.0025", "$0.0025", "$0.0050", Text("0.11x", style="green")),
        ("Gemini 1.5 Pro", "$0.0125", "$0.0100", "$0.0225", Text("0.50x", style="green")),
        ("Gemini 1.5 Flash", "$0.0008", "$0.0006", "$0.0014", Text("0.03x", style="green bold")),
        ("GPT-4 Turbo", "$0.1000", "$0.0600", "$0.1600", Text("3.6x", style="red")),
        ("Claude 3 Opus", "$0.1500", "$0.1500", "$0.3000", Text("6.7x", style="red bold")),
    ]
    for model, inp, out, total, ratio in rows:
        table.add_row(model, inp, out, total, ratio)

    console.print(table)
    svg = console.export_svg(title="tokonomics — cost comparison")
    Path("assets/cost_comparison.svg").write_text(svg)
    print(f"  cost_comparison.svg: {len(svg)//1024}KB")


def gen_budget_tracker():
    console = Console(record=True, width=95)

    table = Table(title="Budget Tracker — Project: chatbot-v2")
    table.add_column("Period", style="cyan")
    table.add_column("Model", style="dim")
    table.add_column("Requests", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Cost", justify="right")
    table.add_column("Budget", justify="right")
    table.add_column("Status", justify="center")

    rows = [
        ("Apr 01–07", "gpt-4o", "12,340", "8.2M", "$36.80", "$50.00", Text("▓▓▓▓▓▓░░ 74%", style="yellow")),
        ("Apr 01–07", "gpt-4o-mini", "45,200", "31.4M", "$4.71", "$10.00", Text("▓▓▓▓░░░░ 47%", style="green")),
        ("Mar 25–31", "gpt-4o", "14,100", "9.8M", "$44.10", "$50.00", Text("▓▓▓▓▓▓▓░ 88%", style="red")),
        ("Mar 25–31", "gpt-4o-mini", "38,900", "26.1M", "$3.92", "$10.00", Text("▓▓▓░░░░░ 39%", style="green")),
        ("Mar 18–24", "gpt-4o", "11,200", "7.5M", "$33.75", "$50.00", Text("▓▓▓▓▓░░░ 68%", style="yellow")),
    ]
    for period, model, reqs, tokens, cost, budget, status in rows:
        table.add_row(period, model, reqs, tokens, cost, budget, status)

    console.print(table)
    svg = console.export_svg(title="tokonomics — budget tracker")
    Path("assets/budget_tracker.svg").write_text(svg)
    print(f"  budget_tracker.svg: {len(svg)//1024}KB")


if __name__ == "__main__":
    gen_cost_comparison()
    gen_budget_tracker()
    print("Done.")
