"""Command-line interface for tokonomics.

Requires the ``cli`` extra: ``pip install tokonomics[cli]``
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

try:
    import click
    _HAS_CLICK = True
except ImportError:
    _HAS_CLICK = False

try:
    from rich.console import Console
    from rich.table import Table
    _console = Console()
    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False
    _console = None  # type: ignore[assignment]


def _read_input(text: str) -> str:
    if text.startswith("@"):
        path = Path(text[1:])
        if not path.is_file():
            print(f"File not found: {path}", file=sys.stderr)
            sys.exit(1)
        return path.read_text(encoding="utf-8")
    return text


def _build_cli() -> Any:
    if not _HAS_CLICK:
        return None

    from tokonomics._types import ModelNotFoundError, Provider
    from tokonomics.compare import cheapest_model as _cheapest_model
    from tokonomics.compare import compare_models as _compare_models
    from tokonomics.compare import format_comparison as _format_comparison
    from tokonomics.cost import estimate_cost as _estimate_cost
    from tokonomics.models import find_models as _find_models
    from tokonomics.models import get_model as _get_model
    from tokonomics.models import list_models as _list_models

    @click.group()
    @click.version_option(package_name="tokonomics")
    def cli() -> None:
        """tokonomics -- LLM token cost management."""

    @cli.command()
    @click.argument("text")
    @click.option("-m", "--model", default="gpt-4o", help="Model ID.")
    def estimate(text: str, model: str) -> None:
        """Estimate the cost of sending TEXT to a model."""
        text = _read_input(text)
        try:
            est = _estimate_cost(text, model)
        except ModelNotFoundError as e:
            click.echo(str(e), err=True)
            raise SystemExit(1)

        if _HAS_RICH:
            table = Table(title=f"Cost Estimate -- {est.model}")
            table.add_column("Metric", style="bold")
            table.add_column("Value", justify="right")
            table.add_row("Provider", est.provider.value)
            table.add_row("Input tokens", f"{est.estimated_input_tokens:,}")
            table.add_row("Input cost", f"${est.estimated_input_cost:.6f}")
            if est.estimated_max_output_cost is not None:
                table.add_row("Max output cost", f"${est.estimated_max_output_cost:.6f}")
            table.add_row("Context window", f"{est.context_window:,}")
            _console.print(table)
        else:
            click.echo(f"Model:          {est.model}")
            click.echo(f"Provider:       {est.provider.value}")
            click.echo(f"Input tokens:   {est.estimated_input_tokens:,}")
            click.echo(f"Input cost:     ${est.estimated_input_cost:.6f}")
            if est.estimated_max_output_cost is not None:
                click.echo(f"Max output cost: ${est.estimated_max_output_cost:.6f}")
            click.echo(f"Context window: {est.context_window:,}")

    @cli.command()
    @click.argument("text")
    @click.option("-o", "--output-tokens", default=500, type=int, help="Assumed output tokens.")
    @click.option("-p", "--providers", default=None, help="Comma-separated provider filter.")
    @click.option("-n", "--top", default=10, type=int, help="Show top N cheapest.")
    def compare(text: str, output_tokens: int, providers: str | None, top: int) -> None:
        """Compare cost across models."""
        text = _read_input(text)
        results = _compare_models(text, output_tokens=output_tokens)
        if providers:
            prov_set = {p.strip().lower() for p in providers.split(",")}
            results = [r for r in results if r["provider"] in prov_set]
        results = results[:top]
        if not results:
            click.echo("No models match the criteria.")
            return
        if _HAS_RICH:
            table = Table(title="Cost Comparison (cheapest first)")
            table.add_column("Model", style="bold")
            table.add_column("Provider")
            table.add_column("Input Tok", justify="right")
            table.add_column("Input $", justify="right")
            table.add_column("Output $", justify="right")
            table.add_column("Total $", justify="right", style="green")
            for r in results:
                table.add_row(
                    r["model"], r["provider"],
                    f"{r['input_tokens']:,}",
                    f"${r['input_cost']:.6f}",
                    f"${r['output_cost']:.6f}",
                    f"${r['total_cost']:.6f}",
                )
            _console.print(table)
        else:
            click.echo(_format_comparison(results))

    @cli.command(name="models")
    @click.option("-p", "--provider", default=None, help="Filter by provider name.")
    def list_models_cmd(provider: str | None) -> None:
        """List all supported models and their pricing."""
        prov = None
        if provider:
            try:
                prov = Provider(provider.lower())
            except ValueError:
                click.echo(f"Unknown provider: {provider}", err=True)
                click.echo(f"Available: {', '.join(p.value for p in Provider)}")
                raise SystemExit(1)
        models = _list_models(prov)
        if _HAS_RICH:
            table = Table(title="Supported Models")
            table.add_column("Model", style="bold")
            table.add_column("Provider")
            table.add_column("Input $/M", justify="right")
            table.add_column("Output $/M", justify="right")
            table.add_column("Context", justify="right")
            for m in models:
                table.add_row(
                    m.model_id, m.provider.value,
                    f"${m.input_per_million}", f"${m.output_per_million}",
                    f"{m.context_window:,}",
                )
            _console.print(table)
        else:
            for m in models:
                click.echo(
                    f"{m.model_id:<30} {m.provider.value:<12} "
                    f"in=${m.input_per_million}/M  out=${m.output_per_million}/M  "
                    f"ctx={m.context_window:,}"
                )

    @cli.command()
    @click.argument("model_id")
    def price(model_id: str) -> None:
        """Show detailed pricing for a specific model."""
        try:
            m = _get_model(model_id)
        except ModelNotFoundError as e:
            click.echo(str(e), err=True)
            matches = _find_models(model_id)
            if matches:
                click.echo("Did you mean:", err=True)
                for match in matches[:5]:
                    click.echo(f"  {match.model_id}", err=True)
            raise SystemExit(1)
        if _HAS_RICH:
            table = Table(title=f"Pricing -- {m.model_id}")
            table.add_column("Attribute", style="bold")
            table.add_column("Value", justify="right")
            table.add_row("Provider", m.provider.value)
            table.add_row("Input (per 1M tokens)", f"${m.input_per_million}")
            table.add_row("Output (per 1M tokens)", f"${m.output_per_million}")
            if m.cached_input_per_million is not None:
                table.add_row("Cached input (per 1M)", f"${m.cached_input_per_million}")
            if m.thinking_output_per_million is not None:
                table.add_row("Thinking output (per 1M)", f"${m.thinking_output_per_million}")
            table.add_row("Context window", f"{m.context_window:,}")
            if m.max_output_tokens is not None:
                table.add_row("Max output tokens", f"{m.max_output_tokens:,}")
            if m.aliases:
                table.add_row("Aliases", ", ".join(m.aliases))
            _console.print(table)
        else:
            click.echo(f"Model:           {m.model_id}")
            click.echo(f"Provider:        {m.provider.value}")
            click.echo(f"Input $/M:       ${m.input_per_million}")
            click.echo(f"Output $/M:      ${m.output_per_million}")
            if m.cached_input_per_million is not None:
                click.echo(f"Cached input $/M: ${m.cached_input_per_million}")
            if m.thinking_output_per_million is not None:
                click.echo(f"Thinking out $/M: ${m.thinking_output_per_million}")
            click.echo(f"Context:         {m.context_window:,}")
            if m.max_output_tokens is not None:
                click.echo(f"Max output:      {m.max_output_tokens:,}")

    @cli.command()
    @click.argument("text")
    @click.option("-p", "--providers", default=None, help="Comma-separated provider filter.")
    @click.option("-c", "--min-context", default=0, type=int, help="Min context window.")
    @click.option("-o", "--output-tokens", default=500, type=int, help="Assumed output tokens.")
    def cheapest(text: str, providers: str | None, min_context: int, output_tokens: int) -> None:
        """Find the cheapest model for the given input."""
        text = _read_input(text)
        prov_list = None
        if providers:
            prov_list = []
            for p in providers.split(","):
                try:
                    prov_list.append(Provider(p.strip().lower()))
                except ValueError:
                    click.echo(f"Unknown provider: {p.strip()}", err=True)
                    raise SystemExit(1)
        try:
            model = _cheapest_model(
                text, providers=prov_list,
                min_context_window=min_context, output_tokens=output_tokens,
            )
        except ValueError as e:
            click.echo(str(e), err=True)
            raise SystemExit(1)
        click.echo(f"{model.model_id} ({model.provider.value}) -- "
                    f"${model.input_per_million}/M in, ${model.output_per_million}/M out")

    return cli


cli = _build_cli()


def main() -> None:
    if cli is None:
        print(
            "The CLI requires extra dependencies. Install with:\n"
            "  pip install tokonomics[cli]",
            file=sys.stderr,
        )
        sys.exit(1)
    cli()


if __name__ == "__main__":
    main()
