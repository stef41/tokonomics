"""Tests for tokonomics.charts — bar chart, SVG export, and table formatting."""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from tokonomics.charts import export_svg_chart, format_bar_chart, format_table

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_comparisons(n: int = 3) -> list[dict]:
    """Build synthetic comparison rows."""
    data = [
        {
            "model": "gpt-4o",
            "provider": "openai",
            "input_tokens": 1000,
            "output_tokens": 500,
            "input_cost": Decimal("0.005000"),
            "output_cost": Decimal("0.015000"),
            "total_cost": Decimal("0.020000"),
            "context_window": 128_000,
        },
        {
            "model": "claude-3-haiku",
            "provider": "anthropic",
            "input_tokens": 1000,
            "output_tokens": 500,
            "input_cost": Decimal("0.000250"),
            "output_cost": Decimal("0.000625"),
            "total_cost": Decimal("0.000875"),
            "context_window": 200_000,
        },
        {
            "model": "gemini-1.5-flash",
            "provider": "google",
            "input_tokens": 1000,
            "output_tokens": 500,
            "input_cost": Decimal("0.000075"),
            "output_cost": Decimal("0.000300"),
            "total_cost": Decimal("0.000375"),
            "context_window": 1_000_000,
        },
    ]
    return data[:n]


# ---------------------------------------------------------------------------
# format_bar_chart
# ---------------------------------------------------------------------------


class TestFormatBarChart:
    def test_basic_output(self):
        result = format_bar_chart(_make_comparisons())
        assert "gpt-4o" in result
        assert "claude-3-haiku" in result
        assert "█" in result

    def test_empty_input(self):
        assert format_bar_chart([]) == "No data to chart."

    def test_single_item(self):
        result = format_bar_chart(_make_comparisons(1))
        lines = result.strip().splitlines()
        assert len(lines) == 1
        assert "gpt-4o" in lines[0]

    def test_metric_tokens(self):
        result = format_bar_chart(_make_comparisons(), metric="input_tokens")
        # Token values should NOT have a $ prefix
        assert "$" not in result
        assert "1,000" in result

    def test_custom_width(self):
        narrow = format_bar_chart(_make_comparisons(), width=10)
        wide = format_bar_chart(_make_comparisons(), width=60)
        # The widest bar in narrow should be shorter
        narrow_max_bar = max(
            line.count("█") for line in narrow.splitlines()
        )
        wide_max_bar = max(
            line.count("█") for line in wide.splitlines()
        )
        assert wide_max_bar >= narrow_max_bar

    def test_all_zero_values(self):
        rows = _make_comparisons(2)
        for r in rows:
            r["total_cost"] = Decimal("0")
        result = format_bar_chart(rows)
        # Should not crash; bars are empty
        assert "gpt-4o" in result
        assert "$0.000000" in result


# ---------------------------------------------------------------------------
# export_svg_chart
# ---------------------------------------------------------------------------


class TestExportSvgChart:
    def test_valid_svg(self):
        svg = export_svg_chart(_make_comparisons())
        assert svg.startswith("<svg")
        assert svg.strip().endswith("</svg>")

    def test_contains_bars(self):
        svg = export_svg_chart(_make_comparisons())
        assert "<rect" in svg
        # Three data rows => at least 3 bar rects (plus the background rect)
        assert svg.count("<rect") >= 4

    def test_empty_input_svg(self):
        svg = export_svg_chart([])
        assert "<svg" in svg
        assert "No data" in svg

    def test_write_to_file(self, tmp_path: Path):
        out = tmp_path / "chart.svg"
        svg = export_svg_chart(_make_comparisons(), path=out)
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert content == svg
        assert content.startswith("<svg")

    def test_metric_tokens_svg(self):
        svg = export_svg_chart(_make_comparisons(), metric="input_tokens")
        assert "1,000" in svg

    def test_title(self):
        svg = export_svg_chart(_make_comparisons(), title="Cost Comparison")
        assert "Cost Comparison" in svg

    def test_html_escape_in_labels(self):
        rows = _make_comparisons(1)
        rows[0]["model"] = "model<br>evil"
        svg = export_svg_chart(rows)
        # Should be escaped, not raw HTML
        assert "&lt;" in svg
        assert "<br>" not in svg


# ---------------------------------------------------------------------------
# format_table
# ---------------------------------------------------------------------------


class TestFormatTable:
    def test_basic_table(self):
        result = format_table(_make_comparisons())
        lines = result.splitlines()
        # header + separator + 3 data rows
        assert len(lines) == 5
        assert "Model" in lines[0]
        assert "Provider" in lines[0]
        assert "Total Cost" in lines[0]

    def test_empty_input(self):
        assert format_table([]) == "No data to display."

    def test_column_alignment(self):
        result = format_table(_make_comparisons())
        lines = result.splitlines()
        # Separator line should be dashes and spaces only
        assert all(ch in "-  " for ch in lines[1])

    def test_values_present(self):
        result = format_table(_make_comparisons())
        assert "$0.020000" in result
        assert "$0.000875" in result
        assert "openai" in result
        assert "anthropic" in result

    def test_single_row(self):
        result = format_table(_make_comparisons(1))
        lines = result.splitlines()
        assert len(lines) == 3  # header + sep + 1 row
