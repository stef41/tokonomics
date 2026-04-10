"""Plain-text and SVG chart export for model cost comparisons."""

from __future__ import annotations

import html
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Unicode block characters for bar rendering (1/8 to full block)
_BLOCKS = " ▏▎▍▌▋▊▉█"

# Pastel colors for SVG bars
_PASTEL_COLORS = [
    "#A8D8EA",  # light blue
    "#AA96DA",  # lavender
    "#FCBAD3",  # pink
    "#FFFFD2",  # pale yellow
    "#B5EAD7",  # mint
    "#FFB7B2",  # salmon
    "#C7CEEA",  # periwinkle
    "#E2F0CB",  # light green
    "#FFDAC1",  # peach
    "#F0E6EF",  # mauve
]


def _extract_metric(row: Dict[str, Any], metric: str) -> float:
    """Pull a numeric value from a comparison row by metric name."""
    key_map = {
        "cost": "total_cost",
        "total_cost": "total_cost",
        "input_cost": "input_cost",
        "output_cost": "output_cost",
        "input_tokens": "input_tokens",
        "output_tokens": "output_tokens",
        "tokens": "input_tokens",
    }
    key = key_map.get(metric, metric)
    val = row.get(key, 0)
    if isinstance(val, Decimal):
        return float(val)
    return float(val)


def _label_for(row: Dict[str, Any]) -> str:
    model = row.get("model", "unknown")
    provider = row.get("provider", "")
    if provider:
        return f"{model} ({provider})"
    return model


def _format_value(value: float, metric: str) -> str:
    if "cost" in metric or metric == "cost":
        return f"${value:.6f}"
    return f"{value:,.0f}"


# ---------------------------------------------------------------------------
# Plain-text horizontal bar chart
# ---------------------------------------------------------------------------


def format_bar_chart(
    comparisons: List[Dict[str, Any]],
    metric: str = "cost",
    width: int = 40,
) -> str:
    """Return a plain-text horizontal bar chart using unicode block characters.

    Parameters
    ----------
    comparisons:
        Output from :func:`compare_models`.
    metric:
        Which metric to chart: ``"cost"``, ``"input_cost"``,
        ``"output_cost"``, ``"input_tokens"``, or ``"output_tokens"``.
    width:
        Maximum bar width in characters.

    Returns
    -------
    str
        Multi-line ASCII art bar chart.
    """
    if not comparisons:
        return "No data to chart."

    values = [_extract_metric(r, metric) for r in comparisons]
    labels = [_label_for(r) for r in comparisons]
    max_val = max(values) if values else 0

    label_width = max(len(l) for l in labels)

    lines: list[str] = []
    for label, val in zip(labels, values):
        if max_val > 0:
            ratio = val / max_val
        else:
            ratio = 0.0

        full_blocks = int(ratio * width)
        remainder = (ratio * width) - full_blocks
        eighth = int(remainder * 8)

        bar = "█" * full_blocks
        if eighth > 0 and full_blocks < width:
            bar += _BLOCKS[eighth]

        formatted = _format_value(val, metric)
        lines.append(f"{label:<{label_width}}  {bar} {formatted}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SVG bar chart
# ---------------------------------------------------------------------------


def export_svg_chart(
    comparisons: List[Dict[str, Any]],
    metric: str = "cost",
    path: Optional[Union[str, Path]] = None,
    bar_height: int = 28,
    max_bar_width: int = 400,
    title: Optional[str] = None,
) -> str:
    """Generate an SVG horizontal bar chart and optionally save it to a file.

    Parameters
    ----------
    comparisons:
        Output from :func:`compare_models`.
    metric:
        Which metric to chart (see :func:`format_bar_chart`).
    path:
        If provided, write the SVG to this file path.
    bar_height:
        Height of each bar in pixels.
    max_bar_width:
        Maximum width of the longest bar in pixels.
    title:
        Optional chart title.

    Returns
    -------
    str
        The complete SVG markup.
    """
    if not comparisons:
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" width="300" height="40">'
            '<text x="10" y="25" font-family="sans-serif" font-size="14" '
            'fill="#666">No data to chart.</text></svg>'
        )
        if path is not None:
            Path(path).write_text(svg, encoding="utf-8")
        return svg

    values = [_extract_metric(r, metric) for r in comparisons]
    labels = [_label_for(r) for r in comparisons]
    max_val = max(values) if values else 0

    gap = 6
    label_area = 220
    value_area = 100
    padding_top = 10
    title_height = 30 if title else 0
    row_height = bar_height + gap
    chart_height = padding_top + title_height + row_height * len(comparisons) + 10
    chart_width = label_area + max_bar_width + value_area + 20

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{chart_width}" height="{chart_height}" '
        f'viewBox="0 0 {chart_width} {chart_height}">'
    )
    parts.append(
        '<rect width="100%" height="100%" fill="#FAFAFA" rx="6" />'
    )

    if title:
        parts.append(
            f'<text x="{chart_width // 2}" y="{padding_top + 18}" '
            f'font-family="sans-serif" font-size="16" font-weight="bold" '
            f'fill="#333" text-anchor="middle">{html.escape(title)}</text>'
        )

    y_start = padding_top + title_height

    for i, (label, val) in enumerate(zip(labels, values)):
        y = y_start + i * row_height
        color = _PASTEL_COLORS[i % len(_PASTEL_COLORS)]
        bar_w = int((val / max_val) * max_bar_width) if max_val > 0 else 0
        # Ensure at least 2px for non-zero values
        if val > 0 and bar_w < 2:
            bar_w = 2

        text_y = y + bar_height * 0.7

        # Label
        parts.append(
            f'<text x="{label_area - 8}" y="{text_y}" '
            f'font-family="sans-serif" font-size="12" fill="#333" '
            f'text-anchor="end">{html.escape(label)}</text>'
        )

        # Bar
        parts.append(
            f'<rect x="{label_area}" y="{y}" width="{bar_w}" '
            f'height="{bar_height}" fill="{color}" rx="3" />'
        )

        # Value label
        formatted = _format_value(val, metric)
        parts.append(
            f'<text x="{label_area + bar_w + 6}" y="{text_y}" '
            f'font-family="sans-serif" font-size="11" fill="#555">'
            f'{html.escape(formatted)}</text>'
        )

    parts.append("</svg>")
    svg = "\n".join(parts)

    if path is not None:
        Path(path).write_text(svg, encoding="utf-8")

    return svg


# ---------------------------------------------------------------------------
# ASCII table
# ---------------------------------------------------------------------------


def format_table(comparisons: List[Dict[str, Any]]) -> str:
    """Return a formatted ASCII table of cost comparison results.

    Columns: Model, Provider, Input Cost, Output Cost, Total Cost.

    Parameters
    ----------
    comparisons:
        Output from :func:`compare_models`.

    Returns
    -------
    str
        A multi-line table string.
    """
    if not comparisons:
        return "No data to display."

    headers = ("Model", "Provider", "Input Cost", "Output Cost", "Total Cost")

    rows: list[tuple[str, str, str, str, str]] = []
    for r in comparisons:
        rows.append((
            str(r.get("model", "")),
            str(r.get("provider", "")),
            f"${float(r.get('input_cost', 0)):.6f}",
            f"${float(r.get('output_cost', 0)):.6f}",
            f"${float(r.get('total_cost', 0)):.6f}",
        ))

    # Compute column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for j, cell in enumerate(row):
            col_widths[j] = max(col_widths[j], len(cell))

    def fmt_row(cells: tuple[str, ...]) -> str:
        parts = []
        for cell, w in zip(cells, col_widths):
            parts.append(f"{cell:<{w}}")
        return "  ".join(parts)

    lines = [fmt_row(headers)]
    lines.append("  ".join("-" * w for w in col_widths))
    for row in rows:
        lines.append(fmt_row(row))

    return "\n".join(lines)
