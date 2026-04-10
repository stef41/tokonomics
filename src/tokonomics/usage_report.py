"""Token usage reporting with daily, weekly, and monthly summaries."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class UsageEntry:
    """A single token usage entry."""

    timestamp: float
    model: str
    tokens: int
    cost: float


class UsageReport:
    """Accumulates usage entries and produces time-based summaries."""

    def __init__(self, entries: list[UsageEntry] | None = None) -> None:
        self._entries: list[UsageEntry] = list(entries) if entries else []

    # -- mutating -----------------------------------------------------------

    def add(
        self,
        model: str,
        tokens: int,
        cost: float,
        timestamp: float | None = None,
    ) -> UsageEntry:
        """Add a usage entry and return it."""
        entry = UsageEntry(
            timestamp=timestamp if timestamp is not None else time.time(),
            model=model,
            tokens=tokens,
            cost=cost,
        )
        self._entries.append(entry)
        return entry

    # -- properties ---------------------------------------------------------

    @property
    def total_cost(self) -> float:
        return sum(e.cost for e in self._entries)

    @property
    def total_tokens(self) -> int:
        return sum(e.tokens for e in self._entries)

    @property
    def entries(self) -> list[UsageEntry]:
        return list(self._entries)

    # -- summaries ----------------------------------------------------------

    def daily_summary(self) -> dict[str, dict[str, Any]]:
        """Aggregate entries by calendar date (YYYY-MM-DD)."""
        return self._summarise_by_key(lambda ts: datetime.fromtimestamp(ts).strftime("%Y-%m-%d"))

    def weekly_summary(self) -> dict[str, dict[str, Any]]:
        """Aggregate entries by ISO week (YYYY-WNN)."""
        return self._summarise_by_key(
            lambda ts: datetime.fromtimestamp(ts).strftime("%Y-W%W")
        )

    def monthly_summary(self) -> dict[str, dict[str, Any]]:
        """Aggregate entries by month (YYYY-MM)."""
        return self._summarise_by_key(lambda ts: datetime.fromtimestamp(ts).strftime("%Y-%m"))

    def by_model(self) -> dict[str, dict[str, Any]]:
        """Aggregate entries by model name."""
        buckets: dict[str, dict[str, Any]] = {}
        for entry in self._entries:
            key = entry.model
            if key not in buckets:
                buckets[key] = {"total_tokens": 0, "total_cost": 0.0, "request_count": 0}
            buckets[key]["total_tokens"] += entry.tokens
            buckets[key]["total_cost"] += entry.cost
            buckets[key]["request_count"] += 1
        return buckets

    # -- helpers ------------------------------------------------------------

    def _summarise_by_key(self, key_fn: Any) -> dict[str, dict[str, Any]]:
        buckets: dict[str, dict[str, Any]] = {}
        for entry in self._entries:
            key = key_fn(entry.timestamp)
            if key not in buckets:
                buckets[key] = {"total_tokens": 0, "total_cost": 0.0, "request_count": 0}
            buckets[key]["total_tokens"] += entry.tokens
            buckets[key]["total_cost"] += entry.cost
            buckets[key]["request_count"] += 1
        return buckets


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_usage_report(report: UsageReport, period: str = "daily") -> str:
    """Return a human-readable text summary of the report.

    *period* can be ``"daily"``, ``"weekly"``, or ``"monthly"``.
    """
    summaries = {
        "daily": report.daily_summary,
        "weekly": report.weekly_summary,
        "monthly": report.monthly_summary,
    }
    if period not in summaries:
        raise ValueError(f"Unknown period {period!r}; expected 'daily', 'weekly', or 'monthly'")

    data = summaries[period]()
    lines: list[str] = [f"Usage Report ({period})", "=" * 40]
    for key in sorted(data):
        bucket = data[key]
        lines.append(
            f"  {key}: {bucket['total_tokens']:,} tokens, "
            f"${bucket['total_cost']:.4f}, "
            f"{bucket['request_count']} requests"
        )
    lines.append("-" * 40)
    lines.append(f"Total: {report.total_tokens:,} tokens, ${report.total_cost:.4f}")
    return "\n".join(lines)


def export_usage_json(report: UsageReport) -> str:
    """Serialize the full report (entries + summaries) to JSON."""
    payload = {
        "entries": [
            {
                "timestamp": e.timestamp,
                "model": e.model,
                "tokens": e.tokens,
                "cost": e.cost,
            }
            for e in report.entries
        ],
        "total_tokens": report.total_tokens,
        "total_cost": report.total_cost,
        "daily_summary": report.daily_summary(),
        "weekly_summary": report.weekly_summary(),
        "monthly_summary": report.monthly_summary(),
        "by_model": report.by_model(),
    }
    return json.dumps(payload, indent=2)
