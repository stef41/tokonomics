"""Tests for tokonomics.usage_report."""

from __future__ import annotations

import json
import time
from datetime import datetime

import pytest

from tokonomics.usage_report import (
    UsageEntry,
    UsageReport,
    export_usage_json,
    format_usage_report,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts(date_str: str) -> float:
    """Convert 'YYYY-MM-DD' to a UNIX timestamp (midnight local)."""
    return datetime.strptime(date_str, "%Y-%m-%d").timestamp()


def _make_report() -> UsageReport:
    """Build a report spanning several days/weeks for reuse."""
    r = UsageReport()
    r.add("gpt-4o", tokens=500, cost=0.01, timestamp=_ts("2025-06-02"))
    r.add("gpt-4o", tokens=300, cost=0.006, timestamp=_ts("2025-06-02"))
    r.add("claude-3.5-sonnet", tokens=1000, cost=0.03, timestamp=_ts("2025-06-03"))
    r.add("gpt-4o", tokens=200, cost=0.004, timestamp=_ts("2025-06-09"))
    return r


# ---------------------------------------------------------------------------
# UsageEntry
# ---------------------------------------------------------------------------

class TestUsageEntry:
    def test_fields(self):
        e = UsageEntry(timestamp=1.0, model="gpt-4o", tokens=100, cost=0.5)
        assert e.model == "gpt-4o"
        assert e.tokens == 100
        assert e.cost == 0.5
        assert e.timestamp == 1.0

    def test_equality(self):
        a = UsageEntry(timestamp=1.0, model="m", tokens=10, cost=0.1)
        b = UsageEntry(timestamp=1.0, model="m", tokens=10, cost=0.1)
        assert a == b


# ---------------------------------------------------------------------------
# UsageReport — construction
# ---------------------------------------------------------------------------

class TestUsageReportConstruction:
    def test_empty_report(self):
        r = UsageReport()
        assert r.total_cost == 0.0
        assert r.total_tokens == 0
        assert r.entries == []

    def test_init_with_entries(self):
        entries = [UsageEntry(1.0, "m", 10, 0.1), UsageEntry(2.0, "m", 20, 0.2)]
        r = UsageReport(entries=entries)
        assert len(r.entries) == 2
        assert r.total_tokens == 30

    def test_add_returns_entry(self):
        r = UsageReport()
        e = r.add("gpt-4o", tokens=100, cost=0.5)
        assert isinstance(e, UsageEntry)
        assert e.model == "gpt-4o"

    def test_add_default_timestamp(self):
        r = UsageReport()
        before = time.time()
        e = r.add("m", tokens=1, cost=0.0)
        after = time.time()
        assert before <= e.timestamp <= after


# ---------------------------------------------------------------------------
# Totals
# ---------------------------------------------------------------------------

class TestTotals:
    def test_total_cost(self):
        r = _make_report()
        assert r.total_cost == pytest.approx(0.05)

    def test_total_tokens(self):
        r = _make_report()
        assert r.total_tokens == 2000


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

class TestDailySummary:
    def test_keys(self):
        r = _make_report()
        daily = r.daily_summary()
        assert "2025-06-02" in daily
        assert "2025-06-03" in daily
        assert "2025-06-09" in daily

    def test_aggregation(self):
        r = _make_report()
        daily = r.daily_summary()
        bucket = daily["2025-06-02"]
        assert bucket["total_tokens"] == 800
        assert bucket["total_cost"] == pytest.approx(0.016)
        assert bucket["request_count"] == 2


class TestWeeklySummary:
    def test_returns_dict(self):
        r = _make_report()
        weekly = r.weekly_summary()
        assert isinstance(weekly, dict)
        assert len(weekly) >= 1

    def test_token_totals_match(self):
        r = _make_report()
        weekly = r.weekly_summary()
        total = sum(v["total_tokens"] for v in weekly.values())
        assert total == r.total_tokens


class TestMonthlySummary:
    def test_single_month(self):
        r = _make_report()
        monthly = r.monthly_summary()
        assert "2025-06" in monthly
        assert monthly["2025-06"]["total_tokens"] == 2000


class TestByModel:
    def test_model_keys(self):
        r = _make_report()
        by_model = r.by_model()
        assert "gpt-4o" in by_model
        assert "claude-3.5-sonnet" in by_model

    def test_model_aggregation(self):
        r = _make_report()
        by_model = r.by_model()
        assert by_model["gpt-4o"]["total_tokens"] == 1000
        assert by_model["gpt-4o"]["request_count"] == 3
        assert by_model["claude-3.5-sonnet"]["total_tokens"] == 1000


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

class TestFormatUsageReport:
    def test_daily_format(self):
        r = _make_report()
        text = format_usage_report(r, period="daily")
        assert "Usage Report (daily)" in text
        assert "2025-06-02" in text
        assert "Total:" in text

    def test_weekly_format(self):
        r = _make_report()
        text = format_usage_report(r, period="weekly")
        assert "Usage Report (weekly)" in text

    def test_monthly_format(self):
        r = _make_report()
        text = format_usage_report(r, period="monthly")
        assert "2025-06" in text

    def test_invalid_period_raises(self):
        r = UsageReport()
        with pytest.raises(ValueError, match="Unknown period"):
            format_usage_report(r, period="yearly")


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------

class TestExportUsageJson:
    def test_valid_json(self):
        r = _make_report()
        raw = export_usage_json(r)
        data = json.loads(raw)
        assert isinstance(data, dict)

    def test_contains_entries(self):
        r = _make_report()
        data = json.loads(export_usage_json(r))
        assert len(data["entries"]) == 4

    def test_contains_summaries(self):
        r = _make_report()
        data = json.loads(export_usage_json(r))
        assert "daily_summary" in data
        assert "weekly_summary" in data
        assert "monthly_summary" in data
        assert "by_model" in data

    def test_totals_in_json(self):
        r = _make_report()
        data = json.loads(export_usage_json(r))
        assert data["total_tokens"] == 2000
        assert data["total_cost"] == pytest.approx(0.05)
