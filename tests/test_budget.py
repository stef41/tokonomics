"""Tests for budget management."""

from decimal import Decimal

import pytest

from tokonomics._types import BudgetExceededError
from tokonomics.budget import Budget


class TestBudget:
    def test_initial_state(self):
        budget = Budget(limit=10.00)
        assert budget.used == Decimal("0")
        assert budget.remaining == Decimal("10.00")
        assert budget.utilization == 0.0

    def test_record_spending(self):
        budget = Budget(limit=10.00)
        budget.record(Decimal("3.50"))
        assert budget.used == Decimal("3.50")
        assert budget.remaining == Decimal("6.50")

    def test_record_multiple(self):
        budget = Budget(limit=10.00)
        budget.record(Decimal("2.00"))
        budget.record(Decimal("3.00"))
        assert budget.used == Decimal("5.00")

    def test_exceeds_budget(self):
        budget = Budget(limit=1.00)
        budget.record(Decimal("0.50"))
        with pytest.raises(BudgetExceededError, match="Budget exceeded"):
            budget.record(Decimal("0.60"))

    def test_exact_limit(self):
        budget = Budget(limit=1.00)
        budget.record(Decimal("1.00"))  # Should not raise
        assert budget.used == Decimal("1.00")
        assert budget.remaining == Decimal("0")

    def test_check_within_budget(self):
        budget = Budget(limit=10.00)
        assert budget.check(Decimal("5.00")) is True

    def test_check_exceeds(self):
        budget = Budget(limit=1.00)
        budget.record(Decimal("0.80"))
        assert budget.check(Decimal("0.30")) is False

    def test_utilization(self):
        budget = Budget(limit=10.00)
        budget.record(Decimal("5.00"))
        assert budget.utilization == pytest.approx(0.5)

    def test_utilization_zero_limit(self):
        budget = Budget(limit=0)
        assert budget.utilization == 1.0

    def test_from_float(self):
        budget = Budget(limit=5.50)
        assert budget.limit == Decimal("5.5")

    def test_reset(self):
        budget = Budget(limit=10.00)
        budget.record(Decimal("5.00"))
        budget.reset()
        assert budget.used == Decimal("0")
        assert budget.remaining == Decimal("10.00")

    def test_repr(self):
        budget = Budget(limit=10.00)
        assert "10" in repr(budget)

    def test_invalid_period(self):
        with pytest.raises(ValueError, match="Invalid period"):
            Budget(limit=10.00, period="yearly")


class TestBudgetPeriods:
    def test_total_period(self):
        budget = Budget(limit=10.00, period="total")
        assert budget.period == "total"

    def test_daily_period(self):
        budget = Budget(limit=10.00, period="daily")
        assert budget.period == "daily"

    @pytest.mark.parametrize(
        "period", ["hourly", "daily", "weekly", "monthly", "total"]
    )
    def test_valid_periods(self, period):
        budget = Budget(limit=10.00, period=period)
        assert budget.period == period


class TestBudgetThresholds:
    def test_threshold_triggers(self):
        budget = Budget(limit=10.00)
        triggered = []
        budget.on_threshold(0.5, lambda b: triggered.append("50%"))
        budget.record(Decimal("3.00"))
        assert triggered == []
        budget.record(Decimal("3.00"))  # Now at 60%
        assert triggered == ["50%"]

    def test_threshold_triggers_once(self):
        budget = Budget(limit=10.00)
        count = []
        budget.on_threshold(0.5, lambda b: count.append(1))
        budget.record(Decimal("6.00"))
        budget.record(Decimal("2.00"))
        assert len(count) == 1  # Should fire only once

    def test_multiple_thresholds(self):
        budget = Budget(limit=10.00)
        triggered = []
        budget.on_threshold(0.5, lambda b: triggered.append("50%"))
        budget.on_threshold(0.8, lambda b: triggered.append("80%"))
        budget.record(Decimal("9.00"))
        assert "50%" in triggered
        assert "80%" in triggered

    def test_threshold_at_boundary(self):
        budget = Budget(limit=10.00)
        triggered = []
        budget.on_threshold(0.5, lambda b: triggered.append(True))
        budget.record(Decimal("5.00"))  # Exactly 50%
        assert len(triggered) == 1
