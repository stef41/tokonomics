"""Budget management with alerts and limits."""

from __future__ import annotations

import threading
import time
from decimal import Decimal
from typing import Callable, List, Optional, Tuple

from tokonomics._types import BudgetExceededError

_PERIOD_SECONDS = {
    "hourly": 3600,
    "daily": 86400,
    "weekly": 604800,
    "monthly": 2592000,  # 30 days
    "total": 0,
}


class Budget:
    """Manage a spending budget with optional time-based resets.

    Parameters
    ----------
    limit:
        Maximum spend in USD for the given period.
    period:
        One of ``"hourly"``, ``"daily"``, ``"weekly"``, ``"monthly"``,
        or ``"total"`` (never resets).

    Examples
    --------
    ::

        budget = Budget(limit=5.00, period="daily")
        budget.record(Decimal("0.05"))
        print(budget.remaining)  # Decimal('4.95')

        budget.on_threshold(0.8, lambda b: print("80% budget used!"))
    """

    def __init__(self, limit: float | Decimal, period: str = "total") -> None:
        if period not in _PERIOD_SECONDS:
            raise ValueError(
                f"Invalid period '{period}'. Choose from: {', '.join(_PERIOD_SECONDS)}"
            )
        self._limit = Decimal(str(limit))
        self._period = period
        self._period_seconds = _PERIOD_SECONDS[period]
        self._used = Decimal("0")
        self._period_start = time.monotonic()
        self._lock = threading.Lock()
        self._thresholds: List[Tuple[Decimal, Callable[["Budget"], None]]] = []
        self._triggered: set[int] = set()

    @property
    def limit(self) -> Decimal:
        """The budget limit."""
        return self._limit

    @property
    def period(self) -> str:
        return self._period

    def _maybe_reset(self) -> None:
        """Reset usage if the current period has elapsed."""
        if self._period_seconds == 0:
            return
        elapsed = time.monotonic() - self._period_start
        if elapsed >= self._period_seconds:
            self._used = Decimal("0")
            self._period_start = time.monotonic()
            self._triggered.clear()

    @property
    def used(self) -> Decimal:
        with self._lock:
            self._maybe_reset()
            return self._used

    @property
    def remaining(self) -> Decimal:
        with self._lock:
            self._maybe_reset()
            return max(Decimal("0"), self._limit - self._used)

    @property
    def utilization(self) -> float:
        """Fraction of budget consumed (0.0 to 1.0)."""
        with self._lock:
            self._maybe_reset()
            if self._limit == 0:
                return 1.0
            return float(self._used / self._limit)

    def check(self, cost: float | Decimal) -> bool:
        """Return True if *cost* can be spent without exceeding the budget."""
        cost_d = Decimal(str(cost))
        with self._lock:
            self._maybe_reset()
            return (self._used + cost_d) <= self._limit

    def record(self, cost: float | Decimal) -> None:
        """Record a cost.  Raises :class:`BudgetExceededError` if the budget is blown."""
        cost_d = Decimal(str(cost))
        callbacks_to_fire: list[Callable[["Budget"], None]] = []

        with self._lock:
            self._maybe_reset()
            new_total = self._used + cost_d
            if new_total > self._limit:
                raise BudgetExceededError(
                    f"Budget exceeded: ${new_total:.6f} > ${self._limit:.6f} "
                    f"({self._period} limit)"
                )
            self._used = new_total

            # Check thresholds
            for idx, (threshold, callback) in enumerate(self._thresholds):
                if idx not in self._triggered and self._limit > 0:
                    if self._used / self._limit >= threshold:
                        self._triggered.add(idx)
                        callbacks_to_fire.append(callback)

        # Fire callbacks outside the lock
        for cb in callbacks_to_fire:
            cb(self)

    def on_threshold(
        self,
        pct: float,
        callback: Callable[["Budget"], None],
    ) -> None:
        """Register *callback* to fire when spending reaches *pct* (0.0–1.0)."""
        with self._lock:
            self._thresholds.append((Decimal(str(pct)), callback))

    def reset(self) -> None:
        """Manually reset the budget."""
        with self._lock:
            self._used = Decimal("0")
            self._period_start = time.monotonic()
            self._triggered.clear()

    def __repr__(self) -> str:
        return (
            f"Budget(limit=${self._limit}, period={self._period!r}, "
            f"used=${self.used}, remaining=${self.remaining})"
        )
