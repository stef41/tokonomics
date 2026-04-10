"""Cost tracking via decorators and context managers."""

from __future__ import annotations

import asyncio
import functools
import threading
import time
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Callable, TypeVar

from tokonomics.cost import calculate_cost

if TYPE_CHECKING:
    from tokonomics._types import UsageRecord

F = TypeVar("F", bound=Callable[..., Any])


class CostTracker:
    """Thread-safe cost tracker that accumulates usage records.

    Can be used as a context manager::

        with CostTracker() as tracker:
            tracker.record("gpt-4o", input_tokens=100, output_tokens=50)
            tracker.record("claude-3.5-sonnet", input_tokens=200, output_tokens=100)

        print(tracker.total_cost)
    """

    def __init__(self) -> None:
        self._records: list[UsageRecord] = []
        self._lock = threading.Lock()

    def __enter__(self) -> CostTracker:
        return self

    def __exit__(self, *args: Any) -> None:
        pass

    def record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
        thinking_tokens: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> UsageRecord:
        """Record a single API call's usage."""
        usage = calculate_cost(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            thinking_tokens=thinking_tokens,
        )
        usage.timestamp = time.time()
        if metadata:
            usage.metadata = metadata
        with self._lock:
            self._records.append(usage)
        return usage

    @property
    def records(self) -> list[UsageRecord]:
        """All recorded usage entries."""
        with self._lock:
            return list(self._records)

    @property
    def total_cost(self) -> Decimal:
        """Sum of all recorded costs."""
        with self._lock:
            return sum((r.total_cost for r in self._records), Decimal("0"))

    @property
    def total_input_tokens(self) -> int:
        with self._lock:
            return sum(r.input_tokens for r in self._records)

    @property
    def total_output_tokens(self) -> int:
        with self._lock:
            return sum(r.output_tokens for r in self._records)

    def by_model(self) -> dict[str, Decimal]:
        """Cost breakdown grouped by model."""
        result: dict[str, Decimal] = {}
        with self._lock:
            for r in self._records:
                result[r.model] = result.get(r.model, Decimal("0")) + r.total_cost
        return dict(sorted(result.items(), key=lambda kv: kv[1], reverse=True))

    def by_provider(self) -> dict[str, Decimal]:
        """Cost breakdown grouped by provider."""
        result: dict[str, Decimal] = {}
        with self._lock:
            for r in self._records:
                key = r.provider.value
                result[key] = result.get(key, Decimal("0")) + r.total_cost
        return dict(sorted(result.items(), key=lambda kv: kv[1], reverse=True))

    def summary(self) -> str:
        """Human-readable summary of all tracked costs."""
        lines = [
            f"Total cost: ${self.total_cost:.6f}",
            f"Total calls: {len(self._records)}",
            f"Input tokens: {self.total_input_tokens:,}",
            f"Output tokens: {self.total_output_tokens:,}",
        ]
        by_model = self.by_model()
        if by_model:
            lines.append("Cost by model:")
            for model, cost in by_model.items():
                lines.append(f"  {model}: ${cost:.6f}")
        return "\n".join(lines)

    def reset(self) -> None:
        """Clear all recorded usage."""
        with self._lock:
            self._records.clear()


# Global tracker for convenience
_global_tracker = CostTracker()


def get_global_tracker() -> CostTracker:
    """Return the module-level global cost tracker."""
    return _global_tracker


def track_cost(
    model: str,
    tracker: CostTracker | None = None,
    input_tokens_key: str = "input_tokens",
    output_tokens_key: str = "output_tokens",
) -> Callable[[F], F]:
    """Decorator that records cost after each function call.

    The decorated function must return a dict-like object (or object with
    attributes) containing token usage. Alternatively you can manually
    record usage through the ``tracker`` attribute on the wrapped function.

    Usage::

        @track_cost(model="gpt-4o")
        def ask(prompt: str) -> dict:
            resp = openai.chat.completions.create(...)
            return {
                "text": resp.choices[0].message.content,
                "input_tokens": resp.usage.prompt_tokens,
                "output_tokens": resp.usage.completion_tokens,
            }

        result = ask("Hi")
        print(ask.last_usage.total_cost)
        print(ask.total_cost)
    """
    target_tracker = tracker or _global_tracker

    def decorator(fn: F) -> F:
        _usage_history: list[UsageRecord] = []
        _last_usage: list[UsageRecord | None] = [None]

        def _extract_tokens(result: Any) -> tuple[int, int]:
            if isinstance(result, dict):
                inp = result.get(input_tokens_key, 0)
                out = result.get(output_tokens_key, 0)
            else:
                inp = getattr(result, input_tokens_key, 0)
                out = getattr(result, output_tokens_key, 0)
            return int(inp), int(out)

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            result = fn(*args, **kwargs)
            inp, out = _extract_tokens(result)
            if inp > 0 or out > 0:
                usage = target_tracker.record(model=model, input_tokens=inp, output_tokens=out)
                _last_usage[0] = usage
                _usage_history.append(usage)
            return result

        @functools.wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            result = await fn(*args, **kwargs)
            inp, out = _extract_tokens(result)
            if inp > 0 or out > 0:
                usage = target_tracker.record(model=model, input_tokens=inp, output_tokens=out)
                _last_usage[0] = usage
                _usage_history.append(usage)
            return result

        wrapper: Any = async_wrapper if asyncio.iscoroutinefunction(fn) else sync_wrapper

        # Attach tracking attributes to the wrapper
        wrapper.tracker = target_tracker
        wrapper.usage_history = _usage_history
        wrapper.last_usage = property(lambda self: _last_usage[0])

        # Make last_usage / total_cost accessible directly
        class _Accessor:
            pass

        _acc = _Accessor()
        wrapper._last_usage = _last_usage
        wrapper._usage_history = _usage_history

        wrapper.__getattribute__ if hasattr(wrapper, "__getattribute__") else None

        # Monkey-patch attribute access on the function
        wrapper.last_usage = None  # placeholder

        def _get_last_usage() -> UsageRecord | None:
            return _last_usage[0]

        def _get_total_cost() -> Decimal:
            return sum((u.total_cost for u in _usage_history), Decimal("0"))

        wrapper.get_last_usage = _get_last_usage
        wrapper.get_total_cost = _get_total_cost
        wrapper.usage_history = _usage_history

        return wrapper  # type: ignore[no-any-return]

    return decorator
