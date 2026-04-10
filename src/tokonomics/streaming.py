"""Streaming cost tracking for async and sync LLM response streams."""

from __future__ import annotations

import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, AsyncIterator, Dict, Iterator, Optional, TypeVar

from tokonomics.models import get_model

T = TypeVar("T")


@dataclass(frozen=True)
class StreamingUsage:
    """Final usage summary after a streaming response completes."""

    model: str
    total_tokens: int
    estimated_cost: Decimal
    duration_ms: float


class StreamingCostTracker:
    """Tracks cost incrementally as tokens stream from an LLM.

    Usage::

        tracker = StreamingCostTracker(model="gpt-4o")
        for token in stream:
            tracker.on_token(token)
        usage = tracker.finalize()
        print(usage.estimated_cost)
    """

    def __init__(self, model: str = "gpt-4o") -> None:
        self._model = model
        self._pricing = get_model(model)
        self._token_count = 0
        self._start_ns = time.monotonic_ns()

    def on_token(self, token_text: str) -> None:
        """Called for each streamed token. Updates running cost."""
        if token_text:
            self._token_count += 1

    def on_chunk(self, chunk_dict: Dict[str, Any]) -> None:
        """Called with a raw OpenAI-style streaming chunk dict.

        Extracts delta content and counts it as one token if present.
        """
        choices = chunk_dict.get("choices") or []
        for choice in choices:
            delta = choice.get("delta") or {}
            content = delta.get("content")
            if content:
                self._token_count += 1

    @property
    def current_cost(self) -> Decimal:
        """Running total estimated cost so far (output tokens only)."""
        return Decimal(str(self._token_count)) * self._pricing.output_per_token

    @property
    def token_count(self) -> int:
        """Number of tokens seen so far."""
        return self._token_count

    def finalize(self) -> StreamingUsage:
        """Return final usage summary."""
        elapsed_ns = time.monotonic_ns() - self._start_ns
        duration_ms = elapsed_ns / 1_000_000
        return StreamingUsage(
            model=self._pricing.model_id,
            total_tokens=self._token_count,
            estimated_cost=self.current_cost,
            duration_ms=round(duration_ms, 2),
        )


class _TrackedAsyncStream:
    """Async iterator wrapper that exposes a ``.tracker`` attribute."""

    def __init__(
        self, async_iterator: AsyncIterator[Any], tracker: StreamingCostTracker
    ) -> None:
        self._it = async_iterator
        self.tracker = tracker

    def __aiter__(self) -> "_TrackedAsyncStream":
        return self

    async def __anext__(self) -> Any:
        chunk = await self._it.__anext__()
        if isinstance(chunk, dict):
            self.tracker.on_chunk(chunk)
        elif isinstance(chunk, str):
            self.tracker.on_token(chunk)
        else:
            try:
                self.tracker.on_chunk(chunk.__dict__)
            except (AttributeError, TypeError):
                self.tracker.on_token(str(chunk))
        return chunk


class _TrackedSyncStream:
    """Iterator wrapper that exposes a ``.tracker`` attribute."""

    def __init__(
        self, iterator: Iterator[Any], tracker: StreamingCostTracker
    ) -> None:
        self._it = iterator
        self.tracker = tracker

    def __iter__(self) -> "_TrackedSyncStream":
        return self

    def __next__(self) -> Any:
        chunk = next(self._it)
        if isinstance(chunk, dict):
            self.tracker.on_chunk(chunk)
        elif isinstance(chunk, str):
            self.tracker.on_token(chunk)
        else:
            try:
                self.tracker.on_chunk(chunk.__dict__)
            except (AttributeError, TypeError):
                self.tracker.on_token(str(chunk))
        return chunk


def async_track_stream(
    async_iterator: AsyncIterator[T],
    model: str = "gpt-4o",
    tracker: Optional[StreamingCostTracker] = None,
) -> _TrackedAsyncStream:
    """Wrap an async iterator to track streaming cost.

    Usage::

        async for chunk in async_track_stream(response_stream, "gpt-4o"):
            print(chunk)

    Access the tracker via the ``tracker`` attribute::

        stream = async_track_stream(response_stream, "gpt-4o")
        async for chunk in stream:
            ...
        usage = stream.tracker.finalize()
    """
    if tracker is None:
        tracker = StreamingCostTracker(model=model)
    return _TrackedAsyncStream(async_iterator, tracker)


def track_stream(
    iterator: Iterator[T],
    model: str = "gpt-4o",
    tracker: Optional[StreamingCostTracker] = None,
) -> _TrackedSyncStream:
    """Wrap an iterator to track streaming cost.

    Usage::

        for chunk in track_stream(response_stream, "gpt-4o"):
            print(chunk)

    Access the tracker via the ``tracker`` attribute::

        stream = track_stream(response_stream, "gpt-4o")
        for chunk in stream:
            ...
        usage = stream.tracker.finalize()
    """
    if tracker is None:
        tracker = StreamingCostTracker(model=model)
    return _TrackedSyncStream(iterator, tracker)
