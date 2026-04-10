"""Tests for streaming cost tracking."""

import asyncio
from decimal import Decimal

import pytest

from tokonomics.streaming import (
    StreamingCostTracker,
    StreamingUsage,
    async_track_stream,
    track_stream,
)


# ---------------------------------------------------------------------------
# StreamingCostTracker
# ---------------------------------------------------------------------------


class TestStreamingCostTracker:
    def test_init_defaults(self):
        tracker = StreamingCostTracker()
        assert tracker.token_count == 0
        assert tracker.current_cost == Decimal("0")

    def test_init_custom_model(self):
        tracker = StreamingCostTracker(model="deepseek-chat")
        assert tracker.token_count == 0

    def test_on_token_increments(self):
        tracker = StreamingCostTracker()
        tracker.on_token("Hello")
        tracker.on_token(" world")
        assert tracker.token_count == 2

    def test_on_token_empty_string_ignored(self):
        tracker = StreamingCostTracker()
        tracker.on_token("")
        assert tracker.token_count == 0

    def test_on_chunk_openai_format(self):
        tracker = StreamingCostTracker()
        chunk = {
            "choices": [{"delta": {"content": "Hello"}, "index": 0}],
        }
        tracker.on_chunk(chunk)
        assert tracker.token_count == 1

    def test_on_chunk_multiple_choices(self):
        tracker = StreamingCostTracker()
        chunk = {
            "choices": [
                {"delta": {"content": "A"}, "index": 0},
                {"delta": {"content": "B"}, "index": 1},
            ],
        }
        tracker.on_chunk(chunk)
        assert tracker.token_count == 2

    def test_on_chunk_no_content(self):
        tracker = StreamingCostTracker()
        # Role-only delta (first chunk in OpenAI streams)
        chunk = {"choices": [{"delta": {"role": "assistant"}, "index": 0}]}
        tracker.on_chunk(chunk)
        assert tracker.token_count == 0

    def test_on_chunk_empty_choices(self):
        tracker = StreamingCostTracker()
        tracker.on_chunk({"choices": []})
        assert tracker.token_count == 0

    def test_on_chunk_missing_choices(self):
        tracker = StreamingCostTracker()
        tracker.on_chunk({})
        assert tracker.token_count == 0

    def test_current_cost_positive_after_tokens(self):
        tracker = StreamingCostTracker(model="gpt-4o")
        for _ in range(100):
            tracker.on_token("tok")
        assert tracker.current_cost > Decimal("0")
        assert tracker.token_count == 100

    def test_current_cost_scales_with_tokens(self):
        tracker = StreamingCostTracker(model="gpt-4o")
        for _ in range(10):
            tracker.on_token("x")
        cost_10 = tracker.current_cost
        for _ in range(10):
            tracker.on_token("x")
        cost_20 = tracker.current_cost
        assert cost_20 == cost_10 * 2

    def test_finalize_returns_streaming_usage(self):
        tracker = StreamingCostTracker(model="gpt-4o")
        for _ in range(5):
            tracker.on_token("hi")
        usage = tracker.finalize()
        assert isinstance(usage, StreamingUsage)
        assert usage.model == "gpt-4o"
        assert usage.total_tokens == 5
        assert usage.estimated_cost > Decimal("0")
        assert usage.duration_ms >= 0

    def test_finalize_zero_tokens(self):
        tracker = StreamingCostTracker()
        usage = tracker.finalize()
        assert usage.total_tokens == 0
        assert usage.estimated_cost == Decimal("0")

    def test_invalid_model_raises(self):
        with pytest.raises(Exception):
            StreamingCostTracker(model="nonexistent-model-xyz")


# ---------------------------------------------------------------------------
# track_stream (sync)
# ---------------------------------------------------------------------------


class TestTrackStreamSync:
    def test_yields_all_string_chunks(self):
        tokens = ["Hello", " ", "world", "!"]
        collected = list(track_stream(iter(tokens), model="gpt-4o"))
        assert collected == tokens

    def test_tracker_attribute(self):
        tokens = ["a", "b", "c"]
        stream = track_stream(iter(tokens), model="gpt-4o")
        list(stream)  # consume
        usage = stream.tracker.finalize()
        assert usage.total_tokens == 3

    def test_dict_chunks(self):
        chunks = [
            {"choices": [{"delta": {"content": "Hello"}, "index": 0}]},
            {"choices": [{"delta": {"content": " world"}, "index": 0}]},
        ]
        stream = track_stream(iter(chunks), model="gpt-4o")
        collected = list(stream)
        assert collected == chunks
        assert stream.tracker.token_count == 2

    def test_empty_stream(self):
        stream = track_stream(iter([]), model="gpt-4o")
        collected = list(stream)
        assert collected == []
        assert stream.tracker.token_count == 0

    def test_custom_tracker(self):
        tracker = StreamingCostTracker(model="gpt-4o")
        tokens = ["x", "y"]
        stream = track_stream(iter(tokens), model="gpt-4o", tracker=tracker)
        list(stream)
        assert tracker.token_count == 2

    def test_mixed_empty_and_content_chunks(self):
        chunks = [
            {"choices": [{"delta": {"role": "assistant"}, "index": 0}]},
            {"choices": [{"delta": {"content": "Hi"}, "index": 0}]},
            {"choices": [{"delta": {}, "index": 0}]},
        ]
        stream = track_stream(iter(chunks), model="gpt-4o")
        list(stream)
        assert stream.tracker.token_count == 1


# ---------------------------------------------------------------------------
# async_track_stream
# ---------------------------------------------------------------------------


class TestAsyncTrackStream:
    @pytest.mark.asyncio
    async def test_yields_all_string_chunks(self):
        async def _gen():
            for t in ["Hello", " ", "world"]:
                yield t

        collected = []
        stream = async_track_stream(_gen(), model="gpt-4o")
        async for chunk in stream:
            collected.append(chunk)
        assert collected == ["Hello", " ", "world"]

    @pytest.mark.asyncio
    async def test_tracker_attribute(self):
        async def _gen():
            for t in ["a", "b", "c", "d"]:
                yield t

        stream = async_track_stream(_gen(), model="gpt-4o")
        async for _ in stream:
            pass
        usage = stream.tracker.finalize()
        assert usage.total_tokens == 4

    @pytest.mark.asyncio
    async def test_dict_chunks(self):
        chunks = [
            {"choices": [{"delta": {"content": "Hi"}, "index": 0}]},
            {"choices": [{"delta": {"content": " there"}, "index": 0}]},
        ]

        async def _gen():
            for c in chunks:
                yield c

        stream = async_track_stream(_gen(), model="gpt-4o")
        collected = []
        async for chunk in stream:
            collected.append(chunk)
        assert collected == chunks
        assert stream.tracker.token_count == 2

    @pytest.mark.asyncio
    async def test_empty_async_stream(self):
        async def _gen():
            return
            yield  # noqa: unreachable — makes this an async generator

        stream = async_track_stream(_gen(), model="gpt-4o")
        collected = []
        async for chunk in stream:
            collected.append(chunk)
        assert collected == []
        assert stream.tracker.token_count == 0

    @pytest.mark.asyncio
    async def test_cost_accumulates_during_stream(self):
        async def _gen():
            for t in ["one", "two", "three"]:
                yield t

        tracker = StreamingCostTracker(model="gpt-4o")
        stream = async_track_stream(_gen(), model="gpt-4o", tracker=tracker)
        costs = []
        async for _ in stream:
            costs.append(tracker.current_cost)
        # Each subsequent cost should be >= the previous
        for i in range(1, len(costs)):
            assert costs[i] >= costs[i - 1]
        assert costs[-1] > Decimal("0")


# ---------------------------------------------------------------------------
# StreamingUsage dataclass
# ---------------------------------------------------------------------------


class TestStreamingUsage:
    def test_frozen(self):
        usage = StreamingUsage(
            model="gpt-4o",
            total_tokens=10,
            estimated_cost=Decimal("0.0001"),
            duration_ms=42.5,
        )
        with pytest.raises(AttributeError):
            usage.model = "other"  # type: ignore[misc]

    def test_fields(self):
        usage = StreamingUsage(
            model="gpt-4o",
            total_tokens=50,
            estimated_cost=Decimal("0.0005"),
            duration_ms=123.45,
        )
        assert usage.model == "gpt-4o"
        assert usage.total_tokens == 50
        assert usage.estimated_cost == Decimal("0.0005")
        assert usage.duration_ms == 123.45
