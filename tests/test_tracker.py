"""Tests for the cost tracker."""

import asyncio
from decimal import Decimal

from tokonomics.tracker import CostTracker, get_global_tracker, track_cost


class TestCostTracker:
    def test_empty_tracker(self):
        tracker = CostTracker()
        assert tracker.total_cost == Decimal("0")
        assert tracker.records == []

    def test_record_single(self):
        tracker = CostTracker()
        usage = tracker.record("gpt-4o", input_tokens=100, output_tokens=50)
        assert usage.total_cost > Decimal("0")
        assert len(tracker.records) == 1
        assert tracker.total_cost == usage.total_cost

    def test_record_multiple(self):
        tracker = CostTracker()
        u1 = tracker.record("gpt-4o", input_tokens=100, output_tokens=50)
        u2 = tracker.record("gpt-4o", input_tokens=200, output_tokens=100)
        assert tracker.total_cost == u1.total_cost + u2.total_cost
        assert len(tracker.records) == 2

    def test_context_manager(self):
        with CostTracker() as tracker:
            tracker.record("gpt-4o", input_tokens=100, output_tokens=50)
        assert tracker.total_cost > Decimal("0")

    def test_by_model(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", input_tokens=100, output_tokens=50)
        tracker.record("gpt-4o", input_tokens=100, output_tokens=50)
        tracker.record("deepseek-chat", input_tokens=100, output_tokens=50)
        breakdown = tracker.by_model()
        assert "gpt-4o" in breakdown
        assert "deepseek-chat" in breakdown
        assert len(breakdown) == 2

    def test_by_provider(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", input_tokens=100, output_tokens=50)
        tracker.record("claude-sonnet-4-20250514", input_tokens=100, output_tokens=50)
        breakdown = tracker.by_provider()
        assert "openai" in breakdown
        assert "anthropic" in breakdown

    def test_total_tokens(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", input_tokens=100, output_tokens=50)
        tracker.record("gpt-4o", input_tokens=200, output_tokens=100)
        assert tracker.total_input_tokens == 300
        assert tracker.total_output_tokens == 150

    def test_reset(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", input_tokens=100, output_tokens=50)
        tracker.reset()
        assert tracker.total_cost == Decimal("0")
        assert tracker.records == []

    def test_summary(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", input_tokens=1000, output_tokens=500)
        summary = tracker.summary()
        assert "Total cost: $" in summary
        assert "gpt-4o" in summary

    def test_metadata(self):
        tracker = CostTracker()
        usage = tracker.record(
            "gpt-4o",
            input_tokens=100,
            output_tokens=50,
            metadata={"request_id": "abc123"},
        )
        assert usage.metadata == {"request_id": "abc123"}

    def test_timestamp_set(self):
        tracker = CostTracker()
        usage = tracker.record("gpt-4o", input_tokens=100, output_tokens=50)
        assert usage.timestamp is not None
        assert usage.timestamp > 0


class TestTrackCostDecorator:
    def test_sync_function(self):
        tracker = CostTracker()

        @track_cost(model="gpt-4o", tracker=tracker)
        def my_fn():
            return {"input_tokens": 100, "output_tokens": 50}

        result = my_fn()
        assert result == {"input_tokens": 100, "output_tokens": 50}
        assert tracker.total_cost > Decimal("0")

    def test_async_function(self):
        tracker = CostTracker()

        @track_cost(model="gpt-4o", tracker=tracker)
        async def my_fn():
            return {"input_tokens": 100, "output_tokens": 50}

        result = asyncio.get_event_loop().run_until_complete(my_fn())
        assert result == {"input_tokens": 100, "output_tokens": 50}
        assert tracker.total_cost > Decimal("0")

    def test_no_tokens_no_record(self):
        tracker = CostTracker()

        @track_cost(model="gpt-4o", tracker=tracker)
        def my_fn():
            return {"input_tokens": 0, "output_tokens": 0}

        my_fn()
        assert tracker.total_cost == Decimal("0")

    def test_usage_history(self):
        tracker = CostTracker()

        @track_cost(model="gpt-4o", tracker=tracker)
        def my_fn():
            return {"input_tokens": 100, "output_tokens": 50}

        my_fn()
        my_fn()
        assert len(my_fn.usage_history) == 2

    def test_get_total_cost(self):
        tracker = CostTracker()

        @track_cost(model="gpt-4o", tracker=tracker)
        def my_fn():
            return {"input_tokens": 100, "output_tokens": 50}

        my_fn()
        assert my_fn.get_total_cost() > Decimal("0")

    def test_get_last_usage(self):
        tracker = CostTracker()

        @track_cost(model="gpt-4o", tracker=tracker)
        def my_fn():
            return {"input_tokens": 100, "output_tokens": 50}

        my_fn()
        last = my_fn.get_last_usage()
        assert last is not None
        assert last.model == "gpt-4o"


class TestGlobalTracker:
    def test_global_tracker_exists(self):
        tracker = get_global_tracker()
        assert isinstance(tracker, CostTracker)
