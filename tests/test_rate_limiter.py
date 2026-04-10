"""Tests for tokonomics.rate_limiter module."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from tokonomics.rate_limiter import (
    PROVIDER_DEFAULTS,
    RateLimitConfig,
    RateLimitState,
    RateLimiter,
    create_limiter,
    format_rate_status,
)


# ── Config / State dataclasses ──────────────────────────────────────

def test_config_defaults():
    cfg = RateLimitConfig(requests_per_minute=60, tokens_per_minute=10_000)
    assert cfg.requests_per_day is None
    assert cfg.cooldown_seconds == 1.0


def test_state_fields():
    st = RateLimitState(remaining_requests=10, remaining_tokens=5000, reset_at=0.0, is_limited=False)
    assert not st.is_limited


# ── Basic acquire / record ──────────────────────────────────────────

def test_acquire_succeeds_within_limits():
    rl = RateLimiter(RateLimitConfig(requests_per_minute=10, tokens_per_minute=1000))
    assert rl.acquire(100) is True


def test_acquire_fails_when_requests_exhausted():
    rl = RateLimiter(RateLimitConfig(requests_per_minute=2, tokens_per_minute=100_000))
    rl.record_usage(1)
    rl.record_usage(1)
    assert rl.acquire(1) is False


def test_acquire_fails_when_tokens_exhausted():
    rl = RateLimiter(RateLimitConfig(requests_per_minute=100, tokens_per_minute=50))
    rl.record_usage(50)
    assert rl.acquire(10) is False


def test_record_usage_increments():
    rl = RateLimiter(RateLimitConfig(requests_per_minute=100, tokens_per_minute=10_000))
    rl.record_usage(500)
    rl.record_usage(300)
    st = rl.state()
    assert st.remaining_requests == 98
    assert st.remaining_tokens == 10_000 - 800


# ── Daily limits ────────────────────────────────────────────────────

def test_daily_limit_blocks_acquire():
    rl = RateLimiter(
        RateLimitConfig(requests_per_minute=1000, tokens_per_minute=1_000_000, requests_per_day=3)
    )
    for _ in range(3):
        rl.record_usage(1)
    assert rl.acquire(1) is False


# ── Window reset ────────────────────────────────────────────────────

def test_window_resets_after_60s():
    rl = RateLimiter(RateLimitConfig(requests_per_minute=1, tokens_per_minute=100))
    rl.record_usage(100)
    assert rl.acquire(1) is False

    # Fast-forward 61 seconds
    rl._window_start -= 61
    assert rl.acquire(1) is True


# ── wait_time ───────────────────────────────────────────────────────

def test_wait_time_zero_when_available():
    rl = RateLimiter(RateLimitConfig(requests_per_minute=100, tokens_per_minute=10_000))
    assert rl.wait_time() == 0.0


def test_wait_time_positive_when_limited():
    rl = RateLimiter(RateLimitConfig(requests_per_minute=1, tokens_per_minute=10_000))
    rl.record_usage(1)
    wt = rl.wait_time()
    assert wt > 0.0
    assert wt <= 60.0


# ── state ───────────────────────────────────────────────────────────

def test_state_is_limited_flag():
    rl = RateLimiter(RateLimitConfig(requests_per_minute=1, tokens_per_minute=10_000))
    assert rl.state().is_limited is False
    rl.record_usage(1)
    assert rl.state().is_limited is True


# ── reset ───────────────────────────────────────────────────────────

def test_reset_clears_counters():
    rl = RateLimiter(RateLimitConfig(requests_per_minute=5, tokens_per_minute=1000))
    for _ in range(5):
        rl.record_usage(100)
    assert rl.acquire(1) is False
    rl.reset()
    assert rl.acquire(1) is True
    st = rl.state()
    assert st.remaining_requests == 5
    assert st.remaining_tokens == 1000


# ── update_from_headers ────────────────────────────────────────────

def test_update_remaining_from_headers():
    rl = RateLimiter(RateLimitConfig(requests_per_minute=100, tokens_per_minute=50_000))
    rl.update_from_headers({
        "x-ratelimit-remaining-requests": "42",
        "x-ratelimit-remaining-tokens": "12345",
    })
    st = rl.state()
    assert st.remaining_requests == 42
    assert st.remaining_tokens == 12345


def test_update_retry_after():
    rl = RateLimiter(RateLimitConfig(requests_per_minute=100, tokens_per_minute=50_000))
    rl.update_from_headers({"Retry-After": "5"})
    assert rl.acquire(1) is False
    assert rl.wait_time() > 0


def test_update_reset_requests_header():
    rl = RateLimiter(RateLimitConfig(requests_per_minute=100, tokens_per_minute=50_000))
    rl.update_from_headers({"x-ratelimit-reset-requests": "30s"})
    # Should not crash; internally adjusts window_start
    st = rl.state()
    assert isinstance(st.reset_at, float)


# ── PROVIDER_DEFAULTS / factory ────────────────────────────────────

def test_provider_defaults_has_all_providers():
    for name in ("openai", "anthropic", "google", "cohere", "mistral"):
        assert name in PROVIDER_DEFAULTS


def test_create_limiter_known():
    rl = create_limiter("openai")
    assert isinstance(rl, RateLimiter)


def test_create_limiter_case_insensitive():
    rl = create_limiter("Anthropic")
    assert isinstance(rl, RateLimiter)


def test_create_limiter_unknown_raises():
    with pytest.raises(KeyError, match="Unknown provider"):
        create_limiter("nonexistent")


# ── format_rate_status ──────────────────────────────────────────────

def test_format_rate_status_ok():
    st = RateLimitState(remaining_requests=50, remaining_tokens=10_000, reset_at=time.time() + 30, is_limited=False)
    text = format_rate_status(st)
    assert "[OK]" in text
    assert "requests=50" in text


def test_format_rate_status_limited():
    st = RateLimitState(remaining_requests=0, remaining_tokens=0, reset_at=time.time() + 10, is_limited=True)
    text = format_rate_status(st)
    assert "[RATE-LIMITED]" in text
