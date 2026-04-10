"""Provider rate limit tracking for LLM API calls."""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""

    requests_per_minute: int
    tokens_per_minute: int
    requests_per_day: int | None = None
    cooldown_seconds: float = 1.0


@dataclass
class RateLimitState:
    """Snapshot of current rate limit state."""

    remaining_requests: int
    remaining_tokens: int
    reset_at: float
    is_limited: bool


class RateLimiter:
    """Sliding-window rate limiter for LLM API providers.

    Tracks request counts and token usage within per-minute windows,
    with optional per-day request caps.
    """

    def __init__(self, config: RateLimitConfig) -> None:
        self._config = config
        self._window_start: float = time.time()
        self._requests_used: int = 0
        self._tokens_used: int = 0
        self._day_start: float = time.time()
        self._daily_requests: int = 0
        self._cooldown_until: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def acquire(self, tokens: int = 1) -> bool:
        """Try to acquire capacity for *tokens* tokens and one request.

        Returns ``True`` if the request is allowed, ``False`` otherwise.
        Does **not** record usage on its own — call :meth:`record_usage`
        after a successful API call.
        """
        self._maybe_reset_window()
        now = time.time()

        if now < self._cooldown_until:
            return False

        if self._requests_used >= self._config.requests_per_minute:
            return False

        if self._tokens_used + tokens > self._config.tokens_per_minute:
            return False

        return not (self._config.requests_per_day is not None and self._daily_requests >= self._config.requests_per_day)

    def wait_time(self) -> float:
        """Seconds until the next request is expected to be allowed."""
        self._maybe_reset_window()
        now = time.time()

        if now < self._cooldown_until:
            return self._cooldown_until - now

        if self._requests_used >= self._config.requests_per_minute:
            return max(0.0, self._window_start + 60.0 - now)

        if self._tokens_used >= self._config.tokens_per_minute:
            return max(0.0, self._window_start + 60.0 - now)

        if (
            self._config.requests_per_day is not None
            and self._daily_requests >= self._config.requests_per_day
        ):
            return max(0.0, self._day_start + 86400.0 - now)

        return 0.0

    def state(self) -> RateLimitState:
        """Return a snapshot of the current rate-limit state."""
        self._maybe_reset_window()
        remaining_req = max(0, self._config.requests_per_minute - self._requests_used)
        remaining_tok = max(0, self._config.tokens_per_minute - self._tokens_used)
        reset_at = self._window_start + 60.0
        is_limited = remaining_req == 0 or remaining_tok == 0
        if (
            self._config.requests_per_day is not None
            and self._daily_requests >= self._config.requests_per_day
        ):
            is_limited = True
        if time.time() < self._cooldown_until:
            is_limited = True
        return RateLimitState(
            remaining_requests=remaining_req,
            remaining_tokens=remaining_tok,
            reset_at=reset_at,
            is_limited=is_limited,
        )

    def record_usage(self, tokens: int) -> None:
        """Record that one request consuming *tokens* tokens was made."""
        self._maybe_reset_window()
        self._requests_used += 1
        self._tokens_used += tokens
        self._daily_requests += 1

    def reset(self) -> None:
        """Reset all counters and windows."""
        now = time.time()
        self._window_start = now
        self._requests_used = 0
        self._tokens_used = 0
        self._day_start = now
        self._daily_requests = 0
        self._cooldown_until = 0.0

    def update_from_headers(self, headers: dict[str, str]) -> None:
        """Update internal state from API response headers.

        Recognised headers (case-insensitive):
        * ``x-ratelimit-remaining-requests``
        * ``x-ratelimit-remaining-tokens``
        * ``x-ratelimit-reset-requests`` (seconds until reset)
        * ``retry-after`` (seconds to wait)
        """
        lower = {k.lower(): v for k, v in headers.items()}

        if "x-ratelimit-remaining-requests" in lower:
            remaining = int(lower["x-ratelimit-remaining-requests"])
            self._requests_used = self._config.requests_per_minute - remaining

        if "x-ratelimit-remaining-tokens" in lower:
            remaining = int(lower["x-ratelimit-remaining-tokens"])
            self._tokens_used = self._config.tokens_per_minute - remaining

        if "x-ratelimit-reset-requests" in lower:
            try:
                secs = float(lower["x-ratelimit-reset-requests"].rstrip("s"))
                self._window_start = time.time() + secs - 60.0
            except ValueError:
                pass

        if "retry-after" in lower:
            with contextlib.suppress(ValueError):
                self._cooldown_until = time.time() + float(lower["retry-after"])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _maybe_reset_window(self) -> None:
        now = time.time()
        if now - self._window_start >= 60.0:
            self._window_start = now
            self._requests_used = 0
            self._tokens_used = 0
        if now - self._day_start >= 86400.0:
            self._day_start = now
            self._daily_requests = 0


# --------------------------------------------------------------------------
# Provider defaults
# --------------------------------------------------------------------------

PROVIDER_DEFAULTS: dict[str, RateLimitConfig] = {
    "openai": RateLimitConfig(
        requests_per_minute=500,
        tokens_per_minute=200_000,
        requests_per_day=10_000,
    ),
    "anthropic": RateLimitConfig(
        requests_per_minute=60,
        tokens_per_minute=100_000,
        requests_per_day=None,
    ),
    "google": RateLimitConfig(
        requests_per_minute=60,
        tokens_per_minute=120_000,
        requests_per_day=1_500,
    ),
    "cohere": RateLimitConfig(
        requests_per_minute=100,
        tokens_per_minute=100_000,
        requests_per_day=5_000,
    ),
    "mistral": RateLimitConfig(
        requests_per_minute=120,
        tokens_per_minute=150_000,
        requests_per_day=None,
    ),
}


def create_limiter(provider: str) -> RateLimiter:
    """Create a :class:`RateLimiter` with default config for *provider*.

    Raises :class:`KeyError` if no defaults are registered for the provider.
    """
    key = provider.lower()
    if key not in PROVIDER_DEFAULTS:
        raise KeyError(
            f"Unknown provider {provider!r}. "
            f"Available: {', '.join(sorted(PROVIDER_DEFAULTS))}"
        )
    return RateLimiter(PROVIDER_DEFAULTS[key])


def format_rate_status(state: RateLimitState) -> str:
    """Return a human-readable summary of a :class:`RateLimitState`."""
    status = "RATE-LIMITED" if state.is_limited else "OK"
    remaining_secs = max(0.0, state.reset_at - time.time())
    return (
        f"[{status}] "
        f"requests={state.remaining_requests} "
        f"tokens={state.remaining_tokens} "
        f"resets_in={remaining_secs:.0f}s"
    )
