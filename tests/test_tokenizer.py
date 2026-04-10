"""Tests for the tokenizer module."""

import pytest

from tokonomics.tokenizer import (
    _estimate_tokens_fallback,
    count_message_tokens,
    count_tokens,
    fits_context,
)


class TestCountTokens:
    def test_empty_string(self):
        assert count_tokens("") == 0

    def test_simple_text(self, sample_text):
        tokens = count_tokens(sample_text, "gpt-4o")
        assert tokens > 0
        assert tokens < 100

    def test_long_text(self, long_text):
        tokens = count_tokens(long_text, "gpt-4o")
        assert tokens > 100

    @pytest.mark.parametrize(
        "model",
        ["gpt-4o", "gpt-4.1", "gpt-4o-mini", "gpt-4-turbo", "gpt-4"],
    )
    def test_various_openai_models(self, sample_text, model):
        tokens = count_tokens(sample_text, model)
        assert tokens > 0

    def test_non_openai_model_still_works(self, sample_text):
        # Should still return a count (using default encoding)
        tokens = count_tokens(sample_text, "claude-3.5-sonnet")
        assert tokens > 0

    def test_deterministic(self, sample_text):
        t1 = count_tokens(sample_text, "gpt-4o")
        t2 = count_tokens(sample_text, "gpt-4o")
        assert t1 == t2

    def test_longer_text_more_tokens(self):
        short = count_tokens("hello", "gpt-4o")
        long = count_tokens("hello world this is a longer sentence", "gpt-4o")
        assert long > short

    def test_unicode(self):
        tokens = count_tokens("こんにちは世界", "gpt-4o")
        assert tokens > 0

    def test_code(self):
        code = "def hello():\n    return 'world'\n"
        tokens = count_tokens(code, "gpt-4o")
        assert tokens > 0

    def test_whitespace_only(self):
        tokens = count_tokens("   ", "gpt-4o")
        assert tokens > 0


class TestFallbackEstimator:
    def test_basic(self):
        estimate = _estimate_tokens_fallback("hello world")
        assert estimate > 0

    def test_empty(self):
        # Empty text has no \S+ matches, so max(1, ...) = 1
        estimate = _estimate_tokens_fallback("")
        assert estimate >= 1

    def test_roughly_correct_magnitude(self):
        text = " ".join(["word"] * 100)
        estimate = _estimate_tokens_fallback(text)
        # Should be in the right ballpark: 100-200 tokens
        assert 50 < estimate < 500


class TestCountMessageTokens:
    def test_simple_messages(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi!"},
        ]
        tokens = count_message_tokens(messages, "gpt-4o")
        assert tokens > 0

    def test_empty_messages(self):
        tokens = count_message_tokens([], "gpt-4o")
        # Overhead for the reply priming
        assert tokens == 3

    def test_more_messages_more_tokens(self):
        short = count_message_tokens(
            [{"role": "user", "content": "Hi"}], "gpt-4o"
        )
        long = count_message_tokens(
            [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
                {"role": "user", "content": "How are you?"},
            ],
            "gpt-4o",
        )
        assert long > short


class TestFitsContext:
    def test_short_text_fits(self, sample_text):
        assert fits_context(sample_text, "gpt-4o") is True

    def test_huge_text_may_not_fit(self):
        # Generate text that's clearly larger than 8192 tokens (gpt-4 context)
        huge = "word " * 100000
        assert fits_context(huge, "gpt-4") is False
