import pytest


@pytest.fixture
def sample_text():
    return "The quick brown fox jumps over the lazy dog."


@pytest.fixture
def long_text():
    return "Hello world. " * 1000
