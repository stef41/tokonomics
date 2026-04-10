"""Tests for the CLI."""

import pytest


@pytest.fixture
def cli_runner():
    click = pytest.importorskip("click")
    from click.testing import CliRunner
    return CliRunner()


@pytest.fixture
def cli_group():
    pytest.importorskip("click")
    pytest.importorskip("rich")
    from tokonomics.cli import cli
    return cli


class TestCLI:
    def test_estimate_basic(self, cli_runner, cli_group):
        result = cli_runner.invoke(cli_group, ["estimate", "hello world", "-m", "gpt-4o"])
        assert result.exit_code == 0

    def test_models_list(self, cli_runner, cli_group):
        result = cli_runner.invoke(cli_group, ["models"])
        assert result.exit_code == 0

    def test_models_filter_provider(self, cli_runner, cli_group):
        result = cli_runner.invoke(cli_group, ["models", "-p", "openai"])
        assert result.exit_code == 0

    def test_price_command(self, cli_runner, cli_group):
        result = cli_runner.invoke(cli_group, ["price", "gpt-4o"])
        assert result.exit_code == 0

    def test_price_not_found(self, cli_runner, cli_group):
        result = cli_runner.invoke(cli_group, ["price", "nonexistent"])
        assert result.exit_code != 0

    def test_compare_basic(self, cli_runner, cli_group):
        result = cli_runner.invoke(cli_group, ["compare", "hello world", "-n", "5"])
        assert result.exit_code == 0

    def test_compare_with_provider_filter(self, cli_runner, cli_group):
        result = cli_runner.invoke(
            cli_group, ["compare", "hello world", "-p", "openai,anthropic"]
        )
        assert result.exit_code == 0

    def test_cheapest_basic(self, cli_runner, cli_group):
        result = cli_runner.invoke(cli_group, ["cheapest", "hello world"])
        assert result.exit_code == 0

    def test_cheapest_with_provider(self, cli_runner, cli_group):
        result = cli_runner.invoke(
            cli_group, ["cheapest", "hello world", "-p", "openai"]
        )
        assert result.exit_code == 0
