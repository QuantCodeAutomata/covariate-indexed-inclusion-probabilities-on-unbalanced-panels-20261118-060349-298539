"""Shared pytest fixtures and marker registration."""

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "market_data: test needs a live QuantX Dataloader / Massive call"
    )


@pytest.fixture(scope="session")
def market_data_slice():
    """A small, reduced slice of the US stock panel used by exp_4/exp_5,
    loaded once per test session and cached between runs.

    Skips with a clear reason when DATALOADER_API_KEY is not set, so tests
    never depend on a live call by default.
    """
    import os

    if not os.environ.get("DATALOADER_API_KEY") or not os.environ.get("MASSIVE_API_KEY"):
        pytest.skip("DATALOADER_API_KEY / MASSIVE_API_KEY not set")
    from data import data_loader

    tickers = ["AAPL", "MSFT", "IBM", "GE", "F"]
    bars, meta = data_loader.load_ohlcv(tickers, "2015-01-01", "2016-12-31", "1d")
    return bars, meta
