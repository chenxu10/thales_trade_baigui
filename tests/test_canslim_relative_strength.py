"""CANSLIM criterion L relative-strength filter — deterministic, no network.

Covered behaviors:
    1. Rank above 80 passes; rank below 80 fails; exact boundary (80) passes.
    2. Missing price history for the candidate => FILTERED "no_price_history".
    3. Universe too small to rank against => "not_enough_universe".
    4. universe_size counts universe names with usable price history.

Mock Object seam: ``screen_relative_strength`` takes an injectable ``fetch``
callable (the same injection pattern as screen_universe's ``score`` param), so
tests supply a fake fetch returning a small fixture DataFrame — no network.
"""
import pandas as pd
import pytest

from fentu.canslim.relative_strength import (
    RelativeStrengthResult,
    rs_rank,
    screen_relative_strength,
    twelve_month_return,
    universe_twelve_month_returns,
)


def close_frame(returns):
    """DataFrame of one year of daily closes whose first->last change is ``returns``."""
    dates = pd.date_range("2025-08-19", periods=260, freq="B")
    columns = {}
    for ticker, ret in returns.items():
        start = 100.0
        end = start * (1 + ret)
        columns[ticker] = [start + (end - start) * i / (len(dates) - 1) for i in range(len(dates))]
    return pd.DataFrame(columns, index=dates)


def fake_fetch(closes):
    def fetch(tickers):
        return closes

    return fetch


def test_rank_above_80_passes():
    closes = close_frame({
        "VRTX": 0.30,
        "PFE": -0.05, "MRK": 0.05, "REGN": 0.10, "GILD": 0.15, "BMY": 0.25,
    })
    result = screen_relative_strength("VRTX", ["PFE", "MRK", "REGN", "GILD", "BMY"], fetch=fake_fetch(closes))
    assert result.passed is True
    assert result.reason == ""
    assert result.return_12m == pytest.approx(0.30)
    assert result.rank == pytest.approx(100.0 * 5 / 6)
    assert result.universe_size == 6


def test_rank_below_80_fails():
    closes = close_frame({
        "VRTX": 0.30,
        "PFE": -0.05, "MRK": 0.05, "REGN": 0.10, "GILD": 0.15, "BMY": 0.35,
    })
    result = screen_relative_strength("VRTX", ["PFE", "MRK", "REGN", "GILD", "BMY"], fetch=fake_fetch(closes))
    assert result.passed is False
    assert result.reason == "below_rank_threshold"
    assert result.rank == pytest.approx(100.0 * 4 / 6)


def test_exact_boundary_rank_80_passes():
    closes = close_frame({
        "VRTX": 0.30,
        "PFE": -0.05, "MRK": 0.05, "REGN": 0.10, "GILD": 0.25,
    })
    result = screen_relative_strength("VRTX", ["PFE", "MRK", "REGN", "GILD"], fetch=fake_fetch(closes))
    assert result.passed is True
    assert result.rank == pytest.approx(80.0)
    assert result.reason == ""


def test_missing_price_history_filtered():
    closes = close_frame({"PFE": 0.05, "MRK": 0.10})
    result = screen_relative_strength("VRTX", ["PFE", "MRK"], fetch=fake_fetch(closes))
    assert result.passed is False
    assert result.reason == "no_price_history"
    assert result.return_12m is None
    assert result.rank is None
    assert result.universe_size == 2


def test_not_enough_universe_filtered():
    closes = close_frame({"VRTX": 0.30, "PFE": 0.05})
    result = screen_relative_strength("VRTX", ["PFE"], fetch=fake_fetch(closes))
    assert result.passed is False
    assert result.reason == "not_enough_universe"


def test_universe_size_counts_names_with_price_history():
    closes = close_frame({"VRTX": 0.30, "PFE": 0.05, "MRK": 0.10})
    result = screen_relative_strength("VRTX", ["PFE", "MRK", "REGN"], fetch=fake_fetch(closes))
    assert result.universe_size == 3
    assert result.rank == pytest.approx(100.0 * 2 / 3)


def test_twelve_month_return_uses_first_and_last_close():
    assert twelve_month_return([100.0, 105.0, 130.0]) == pytest.approx(0.30)


def test_twelve_month_return_uses_full_span_beyond_252_sessions():
    assert twelve_month_return([100.0] * 252 + [130.0]) == pytest.approx(0.30)


def test_twelve_month_return_none_when_degenerate():
    assert twelve_month_return([]) is None
    assert twelve_month_return([100.0]) is None


def test_rs_rank_percentile_of_strictly_below():
    assert rs_rank([0.05, 0.10, 0.15, 0.25, 0.35], 0.30) == pytest.approx(80.0)
    assert rs_rank([0.05, 0.10, 0.15], 0.35) == pytest.approx(100.0)
    assert rs_rank([0.05, 0.10, 0.15], 0.05) == pytest.approx(0.0)


def test_universe_returns_map_is_order_preserving():
    closes = close_frame({"VRTX": 0.30, "PFE": -0.05, "MRK": 0.10})
    pairs = universe_twelve_month_returns(closes, ["MRK", "VRTX", "PFE"])
    assert [t for t, _ in pairs] == ["MRK", "VRTX", "PFE"]
    assert [r for _, r in pairs] == [pytest.approx(0.10), pytest.approx(0.30), pytest.approx(-0.05)]