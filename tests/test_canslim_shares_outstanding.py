"""
CANSLIM criterion S screen — shares outstanding -> capitalization verdict.

Covered behaviors:
    1. Pass: 4.6M shares — the median of O'Neil's best performers (Market
       Wizards 2006) — clears the default 25M line.
    2. Fail: 250M shares (a large-cap) -> reason "too_many_shares".
    3. Fail honestly when no share data exists -> reason "no_share_data",
       never a fabricated count.
    4. Trailing-min mode: a series that spiked recently (diluted above the
       line) still passes via the minimum over the trailing 12 months, and
       as_of reports the date of that minimum.
    5. Fetch fallback: get_shares_full empty -> get_shares (patched
       yfinance.Ticker, same pattern as test_canslim_screen).

Mock Object seam: the screen takes the fetch callable as an injected
parameter (like screen_universe's `score`), defaulting to fetch_shares_history.
"""
from datetime import datetime
from unittest.mock import patch

import pandas as pd
import pytest

from fentu.canslim.shares_outstanding import (
    DEFAULT_MAX_SHARES,
    latest_share_count,
    screen_shares_outstanding,
    trailing_min_share_count,
)

MEDIAN = 4_600_000


def share_series(entries):
    """Series of (year, month, day, shares) tuples; shares may be None -> NaN."""
    index = pd.DatetimeIndex([datetime(*d[:3]) for d in entries])
    values = [None if d[3] is None else float(d[3]) for d in entries]
    return pd.Series(values, index=index)


def test_pass_when_median_best_performer_shares():
    fetch = lambda t: share_series([(2025, 6, 30, MEDIAN)])
    result = screen_shares_outstanding("VRTX", fetch=fetch)
    assert result.passed is True
    assert result.shares == MEDIAN
    assert result.as_of == "2025-06-30"
    assert result.reason == ""


def test_pass_at_exactly_the_max_shares_line():
    fetch = lambda t: share_series([(2025, 6, 30, DEFAULT_MAX_SHARES)])
    result = screen_shares_outstanding("VRTX", fetch=fetch)
    assert result.passed is True


def test_fail_when_large_cap_250_million_shares():
    fetch = lambda t: share_series([(2025, 6, 30, 250_000_000)])
    result = screen_shares_outstanding("PFE", fetch=fetch)
    assert result.passed is False
    assert result.reason == "too_many_shares"
    assert result.shares == 250_000_000


def test_fail_with_custom_stricter_cap():
    fetch = lambda t: share_series([(2025, 6, 30, MEDIAN)])
    result = screen_shares_outstanding("VRTX", max_shares=1_000_000, fetch=fetch)
    assert result.passed is False
    assert result.reason == "too_many_shares"


def test_fail_honestly_when_fetch_returns_none():
    result = screen_shares_outstanding("XFOR", fetch=lambda t: None)
    assert result.passed is False
    assert result.reason == "no_share_data"
    assert result.shares is None
    assert result.as_of is None


def test_fail_honestly_when_series_is_empty():
    fetch = lambda t: pd.Series(dtype="float64")
    result = screen_shares_outstanding("XFOR", fetch=fetch)
    assert result.passed is False
    assert result.reason == "no_share_data"


def test_latest_share_count_skips_nan_entries():
    series = share_series([(2025, 1, 1, 5_000_000), (2025, 2, 1, None), (2025, 3, 1, 6_000_000)])
    shares, when = latest_share_count(series)
    assert shares == 6_000_000
    assert when == "2025-03-01"


def test_trailing_min_uses_minimum_not_latest():
    entries = [
        (2024, 1, 31, 2_500_000),
        (2025, 1, 31, 3_000_000),
        (2025, 6, 1, 40_000_000),
    ]
    fetch = lambda t: share_series(entries)

    result = screen_shares_outstanding("VRTX", fetch=fetch)
    assert result.passed is False
    assert result.reason == "too_many_shares"
    assert result.shares == 40_000_000

    trailing = screen_shares_outstanding("VRTX", use_trailing_min=True, fetch=fetch)
    assert trailing.passed is True
    assert trailing.shares == 3_000_000
    assert trailing.as_of == "2025-01-31"


def test_trailing_min_ignores_observations_older_than_window():
    series = share_series([(2024, 1, 31, 1_000_000), (2025, 6, 1, 40_000_000)])
    shares, when = trailing_min_share_count(series)
    assert shares == 40_000_000
    assert when == "2025-06-01"


def test_fetch_shares_history_falls_back_to_get_shares():
    fallback = pd.Series([1_000_000, 2_000_000], index=pd.DatetimeIndex([datetime(2025, 1, 1), datetime(2025, 6, 30)]))
    with patch("yfinance.Ticker") as ticker:
        ticker.return_value.get_shares_full.return_value = None
        ticker.return_value.get_shares.return_value = fallback
        from fentu.canslim.shares_outstanding import fetch_shares_history

        series = fetch_shares_history("VRTX")
    assert list(series) == [1_000_000, 2_000_000]


def test_main_returns_pass_exit_code_with_injected_screen(monkeypatch):
    from fentu.canslim import shares_outstanding as mod

    def fake_screen(ticker, max_shares, use_trailing_min):
        passed = ticker == "VRTX"
        return mod.SharesResult(ticker=ticker, shares=MEDIAN if passed else 250_000_000, as_of="2025-06-30", passed=passed, reason="" if passed else "too_many_shares")

    monkeypatch.setattr(mod, "screen_shares_outstanding", fake_screen)
    assert mod.main(["VRTX"]) == 0
    assert mod.main(["PFE"]) == 1
