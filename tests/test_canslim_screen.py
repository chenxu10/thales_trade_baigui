"""
CANSLIM criterion C screen — yfinance quarterly EPS -> CurrentEpsCriterion verdict.

Covered behaviors:
    1. Pass: quarterly Diluted EPS up >= 20% YoY vs same quarter last year.
    2. Fail: growth below threshold (declining EPS).
    3. Fail honestly when no quarterly data exists (HK 18A/8.05 names often
       report semi-annually; yfinance serves annual only) — reason
       "no_quarterly_data", never a fabricated quarter.

Mock Object seam: `screen.fetch_quarterly_eps` imports yfinance lazily, so we
patch yfinance.Ticker directly (same pattern as option_quotes).
"""
from datetime import datetime

import pandas as pd
import pytest
from unittest.mock import patch

from fentu.canslim.screen import align_same_quarter_last_year, screen_current_eps


def fake_statement(eps_values, dates):
    return pd.DataFrame(
        {datetime(*d): [eps] for d, eps in zip(dates, eps_values)},
        index=["Diluted EPS"],
    )


def test_pass_when_current_quarter_up_30_percent_yoy():
    stmt = fake_statement([0.65, 0.50], [(2025, 3, 31), (2024, 3, 31)])
    with patch("yfinance.Ticker") as ticker:
        ticker.return_value.quarterly_income_stmt = stmt
        result = screen_current_eps("3696.HK")
    assert result.passed is True
    assert result.current_eps == 0.65
    assert result.prior_year_eps == 0.50
    assert result.growth == pytest.approx(0.30)
    assert result.reason == ""


def test_fail_when_current_quarter_declining():
    stmt = fake_statement([0.55, 0.60], [(2025, 3, 31), (2024, 3, 31)])
    with patch("yfinance.Ticker") as ticker:
        ticker.return_value.quarterly_income_stmt = stmt
        result = screen_current_eps("3696.HK")
    assert result.passed is False
    assert result.reason == "below_threshold"


def test_fail_honestly_when_no_quarterly_data():
    with patch("yfinance.Ticker") as ticker:
        ticker.return_value.quarterly_income_stmt = pd.DataFrame()
        result = screen_current_eps("3696.HK")
    assert result.passed is False
    assert result.reason == "no_quarterly_data"
    assert result.current_eps is None
    assert result.growth is None


def test_align_picks_nearest_period_not_first_within_tolerance():
    stmt = fake_statement(
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [(2026, 6, 30), (2026, 3, 31), (2025, 12, 31), (2025, 9, 30), (2025, 6, 30)],
    )
    current, prior = align_same_quarter_last_year(stmt)
    assert current == datetime(2026, 6, 30)
    assert prior == datetime(2025, 6, 30)