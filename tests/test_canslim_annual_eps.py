"""
CANSLIM criterion A — annual EPS growth -> AnnualEpsResult verdict.

Covered behaviors:
    1. Pass: 5-year compounded EPS growth ~30%/yr, every year up.
    2. Fail: growth below threshold (flat EPS) — reason "below_threshold".
    3. Fail: growth above threshold but a down year mid-streak —
       reason "not_every_year_up".
    4. Fail: loss-making base year (negative or zero EPS) — "negative_base".
    5. Fail honestly on missing data: no income statement -> "no_annual_data",
       unusable EPS row -> "missing_eps", fewer than two years -> "too_few_years".
    6. Only the most recent up-to-5 years are scored (older history ignored).
    7. CLI: exit 0 on PASS, exit 1 on FAIL.

Mock Object seam: `annual_eps.fetch_annual_eps` imports yfinance lazily, so we
patch yfinance.Ticker directly (same pattern as test_canslim_screen.py); the
pipeline is also reachable by injecting `fetch` into screen_annual_eps (same
pattern as screen_universe's `score` param).
"""
from datetime import datetime
from unittest.mock import patch

import pandas as pd
import pytest

from fentu.canslim.annual_eps import (
    compute_cagr,
    consecutive_up_years,
    extract_annual_eps,
    main,
    screen_annual_eps,
)

CAGR_30_YEARS = [1.00, 1.30, 1.69, 2.20, 2.86, 3.71]
YEARS_2020_2025 = [(2020, 12, 31), (2021, 12, 31), (2022, 12, 31), (2023, 12, 31), (2024, 12, 31), (2025, 12, 31)]


def fake_statement(eps_values, dates):
    return pd.DataFrame(
        {datetime(*d): [eps] for d, eps in zip(dates, eps_values)},
        index=["Diluted EPS"],
    )


def screen_with(stmt, ticker="VRTX", **kwargs):
    return screen_annual_eps(ticker, fetch=lambda t: stmt, **kwargs)


class TestAnnualEpsCriterionPass:
    def test_30_percent_cagr_every_year_up_passes(self):
        result = screen_with(fake_statement(CAGR_30_YEARS, YEARS_2020_2025))
        assert result.passed is True
        assert result.cagr == pytest.approx(3.71 ** 0.2 - 1)
        assert result.up_years == 5
        assert result.total_years == 5
        assert result.reason == ""

    def test_exactly_threshold_24_percent_passes(self):
        eps = [1.00, 1.24, 1.5376, 1.906624, 2.36421376, 2.9316250624]
        result = screen_with(fake_statement(eps, YEARS_2020_2025))
        assert result.passed is True
        assert result.cagr == pytest.approx(2.9316250624 ** 0.2 - 1)

    def test_min_up_years_override_allows_one_down_year(self):
        eps = [1.00, 1.30, 0.90, 1.60, 2.20, 3.00]
        stmt = fake_statement(eps, YEARS_2020_2025)
        assert screen_with(stmt).passed is False
        assert screen_with(stmt, min_up_years=3).passed is True


class TestAnnualEpsCriterionFail:
    def test_flat_eps_fails_below_threshold(self):
        eps = [1.00, 1.05, 1.10, 1.15, 1.20, 1.20]
        result = screen_with(fake_statement(eps, YEARS_2020_2025))
        assert result.passed is False
        assert result.reason == "below_threshold"
        assert result.cagr == pytest.approx(1.20 ** 0.2 - 1)
        assert result.up_years == 0

    def test_down_year_mid_streak_fails_not_every_year_up(self):
        eps = [1.00, 1.30, 0.90, 1.60, 2.20, 3.00]
        result = screen_with(fake_statement(eps, YEARS_2020_2025))
        assert result.passed is False
        assert result.reason == "not_every_year_up"
        assert result.cagr == pytest.approx(3.00 ** 0.2 - 1)
        assert result.cagr >= 0.24
        assert result.up_years == 3
        assert result.total_years == 5

    def test_flat_final_year_breaks_streak(self):
        eps = [1.00, 1.30, 1.69, 2.20, 2.86, 2.86]
        result = screen_with(fake_statement(eps, YEARS_2020_2025))
        assert result.passed is False
        assert result.reason == "below_threshold"
        assert result.up_years == 0


class TestAnnualEpsCriterionNegativeBase:
    @pytest.mark.parametrize("base", [-0.50, 0.00])
    def test_negative_or_zero_base_year_fails(self, base):
        eps = [base, 0.80, 1.10, 1.40, 1.80, 2.30]
        result = screen_with(fake_statement(eps, YEARS_2020_2025))
        assert result.passed is False
        assert result.reason == "negative_base"
        assert result.cagr is None


class TestAnnualEpsMissingData:
    def test_no_annual_data_fails_honestly(self):
        result = screen_with(pd.DataFrame())
        assert result.passed is False
        assert result.reason == "no_annual_data"
        assert result.eps_by_year == ()

    def test_missing_eps_row_fails(self):
        stmt = pd.DataFrame({datetime(2025, 12, 31): [None]}, index=["Diluted EPS"])
        result = screen_with(stmt)
        assert result.passed is False
        assert result.reason == "missing_eps"

    def test_too_few_years_fails(self):
        stmt = fake_statement([1.00], [(2025, 12, 31)])
        result = screen_with(stmt)
        assert result.passed is False
        assert result.reason == "too_few_years"


class TestAnnualEpsPureFunctions:
    def test_extract_sorts_by_date_and_drops_nan(self):
        stmt = fake_statement([1.30, None, 1.00], [(2021, 12, 31), (2022, 12, 31), (2020, 12, 31)])
        pairs = extract_annual_eps(stmt)
        assert [p.date().year for p, _ in pairs] == [2020, 2021]
        assert [eps for _, eps in pairs] == [1.00, 1.30]

    def test_extract_empty_for_no_statement(self):
        assert extract_annual_eps(pd.DataFrame()) == ()

    def test_compute_cagr_caps_window_at_five_years(self):
        eps = [0.50] + CAGR_30_YEARS
        cagr, reason = compute_cagr(tuple(eps))
        assert reason is None
        assert cagr == pytest.approx(3.71 ** 0.2 - 1)

    def test_compute_cagr_too_few_years(self):
        assert compute_cagr((1.00,)) == (None, "too_few_years")

    def test_compute_cagr_negative_base(self):
        assert compute_cagr((-0.50, 1.00, 1.30)) == (None, "negative_base")

    def test_consecutive_up_years_streak(self):
        assert consecutive_up_years(tuple(CAGR_30_YEARS)) == 5
        assert consecutive_up_years((1.00, 1.30, 0.90, 1.60, 2.20, 3.00)) == 3
        assert consecutive_up_years((1.00, 2.00, 3.00, 2.50)) == 0


class TestAnnualEpsWindowAndFetchSeam:
    def test_only_recent_five_years_are_scored(self):
        eps = [2.00, 1.00, 1.30, 1.69, 2.20, 2.86, 3.71]
        dates = [(2019, 12, 31)] + YEARS_2020_2025
        result = screen_with(fake_statement(eps, dates))
        assert result.passed is True
        assert result.total_years == 5
        assert result.up_years == 5

    def test_screen_patches_yfinance_ticker_like_criterion_c(self):
        stmt = fake_statement(CAGR_30_YEARS, YEARS_2020_2025)
        with patch("yfinance.Ticker") as ticker:
            ticker.return_value.income_stmt = stmt
            result = screen_annual_eps("VRTX")
        assert result.passed is True


class TestAnnualEpsCli:
    def test_cli_pass_exit_zero(self, capsys):
        stmt = fake_statement(CAGR_30_YEARS, YEARS_2020_2025)
        with patch("yfinance.Ticker") as ticker:
            ticker.return_value.income_stmt = stmt
            exit_code = main(["VRTX"])
        out = capsys.readouterr().out
        assert exit_code == 0
        assert "PASS  VRTX  criterion A" in out
        assert "30.0%" in out
        assert "5/5 up YoY" in out

    def test_cli_fail_exit_one(self, capsys):
        stmt = fake_statement([1.00, 1.05, 1.10, 1.15, 1.20, 1.20], YEARS_2020_2025)
        with patch("yfinance.Ticker") as ticker:
            ticker.return_value.income_stmt = stmt
            exit_code = main(["VRTX"])
        out = capsys.readouterr().out
        assert exit_code == 1
        assert "FAIL  VRTX  criterion A" in out
        assert "below_threshold" in out

    def test_cli_min_cagr_override_fails_30_percent_name_at_40(self):
        stmt = fake_statement(CAGR_30_YEARS, YEARS_2020_2025)
        with patch("yfinance.Ticker") as ticker:
            ticker.return_value.income_stmt = stmt
            exit_code = main(["VRTX", "--min-cagr", "0.40"])
        assert exit_code == 1