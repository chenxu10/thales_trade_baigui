"""CANSLIM criterion A screen: annual EPS growth from the yfinance income statement.

O'Neil (interviewed in Schwager's Market Wizards, 2006): "The 'A' in our
formula stands for annual earnings per share. In our studies, the prior
five-year average annual compounded earnings growth rate of outstanding
performing stocks at their early emerging stage was 24 percent. Ideally,
each year's earnings per share should show an increase over the prior year's
earnings. It is a unique combination of both strong current earnings and
high average earnings growth that creates a superb stock."

PASS requires BOTH:
    - a 5-year average annual COMPOUNDED EPS growth rate >= --min-cagr
      (default 0.24), computed as the geometric mean of year-over-year
      growth factors over the most recent up-to-5 years, and
    - every year's EPS up over the prior year's (--min-up-years, default:
      all years in the window), measured as the consecutive streak of YoY
      increases ending at the most recent fiscal year.

A negative or zero EPS year inside the window always fails with reason
"negative_base" — the geometric mean is undefined over a loss-making base,
the same trap criterion C guards against. Gaps (NaN cells) in the Diluted
EPS row are dropped before scoring.

Usage:
    uv run python -m fentu.canslim.annual_eps VRTX
    uv run python -m fentu.canslim.annual_eps VRTX --min-cagr 0.20 --min-up-years 3
"""
import argparse
import itertools
import operator
import sys
from dataclasses import dataclass
from functools import reduce
from typing import Callable, Optional

import pandas as pd

DILUTED_EPS_ROW = "Diluted EPS"
DEFAULT_MIN_CAGR = 0.24
MAX_CAGR_YEARS = 5
MIN_ANNUAL_POINTS = 2
_EPSILON = 1e-9


@dataclass(frozen=True)
class AnnualEpsResult:
    ticker: str
    periods: tuple
    eps_by_year: tuple
    cagr: Optional[float]
    up_years: int
    total_years: int
    passed: bool
    reason: str


def fetch_annual_eps(ticker: str) -> pd.DataFrame:
    import yfinance as yf

    return yf.Ticker(ticker).income_stmt


def extract_annual_eps(income_stmt: pd.DataFrame) -> tuple:
    """(period, EPS) pairs sorted by fiscal period end date, most recent last; NaN cells dropped."""
    if income_stmt is None or income_stmt.empty or DILUTED_EPS_ROW not in income_stmt.index:
        return ()
    return tuple(
        (period, float(eps))
        for period, eps in sorted(income_stmt.loc[DILUTED_EPS_ROW].items(), key=lambda item: item[0])
        if not pd.isna(eps)
    )


def _growth_window(eps_by_year: tuple, max_years: int = MAX_CAGR_YEARS) -> tuple:
    """Most recent up-to-(max_years + 1) annual EPS points, oldest first."""
    return eps_by_year[-(max_years + 1):]


def compute_cagr(eps_by_year: tuple, max_years: int = MAX_CAGR_YEARS) -> tuple:
    """(cagr, None) or (None, reason): the 5-year average annual COMPOUNDED EPS growth.

    Geometric mean of the year-over-year growth factors over the most recent
    up-to-5 years. "too_few_years" with fewer than MIN_ANNUAL_POINTS points;
    "negative_base" when any year in the window is <= 0 (geometric mean
    undefined over a loss-making base).
    """
    window = _growth_window(eps_by_year, max_years)
    if len(window) < MIN_ANNUAL_POINTS:
        return None, "too_few_years"
    if any(eps <= 0 for eps in window):
        return None, "negative_base"
    factors = [curr / prev for prev, curr in zip(window, window[1:])]
    geometric_mean = reduce(operator.mul, factors, 1.0) ** (1.0 / len(factors))
    return geometric_mean - 1.0, None


def consecutive_up_years(eps_by_year: tuple, max_years: int = MAX_CAGR_YEARS) -> int:
    """Consecutive YoY EPS increases ending at the most recent year (the current streak)."""
    window = _growth_window(eps_by_year, max_years)
    deltas = [curr - prev for prev, curr in zip(window, window[1:])]
    return len(list(itertools.takewhile(lambda delta: delta > 0, reversed(deltas))))


def score_annual_eps(
    cagr: float,
    up_years: int,
    total_years: int,
    min_cagr: float = DEFAULT_MIN_CAGR,
    min_up_years: Optional[int] = None,
) -> tuple:
    """Pure function: threshold test on CAGR plus the YoY up-streak. Returns (passed, reason or None)."""
    threshold = total_years if min_up_years is None else min_up_years
    if cagr < min_cagr - _EPSILON:
        return False, "below_threshold"
    if up_years < threshold:
        return False, "not_every_year_up"
    return True, None


def _failure(ticker: str, reason: str, periods: tuple = (), eps_by_year: tuple = ()) -> AnnualEpsResult:
    return AnnualEpsResult(
        ticker=ticker,
        periods=periods,
        eps_by_year=eps_by_year,
        cagr=None,
        up_years=0,
        total_years=0,
        passed=False,
        reason=reason,
    )


def _guard_annual_data(ticker: str, income_stmt: pd.DataFrame):
    """Return ((periods, eps_by_year), None) or (None, failure result) for every bail-out path."""
    if income_stmt is None or income_stmt.empty or DILUTED_EPS_ROW not in income_stmt.index:
        return None, _failure(ticker, "no_annual_data")
    pairs = extract_annual_eps(income_stmt)
    if not pairs:
        return None, _failure(ticker, "missing_eps")
    periods = tuple(period for period, _ in pairs)
    eps_by_year = tuple(eps for _, eps in pairs)
    if len(eps_by_year) < MIN_ANNUAL_POINTS:
        return None, _failure(ticker, "too_few_years")
    return (periods, eps_by_year), None


def screen_annual_eps(
    ticker: str,
    min_cagr: float = DEFAULT_MIN_CAGR,
    min_up_years: Optional[int] = None,
    fetch: Callable[[str], pd.DataFrame] = fetch_annual_eps,
) -> AnnualEpsResult:
    income_stmt = fetch(ticker)
    data, failure = _guard_annual_data(ticker, income_stmt)
    if failure is not None:
        return failure
    periods, eps_by_year = data
    cagr, cagr_reason = compute_cagr(eps_by_year)
    if cagr_reason is not None:
        return _failure(ticker, cagr_reason, periods=periods, eps_by_year=eps_by_year)
    total_years = len(_growth_window(eps_by_year)) - 1
    up_years = consecutive_up_years(eps_by_year)
    passed, reason = score_annual_eps(cagr, up_years, total_years, min_cagr, min_up_years)
    return AnnualEpsResult(
        ticker=ticker,
        periods=periods,
        eps_by_year=eps_by_year,
        cagr=cagr,
        up_years=up_years,
        total_years=total_years,
        passed=passed,
        reason=reason or "",
    )


def _format_cagr(cagr: Optional[float]) -> str:
    if cagr is None:
        return "-"
    return f"{cagr * 100:.1f}%"


def _format_periods(periods: tuple) -> str:
    return ", ".join(str(p.date()) for p in periods) or "-"


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="CANSLIM criterion A screen (annual EPS growth)")
    parser.add_argument("ticker", help="yfinance ticker, e.g. VRTX")
    parser.add_argument("--min-cagr", type=float, default=DEFAULT_MIN_CAGR, help="5-year compounded annual EPS growth threshold (default 0.24)")
    parser.add_argument("--min-up-years", type=int, default=None, help="required consecutive YoY EPS up-years (default: every year in the window)")
    args = parser.parse_args(argv)

    result = screen_annual_eps(args.ticker, min_cagr=args.min_cagr, min_up_years=args.min_up_years)
    verdict = "PASS" if result.passed else "FAIL"
    print(f"{verdict}  {result.ticker}  criterion A (min cagr {args.min_cagr * 100:.0f}%)")
    print(f"  reason : {result.reason or 'meets threshold'}")
    print(f"  cagr   : {_format_cagr(result.cagr)}")
    print(f"  years  : {result.up_years}/{result.total_years} up YoY")
    print(f"  period : {_format_periods(result.periods)}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
