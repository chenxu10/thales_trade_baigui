"""CANSLIM criterion C screen: pull quarterly EPS from yfinance and apply it.

Usage:
    uv run python -m fentu.canslim.screen 3696.HK
    uv run python -m fentu.canslim.screen 3696.HK --min-growth 0.25
    uv run python -m fentu.canslim.screen ONC    # BeOne Medicines (ex-BeiGene), Nasdaq
"""
import argparse
import sys
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

import pandas as pd

from fentu.canslim.current_eps import CurrentEpsResult, score_current_eps

DILUTED_EPS_ROW = "Diluted EPS"
SAME_QUARTER_TOLERANCE_DAYS = 100


@dataclass(frozen=True)
class ScreenResult:
    ticker: str
    current_eps: Optional[float]
    prior_year_eps: Optional[float]
    current_period: Optional[str]
    prior_period: Optional[str]
    growth: Optional[float]
    passed: bool
    reason: str


def fetch_quarterly_eps(ticker: str) -> pd.DataFrame:
    import yfinance as yf

    return yf.Ticker(ticker).quarterly_income_stmt


def align_same_quarter_last_year(income_stmt: pd.DataFrame, tolerance_days: int = SAME_QUARTER_TOLERANCE_DAYS) -> tuple:
    """Most recent period end and the period nearest one year earlier, or None if unmatched."""
    periods = sorted(income_stmt.columns)
    current_period = periods[-1]
    target = current_period - timedelta(days=365)
    candidates = [c for c in periods[:-1] if abs((c - target).days) <= tolerance_days]
    if not candidates:
        return current_period, None
    prior_period = min(candidates, key=lambda c: abs((c - target).days))
    return current_period, prior_period


def _failure(ticker: str, reason: str, current_period: Optional[str] = None, prior_period: Optional[str] = None) -> ScreenResult:
    return ScreenResult(
        ticker=ticker,
        current_eps=None,
        prior_year_eps=None,
        current_period=current_period,
        prior_period=prior_period,
        growth=None,
        passed=False,
        reason=reason,
    )


def _screen_reason(result: CurrentEpsResult) -> str:
    if result.reason is not None:
        return result.reason
    return "" if result.passed else "below_threshold"


def _guard_quarterly_data(ticker: str, income_stmt: pd.DataFrame):
    """Return (aligned EPS pair, None) or (None, failure result) for every bail-out path."""
    if income_stmt is None or income_stmt.empty or DILUTED_EPS_ROW not in income_stmt.index:
        return None, _failure(ticker, "no_quarterly_data")
    eps_series = income_stmt.loc[DILUTED_EPS_ROW]
    current_period, prior_period = align_same_quarter_last_year(income_stmt)
    if prior_period is None:
        return None, _failure(ticker, "no_prior_year_quarter", current_period=str(current_period.date()))
    current_eps = float(eps_series[current_period])
    prior_year_eps = float(eps_series[prior_period])
    if pd.isna(current_eps) or pd.isna(prior_year_eps):
        return None, _failure(
            ticker,
            "missing_eps",
            current_period=str(current_period.date()),
            prior_period=str(prior_period.date()),
        )
    return (current_eps, prior_year_eps, str(current_period.date()), str(prior_period.date())), None


def screen_current_eps(ticker: str, min_growth: float = 0.20) -> ScreenResult:
    income_stmt = fetch_quarterly_eps(ticker)
    data, failure = _guard_quarterly_data(ticker, income_stmt)
    if failure is not None:
        return failure
    current_eps, prior_year_eps, current_period, prior_period = data
    result = score_current_eps(current_eps, prior_year_eps, min_growth)
    return ScreenResult(
        ticker=ticker,
        current_eps=current_eps,
        prior_year_eps=prior_year_eps,
        current_period=current_period,
        prior_period=prior_period,
        growth=result.growth,
        passed=result.passed,
        reason=_screen_reason(result),
    )


def _format_growth(growth: Optional[float]) -> str:
    if growth is None:
        return "-"
    return f"{growth * 100:.1f}%"


def _format_eps(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return str(value)


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="CANSLIM criterion C screen (current quarterly EPS)")
    parser.add_argument("ticker", help="yfinance ticker, e.g. 3696.HK")
    parser.add_argument("--min-growth", type=float, default=0.20, help="YoY growth threshold (default 0.20)")
    args = parser.parse_args(argv)

    result = screen_current_eps(args.ticker, args.min_growth)
    verdict = "PASS" if result.passed else "FAIL"
    print(f"{verdict}  {result.ticker}  criterion C (min growth {args.min_growth * 100:.0f}%)")
    print(f"  reason : {result.reason or 'meets threshold'}")
    print(f"  quarter: {result.current_period or '-'}  EPS {_format_eps(result.current_eps)}")
    print(f"  prior  : {result.prior_period or '-'}  EPS {_format_eps(result.prior_year_eps)}")
    print(f"  growth : {_format_growth(result.growth)}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())