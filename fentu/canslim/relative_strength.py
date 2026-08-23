"""CANSLIM criterion L screen: twelve-month relative strength (leader vs laggard).

O'Neil (Market Wizards, 2006): the 500 best-performing stocks of 1953-1985 had
an average relative strength of 87 before their major price increase began;
relative strength measures a stock's 12-month price performance against the
rest of the market, and O'Neil restricts purchases to names with relative
strength ranks above 80. This screen implements that rule: PASS iff the
candidate's trailing-12-month return ranks at or above ``--min-rank`` (default
80) within an explicitly supplied universe.

Usage:
    uv run python -m fentu.canslim.relative_strength VRTX --universe PFE MRK REGN GILD
    uv run python -m fentu.canslim.relative_strength VRTX --universe PFE MRK REGN --min-rank 85
"""
import argparse
import math
import sys
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

DEFAULT_MIN_RANK = 80.0
MIN_UNIVERSE_SIZE = 3
"""Smallest usable universe (candidate + 2 comparables) before a rank means anything."""
_EPSILON = 1e-9


@dataclass(frozen=True)
class RelativeStrengthResult:
    ticker: str
    return_12m: Optional[float]
    rank: Optional[float]
    universe_size: int
    passed: bool
    reason: str


def twelve_month_return(closes: Sequence[float]) -> Optional[float]:
    """Percentage change from the first close of the trailing 12 months to the latest close.

    Uses the full span of the series (which may run longer than 252 sessions).
    Returns None when the series is too short or degenerate to price a return.
    """
    prices = [float(c) for c in closes if c is not None and math.isfinite(float(c))]
    if len(prices) < 2 or prices[0] == 0:
        return None
    return (prices[-1] - prices[0]) / prices[0]


def rs_rank(returns: Sequence[float], own_return: float) -> float:
    """Percentile rank 0-100: fraction of the universe with return strictly below own_return, times 100.

    Rank 80 means the stock outperformed 80% of the universe. A stock is never
    counted as below itself; with the own return included in ``returns`` the
    denominator grows by one, which is immaterial for a market-sized universe.
    Returns 0.0 for an empty universe (guarded upstream as "not_enough_universe").
    """
    valid = [r for r in returns if r is not None and math.isfinite(float(r))]
    if not valid:
        return 0.0
    below = sum(1 for r in valid if r < own_return)
    return 100.0 * below / len(valid)


def fetch_universe_closes(tickers: Iterable[str]) -> pd.DataFrame:
    """Trailing-12-month daily closes for a universe (columns=tickers, datetime index).

    Missing tickers simply do not appear as columns; the callers treat an
    absent column as no price history.
    """
    import yfinance as yf

    names = list(tickers)
    if not names:
        return pd.DataFrame()
    closes = yf.download(names, period="1y", auto_adjust=False)["Close"]
    if isinstance(closes, pd.Series):
        closes = closes.to_frame()
    return closes if closes is not None and not closes.empty else pd.DataFrame()


def _prices_for(closes: pd.DataFrame, ticker: str) -> List[float]:
    if ticker not in closes.columns:
        return []
    return [float(value) for value in closes[ticker].dropna()]


def universe_twelve_month_returns(
    closes: pd.DataFrame, tickers: Iterable[str]
) -> List[Tuple[str, Optional[float]]]:
    """Order-preserving map of the 12-month return over the universe columns."""
    return list(map(lambda t: (t, twelve_month_return(_prices_for(closes, t))), tickers))


def _failure(ticker: str, reason: str, universe_size: int = 0) -> RelativeStrengthResult:
    return RelativeStrengthResult(
        ticker=ticker,
        return_12m=None,
        rank=None,
        universe_size=universe_size,
        passed=False,
        reason=reason,
    )


def screen_relative_strength(
    ticker: str,
    universe: Iterable[str],
    min_rank: float = DEFAULT_MIN_RANK,
    fetch: Callable[[Iterable[str]], pd.DataFrame] = fetch_universe_closes,
) -> RelativeStrengthResult:
    """Score one ticker's trailing-12-month return against the universe's.

    The candidate is fetched with the universe (own ticker first, deduped); its
    rank is the fraction of the universe with a strictly lower 12-month return.
    """
    names = list(dict.fromkeys([ticker, *universe]))
    closes = fetch(names)
    returns = dict(universe_twelve_month_returns(closes, names))
    own_return = returns.get(ticker)
    valid = {name: ret for name, ret in returns.items() if ret is not None}
    universe_size = len(valid)
    if own_return is None:
        return _failure(ticker, "no_price_history", universe_size=universe_size)
    if universe_size < MIN_UNIVERSE_SIZE:
        return _failure(ticker, "not_enough_universe", universe_size=universe_size)
    rank = rs_rank(list(valid.values()), own_return)
    passed = rank >= min_rank - _EPSILON
    return RelativeStrengthResult(
        ticker=ticker,
        return_12m=own_return,
        rank=rank,
        universe_size=universe_size,
        passed=passed,
        reason="" if passed else "below_rank_threshold",
    )


def _format_return(value: Optional[float]) -> str:
    return "-" if value is None else f"{value * 100:+.1f}%"


def _format_rank(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.1f}"


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="CANSLIM criterion L screen (12-month relative strength)")
    parser.add_argument("ticker", help="candidate ticker, e.g. VRTX")
    parser.add_argument("--universe", nargs="+", required=True, help="universe tickers to rank against, e.g. PFE MRK REGN GILD")
    parser.add_argument("--min-rank", type=float, default=DEFAULT_MIN_RANK, help="relative strength rank threshold (default 80, O'Neil's line; winners averaged 87)")
    args = parser.parse_args(argv)

    result = screen_relative_strength(args.ticker, args.universe, min_rank=args.min_rank)
    verdict = "PASS" if result.passed else "FAIL"
    print(f"{verdict}  {result.ticker}  criterion L (min rank {args.min_rank:.0f})")
    print(f"  reason : {result.reason or 'meets threshold'}")
    print(f"  12m ret: {_format_return(result.return_12m)}")
    print(f"  rank   : {_format_rank(result.rank)}")
    print(f"  universe: {result.universe_size} names")
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())