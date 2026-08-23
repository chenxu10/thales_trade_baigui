"""CANSLIM criterion S: shares outstanding (O'Neil, Market Wizards 2006).

"The 'S' in the formula stands for shares outstanding. Ninety-five percent of
the stocks that performed best in our studies had less than twenty-five
million shares of capitalization during the period when they had their best
performance. The average capitalization of all of these stocks was 11.8
million shares, while the median figure was only 4.6 million. Many
institutional investors handicap themselves by restricting their purchases to
only large-capitalization companies. By doing so, they automatically eliminate
some of the best growth companies."

The default verdict compares the latest share count against the 25M line
(``--max-shares``, default 25,000,000 — O'Neil's line; the study average was
11.8M and the median 4.6M). Because O'Neil measures capitalization "during
the period when they had their best performance", ``--use-trailing-min``
switches the verdict to the minimum share count observed over the trailing 12
months (252 days) — a stricter reading that catches names which only recently
diluted above the line.

Usage:
    uv run python -m fentu.canslim.shares_outstanding VRTX
    uv run python -m fentu.canslim.shares_outstanding VRTX --max-shares 11800000
    uv run python -m fentu.canslim.shares_outstanding VRTX --use-trailing-min
"""
import argparse
import sys
from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd

DEFAULT_MAX_SHARES = 25_000_000
TRAILING_WINDOW_DAYS = 252
FAIL_TOO_MANY_SHARES = "too_many_shares"
FAIL_NO_DATA = "no_share_data"


@dataclass(frozen=True)
class SharesResult:
    ticker: str
    shares: Optional[float]
    as_of: Optional[str]
    passed: bool
    reason: str


def fetch_shares_history(ticker: str) -> Optional[pd.Series]:
    """Share count history (datetime index -> share counts), or None if both sources fail.

    Primary source is ``get_shares_full`` (roughly the last 18 months); falls
    back to ``get_shares`` when it is None or empty, normalizing DataFrame /
    dict payloads to a Series.
    """
    import yfinance as yf

    ticker_obj = yf.Ticker(ticker)
    series = ticker_obj.get_shares_full()
    if series is None or len(series) == 0:
        series = ticker_obj.get_shares()
    if series is None or len(series) == 0:
        return None
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    if isinstance(series, dict):
        series = pd.Series(series)
    return series


def _as_of(value) -> str:
    return str(pd.Timestamp(value).date())


def latest_share_count(shares: Optional[pd.Series]) -> tuple:
    """Last non-null share count with its as-of date, or (None, None)."""
    if shares is None:
        return None, None
    valid = shares.dropna()
    if valid.empty:
        return None, None
    return float(valid.iloc[-1]), _as_of(valid.index[-1])


def trailing_min_share_count(shares: Optional[pd.Series], window_days: int = TRAILING_WINDOW_DAYS) -> tuple:
    """Lowest share count over the trailing ``window_days`` calendar days, with its date.

    Mirrors ``min(series[-252d:])``; a window with no data yields (None, None).
    """
    if shares is None or shares.empty:
        return None, None
    last = pd.Timestamp(shares.index[-1])
    window = shares[shares.index >= last - pd.Timedelta(days=window_days)]
    valid = window.dropna()
    if valid.empty:
        return None, None
    when = valid.idxmin()
    return float(valid[when]), _as_of(when)


def screen_shares_outstanding(
    ticker: str,
    max_shares: float = DEFAULT_MAX_SHARES,
    use_trailing_min: bool = False,
    fetch: Callable[[str], Optional[pd.Series]] = fetch_shares_history,
) -> SharesResult:
    """Pure screen: shares outstanding against the O'Neil capitalization cap."""
    series = fetch(ticker)
    shares, when = trailing_min_share_count(series) if use_trailing_min else latest_share_count(series)
    if shares is None:
        return SharesResult(ticker=ticker, shares=None, as_of=None, passed=False, reason=FAIL_NO_DATA)
    passed = shares <= max_shares
    return SharesResult(
        ticker=ticker,
        shares=shares,
        as_of=when,
        passed=passed,
        reason="" if passed else FAIL_TOO_MANY_SHARES,
    )


def _format_shares(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:,.0f}"


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="CANSLIM criterion S screen (shares outstanding)")
    parser.add_argument("ticker", help="yfinance ticker, e.g. VRTX")
    parser.add_argument(
        "--max-shares",
        type=float,
        default=DEFAULT_MAX_SHARES,
        help="capitalization cap in shares (O'Neil 25M line; average 11.8M, median 4.6M; default 25000000)",
    )
    parser.add_argument(
        "--use-trailing-min",
        action="store_true",
        help="use the min share count over the trailing 12 months instead of the latest",
    )
    args = parser.parse_args(argv)

    result = screen_shares_outstanding(args.ticker, max_shares=args.max_shares, use_trailing_min=args.use_trailing_min)
    mode = "trailing-12m min" if args.use_trailing_min else "latest"
    verdict = "PASS" if result.passed else "FAIL"
    print(f"{verdict}  {result.ticker}  criterion S ({mode} shares, cap {args.max_shares:,.0f})")
    print(f"  reason : {result.reason or 'meets threshold'}")
    print(f"  shares : {_format_shares(result.shares)}")
    print(f"  as-of  : {result.as_of or '-'}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
