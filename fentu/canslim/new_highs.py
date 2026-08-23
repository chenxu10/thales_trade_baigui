"""CANSLIM criterion N: something new — new highs, new developments (O'Neil).

O'Neil on the "N" (Market Wizards, 2006): it stands for something new — a new
product or service, a change in the industry, or new management — and for a
new high price. 95% of the greatest winners had something new in those
categories, and yet 98% of investors refuse to buy a stock at a new high; "what
seems too high usually goes higher." This module scores one ticker on both legs:

    1. Price: latest close within `tolerance` (default 5%) of the trailing
       52-week high (max High over the year of daily history).
    2. News: at least one item published in the last `news_days` (default 30)
       whose title matches a development keyword set (fda, approval, phase,
       trial, license, partnership, breakthrough, ind, ...) — a proxy for a
       new product, an industry change, or new management.

PASS iff the new-high signal OR at least one fresh development news item.
Final verdict reasons: "no_price_history", "no_news", "below_high_and_no_news",
or "" on PASS. The individual signal reasons "below_high" and
"no_fresh_development" are exposed by score_new_high / score_fresh_news.

Usage:
    uv run python -m fentu.canslim.new_highs VRTX
    uv run python -m fentu.canslim.new_highs VRTX --tolerance 0.03 --news-days 45
"""
import argparse
import sys
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import pandas as pd

DEFAULT_TOLERANCE = 0.05
DEFAULT_NEWS_DAYS = 30
DEVELOPMENT_KEYWORDS = (
    "fda",
    "approval",
    "approved",
    "phase",
    "trial",
    "license",
    "licensing",
    "partnership",
    "breakthrough",
    "ind",
    "fda-approved",
    "510k",
    "label",
    "guidance",
)
_SECONDS_PER_DAY = 86400


@dataclass(frozen=True)
class NewHighsResult:
    ticker: str
    high: Optional[float]
    close: Optional[float]
    distance_from_high: Optional[float]
    days_since_high: Optional[int]
    fresh_news_count: int
    passed: bool
    reason: str


def fetch_price_history(ticker: str) -> pd.DataFrame:
    import yfinance as yf

    return yf.Ticker(ticker).history(period="1y", auto_adjust=False)


def fetch_news(ticker: str) -> list:
    import yfinance as yf

    return yf.Ticker(ticker).news


def compute_high_stats(history: pd.DataFrame) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[int], Optional[str]]:
    """Pure: 52-week high, latest close, distance from the high, days since it was set.

    Returns (high, close, distance_from_high, days_since_high, reason); reason is
    "no_price_history" when the history is unusable, else None.
    """
    if history is None or history.empty or "High" not in history.columns or "Close" not in history.columns:
        return None, None, None, None, "no_price_history"
    high = float(history["High"].max())
    close = float(history["Close"].iloc[-1])
    if pd.isna(high) or pd.isna(close):
        return None, None, None, None, "no_price_history"
    distance = (high - close) / high
    high_date = history["High"].idxmax()
    days_since_high = (history.index[-1] - high_date).days
    return high, close, distance, days_since_high, None


def score_new_high(
    history: pd.DataFrame, tolerance: float = DEFAULT_TOLERANCE
) -> Tuple[bool, Optional[float], Optional[float], Optional[float], Optional[int], Optional[str]]:
    """Pure: new-high signal, latest close within `tolerance` of the 52-week high.

    Returns (passed, high, close, distance_from_high, days_since_high, reason);
    reason is "below_high" when the close is beyond tolerance, None on a new high.
    """
    high, close, distance, days_since_high, reason = compute_high_stats(history)
    if reason is not None:
        return False, high, close, distance, days_since_high, reason
    passed = distance <= tolerance
    return passed, high, close, distance, days_since_high, None if passed else "below_high"


def score_fresh_news(
    news: list, news_days: int = DEFAULT_NEWS_DAYS, now: Optional[float] = None
) -> Tuple[int, bool, Optional[str]]:
    """Pure: count news items from the last `news_days` matching a development keyword.

    Returns (fresh_news_count, passed, reason); reason is "no_news" when the news
    data is absent, "no_fresh_development" when nothing fresh matches, else None.
    """
    if not news:
        return 0, False, "no_news"
    current = time.time() if now is None else now
    cutoff = current - news_days * _SECONDS_PER_DAY

    def is_fresh(item: dict) -> bool:
        published = item.get("providerPublishTime")
        if not isinstance(published, (int, float)):
            return False
        title = (item.get("title") or "").lower()
        return published >= cutoff and any(keyword in title for keyword in DEVELOPMENT_KEYWORDS)

    count = sum(1 for item in news if is_fresh(item))
    return count, count >= 1, None if count >= 1 else "no_fresh_development"


def _verdict_reason(passed: bool, new_high_reason: Optional[str], news_reason: Optional[str]) -> str:
    if passed:
        return ""
    if new_high_reason == "no_price_history":
        return "no_price_history"
    if news_reason == "no_news":
        return "no_news"
    return "below_high_and_no_news"


def screen_new_highs(
    ticker: str,
    tolerance: float = DEFAULT_TOLERANCE,
    news_days: int = DEFAULT_NEWS_DAYS,
    now: Optional[float] = None,
    fetch_history: Callable[[str], pd.DataFrame] = fetch_price_history,
    fetch_news: Callable[[str], list] = fetch_news,
) -> NewHighsResult:
    """Score one ticker on criterion N; fetch functions are injectable for tests."""
    history = fetch_history(ticker)
    news = fetch_news(ticker)
    passed_high, high, close, distance, days_since_high, high_reason = score_new_high(history, tolerance)
    fresh_count, passed_news, news_reason = score_fresh_news(news, news_days, now)
    passed = passed_high or passed_news
    return NewHighsResult(
        ticker=ticker,
        high=high,
        close=close,
        distance_from_high=distance,
        days_since_high=days_since_high,
        fresh_news_count=fresh_count,
        passed=passed,
        reason=_verdict_reason(passed, high_reason, news_reason),
    )


def _format_price(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.2f}"


def _format_distance(value: Optional[float]) -> str:
    return "-" if value is None else f"{value * 100:.1f}%"


def _format_days(value: Optional[int]) -> str:
    return "-" if value is None else str(value)


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="CANSLIM criterion N screen (52-week new highs / fresh developments)")
    parser.add_argument("ticker", help="yfinance ticker, e.g. VRTX")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE, help="max distance from the 52-week high to count as a new high (default 0.05)")
    parser.add_argument("--news-days", type=int, default=DEFAULT_NEWS_DAYS, help="fresh-development news lookback window in days (default 30)")
    args = parser.parse_args(argv)

    result = screen_new_highs(args.ticker, tolerance=args.tolerance, news_days=args.news_days)
    verdict = "PASS" if result.passed else "FAIL"
    print(f"{verdict}  {result.ticker}  criterion N (new high within {args.tolerance * 100:.0f}%, news window {args.news_days}d)")
    print(f"  reason : {result.reason or 'new high or fresh development'}")
    print(f"  52w high: {_format_price(result.high)}  close {_format_price(result.close)}  distance {_format_distance(result.distance_from_high)}  days since high {_format_days(result.days_since_high)}")
    print(f"  fresh news: {result.fresh_news_count}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
