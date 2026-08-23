"""CANSLIM criterion N: something new — new highs, new developments (O'Neil).

O'Neil on the "N" (Market Wizards, 2006): it stands for something new — a new
product or service, a change in the industry, or new management — and for a
new high price. 95% of the greatest winners had something new; yet "98
percent of investors are afraid to buy a stock that is beginning to go into
new high ground... what seems too high usually goes higher and what seems
too low usually goes lower."

Scoring model — the price leg is the gate, the news leg confirms or vetoes:

    Price gate (all required):
        1. Latest close within `tolerance` (default 5%) of the 52-week
           closing high. O'Neil reads closing prices; a stock drifting
           sideways under a months-old high is not "emerging from a base".
        2. That high is recent: set within `max_high_age_days` (default 30).
           O'Neil buys the breakout, not the aftermath — `days_since_high`
           separates a fresh breakout from a stale one.
        3. Breakout volume >= `volume_multiplier` (default 1.5x) the mean of
           the prior `volume_baseline` sessions. "When a stock is beginning
           to move into new high ground, volume should increase by at least
           50 percent over the average daily volume in recent months." A
           low-volume new high is a warning (Ryan: "if the volume is only up
           10 percent, I would be wary"). Unmeasurable volume (no Volume
           column / no baseline) does not veto.

    News leg (pure headline classification, word-boundary matched):
        - positive: fda approval / partnership / launch / ... → confirms
        - negative: fail / reject / clinical hold / dilution / ... → veto
          (a crashed biotech with a failed readout is NOT criterion N)
        - neutral (earnings transcripts, price notes) → ignored

    PASS iff the price gate holds AND no fresh negative headline.

Failure reasons: "no_price_history", "below_high", "stale_high",
"low_volume_breakout", "negative_news"; "" on PASS.

Usage:
    uv run python -m fentu.canslim.new_highs VRTX
    uv run python -m fentu.canslim.new_highs VRTX --tolerance 0.03 --max-high-age-days 45
"""
import argparse
import re
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Callable, Optional, Tuple

import pandas as pd

DEFAULT_TOLERANCE = 0.05
DEFAULT_NEWS_DAYS = 30
DEFAULT_MAX_HIGH_AGE_DAYS = 30
DEFAULT_VOLUME_MULTIPLIER = 1.5
VOLUME_BASELINE_SESSIONS = 50
_SECONDS_PER_DAY = 86400

POSITIVE_KEYWORDS = (
    "fda approval",
    "approved",
    "approves",
    "approval",
    "clearance",
    "cleared",
    "breakthrough",
    "fast track",
    "priority review",
    "partnership",
    "collaboration",
    "licensing",
    "license",
    "launch",
    "positive",
    "510k",
    "ind",
)
NEGATIVE_KEYWORDS = (
    "fail",
    "fails",
    "failed",
    "failure",
    "miss",
    "missed",
    "misses",
    "reject",
    "rejected",
    "rejection",
    "complete response letter",
    "crl",
    "halt",
    "halted",
    "clinical hold",
    "downgrade",
    "dilution",
    "dilutive",
    "offering",
    "recall",
    "warning letter",
    "lawsuit",
)


def _keyword_regex(keywords: Tuple[str, ...]) -> "re.Pattern":
    return re.compile(r"\b(?:%s)\b" % "|".join(map(re.escape, keywords)), re.IGNORECASE)


_POSITIVE_RE = _keyword_regex(POSITIVE_KEYWORDS)
_NEGATIVE_RE = _keyword_regex(NEGATIVE_KEYWORDS)


@dataclass(frozen=True)
class PriceSignal:
    """52-week closing-high stats plus the price-gate verdict."""

    high: Optional[float]
    close: Optional[float]
    distance_from_high: Optional[float]
    days_since_high: Optional[int]
    volume_confirmed: Optional[bool]
    reason: Optional[str]


_NO_HISTORY_SIGNAL = PriceSignal(None, None, None, None, None, "no_price_history")


@dataclass(frozen=True)
class NewsSignal:
    """Fresh-headline counts by class within the lookback window."""

    positive: int
    negative: int
    reason: Optional[str]


@dataclass(frozen=True)
class NewHighsResult:
    ticker: str
    high: Optional[float]
    close: Optional[float]
    distance_from_high: Optional[float]
    days_since_high: Optional[int]
    volume_confirmed: Optional[bool]
    fresh_news_count: int
    fresh_negative_count: int
    passed: bool
    reason: str


def _yf_ticker(ticker: str):
    """Lazy yfinance import keeps the module importable without network deps."""
    import yfinance as yf

    return yf.Ticker(ticker)


def fetch_price_history(ticker: str) -> pd.DataFrame:
    return _yf_ticker(ticker).history(period="1y", auto_adjust=True)


def fetch_news(ticker: str) -> list:
    return _yf_ticker(ticker).news


def _volume_confirmation(history: pd.DataFrame, breakout_idx: int, volume_multiplier: float) -> Optional[bool]:
    """Breakout-day volume vs the mean of the prior baseline sessions; None when unmeasurable."""
    if "Volume" not in history.columns:
        return None
    baseline = history["Volume"].iloc[max(0, breakout_idx - VOLUME_BASELINE_SESSIONS):breakout_idx]
    baseline_mean = baseline.mean()
    if baseline.empty or pd.isna(baseline_mean) or baseline_mean <= 0:
        return None
    breakout_volume = history["Volume"].iloc[breakout_idx]
    return bool(breakout_volume >= volume_multiplier * baseline_mean)


def compute_high_stats(
    history: pd.DataFrame,
    volume_multiplier: float = DEFAULT_VOLUME_MULTIPLIER,
) -> PriceSignal:
    """Pure: 52-week closing high, latest close, distance, recency, volume confirmation."""
    usable = history is not None and not history.empty and {"High", "Close"} <= set(getattr(history, "columns", []))
    if not usable:
        return _NO_HISTORY_SIGNAL
    closes = history["Close"].dropna()
    if closes.empty:
        return _NO_HISTORY_SIGNAL
    breakout_idx = int(closes.index.get_indexer([closes.idxmax()])[0])
    high = float(closes.max())
    close = float(closes.iloc[-1])
    return PriceSignal(
        high=high,
        close=close,
        distance_from_high=(high - close) / high,
        days_since_high=int((closes.index[-1] - closes.idxmax()).days),
        volume_confirmed=_volume_confirmation(history, breakout_idx, volume_multiplier),
        reason=None,
    )


def score_new_high(
    history: pd.DataFrame,
    tolerance: float = DEFAULT_TOLERANCE,
    max_high_age_days: int = DEFAULT_MAX_HIGH_AGE_DAYS,
    volume_multiplier: float = DEFAULT_VOLUME_MULTIPLIER,
) -> PriceSignal:
    """Pure: the price gate — near the high, freshly set, on confirmed volume."""
    stats = compute_high_stats(history, volume_multiplier)
    if stats.reason is not None:
        return stats
    failures = (
        ("below_high", stats.distance_from_high > tolerance),
        ("stale_high", stats.days_since_high > max_high_age_days),
        ("low_volume_breakout", stats.volume_confirmed is False),
    )
    return replace(stats, reason=next((name for name, failed in failures if failed), None))


def _to_timestamp(published) -> Optional[float]:
    """Unix seconds from either a unix int/float or an ISO-8601 string (old/new yfinance)."""
    if isinstance(published, (int, float)) and not pd.isna(published):
        return float(published)
    if isinstance(published, str):
        try:
            parsed = datetime.fromisoformat(published.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    return None


def parse_news_item(item) -> Optional[Tuple[str, float]]:
    """Pure: (title, unix_ts) from either yfinance schema — content-pubDate or legacy top-level."""
    if not isinstance(item, dict):
        return None
    content = item.get("content") if isinstance(item.get("content"), dict) else {}
    title = content.get("title") or item.get("title")
    timestamp = _to_timestamp(content.get("pubDate") or item.get("providerPublishTime"))
    return (str(title), timestamp) if title and timestamp is not None else None


def classify_headline(title: str) -> str:
    """Pure: 'negative' | 'positive' | 'neutral' — negative wins ties (a failed trial is not criterion N)."""
    if _NEGATIVE_RE.search(title):
        return "negative"
    if _POSITIVE_RE.search(title):
        return "positive"
    return "neutral"


def score_fresh_news(
    news, news_days: int = DEFAULT_NEWS_DAYS, now: Optional[float] = None
) -> NewsSignal:
    """Pure: count fresh (within `news_days`) headlines by class; word-boundary keyword matching."""
    if not news:
        return NewsSignal(0, 0, "no_news")
    current = time.time() if now is None else now
    cutoff = current - news_days * _SECONDS_PER_DAY
    headlines = (parse_news_item(item) for item in news)
    fresh = tuple(classify_headline(title) for title, ts in headlines if ts is not None and ts >= cutoff)
    return NewsSignal(
        positive=sum(1 for c in fresh if c == "positive"),
        negative=sum(1 for c in fresh if c == "negative"),
        reason=None,
    )


def _verdict(price: PriceSignal, news: NewsSignal) -> Tuple[bool, str]:
    """Price gate first; fresh negative headlines veto a passing price leg."""
    if price.reason is not None:
        return False, price.reason
    if news.negative > 0:
        return False, "negative_news"
    return True, ""


def screen_new_highs(
    ticker: str,
    tolerance: float = DEFAULT_TOLERANCE,
    news_days: int = DEFAULT_NEWS_DAYS,
    max_high_age_days: int = DEFAULT_MAX_HIGH_AGE_DAYS,
    volume_multiplier: float = DEFAULT_VOLUME_MULTIPLIER,
    now: Optional[float] = None,
    fetch_history: Callable[[str], pd.DataFrame] = fetch_price_history,
    fetch_news: Callable[[str], list] = fetch_news,
) -> NewHighsResult:
    """Score one ticker on criterion N; fetch functions are injectable for tests."""
    price = score_new_high(fetch_history(ticker), tolerance, max_high_age_days, volume_multiplier)
    news = score_fresh_news(fetch_news(ticker), news_days, now)
    passed, reason = _verdict(price, news)
    return NewHighsResult(
        ticker=ticker,
        high=price.high,
        close=price.close,
        distance_from_high=price.distance_from_high,
        days_since_high=price.days_since_high,
        volume_confirmed=price.volume_confirmed,
        fresh_news_count=news.positive,
        fresh_negative_count=news.negative,
        passed=passed,
        reason=reason,
    )


def _dash_for_none(value, render):
    return "-" if value is None else render(value)


def _format_price(value: Optional[float]) -> str:
    return _dash_for_none(value, lambda v: f"{v:.2f}")


def _format_distance(value: Optional[float]) -> str:
    return _dash_for_none(value, lambda v: f"{v * 100:.1f}%")


def _format_days(value: Optional[int]) -> str:
    return _dash_for_none(value, str)


def _format_flag(value: Optional[bool]) -> str:
    return {True: "yes", False: "NO", None: "n/a"}[value]


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="CANSLIM criterion N screen (fresh new highs + headline confirmation)")
    parser.add_argument("ticker", help="yfinance ticker, e.g. VRTX")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE, help="max distance from the 52-week closing high (default 0.05)")
    parser.add_argument("--news-days", type=int, default=DEFAULT_NEWS_DAYS, help="fresh-headline lookback window in days (default 30)")
    parser.add_argument("--max-high-age-days", type=int, default=DEFAULT_MAX_HIGH_AGE_DAYS, help="max age of the 52-week high to count as a fresh breakout (default 30)")
    parser.add_argument("--volume-multiplier", type=float, default=DEFAULT_VOLUME_MULTIPLIER, help="breakout-day volume vs prior-session mean required to confirm (default 1.5)")
    args = parser.parse_args(argv)

    result = screen_new_highs(
        args.ticker,
        tolerance=args.tolerance,
        news_days=args.news_days,
        max_high_age_days=args.max_high_age_days,
        volume_multiplier=args.volume_multiplier,
    )
    verdict = "PASS" if result.passed else "FAIL"
    print(f"{verdict}  {result.ticker}  criterion N (within {args.tolerance * 100:.0f}%, high <= {args.max_high_age_days}d old, volume >= {args.volume_multiplier}x)")
    print(f"  reason : {result.reason or 'fresh new high, no negative headlines'}")
    print(f"  52w close high {_format_price(result.high)}  close {_format_price(result.close)}  distance {_format_distance(result.distance_from_high)}  high set {_format_days(result.days_since_high)}d ago")
    print(f"  volume confirmed: {_format_flag(result.volume_confirmed)}")
    print(f"  fresh headlines: {result.fresh_news_count} positive, {result.fresh_negative_count} negative")
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
