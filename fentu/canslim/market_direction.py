"""CANSLIM criterion M: market direction filter (O'Neil, Market Wizards 2006).

"Three out of four stocks will go in the same direction as a significant move
in the market averages" — criterion M gates the whole portfolio, not
individual names. But the residual one-in-four is dominated by sector
effects, so this module runs a DUAL GATE:

    1. General market (^GSPC): O'Neil's M — the broad tape.
    2. Sector (XBI): Ryan's version of M — "how are my leaders doing?" For a
       biotech book, XBI constituents ARE the leaders; biotech routinely
       diverges from the S&P for months at a time.

PASS requires BOTH gates. The market leg scores O'Neil's two top-formation
signals:

    1. The average made a new high on poor demand — the index sits below its
       50-day SMA, or the SMA has stopped rising.
    2. Heavy-volume down days ("distribution days") — volume surges for
       several days with little or no upside price progress. O'Neil: after a
       few such days the market is under pressure. Each session is compared
       to ITS OWN trailing 50-session mean volume (no self-referential
       baseline), and only the most recent quarter of the 25-day window can
       trip the gate — O'Neil cares about clustered, recent distribution,
       not stale one-offs.

Failure reasons, first failing signal wins per leg: "no_price_history",
"below_50ma", "ma_not_rising", "distribution_days_high". Combined verdict
reasons: "market:<reason>" / "sector:<reason>" / "market:<reason> sector:<reason>".

Usage:
    uv run python -m fentu.canslim.market_direction
    uv run python -m fentu.canslim.market_direction --market ^GSPC --sector XBI
    uv run python -m fentu.canslim.market_direction --max-distribution-days 3
"""
import argparse
import sys
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import pandas as pd

DEFAULT_MARKET_INDEX = "^GSPC"
DEFAULT_SECTOR_INDEX = "XBI"
DEFAULT_MAX_DISTRIBUTION_DAYS = 4
DEFAULT_MA_WINDOW = 50
DEFAULT_RISE_LOOKBACK = 5
DEFAULT_DIST_LOOKBACK = 25
DEFAULT_DIST_RECENCY = 10
DEFAULT_VOLUME_MULT = 1.25
DEFAULT_VOLUME_BASELINE = 50


@dataclass(frozen=True)
class MarketResult:
    index: str
    close: Optional[float]
    sma50: Optional[float]
    ma_rising: bool
    distribution_days: int
    passed: bool
    reason: str


@dataclass(frozen=True)
class DualMarketResult:
    market: MarketResult
    sector: MarketResult
    passed: bool
    reason: str


def fetch_index_history(index: str) -> pd.DataFrame:
    """Daily OHLCV for the index (Open/High/Low/Close/Volume), the only I/O."""
    import yfinance as yf

    return yf.Ticker(index).history(period="6mo", auto_adjust=True)


def sma(closes: List[float], window: int) -> List[float]:
    """Rolling simple moving average; leading ``window - 1`` entries are NaN."""
    return pd.Series(closes).rolling(window).mean().tolist()


def _ma_rising(closes: List[float], window: int = DEFAULT_MA_WINDOW, rise_lookback: int = DEFAULT_RISE_LOOKBACK) -> bool:
    if len(closes) < window + rise_lookback:
        return False
    values = sma(closes, window)
    return values[-1] > values[-1 - rise_lookback]


def above_rising_ma(closes: List[float], window: int = DEFAULT_MA_WINDOW, rise_lookback: int = DEFAULT_RISE_LOOKBACK) -> bool:
    """Latest close above the SMA AND the SMA itself is rising (vulnerable rally test)."""
    if len(closes) < window + rise_lookback:
        return False
    values = sma(closes, window)
    return closes[-1] > values[-1] and values[-1] > values[-1 - rise_lookback]


def _is_distribution_day(
    closes: List[float],
    volumes: List[float],
    i: int,
    volume_mult: float,
    volume_baseline: int,
) -> bool:
    """Session i closed down on volume well above ITS OWN trailing baseline mean."""
    if i < 1 or closes[i] >= closes[i - 1]:
        return False
    baseline = volumes[max(0, i - volume_baseline):i]
    if not baseline:
        return False
    baseline_mean = sum(baseline) / len(baseline)
    return baseline_mean > 0 and volumes[i] > volume_mult * baseline_mean


def distribution_days(
    ohlcv: pd.DataFrame,
    lookback: int = DEFAULT_DIST_LOOKBACK,
    volume_mult: float = DEFAULT_VOLUME_MULT,
    volume_baseline: int = DEFAULT_VOLUME_BASELINE,
) -> int:
    """Heavy-volume down sessions in the trailing ``lookback``.

    O'Neil's distribution concept: close below the prior close on volume well
    above the recent average — volume surges with no upside price progress.
    Each day is scored against its own trailing baseline (the scored day is
    excluded), so a huge-volume day cannot dilute its own comparison.
    """
    closes = [float(c) for c in ohlcv["Close"].tolist()]
    volumes = [float(v) for v in ohlcv["Volume"].tolist()]
    start = max(len(closes) - lookback, 1)
    return sum(
        1
        for i in range(start, len(closes))
        if _is_distribution_day(closes, volumes, i, volume_mult, volume_baseline)
    )


def recent_distribution_days(
    ohlcv: pd.DataFrame,
    recency: int = DEFAULT_DIST_RECENCY,
    volume_mult: float = DEFAULT_VOLUME_MULT,
    volume_baseline: int = DEFAULT_VOLUME_BASELINE,
) -> int:
    """Distribution days in only the most recent ``recency`` sessions (the cluster test)."""
    return distribution_days(ohlcv, lookback=recency, volume_mult=volume_mult, volume_baseline=volume_baseline)


def market_verdict(
    closes: List[float],
    ohlcv: pd.DataFrame,
    max_distribution_days: int = DEFAULT_MAX_DISTRIBUTION_DAYS,
    ma_window: int = DEFAULT_MA_WINDOW,
    rise_lookback: int = DEFAULT_RISE_LOOKBACK,
    dist_lookback: int = DEFAULT_DIST_LOOKBACK,
    volume_mult: float = DEFAULT_VOLUME_MULT,
) -> Tuple[bool, str]:
    """Pure sub-signal composition: (passed, reason). First failing signal wins."""
    if not above_rising_ma(closes, ma_window, rise_lookback):
        last_sma = sma(closes, ma_window)[-1]
        if closes[-1] <= last_sma:
            return False, "below_50ma"
        return False, "ma_not_rising"
    if distribution_days(ohlcv, dist_lookback, volume_mult) > max_distribution_days:
        return False, "distribution_days_high"
    return True, ""


def _failure(index: str) -> MarketResult:
    return MarketResult(
        index=index,
        close=None,
        sma50=None,
        ma_rising=False,
        distribution_days=0,
        passed=False,
        reason="no_price_history",
    )


def _sma50(closes: List[float], window: int = DEFAULT_MA_WINDOW) -> Optional[float]:
    values = sma(closes, window)
    last = values[-1]
    return None if pd.isna(last) else float(last)


def market_direction(
    index: str,
    fetch: Callable[[str], pd.DataFrame] = fetch_index_history,
    max_distribution_days: int = DEFAULT_MAX_DISTRIBUTION_DAYS,
) -> MarketResult:
    """Verdict for one index tape: OHLCV -> MarketResult."""
    ohlcv = fetch(index)
    if ohlcv is None or ohlcv.empty or not {"Close", "Volume"}.issubset(ohlcv.columns):
        return _failure(index)
    closes = [float(c) for c in ohlcv["Close"].tolist()]
    passed, reason = market_verdict(closes, ohlcv, max_distribution_days)
    return MarketResult(
        index=index,
        close=closes[-1],
        sma50=_sma50(closes),
        ma_rising=_ma_rising(closes),
        distribution_days=distribution_days(ohlcv),
        passed=passed,
        reason=reason,
    )


def _dual_reason(market: MarketResult, sector: MarketResult) -> str:
    failures = [
        f"{label}:{r.reason}"
        for label, r in (("market", market), ("sector", sector))
        if not r.passed
    ]
    return " ".join(failures)


def dual_market_direction(
    market_index: str = DEFAULT_MARKET_INDEX,
    sector_index: str = DEFAULT_SECTOR_INDEX,
    fetch: Callable[[str], pd.DataFrame] = fetch_index_history,
    max_distribution_days: int = DEFAULT_MAX_DISTRIBUTION_DAYS,
) -> DualMarketResult:
    """O'Neil's M for a sector-concentrated book: general tape AND sector tape must both pass."""
    market = market_direction(market_index, fetch=fetch, max_distribution_days=max_distribution_days)
    sector = market_direction(sector_index, fetch=fetch, max_distribution_days=max_distribution_days)
    passed = market.passed and sector.passed
    return DualMarketResult(
        market=market,
        sector=sector,
        passed=passed,
        reason="" if passed else _dual_reason(market, sector),
    )


def _format_value(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.2f}"


def _format_leg(label: str, result: MarketResult) -> str:
    verdict = "PASS" if result.passed else "FAIL"
    return (
        f"  {label:6s} {result.index:8s} {verdict}  reason: {result.reason or 'uptrend intact'}\n"
        f"         close {result.close if result.close is None else round(result.close, 2)}  "
        f"50-day SMA {_format_value(result.sma50)}  rising {result.ma_rising}  "
        f"distribution days {result.distribution_days}"
    )


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="CANSLIM criterion M screen (dual gate: general market + sector)")
    parser.add_argument("--market", default=DEFAULT_MARKET_INDEX, help="general market index (default ^GSPC)")
    parser.add_argument("--sector", default=DEFAULT_SECTOR_INDEX, help="sector index (default XBI; use '' to skip)")
    parser.add_argument("--max-distribution-days", type=int, default=DEFAULT_MAX_DISTRIBUTION_DAYS, help="max heavy-volume down days (default 4)")
    args = parser.parse_args(argv)

    if args.sector:
        result = dual_market_direction(
            args.market, args.sector, fetch=fetch_index_history, max_distribution_days=args.max_distribution_days
        )
        verdict = "PASS" if result.passed else "FAIL"
        print(f"{verdict}  criterion M (market direction — dual gate)")
        print(_format_leg("market", result.market), end="\n")
        print(_format_leg("sector", result.sector))
        if result.reason:
            print(f"  failing legs: {result.reason}")
        return 0 if result.passed else 1

    market = market_direction(args.market, fetch=fetch_index_history, max_distribution_days=args.max_distribution_days)
    verdict = "PASS" if market.passed else "FAIL"
    print(f"{verdict}  {market.index}  criterion M (market direction)")
    print(_format_leg("market", market))
    return 0 if market.passed else 1


if __name__ == "__main__":
    sys.exit(main())
