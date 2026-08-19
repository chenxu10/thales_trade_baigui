"""CANSLIM criterion I screen: institutional sponsorship from yfinance.

O'Neil (Market Wizards, 2006): "The 'I' in the formula stands for
institutional sponsorship. The institutional buyers are by far the largest
source of demand for stocks. Leading stocks usually have institutional
backing. However, although some institutional sponsorship is desired,
excessive sponsorship is not, because it would be a source of large selling
if anything went wrong with the company or the market in general."

The rule is therefore two-sided: some institutional backing is required
(min_holders holders AND at least a 1% ownership floor), but heavy ownership
beyond max_held_pct is a FAIL — by the time nearly every institution owns a
stock, it is probably too late to buy.

Usage:
    uv run python -m fentu.canslim.institutional_sponsorship VRTX
    uv run python -m fentu.canslim.institutional_sponsorship VRTX --min-holders 5
    uv run python -m fentu.canslim.institutional_sponsorship VRTX --max-held-pct 0.80
"""
import argparse
import sys
from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd

DEFAULT_MIN_HOLDERS = 1
DEFAULT_MAX_HELD_PCT = 0.85
SPARSE_SPONSORSHIP_FLOOR = 0.01


@dataclass(frozen=True)
class HoldingsData:
    """Raw institutional sponsorship numbers pulled from yfinance."""

    held_pct: Optional[float]
    holder_count: Optional[int]


@dataclass(frozen=True)
class SponsorshipVerdict:
    passed: bool
    reason: str = ""


@dataclass(frozen=True)
class SponsorshipResult:
    ticker: str
    held_pct: Optional[float]
    holder_count: int
    passed: bool
    reason: str


def _major_holders_value(major: Optional[pd.DataFrame], breakdown: str) -> Optional[float]:
    """Value of a Breakdown row (as index or column), or None when absent."""
    if major is None or major.empty or "Value" not in major.columns:
        return None
    if "Breakdown" in major.columns:
        values = major.loc[major["Breakdown"].astype(str) == breakdown, "Value"]
    else:
        values = major.loc[major.index.astype(str) == breakdown, "Value"]
    return None if values.empty else float(values.iloc[0])


def fetch_holdings(ticker: str) -> HoldingsData:
    """Single I/O seam: pull institutional sponsorship raw numbers from yfinance."""
    import yfinance as yf

    inst = yf.Ticker(ticker)
    held_pct = _major_holders_value(inst.major_holders, "institutionsPercentHeld")

    holders = inst.institutional_holders
    if holders is not None and not holders.empty and "Holder" in holders.columns:
        holder_count = int(len(holders))
    else:
        count = _major_holders_value(inst.major_holders, "institutionsCount")
        holder_count = None if count is None else int(count)
    return HoldingsData(held_pct=held_pct, holder_count=holder_count)


def sponsorship_verdict(
    held_pct: Optional[float],
    holder_count: Optional[int],
    min_holders: int = DEFAULT_MIN_HOLDERS,
    max_held_pct: float = DEFAULT_MAX_HELD_PCT,
) -> SponsorshipVerdict:
    """Pure function: O'Neil's two-sided institutional sponsorship rule.

    PASS iff at least min_holders institutions own the stock AND ownership
    sits in [0.01, max_held_pct] — enough to confirm institutional demand,
    not so much that the name is a crowded sell.
    """
    if held_pct is None or holder_count is None:
        return SponsorshipVerdict(passed=False, reason="no_sponsorship_data")
    if holder_count <= 0 or held_pct <= 0:
        return SponsorshipVerdict(passed=False, reason="no_institutional_holders")
    if held_pct > max_held_pct:
        return SponsorshipVerdict(passed=False, reason="excessive_sponsorship")
    if held_pct < SPARSE_SPONSORSHIP_FLOOR or holder_count < min_holders:
        return SponsorshipVerdict(passed=False, reason="sparse_sponsorship")
    return SponsorshipVerdict(passed=True)


def screen_institutional_sponsorship(
    ticker: str,
    min_holders: int = DEFAULT_MIN_HOLDERS,
    max_held_pct: float = DEFAULT_MAX_HELD_PCT,
    fetch: Callable[[str], HoldingsData] = fetch_holdings,
) -> SponsorshipResult:
    """Screen one ticker on criterion I; `fetch` is injectable for tests."""
    data = fetch(ticker)
    verdict = sponsorship_verdict(data.held_pct, data.holder_count, min_holders, max_held_pct)
    return SponsorshipResult(
        ticker=ticker,
        held_pct=data.held_pct,
        holder_count=0 if data.holder_count is None else data.holder_count,
        passed=verdict.passed,
        reason=verdict.reason,
    )


def _format_held_pct(held_pct: Optional[float]) -> str:
    return "-" if held_pct is None else f"{held_pct * 100:.1f}%"


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="CANSLIM criterion I screen (institutional sponsorship)")
    parser.add_argument("ticker", help="yfinance ticker, e.g. VRTX")
    parser.add_argument("--min-holders", type=int, default=DEFAULT_MIN_HOLDERS, help="minimum institutional holders (default 1)")
    parser.add_argument("--max-held-pct", type=float, default=DEFAULT_MAX_HELD_PCT, help="maximum institutional ownership share (default 0.85)")
    args = parser.parse_args(argv)

    result = screen_institutional_sponsorship(args.ticker, args.min_holders, args.max_held_pct)
    verdict = "PASS" if result.passed else "FAIL"
    print(f"{verdict}  {result.ticker}  criterion I (min holders {args.min_holders}, max held {args.max_held_pct * 100:.0f}%)")
    print(f"  reason : {result.reason or 'meets threshold'}")
    print(f"  held % : {_format_held_pct(result.held_pct)}")
    print(f"  holders: {result.holder_count}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
