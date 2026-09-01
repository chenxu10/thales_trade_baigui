"""Shared yfinance option-quote utilities (DRY seam for tail_plot, volcalculator).

All fetchers here touch the network; keep them thin and test the pure helpers
(mid, atm_strike, otm_strike, pick_expiry) with fake data.
"""

from __future__ import annotations

from datetime import datetime


def mid(row) -> float:
    """Mid-market price from a yfinance chain row (bid, ask); lastPrice when unquoted.

    Yahoo's options feed can zero bid/ask (and IV) even mid-session while
    `lastPrice` stays real — collapsing to 0 would void every quote and skip
    the whole chart, so the mid falls back to the last traded price.
    """
    bid, ask = float(row["bid"]), float(row["ask"])
    if bid > 0 or ask > 0:
        return (bid + ask) / 2.0
    last = float(row.get("lastPrice") or 0.0)
    return last if last > 0 else 0.0


def fetch_spot(symbol: str) -> float:
    """Last close of `symbol` as of the latest trading day."""
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    return float(ticker.history(period="1d")["Close"].iloc[-1])


def nearest_strike(side, strike) -> float | None:
    """Closest available strike to `strike` in a chain side (calls or puts).

    A rounded ``otm_strike`` can land on a strike the chain does not carry —
    e.g. QQQ rallies so the far-OTM wing exits the chain's strike range. The
    old exact ``==`` lookup then indexed an empty frame (``IndexError``).
    Returns the nearest present strike instead, or ``None`` for an empty side.
    """
    if side is None or side.empty:
        return None
    strikes = side["strike"].astype(float)
    if strikes.empty:
        return None
    return float(strikes.iloc[(strikes - float(strike)).abs().argmin()])


def pick_expiry(ticker, target_days: int, max_dte_factor: float = 1.7) -> str | None:
    """Nearest yfinance expiration string to `target_days` DTE (within target*max_dte_factor)."""
    today = datetime.now().date()
    expiry = None
    for exp in ticker.options:
        exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
        dte = (exp_date - today).days
        if 0 < dte <= target_days * max_dte_factor and (
            expiry is None
            or abs(dte - target_days)
            < abs((datetime.strptime(expiry, "%Y-%m-%d").date() - today).days - target_days)
        ):
            expiry = exp
    return expiry


def days_to_expiry(expiry: str) -> int:
    return (datetime.strptime(expiry, "%Y-%m-%d").date() - datetime.now().date()).days


def atm_strike(chain, spot: float) -> float:
    """Strike nearest to spot among the chain's call strikes."""
    return min(set(chain.calls["strike"]), key=lambda k: abs(k - spot))


def straddle_mid(chain, strike: float) -> float:
    """ATM straddle mid: call mid + put mid at `strike` (nearest when absent)."""
    total = 0.0
    for side in (chain.calls, chain.puts):
        present = nearest_strike(side, strike)
        if present is not None:
            total += mid(side[side["strike"] == present].iloc[0])
    return total


def otm_strike(spot: float, otm_pct: float, step: float = 5.0) -> float:
    """Strike `otm_pct` below spot, rounded to the chain's `step`."""
    return float(int(round(spot * (1 - otm_pct) / step)) * step)


def otm_put_mid(chain, strike: float) -> float:
    """Mid price of the put at `strike` (nearest strike when `strike` is absent)."""
    present = nearest_strike(chain.puts, strike)
    if present is None:
        return 0.0
    return mid(chain.puts[chain.puts["strike"] == present].iloc[0])


def put_iv(chain, strike: float) -> float:
    """Implied volatility (decimal) of the put at `strike` (nearest when absent)."""
    present = nearest_strike(chain.puts, strike)
    if present is None:
        return 0.0
    return float(chain.puts[chain.puts["strike"] == present].iloc[0]["impliedVolatility"])


def call_iv(chain, strike: float) -> float:
    """Implied volatility (decimal) of the call at `strike` (nearest when absent)."""
    present = nearest_strike(chain.calls, strike)
    if present is None:
        return 0.0
    return float(chain.calls[chain.calls["strike"] == present].iloc[0]["impliedVolatility"])
