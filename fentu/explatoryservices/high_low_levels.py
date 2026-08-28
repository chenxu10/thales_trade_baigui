"""High/low levels over decision windows — where the stop clusters sit.

Paul Tudor Jones on moving volume:

    "Tullis taught me about moving volume. When you are trading size, you
    have to get out when the market lets you out, not when you want to get
    out. He taught me that if you want to move a large position, you don't
    wait until the market is in new high or low ground because very little
    volume may trade there if it is a turning point."

    "One thing I learned as a floor trader was that if, for example, the old
    high was at 56.80, there are probably going to be a lot of buy stops at
    56.85. ... I will always liquidate half my position below new highs or
    lows and the remaining half beyond that point."

    "By watching Eli, I learned that even though markets look their very best
    when they are setting new highs, that is often the best time to sell."

Built for USO (oil, the war instrument), but `instrument` is any yfinance
ticker — the same levels logic applies to BNO, CL=F, GLD, SPY, BRK-B, TQQQ, ...

CLI
---
* ``uv run python -m fentu.explatoryservices.high_low_levels [TICKER] --plot``
"""

from __future__ import annotations

import sys
from datetime import date

import pandas as pd

from fentu.explatoryservices.volcalculator import ReturnsRepository

DEFAULT_INSTRUMENT = "USO"
WAR_START = date(2026, 2, 28)  # US-Iran war start
LOOKBACK_MONTHS = 2
LOOKBACK_MONTHS_LONG = 6
STOP_BUFFER_PCT = 0.01  # illustrative stop-cluster width just beyond a level


def stop_zone(level, side):
    """(lower, upper) band where stops cluster just beyond `level`.

    PTJ's floor lesson: old high 56.80 -> buy stops at 56.85. Buy stops sit
    ABOVE an old high (shorts cover / breakout buyers trigger there); sell
    stops sit BELOW an old low. The 1% width shades the cluster zone for the
    chart; it is illustrative, not a precise estimate of stop density.
    """
    if side == "buy":
        return (level, level * (1.0 + STOP_BUFFER_PCT))
    if side == "sell":
        return (level * (1.0 - STOP_BUFFER_PCT), level)
    raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")


def window_high_low(open_high_low_close, start, end):
    """High/low prints and their dates for rows with start <= index <= end."""
    window = open_high_low_close[(open_high_low_close.index >= pd.Timestamp(start)) & (open_high_low_close.index <= pd.Timestamp(end))]
    if window.empty:
        return None
    high = float(window["High"].max())
    low = float(window["Low"].min())
    return {
        "high": high,
        "high_date": window["High"].idxmax().date(),
        "low": low,
        "low_date": window["Low"].idxmin().date(),
    }


def _windows(today, instrument):
    """(short_name, label, start) per decision window, both ending at `today`.

    Every instrument reports the trailing 2mo and 6mo windows; USO (the war
    instrument) additionally reports the regime since the 2026-02-28 US-Iran
    war start. The 6mo window is universal — it is the source of every
    ticker's resistance/support levels.
    """
    two_mo_back = (pd.Timestamp(today) - pd.DateOffset(months=LOOKBACK_MONTHS)).date()
    six_mo_back = (pd.Timestamp(today) - pd.DateOffset(months=LOOKBACK_MONTHS_LONG)).date()
    windows = [("2mo", f"past {LOOKBACK_MONTHS}mo (since {two_mo_back})", two_mo_back)]
    if instrument == DEFAULT_INSTRUMENT:
        windows.append(("war", f"since {WAR_START} (war start)", WAR_START))
    windows.append(("6mo", f"past {LOOKBACK_MONTHS_LONG}mo (since {six_mo_back})", six_mo_back))
    return windows


def levels_view(open_high_low_close, instrument=DEFAULT_INSTRUMENT, today=None):
    """View-model for the chart: one entry per DISTINCT old high/low.

    Each entry: {label, short, kind, price, date, stop_side, stop_zone} where
    stop_side is "buy" above old highs and "sell" below old lows — where
    PTJ says the stops cluster. `short` is the first window quoting the
    level; it keys the line style.

    Two windows can print the SAME level (e.g. the 6mo high fell inside the
    trailing 2mo window -> 2mo high == 6mo high, the same print on the same
    date). Drawing both would stack two styles at one price — the second
    hides the first while the legend still lists both, so the swatch colors
    stop matching the screen. Coincident levels merge into one entry whose
    label joins the window names ("2mo+6mo high").
    """
    today = today if today is not None else date.today()
    entries = []
    for short, _label, start in _windows(today, instrument):
        stats = window_high_low(open_high_low_close, start, today)
        if stats is not None:
            entries.append(_level_entry(short, "high", stats))
            entries.append(_level_entry(short, "low", stats))
    return _merge_coincident_levels(entries)


def _level_entry(short, kind, stats):
    stop_side = "buy" if kind == "high" else "sell"
    price = stats[kind]
    name = kind
    if short == "6mo":
        name = "resistance" if kind == "high" else "support"
    return {
        "label": f"{short} {name}",
        "short": short,
        "kind": kind,
        "price": price,
        "date": stats[f"{kind}_date"],
        "stop_side": stop_side,
        "stop_zone": stop_zone(price, stop_side),
    }


def _merge_coincident_levels(entries):
    """One entry per (kind, price): same-print window levels join labels."""
    merged = []
    by_key = {}
    for entry in entries:
        first = by_key.get((entry["kind"], entry["price"]))
        if first is None:
            by_key[(entry["kind"], entry["price"])] = entry
            merged.append(entry)
        else:
            first["label"] = f"{first['short']}+{entry['short']} {first['kind']}"
    return merged


def _window_line(label, stats, last):
    if stats is None:
        return f"{label}: no data"
    high_pct = (stats["high"] / last - 1.0) * 100.0
    low_pct = (stats["low"] / last - 1.0) * 100.0
    return (
        f"{label}: high {stats['high']:.2f} ({stats['high_date']}, {high_pct:+.1f}% above)"
        f" | low {stats['low']:.2f} ({stats['low_date']}, {low_pct:+.1f}% below)"
    )


def _report_from_open_high_low_close(open_high_low_close, instrument, today):
    """The report from a pre-built frame (None/empty -> "<ticker> unavailable")."""
    if open_high_low_close is None or open_high_low_close.empty:
        return f"{instrument} unavailable"
    last = float(open_high_low_close["Close"].iloc[-1])
    lines = [f"{instrument} @ {last:.2f}"]
    for _short, label, start in _windows(today, instrument):
        lines.append(_window_line(label, window_high_low(open_high_low_close, start, today), last))
    return "\n".join(lines)



_LEVEL_STYLE = {
    ("2mo", "high"): {"color": "darkred", "ls": "--"},
    ("war", "high"): {"color": "red", "ls": ":"},
    ("6mo", "high"): {"color": "red", "ls": "-"},
    ("2mo", "low"): {"color": "darkgreen", "ls": "--"},
    ("war", "low"): {"color": "green", "ls": ":"},
    ("6mo", "low"): {"color": "green", "ls": "-"},
}


def plot_high_low_levels(open_high_low_close, instrument, today=None, ax=None, show=True):
    """Chart the regime: close since (war start | 6mo ago), old highs/lows.

    Each DISTINCT level is a signal line (2mo dashed, war dotted, the
    trailing-6mo resistance/support SOLID — the six-month high/low, named
    support/resistance regardless of ticker) with a marker at its print date.
    Windows printing the same level share one merged line ("2mo+6mo high"), so
    every legend swatch matches a visible line. The shaded band JUST BEYOND a
    level is where PTJ says the stops cluster — "buy stops" above old highs,
    "sell stops" below old lows. Pure of fetching: pass a pre-built
    open_high_low_close frame.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(11, 6))
    today = today if today is not None else date.today()
    close = open_high_low_close["Close"]
    regime_start = WAR_START if instrument == DEFAULT_INSTRUMENT else (
        pd.Timestamp(today) - pd.DateOffset(months=LOOKBACK_MONTHS_LONG)
    )
    regime = close[close.index >= pd.Timestamp(regime_start)]
    if regime.empty:
        regime = close

    ax.plot(regime.index, regime.values, color="black", lw=1.2, label="close")
    for entry in levels_view(open_high_low_close, instrument=instrument, today=today):
        style = _LEVEL_STYLE[(entry["short"], entry["kind"])]
        _draw_level(ax, entry, style, regime.index[-1])

    ax.set_title(f"{instrument} — old highs/lows & stop clusters (PTJ moving volume)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    if show:
        plt.show()
    return ax


def _draw_level(ax, entry, style, right_edge):
    """One signal line + its print-date marker + the stop-cluster band."""
    color, price = style["color"], entry["price"]
    ax.axhline(price, color=color, ls=style["ls"], lw=1.1,
               label=f"{entry['label']} {price:.2f}")
    ax.plot([pd.Timestamp(entry["date"])], [price], marker="o", color=color, ms=6)
    lo, hi = entry["stop_zone"]
    ax.axhspan(lo, hi, color=color, alpha=0.12)
    ax.text(right_edge, (lo + hi) / 2.0, f"{entry['stop_side']} stops",
            color=color, fontsize=8, ha="right", va="center")


def main(argv=None):
    args = list(argv if argv is not None else sys.argv[1:])
    want_plot = "--plot" in args
    tickers = [a for a in args if not a.startswith("-")]
    instrument = tickers[0] if tickers else DEFAULT_INSTRUMENT
    open_high_low_close = ReturnsRepository().try_fetch_open_high_low_close(instrument)
    print(_report_from_open_high_low_close(open_high_low_close, instrument, date.today()))
    if want_plot and open_high_low_close is not None and not open_high_low_close.empty:
        plot_high_low_levels(open_high_low_close, instrument)


if __name__ == "__main__":
    main()
