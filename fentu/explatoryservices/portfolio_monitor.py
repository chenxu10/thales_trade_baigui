"""Portfolio signal monitor — Taleb's Bloomberg trick, Fooled by Randomness p.166.

The trick (``teams/2005-01-01-nassim-nicolas-taleb-fooled-by-randomness.pdf``,
PDF page 93 / book page 166):

    "I have set up my Bloomberg monitor to display the price and percentage
    change of all relevant prices in the world ... The trick is to look only
    at the large percentage changes. Unless something moves by more than its
    usual daily percentage change, the event is deemed to be noise. ... the
    interpretation is not linear; a 2% move is not twice as significant an
    event as 1%, it is rather like four times."

Three rules implemented here:

1. FIXED PANEL — the same holdings in the same positions every day, so the
   trader builds the instinctive feel Taleb describes. Default:
   TQQQ upper-left, USO upper-right, IAU lower-left, BRKB lower-right.
2. NOISE FILTER — the "usual percentage change" is the MAD of the holding's
   NON-OVERLAPPING calendar-period log returns (the project's headline
   volatility metric; STD is forbidden under fat tails). Bars inside ±1 MAD
   are gray noise; only moves beyond the band are highlighted (red down /
   green up). The latest move never calibrates its own denominator (same
   discipline as ``morning_brief``).
3. NON-LINEAR SIGNIFICANCE — significance scales with the SQUARE of the
   MAD-multiple: a 2-MAD move is reported as ~4x the event of a 1-MAD move,
   not 2x.

Bars are one calendar period each (trading day / week ending Friday / month /
year), never overlapping rolling windows: adjacent rolling 21-day bars share
~95% and rolling 252-day bars ~99.6% of their content, so a rolling "yearly"
panel is one number drifting over a quarter while pretending to be 60
observations — and its MAD borrows a fake-large sample. Yearly panels are
UNCAPPED (no lookback truncation): cutting old years would amputate exactly
the deep-tail observations (1929, 1987) this monitor exists to reveal.

Reuse: all network I/O goes through ``ReturnsRepository`` (Seam 1 of
``volcalculator`` — the only object that touches yfinance); calendar
resampling is pure pandas on the fetched prices; the scale is computed by
``DailyVolatility`` with its default ``MeanAbsoluteDeviationVolatility``
calculator.

CLI: ``see_change daily portfolio`` (wired in ``seechange.py``).
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
from matplotlib.patches import Patch

from fentu.explatoryservices.volcalculator import (
    DailyVolatility,
    ReturnsRepository,
    non_overlapping_period_returns,
)

# (display label, yfinance ticker) — fixed positions, per the trick.
DEFAULT_PORTFOLIO = (
    ("TQQQ", "TQQQ"),
    ("USO", "USO"),
    ("IAU", "IAU"),
    ("BRKB", "BRK-B"),
)

# Per-period windowing + rendering. Bars are non-overlapping calendar periods:
# `resample` is the pandas period-end rule applied to closes before taking log
# returns (None = trading days, no resampling). `lookback` is the number of
# bars kept per panel; None (yearly) means never truncate — the deep-tail
# years are the point of the exercise. `bar_width_days` sizes bars on the
# date axis; `incomplete_label` tags the still-open current period (YTD/MTD/WTD).
PERIOD_INFO = {
    "daily":   {"unit": "day",   "resample": None,    "lookback": 60,
                "bar_width_days": 1.0,   "incomplete_label": None},
    "weekly":  {"unit": "week",  "resample": "W-FRI", "lookback": 60,
                "bar_width_days": 5.0,   "incomplete_label": "WTD"},
    "monthly": {"unit": "month", "resample": "ME",    "lookback": 60,
                "bar_width_days": 25.0,  "incomplete_label": "MTD"},
    "yearly":  {"unit": "year",  "resample": "YE",    "lookback": None,
                "bar_width_days": 300.0, "incomplete_label": "YTD"},
}

# Below this many completed periods the MAD band is a small-sample guess;
# the panel says so instead of pretending precision (honest small sample
# beats the fake precision of overlapping windows).
MIN_CALIBRATION_PERIODS = 5

NOISE_COLOR = "0.75"  # gray: inside the usual band, deemed noise
UP_COLOR = "green"
DOWN_COLOR = "red"
BAND_COLOR = "0.9"


def noise_multiple(move, usual):
    """Signed MAD-multiple of a move vs the usual percentage change."""
    if not usual or usual != usual:  # 0.0 or NaN
        return None
    return move / usual


def significance(move, usual):
    """Non-linear (quadratic) significance: a 2x-usual move is a 4x event."""
    multiple = noise_multiple(move, usual)
    if multiple is None:
        return None
    return multiple ** 2


def is_signal(move, usual):
    """True only when the move exceeds its usual daily percentage change."""
    multiple = noise_multiple(move, usual)
    return multiple is not None and abs(multiple) > 1.0


def _bar_color(move, usual):
    if not is_signal(move, usual):
        return NOISE_COLOR
    return UP_COLOR if move > 0 else DOWN_COLOR


class PortfolioMonitor:
    """Taleb-trick monitor over a fixed panel of holdings.

    `repository` is injectable (defaults to a fresh `ReturnsRepository`,
    which does NO I/O until asked). `volatility` defaults to the project's
    headline MAD calculator via `DailyVolatility`.
    """

    def __init__(self, holdings=DEFAULT_PORTFOLIO, period="daily",
                 repository=None, volatility=None, lookback=None):
        if period not in PERIOD_INFO:
            raise ValueError(f"Period must be one of {list(PERIOD_INFO)}")
        self.holdings = holdings
        self.period = period
        self._info = PERIOD_INFO[period]
        self._repository = repository or ReturnsRepository()
        self._volatility = volatility or DailyVolatility()
        # None = period default; the yearly default is itself None = all history.
        self.lookback = self._info["lookback"] if lookback is None else lookback

    # --- data (view-model; the only place the network is touched) ----------

    def prepare_panels(self):
        return [self._prepare_panel(label, ticker)
                for label, ticker in self.holdings]

    def _prepare_panel(self, label, ticker):
        data = self._fetch_panel_data(ticker)
        if data is None:
            return {"label": label, "available": False}
        returns, prices = data
        calibration = returns.iloc[:-1]  # the event never sets its own scale
        usual = float(
            self._volatility.calculate_1std_daily_volatility(calibration))
        window = (returns if self.lookback is None
                  else returns.iloc[-self.lookback:])
        last_move = float(returns.iloc[-1])
        return {
            "label": label,
            "available": True,
            "unit": self._info["unit"],
            "window": window,
            "n_calibration": len(calibration),
            "insufficient_history": len(calibration) < MIN_CALIBRATION_PERIODS,
            "incomplete_label": self._info["incomplete_label"],
            "bar_width_days": self._info["bar_width_days"],
            "last_price": float(prices.iloc[-1]),
            "last_date": prices.index[-1],
            "last_move": last_move,
            "usual": usual,
            "multiple": noise_multiple(last_move, usual),
            "significance": significance(last_move, usual),
            "signal": is_signal(last_move, usual),
        }

    def _fetch_panel_data(self, ticker):
        """(returns, prices) in percent, or None on any fetch failure or
        empty history — a spurious yfinance hiccup on one holding must
        never crash the whole monitor (morning_brief discipline)."""
        try:
            prices = self._repository.get_prices(ticker)
            returns = self._period_returns(prices) * 100.0  # display in percent
        except Exception:
            return None
        if returns.empty or prices.empty:
            return None
        return returns, prices

    def _period_returns(self, prices):
        """Non-overlapping calendar-period log returns from daily closes.

        Daily keeps trading-day returns; weekly/monthly/yearly first resample
        closes to period-end (Friday / month-end / year-end), so each bar is
        one real period — never an overlapping rolling window. Shared helper
        lives in ``volcalculator.non_overlapping_period_returns`` (Seam 1).
        """
        return non_overlapping_period_returns(prices, self._info["resample"])

    # --- presentation (pure render from the view-model) --------------------

    def visualize(self, panels=None):
        """Render the 2x2 signal panel. `panels` may be prebuilt (from
        ``prepare_panels``) so a caller that already fetched them for its own
        report does not fetch a second time."""
        if panels is None:
            panels = self.prepare_panels()
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        for ax, panel in zip(axes.flat, panels):
            plot_signal_panel(ax, panel)
        unit = self._info["unit"]
        fig.suptitle(
            f"Portfolio {self.period} signal monitor — Taleb noise filter "
            "(Fooled by Randomness p.166):\n"
            f"each bar = one {unit}'s % change; inside ±1 usual {unit} "
            "(MAD) is gray noise, beyond is signal; significance ∝ move²",
            fontsize=11)
        fig.legend(handles=legend_handles(), loc="lower center", ncol=4,
                   frameon=False, fontsize=9)
        fig.tight_layout(rect=[0, 0.04, 1, 0.93])
        plt.show()
        return fig


def span_label(window):
    """Human date-span of the window, e.g. '2006–2026' or 'Apr–Jul 2026'."""
    start, end = window.index[0], window.index[-1]
    if end.year - start.year >= 2:
        return f"{start:%Y}–{end:%Y}"
    if start.year == end.year:
        return f"{start:%b}–{end:%b %Y}"
    return f"{start:%b %Y} – {end:%b %Y}"


def legend_handles():
    """Proxy artists for the figure-level legend: band, noise, up/down signal."""
    return [
        Patch(facecolor=BAND_COLOR, edgecolor="0.6",
              label="usual band (±1 MAD)"),
        Patch(facecolor=NOISE_COLOR, edgecolor="0.5",
              label="noise (inside usual move)"),
        Patch(facecolor=UP_COLOR, label="up signal (beyond usual move)"),
        Patch(facecolor=DOWN_COLOR, label="down signal (beyond usual move)"),
    ]


def plot_signal_panel(ax, panel):
    """Render one holding: gray noise bars inside ±1 MAD, highlighted signals.

    Bars sit at their period-end date on a real time axis; the title carries
    the holding plus the window's actual date span, so the chart itself
    answers "what period does this cover?".
    """
    if not panel.get("available", True):
        ax.text(0.5, 0.5, f"{panel['label']} unavailable",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title(panel["label"])
        return
    window = panel["window"]
    usual = panel["usual"]
    colors = [_bar_color(move, usual) for move in window]
    ax.bar(window.index, window.values,
           width=panel.get("bar_width_days", 1.0), color=colors)
    ax.axhspan(-usual, usual, color=BAND_COLOR, zorder=0)
    ax.axhline(0, color="black", lw=0.5)
    locator = AutoDateLocator(minticks=3, maxticks=6)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(ConciseDateFormatter(locator))
    unit = panel.get("unit", "day")
    ax.set_title(f"{panel['label']}  ({span_label(window)})", fontsize=10)
    ax.set_xlabel(f"{unit.capitalize()} ending")
    ax.set_ylabel(f"% change per {unit}")
    _annotate(ax, panel)


def _annotate(ax, panel):
    verdict = "SIGNAL" if panel["signal"] else "noise"
    verdict_color = "green" if panel["signal"] else "red"
    ax.text(0.99, 0.97, verdict, transform=ax.transAxes, ha="right", va="top",
            fontsize=10, fontweight="bold", color=verdict_color)
    if panel["multiple"] is None:
        reading = "usual change undefined"
    else:
        open_period = panel.get("incomplete_label")
        prefix = f"{open_period} " if open_period else ""
        reading = (f"{prefix}{panel['last_move']:+.2f}% = "
                   f"{panel['multiple']:+.1f}x usual\n"
                   f"significance {panel['significance']:.1f}x")
    unit = panel.get("unit", "day")
    usual_line = f"usual {panel['usual']:.2f}% (MAD"
    n = panel.get("n_calibration")
    if n is not None:
        usual_line += f" of {n} {unit}s"
    usual_line += ")"
    lines = [f"as of {panel['last_date']:%d %b %Y}",
             f"last {panel['last_price']:,.2f}", reading, usual_line]
    if panel.get("insufficient_history"):
        lines.append("insufficient history — band unreliable")
    ax.text(0.99, 0.87, "\n".join(lines),
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(facecolor="white", alpha=0.8,
                      edgecolor="black" if panel["signal"] else NOISE_COLOR))
