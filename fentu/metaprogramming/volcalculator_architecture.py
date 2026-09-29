"""Architecture-diagram generator for ``fentu/explatoryservices/volcalculator.py``.

``volcalculator.py``'s module docstring used to carry a hand-maintained ASCII
topology diagram. Hand-drawn ASCII drifts out of date (the old one still marked
``curl_cffi.requests`` as "unused" and predated ``try_fetch_open_high_low_close``),
so the topology now lives in a rendered figure instead:

    figures/volcalculator_architecture.png

CLI
---
* ``uv run python -m fentu.metaprogramming.volcalculator_architecture``
  regenerates the figure at the default path.
* ``... --out <path> --dpi <n>`` writes elsewhere / at another resolution.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive; safe in hooks / CI
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

DEFAULT_OUT = Path("figures/volcalculator_architecture.png")

X_LEFT, X_RIGHT = 1.5, 98.5
Y_TOP, Y_BOTTOM = 200.0, 0.0

INK = "#1f2933"
GREY = "#5f6b76"
BLUE = ("#e8f0fb", "#2f6fb3")
GREEN = ("#e7f6ec", "#2f8f52")
PURPLE = ("#f1ebfa", "#6b4fa8")
AMBER = ("#fdf3e2", "#b8791b")
SLATE = ("#eef1f4", "#5b6b7a")
PLAIN = ("#ffffff", "#8a949e")

DEPS = [
    ("yfinance", "price history\n(yf.Ticker)"),
    ("pandas", "OHLC frame\ncalendar resampling"),
    ("numpy", "log returns"),
    ("scipy.stats", "norm / t\n(indirect: via ps)"),
    ("curl_cffi.requests", "HTTP session\n(impersonate chrome)"),
    ("matplotlib.pyplot", "figure canvas\n+ gridspec"),
    ("plotting_service (ps)", "qq_plot()\nhistgram_plot()"),
    ("see_power_law (spl)", "plot_loglog_with_fit()"),
]

SEAMS = [
    (GREEN, "Seam 1 · ReturnsRepository", "network I/O — the ONLY object that touches the network", [
        "__init__(start_date, end_date)   <- cheap, no I/O",
        "_raw_open_high_low_close(instrument)",
        "    -> yf.Ticker + curl_cffi session; strips tz",
        "try_fetch_open_high_low_close(instrument) -> None on hiccup",
        "get_prices(instrument) -> _raw... + start/end window",
        "get_returns(instrument, n) -> np.log(prices/shift(n))",
        "get_period_returns(instrument, period)",
        "    -> non-overlapping calendar log returns",
        "get_vix_open_high_low_close() / get_vix_prices()",
        "    -> full ^VIX history, UN-windowed",
    ]),
    (BLUE, "Seam 2 · MarketClock", "DST / market-open logic, pure of I/O", [
        "now_eastern()",
        "    -> delegates to module _now_eastern()",
        "market_opened_today(last, now)",
        "    -> last == now.date() and >= 9:30 ET",
        "current_vix_value(ohlc, now_et) -> (label, value)",
        "    -> open if market opened, else last close",
    ]),
    (PURPLE, "Seam 3 · VolatilityDashboard", "presentation, pure of fetching", [
        "show_panel_unavailable(ax, ...)",
        "    -> centered \"unavailable\" note",
        "plot_vix_panel(ax, ohlc, current_value)",
        "    -> pure render from prebuilt ohlc +",
        "       optional (label, value) current value",
    ]),
]

FACADE_STATE = [
    "instrument | start_date | end_date",
    "_repository | _clock | _dashboard   <- injected, default-constructed",
    "_returns_cache                       <- lazy; populated on first access",
    "",
    "Construction does NO network I/O.",
    "daily/weekly/monthly/yearly_returns are @property + setter, cached.",
    "return_periods is a @property building the dict lazily.",
    "Each former private helper (_get_prices, _get_vix_open_high_low_close,",
    "_get_current_vix_value, _plot_*_panel) is kept as a delegating shim so",
    "existing callers/tests stay green.",
]

FACADE_CAPS = [
    ("[Volatility]", ["calculate_daily_volatility() -> DailyVolatility"]),
    ("[Extreme]", ["find_negative/positive_extreme_returns(k | threshold)"]),
    ("[Visualization]", [
        "visualize_percentage_change(period)",
        "  +-> _prepare_percentage_change_data()   (data view-model)",
        "  +-> _plot_percentage_change()",
        "       +-> ps.qq_plot / ps.histgram_plot / spl.plot_loglog_with_fit",
        "       +-> _plot_vix_panel      (delegates -> dashboard)",
        "       +-> matplotlib 3x2 gridspec + suptitle",
    ]),
    ("[Reporting]", ["get_past_week_price_and_log_returns()"]),
]

CLI_LINES = [
    "$ uv run fentu/explatoryservices/seechange.py <timeframe> <ticker>",
    "  timeframes: daily | weekly | monthly | yearly",
    "  e.g. uv run fentu/explatoryservices/seechange.py monthly QQQ",
    "  e.g. uv run fentu/explatoryservices/seechange.py daily portfolio",
]


def _text(ax, x, y, s, size=9.0, weight="normal", color=INK, ha="left",
          va="top", style="normal", family=None, spacing=1.35):
    ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha,
            va=va, style=style, family=family, linespacing=spacing, zorder=3)


def _box(ax, x0, y0, x1, y1, palette, radius=1.3, lw=1.4):
    face, edge = palette
    patch = FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=lw, edgecolor=edge, facecolor=face, zorder=2,
    )
    ax.add_patch(patch)


def _header(ax, y, s):
    _text(ax, X_LEFT, y, s, size=11.5, weight="bold", color=INK)


def _arrow(ax, x0, y0, x1, y1, label=None):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=1,
                arrowprops=dict(arrowstyle="-|>", lw=1.4, color=INK,
                                shrinkA=0, shrinkB=0))
    if label:
        _text(ax, x0 + 1.4, (y0 + y1) / 2.0, label, size=8.5, style="italic",
              color=GREY, va="center")


def _draw_title(ax):
    _text(ax, 50, Y_TOP - 2, "volcalculator.py — Volatility Topology",
          size=17, weight="bold", ha="center")
    _text(ax, 50, Y_TOP - 8.5, "How do extreme values manifest themselves in real data?",
          size=10.5, color=GREY, ha="center", style="italic")


def _draw_dependencies(ax):
    _header(ax, 185, "1 · External dependencies")
    gap = 0.8
    width = (X_RIGHT - X_LEFT - gap * (len(DEPS) - 1)) / len(DEPS)
    for i, (name, role) in enumerate(DEPS):
        x0 = X_LEFT + i * (width + gap)
        _box(ax, x0, 173, x0 + width, 182, SLATE, radius=1.0)
        _text(ax, x0 + width / 2, 181, name, size=8.4, weight="bold", ha="center")
        _text(ax, x0 + width / 2, 178.6, role, size=7.4, color=GREY, ha="center")


def _draw_classes(ax):
    _header(ax, 167, "2 · Class hierarchy")
    _box(ax, 30, 152, 70, 163, BLUE)
    _text(ax, 50, 161.6, "VolatilityCalculator", size=11, weight="bold", ha="center")
    _text(ax, 50, 157.8, "<<abstract base>>", size=8.4, color=GREY, ha="center", style="italic")
    _text(ax, 50, 154.6, "calculate_volatility() -> NotImplementedError", size=8.2, ha="center")

    _arrow(ax, 50, 152, 50, 148, "inherits")
    _box(ax, 2, 135, 48.5, 147.5, BLUE)
    _text(ax, 25.2, 146, "MeanAbsoluteDeviationVolatility", size=10, weight="bold", ha="center")
    _text(ax, 25.2, 142, "headline; MAD survives fat tails", size=8.2, color=GREY, ha="center")
    _text(ax, 25.2, 138.2, "mean(|r - mean(r)|)", size=8.2, ha="center", family="monospace")
    _box(ax, 51.5, 135, 98, 147.5, PLAIN)
    _text(ax, 74.7, 146, "StandardDeviationVolatility", size=10, weight="bold", ha="center")
    _text(ax, 74.7, 142, "kept for *_gaussian_only comparison", size=8.2, color=GREY, ha="center")
    _text(ax, 74.7, 138.2, "std(r)", size=8.2, ha="center", family="monospace")

    _arrow(ax, 50, 135, 50, 130.6, "used-by (Strategy)")
    _box(ax, 28, 120, 72, 130.5, PLAIN)
    _text(ax, 50, 129, "DailyVolatility   <<context>>", size=10, weight="bold", ha="center")
    _text(ax, 50, 125, "- calculator: VolatilityCalculator", size=8.2, ha="center", family="monospace")
    _text(ax, 50, 121.8, "calculate_1std_daily_volatility(daily_returns)", size=8.2, ha="center", family="monospace")


def _draw_seams(ax):
    _header(ax, 113, "3 · Three injectable seams (extracted from the former God-Object facade)")
    gap = 1.5
    width = (X_RIGHT - X_LEFT - gap * 2) / 3.0
    for i, (palette, title, sub, lines) in enumerate(SEAMS):
        x0 = X_LEFT + i * (width + gap)
        _box(ax, x0, 79, x0 + width, 108, palette)
        _text(ax, x0 + 1.6, 106, title, size=9.6, weight="bold")
        _text(ax, x0 + 1.6, 102.5, sub, size=7.6, color=GREY, style="italic")
        _text(ax, x0 + 1.6, 98.5, "\n".join(lines), size=7.0, family="monospace", spacing=1.5)
    _arrow(ax, 50, 79, 50, 72.5, "composed by")


def _draw_facade(ax):
    _box(ax, X_LEFT, 16, X_RIGHT, 70, AMBER, lw=1.8)
    _text(ax, X_LEFT + 2, 67, "VolatilityFacade  (thin orchestrator)", size=12.5, weight="bold")
    _box(ax, 3.5, 21, 47, 62.5, PLAIN, radius=1.0, lw=1.0)
    _text(ax, 5, 60.8, "\n".join(FACADE_STATE), size=7.6, family="monospace", spacing=1.6)
    _box(ax, 49, 21, 96.5, 62.5, PLAIN, radius=1.0, lw=1.0)
    y = 60.8
    for tag, lines in FACADE_CAPS:
        _text(ax, 50.5, y, tag, size=8.2, weight="bold")
        _text(ax, 50.5, y - 3.2, "\n".join(lines), size=7.4, family="monospace", spacing=1.5)
        y -= 3.2 + 2.6 * len(lines) + 1.4


def _draw_cli(ax):
    _box(ax, X_LEFT, 2.5, X_RIGHT, 13.5, SLATE)
    _text(ax, 3, 12, "Run it", size=9.6, weight="bold")
    _text(ax, 3, 9, "\n".join(CLI_LINES), size=7.8, family="monospace", spacing=1.5)


def build_figure():
    fig = plt.figure(figsize=(13.5, 19.0), dpi=150)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(X_LEFT - 2, X_RIGHT + 2)
    ax.set_ylim(Y_BOTTOM, Y_TOP)
    ax.axis("off")
    _draw_title(ax)
    _draw_dependencies(ax)
    _draw_classes(ax)
    _draw_seams(ax)
    _draw_facade(ax)
    _draw_cli(ax)
    _text(ax, 50, 0.8, "regenerate: uv run python -m fentu.metaprogramming.volcalculator_architecture",
          size=7.2, color=GREY, ha="center")
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args(argv)

    fig = build_figure()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, facecolor="white", bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
