"""
This script calculates volatility and return metrics for financial instruments.

Topology Diagram (ASCII)
========================

 External Dependencies
 +---------------------------------------------------------------------------+
 | yfinance | pandas | numpy | scipy.stats(norm,t) | curl_cffi.requests     |
 | matplotlib.pyplot | plotting_service (ps) | see_power_law (spl)          |
 +---------------------------------------------------------------------------+
      |            |          |            |               |
      | prices     | data     | tails/log  | (unused)      | plotting
      v            v          v            v               v

 Class Hierarchy & Relationships
 +-------------------------------+
 |   VolatilityCalculator        |  <<abstract base>>
 |   calculate_volatility()      |  -> NotImplementedError
 +---------------+---------------+
                 |  inherits
                 v
 +-------------------------------+
 | MeanAbsoluteDeviationVolatility|  (headline; MAD survives fat tails)
 | StandardDeviationVolatility   |  (kept for *_gaussian_only comparison)
 +---------------+---------------+
                 |  used-by (Strategy pattern)
                 v
 +-------------------------------+
 | DailyVolatility               |  <<context>>
 | - calculator: VolatilityCalc  |
 | calculate_1std_daily_vol()    |
 +---------------+---------------+

 Three injectable seams (extracted from the former God-Object facade):
 +---------------------------------------------------------------------------+
 | ReturnsRepository   (Seam 1 — the ONLY object that touches the network)   |
 |   __init__(start_date, end_date)   <- cheap, no I/O                       |
 |   _raw_open_high_low_close(instrument)            -> yf.Ticker + curl_cffi session;      |
 |                                       strips tz from the index            |
 |   get_prices(instrument)           -> _raw_open_high_low_close + start/end window        |
 |   get_returns(instrument, period)  -> np.log(prices/shift)[period:]       |
 |   get_vix_open_high_low_close() / get_vix_prices()-> full ^VIX history, UN-windowed      |
 +---------------------------------------------------------------------------+
 +---------------------------------------------------------------------------+
 | MarketClock          (Seam 2 — DST / market-open logic, pure of I/O)      |
 |   now_eastern()                   -> delegates to module _now_eastern()   |
 |   market_opened_today(last, now)  -> last==now.date() and >=9:30 ET       |
 |   current_vix_value(open_high_low_close, now_et) -> (label, value): open if opened else  |
 |                                       last close                          |
 +---------------------------------------------------------------------------+
 +---------------------------------------------------------------------------+
 | VolatilityDashboard  (Seam 3 — presentation, pure of fetching)            |
 |   show_panel_unavailable(ax,...)  -> centered "unavailable" note          |
 |   plot_vix_panel(ax, open_high_low_close, current_value) -> pure render from prebuilt    |
 |       open_high_low_close + optional (label, value) current-value pair   |
 +---------------------------------------------------------------------------+
                 |  composed by
                 v
 +---------------------------------------------------------------------------+
 |                   VolatilityFacade  (thin orchestrator)                   |
 |  instrument | start_date | end_date                                       |
 |  _repository | _clock | _dashboard   <- injected, default-constructed     |
 |  _returns_cache                       <- lazy; populated on first access  |
 |                                                                           |
 |  Construction does NO network I/O.                                        |
 |  daily/weekly/monthly/yearly_returns are @property + setter, cached.      |
 |  return_periods is a @property building the dict lazily.                  |
 |                                                                           |
 |  Each former private helper (_get_prices, _get_vix_open_high_low_close,   |
 |  _get_current_vix_value, _plot_*_panel) is kept                          |
 |  as a delegating shim so existing callers/tests stay green.               |
 |                                                                           |
 |  [Volatility]    calculate_daily_volatility() -> DailyVolatility          |
 |  [Extreme]       find_negative/positive_extreme_returns(k|threshold)      |
  |  [Visualization] visualize_percentage_change(period)                      |
 |     +-> _prepare_percentage_change_data() (data view-model)               |
 |     +-> _plot_percentage_change()                                         |
 |           +-> ps.qq_plot / ps.histgram_plot / spl.plot_loglog_with_fit    |
 |           +-> _plot_vix_panel             (delegates -> dashboard)        |
 |           +-> matplotlib 3x2 gridspec + suptitle                          |
 |  [Reporting]     get_past_week_price_and_log_returns()                    |
 +---------------------------------------------------------------------------+

 Entry Point
 +---------------------------------------------------------------------------+
 | __main__: VolatilityFacade("ILS")                                         |
 |   -> get_past_week_price_and_log_returns()                                |
 |   -> calculate_daily_volatility()                                         |
 |   -> find_negative/positive_extreme_returns() (threshold=+-0.2)           |
 +---------------------------------------------------------------------------+
"""

import yfinance as yf
import pandas as pd
pd.set_option('display.max_rows', None)

import fentu.explatoryservices.plotting_service as ps
import fentu.explatoryservices.see_power_law as spl
import matplotlib.pyplot as plt
import numpy as np
from curl_cffi import requests
from datetime import datetime, timezone, timedelta, time as _dtime

VIX_TICKER = "^VIX"
TIME_MARKET_OPEN = _dtime(9, 30)  # US equity market open, Eastern Time


def _is_us_dst(dt_utc):
    """True if `dt_utc` (tz-aware UTC) falls within US daylight saving time.

    US DST: second Sunday of March -> first Sunday of November. Stdlib-only so
    this works on Python 3.8 (no zoneinfo).
    """
    year = dt_utc.year
    mar1 = datetime(year, 3, 1, tzinfo=timezone.utc)
    first_sunday_mar = mar1 + timedelta(days=(6 - mar1.weekday()) % 7)
    second_sunday_mar = first_sunday_mar + timedelta(days=7)
    nov1 = datetime(year, 11, 1, tzinfo=timezone.utc)
    first_sunday_nov = nov1 + timedelta(days=(6 - nov1.weekday()) % 7)
    return second_sunday_mar <= dt_utc < first_sunday_nov


def _now_eastern():
    """Current time in US Eastern Time (EST/EDT), tz-aware."""
    now_utc = datetime.now(timezone.utc)
    offset_h = -4 if _is_us_dst(now_utc) else -5
    return now_utc.astimezone(timezone(timedelta(hours=offset_h), "ET"))


class VolatilityCalculator:
    """Base class for different volatility calculation strategies"""
    def calculate_volatility(self, returns_data):
        raise NotImplementedError

class MeanAbsoluteDeviationVolatility(VolatilityCalculator):
    """Headline daily-volatility metric.

    Per the taleb-convexity-tailhedge skill: SD breaks under fat tails
    (one day = 80% of SP500 kurtosis over 56 years), while MAD exists
    whenever the mean exists. MAD is the headline; STD is kept only for a
    `*_gaussian_only` comparison via StandardDeviationVolatility below.
    """
    def calculate_volatility(self, returns_data):
        return (returns_data - returns_data.mean()).abs().mean()

class StandardDeviationVolatility(VolatilityCalculator):
    """Gaussian-only comparison volatility (kept for reference, not headline).  """
    def calculate_volatility(self, returns_data):
        return returns_data.std()

class DailyVolatility:
    def __init__(self, calculator: VolatilityCalculator = None):
        self.calculator = calculator or MeanAbsoluteDeviationVolatility()
    
    def calculate_1std_daily_volatility(self, daily_returns):
        return self.calculator.calculate_volatility(daily_returns)


# ---------------------------------------------------------------------------
# Seam 1: ReturnsRepository — the only object that touches the network.
# ---------------------------------------------------------------------------


class ReturnsRepository:
    """Owns all yfinance open_high_low_close fetching and the start/end date window.

    Constructed cheaply (no I/O); fetches happen lazily on demand. The VIX
    helpers deliberately ignore the ETF's start/end window so the VIX subplot
    always shows the full 1990 -> today history.
    """

    def __init__(self, start_date=None, end_date=None):
        self.start_date = start_date
        self.end_date = end_date

    def _raw_open_high_low_close(self, instrument):
        """Fetch full open_high_low_close history for `instrument` with no date filtering.

        The shared network fetch; callers that want the repository's date
        window apply start_date/end_date themselves.

        A spurious yfinance failure ("possibly delisted; no price data
        found") yields an EMPTY DataFrame whose index is a plain Index with
        no .tz attribute -- only strip tz from a real DatetimeIndex.
        """
        session = requests.Session(impersonate="chrome")
        ticker = yf.Ticker(instrument, session=session)
        open_high_low_close = ticker.history(period="max")
        if isinstance(open_high_low_close.index, pd.DatetimeIndex) and open_high_low_close.index.tz is not None:
            open_high_low_close.index = open_high_low_close.index.tz_localize(None)
        return open_high_low_close

    def try_fetch_open_high_low_close(self, instrument):
        """Fetch open_high_low_close, return None on any hiccup instead of raising.

        ``_raw_open_high_low_close`` assumes yfinance returns a tz-aware
        DatetimeIndex; tickers that 404 yield a plain Index and crash the
        helper. Callers that want a do-or-die result should use
        ``_raw_open_high_low_close`` directly; callers that must stay green on a
        network hiccup (CLIs, cron) should call this safe wrapper.
        """
        try:
            return self._raw_open_high_low_close(instrument)
        except Exception:
            return None

    def get_prices(self, instrument):
        open_high_low_close = self._raw_open_high_low_close(instrument)
        prices = open_high_low_close['Close']
        if self.start_date is not None:
            prices = prices[prices.index >= pd.Timestamp(self.start_date)]
        if self.end_date is not None:
            prices = prices[prices.index <= pd.Timestamp(self.end_date)]
        return prices

    def get_returns(self, instrument, period_length):
        prices = self.get_prices(instrument)
        return np.log(prices / prices.shift(period_length))[period_length:]

    def get_vix_open_high_low_close(self):
        """Full ^VIX open_high_low_close history (1990 -> today), unfiltered."""
        return self._raw_open_high_low_close(VIX_TICKER)

    def get_vix_prices(self):
        """Full ^VIX daily close history (1990 -> today), unfiltered."""
        return self.get_vix_open_high_low_close()['Close']


# ---------------------------------------------------------------------------
# Seam 2: MarketClock — DST / market-open logic, pure of I/O.
# ---------------------------------------------------------------------------


class MarketClock:
    """US Eastern Time clock + market-open detection for the VIX annotation.

    `now_eastern` delegates to the module-level `_now_eastern()` so existing
    tests that patch `volcalculator._now_eastern` keep working.
    """

    def now_eastern(self):
        return _now_eastern()

    def market_opened_today(self, last_date, now_et):
        """True if `last_date` (a date) is `now_et`'s date and time >= 9:30 ET."""
        return last_date == now_et.date() and now_et.time() >= TIME_MARKET_OPEN

    def current_vix_value(self, vix_open_high_low_close, now_et=None):
        """Return (label, value) for the VIX "current value" annotation.

        If the US market has opened today — i.e. the data's last trading day is
        today and the time is >= 9:30 ET — return today's open (the 9:30 ET
        print). Otherwise return the last trading day's close.
        """
        if now_et is None:
            now_et = self.now_eastern()
        if vix_open_high_low_close.empty:
            return None
        last_row = vix_open_high_low_close.iloc[-1]
        last_date = vix_open_high_low_close.index[-1].date()
        if self.market_opened_today(last_date, now_et):
            return (
                f"VIX @ open {last_date.isoformat()} 9:30 ET",
                float(last_row['Open']),
            )
        return (
            f"VIX @ last close {last_date.isoformat()}",
            float(last_row['Close']),
        )


# ---------------------------------------------------------------------------
# Seam 3: VolatilityDashboard — presentation, pure of fetching.
# ---------------------------------------------------------------------------


class VolatilityDashboard:
    """Renders the percentage-change figure's panels from pre-built data.

    Never fetches. `plot_vix_panel` takes pre-built open_high_low_close + an optional
    `(label, value)` current-value pair.
    """

    def show_panel_unavailable(self, ax, title, message, detail=""):
        text = message
        if detail:
            text = f"{message}\n{detail}"
        ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)

    def plot_vix_panel(self, ax, vix_open_high_low_close, current_value=None):
        if vix_open_high_low_close.empty:
            self.show_panel_unavailable(ax, "VIX Index", "VIX unavailable")
            return
        close = vix_open_high_low_close['Close']
        ax.plot(close.index, close.values, color="purple", lw=1.2, label="VIX close")
        ax.set_title("VIX Index")
        ax.set_ylabel("VIX")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        if current_value is not None:
            label, value = current_value
            ax.text(
                0.99, 0.95, f"{label}\n{value:.2f}",
                transform=ax.transAxes, ha="right", va="top",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="purple"),
            )


class VolatilityFacade:
    """
    This class gets daily percentage change of an instrument.

    Thin orchestrator over three injectable seams:
      - ReturnsRepository  (network I/O; lazy)
      - MarketClock        (DST / market-open logic)
      - VolatilityDashboard (presentation)

    Construction does NO network I/O. Returns are computed lazily on first
    access and cached. Each former private helper is preserved as a delegating
    shim so existing callers/tests keep working.
    """

    def __init__(self, instrument, start_date=None, end_date=None,
                 repository=None, clock=None, dashboard=None):
        self.instrument = instrument
        self.start_date = start_date
        self.end_date = end_date
        self._repository = repository or ReturnsRepository(
            start_date=start_date, end_date=end_date)
        self._clock = clock or MarketClock()
        self._dashboard = dashboard or VolatilityDashboard()
        self.daily_volatility = DailyVolatility()
        self._returns_cache = {}

    # --- lazy, cached return periods ----------------------------------------

    def _get_return_period(self, key, period_length):
        cached = self._returns_cache.get(key)
        if cached is None:
            self._returns_cache[key] = self._repository.get_returns(
                self.instrument, period_length)
        return self._returns_cache[key]

    @property
    def daily_returns(self):
        return self._get_return_period('daily', 1)

    @daily_returns.setter
    def daily_returns(self, value):
        self._returns_cache['daily'] = value

    @property
    def weekly_returns(self):
        return self._get_return_period('weekly', 5)

    @weekly_returns.setter
    def weekly_returns(self, value):
        self._returns_cache['weekly'] = value

    @property
    def monthly_returns(self):
        return self._get_return_period('monthly', 21)

    @monthly_returns.setter
    def monthly_returns(self, value):
        self._returns_cache['monthly'] = value

    @property
    def yearly_returns(self):
        return self._get_return_period('yearly', 252)

    @yearly_returns.setter
    def yearly_returns(self, value):
        self._returns_cache['yearly'] = value

    @property
    def return_periods(self):
        return {
            'daily': self.daily_returns,
            'weekly': self.weekly_returns,
            'monthly': self.monthly_returns,
            'yearly': self.yearly_returns,
        }

    # --- delegating shims (preserve pre-refactor surface) -------------------

    def _get_prices(self, instrument):
        return self._repository.get_prices(instrument)

    def _get_vix_open_high_low_close(self):
        return self._repository.get_vix_open_high_low_close()

    def _get_current_vix_value(self, vix_open_high_low_close=None, now_et=None):
        if vix_open_high_low_close is None:
            vix_open_high_low_close = self._get_vix_open_high_low_close()
        return self._clock.current_vix_value(vix_open_high_low_close, now_et=now_et)

    def _plot_vix_panel(self, ax):
        try:
            vix_open_high_low_close = self._get_vix_open_high_low_close()
            current = (self._get_current_vix_value(vix_open_high_low_close=vix_open_high_low_close)
                       if not vix_open_high_low_close.empty else None)
            self._dashboard.plot_vix_panel(ax, vix_open_high_low_close, current_value=current)
        except Exception:
            self._dashboard.show_panel_unavailable(ax, "VIX Index", "VIX unavailable")

    def calculate_daily_volatility(self):
        return self.daily_volatility.calculate_1std_daily_volatility(self.daily_returns)
    
    def _prepare_percentage_change_data(self, period='daily'):
        """
        Prepare data for percentage change visualization.

        Returns:
            dict with 'returns', 'tails', 'period', 'instrument'
        """
        if period not in self.return_periods:
            raise ValueError(f"Period must be one of {list(self.return_periods.keys())}")

        returns_data = self.return_periods[period]
        left_tail = np.abs(returns_data[returns_data < 0].values)
        right_tail = returns_data[returns_data > 0].values

        return {
            'returns': returns_data,
            'tails': [
                {'data': left_tail, 'x_min': np.min(left_tail) if len(left_tail) > 0 else None,
                 'title': 'Left Tail (Negative Returns)'},
                {'data': right_tail, 'x_min': np.min(right_tail) if len(right_tail) > 0 else None,
                 'title': 'Right Tail (Positive Returns)'},
            ],
            'period': period,
            'instrument': self.instrument,
        }

    def _plot_tail_fits(self, tails, axes):
        """Render left/right tail log-log fits onto the two bottom-row axes."""
        for tail, ax in zip(tails, axes):
            if tail['x_min'] is not None:
                spl.plot_loglog_with_fit(
                    tail['data'], tail['x_min'], ax=ax,
                    title=tail['title'],
                )

    def _build_percentage_change_figure_layout(self):
        """Build the 2x2-on-top + full-width VIX figure.

        Returns: (fig, ax_qq, ax_hist, ax_left, ax_right, ax_vix) where
        ax_vix spans the full width of its bottom row.
        """
        fig = plt.figure(figsize=(12, 12))
        gs = fig.add_gridspec(
            3, 2,
            height_ratios=[1, 1, 1],
            hspace=0.45, wspace=0.25,
        )
        ax_qq = fig.add_subplot(gs[0, 0])
        ax_hist = fig.add_subplot(gs[0, 1])
        ax_left = fig.add_subplot(gs[1, 0])
        ax_right = fig.add_subplot(gs[1, 1])
        ax_vix = fig.add_subplot(gs[2, :])
        return fig, ax_qq, ax_hist, ax_left, ax_right, ax_vix

    def _plot_percentage_change(self, data):
        """
        Plot percentage change visualizations.

        Layout: a 2x2 block of QQ / histogram / left-tail / right-tail on top,
        with the VIX index curve occupying the full-width row beneath.

        Args:
            data: dict from _prepare_percentage_change_data
        """
        fig, ax_qq, ax_hist, ax_left, ax_right, ax_vix = self._build_percentage_change_figure_layout()

        ps.qq_plot(data['returns'], ax=ax_qq, show=False)
        ps.histgram_plot(data['returns'], ax=ax_hist, show=False)
        self._plot_tail_fits(data['tails'], [ax_left, ax_right])
        self._plot_vix_panel(ax_vix)

        fig.suptitle(f"{data['instrument']} {data['period'].capitalize()} Returns")
        plt.show()

    def visualize_percentage_change(self, period='daily'):
        """
        Visualize percentage changes for a specific period using QQ plot, histogram,
        and log-log plots for left and right tail analysis. Tail alpha is
        estimated by the CSN (2009) fitter (MLE + KS-selected x_min).

        Args:
            period: str, one of 'daily', 'weekly', 'monthly', 'yearly'
        """
        data = self._prepare_percentage_change_data(period)
        self._plot_percentage_change(data)

    def _find_extreme_returns(self, period='daily', k=None, threshold=None, side='negative'):
        """
        Find extreme returns for a specific period either by count (k) or threshold.

        Args:
            period: str, one of 'daily', 'weekly', 'monthly', 'yearly'
            k: int, number of extreme returns to find (mutually exclusive with threshold)
            threshold: float, threshold beyond which returns are considered extreme
            side: str, 'negative' (worst, sorted ascending) or 'positive' (best, sorted descending)

        Returns:
            pandas.Series: Filtered returns sorted from most extreme to least extreme
        """
        if period not in self.return_periods:
            raise ValueError(f"Period must be one of {list(self.return_periods.keys())}")
        returns = self.return_periods[period]
        ascending = (side == 'negative')
        compare = (lambda r, t: r < t) if side == 'negative' else (lambda r, t: r > t)
        if k is not None:
            return returns.sort_values(ascending=ascending).head(k)
        if threshold is not None:
            return returns.loc[compare(returns, threshold)].sort_values(ascending=ascending)
        raise ValueError("Either k or threshold must be specified")

    def find_negative_extreme_returns(self, period='daily', k=None, threshold=None):
        """
        Find negative extreme returns for a specific period either by count (k) or threshold

        Args:
            period: str, one of 'daily', 'weekly', 'monthly', 'yearly'
            k: int, number of worst returns to find (mutually exclusive with threshold)
            threshold: float, threshold below which returns are considered "worst"

        Returns:
            pandas.Series: Filtered returns sorted from worst to best
        """
        return self._find_extreme_returns(period, k, threshold, side='negative')

    def find_positive_extreme_returns(self, period='daily', k=None, threshold=None):
        """
        Find positive extreme returns for a specific period either by count (k) or threshold

        Args:
            period: str, one of 'daily', 'weekly', 'monthly', 'yearly'
            k: int, number of best returns to find (mutually exclusive with threshold)
            threshold: float, threshold above which returns are considered "best"

        Returns:
            pandas.Series: Filtered returns sorted from best to worst
        """
        return self._find_extreme_returns(period, k, threshold, side='positive')

    def get_past_week_price_and_log_returns(self):
        """Get most recent 5 trading days prices with daily log returns"""
        prices = self._get_prices(self.instrument).tail(5)
        log_returns = np.log(prices / prices.shift(1))
        return pd.DataFrame({'price': prices, 'log_return': log_returns})

if __name__ == "__main__":
    volatility = VolatilityFacade("ILS")
    # Visualize different time-frame return distributions
    # volatility.visualize_percentage_change('weekly')
    print(volatility.get_past_week_price_and_log_returns())

    # Calculate volatility metrics
    print(f"Daily volatility (MAD): {volatility.calculate_daily_volatility()}")

    # Find extreme returns
    #print(f"Worst months: {volatility.find_negative_extreme_returns('monthly', k=3)}")
    print(f"Worst days (below -20%): {volatility.find_negative_extreme_returns('daily', threshold=-0.2)}")
    print(f"Worst months (below -20%): {volatility.find_negative_extreme_returns('monthly', threshold=-0.2)}")
    print(f"Best days (above +20%): {volatility.find_positive_extreme_returns('daily', threshold=0.2)}")
    print(f"Best months (above +20%): {volatility.find_positive_extreme_returns('monthly', threshold=0.2)}")

