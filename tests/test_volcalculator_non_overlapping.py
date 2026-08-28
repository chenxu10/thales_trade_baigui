"""
Fix #2: weekly/monthly/yearly returns must be NON-overlapping.

The overlapping rolling window ``log(p / p.shift(252))`` at every daily point
let adjacent "yearly" observations share ~99.6% of their content — a
pseudo-replicated fake-large sample that overstates the confidence of the
tail alpha and extreme-return lists on the see_change chart. This test pins
the corrected behavior: weekly/monthly/yearly resample closes to period-end
(Friday / month-end / year-end) so each observation is one real calendar
period. Daily keeps trading-day returns (already non-overlapping).
"""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

from fentu.explatoryservices.volcalculator import (
    VolatilityFacade,
    ReturnsRepository,
    non_overlapping_period_returns,
)


def _biz_daily_prices(periods, start="2022-01-03"):
    """Deterministic random-walk closes on a business-day calendar."""
    rng = np.random.default_rng(0)
    dates = pd.date_range(start, periods=periods, freq="B")
    prices = 100 + np.cumsum(rng.normal(0, 0.5, periods))
    return pd.Series(prices, index=dates, name="Close")


def _facade_with_prices(prices):
    repo = ReturnsRepository()
    repo._raw_open_high_low_close = MagicMock(
        return_value=pd.DataFrame({"Close": prices}))
    return VolatilityFacade("FAKE", repository=repo)


class TestNonOverlappingPeriodReturns:
    def test_weekly_resamples_to_fridays_only(self):
        prices = _biz_daily_prices(600)
        weekly = non_overlapping_period_returns(prices, "W-FRI")
        assert not weekly.empty
        assert (weekly.index.weekday == 4).all()  # all Fridays
        # 600 business days ~ 120 weeks; far fewer than the 599 rolling bars.
        assert 80 < len(weekly) < 130

    def test_monthly_resamples_to_month_ends(self):
        prices = _biz_daily_prices(800)
        monthly = non_overlapping_period_returns(prices, "ME")
        assert not monthly.empty
        assert (monthly.index.day >= 26).all()  # month-end buckets
        assert len(monthly) < 40  # ~26 months, not 799 rolling bars

    def test_yearly_resamples_to_year_ends(self):
        prices = _biz_daily_prices(1200)  # ~4.8 years
        yearly = non_overlapping_period_returns(prices, "YE")
        assert len(yearly) < 6  # a handful of years, never 1199 rolling bars
        assert (yearly.index.month == 12).all()

    def test_none_rule_returns_daily_log_returns(self):
        prices = pd.Series(
            [100.0, 110.0, 121.0],
            index=pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
        )
        daily = non_overlapping_period_returns(prices, None)
        assert len(daily) == 2
        assert daily.iloc[0] == pytest.approx(np.log(1.1))


class TestFacadeNonOverlapping:
    def test_daily_still_uses_single_day_shift(self):
        """Daily is already non-overlapping; keep the existing repo.get_returns seam."""
        prices = _biz_daily_prices(10)
        facade = _facade_with_prices(prices)
        daily = facade.daily_returns
        assert len(daily) == 9  # 10 prices -> 9 one-day log returns

    def test_yearly_returns_are_year_ends_not_rolling_252(self):
        prices = _biz_daily_prices(1300)  # ~5.2 years of business days
        facade = _facade_with_prices(prices)
        yearly = facade.yearly_returns
        assert len(yearly) < 6
        assert (yearly.index.month == 12).all()

    def test_weekly_returns_are_fridays(self):
        prices = _biz_daily_prices(520)
        facade = _facade_with_prices(prices)
        weekly = facade.weekly_returns
        assert (weekly.index.weekday == 4).all()

    def test_monthly_returns_are_month_ends(self):
        prices = _biz_daily_prices(800)
        facade = _facade_with_prices(prices)
        monthly = facade.monthly_returns
        assert len(monthly) < 40
        assert (monthly.index.day >= 26).all()

    def test_cache_persists_without_refetch(self):
        prices = _biz_daily_prices(600)
        facade = _facade_with_prices(prices)
        _ = facade.yearly_returns
        first = facade._repository._raw_open_high_low_close.call_count
        _ = facade.yearly_returns
        assert facade._repository._raw_open_high_low_close.call_count == first


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
