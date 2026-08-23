"""
CANSLIM criterion M market direction — dual gate (^GSPC + XBI) -> DualMarketResult.

Covered behaviors:
    1. Pass: close above a rising 50-day SMA with no heavy-volume down days.
    2. Fail: close below the 50-day SMA ("below_50ma").
    3. Fail: close above the SMA but the SMA has stopped rising ("ma_not_rising").
    4. Fail: more than the allowed heavy-volume down days ("distribution_days_high").
    5. Fail honestly on missing price data ("no_price_history"), never a verdict.
    6. distribution_days counts heavy-volume down sessions in the trailing
       window, each scored against ITS OWN trailing 50-session baseline (the
       scored day excluded), ignoring window outsiders, normal-volume down
       days and heavy-volume up days.
    7. sma / above_rising_ma unit checks; CLI exit codes via patched fetch.
    8. Dual gate: PASS requires market AND sector legs; failure reasons are
       labeled "market:<reason> sector:<reason>"; single-leg CLI mode works.

Mock Object seam: ``market_direction`` / ``dual_market_direction`` accept an
injected ``fetch`` callable (same pattern as screen_universe's ``score``
param); no network anywhere.
"""
from unittest.mock import patch

import pandas as pd
import pytest

from fentu.canslim.market_direction import (
    above_rising_ma,
    distribution_days,
    dual_market_direction,
    main,
    market_direction,
    market_verdict,
    sma,
)


def make_ohlcv(closes, volumes=None, start="2025-01-01"):
    index = pd.date_range(start, periods=len(closes), freq="B")
    if volumes is None:
        volumes = [1_000_000] * len(closes)
    return pd.DataFrame(
        {
            "Open": closes,
            "High": closes,
            "Low": closes,
            "Close": closes,
            "Volume": volumes,
        },
        index=index,
    )


def rising_closes(n=130, start=100.0, step=0.3):
    return [start + step * i for i in range(n)]


class TestSmaAndAboveRisingMa:
    def test_sma_rolling_mean_with_nan_warmup(self):
        values = sma([1.0, 2.0, 3.0, 4.0], window=3)
        assert pd.isna(values[0]) and pd.isna(values[1])
        assert values[2:] == pytest.approx([2.0, 3.0])

    def test_above_rising_ma_true_on_steady_uptrend(self):
        assert above_rising_ma(rising_closes()) is True

    def test_above_rising_ma_false_when_sma_flat_though_close_above(self):
        closes = [100.0] * 74 + [120.0] * 6 + [100.0] * 44 + [110.0] * 6
        assert above_rising_ma(closes) is False

    def test_above_rising_ma_false_when_too_little_data(self):
        assert above_rising_ma([100.0] * 10) is False


class TestDistributionDayCounting:
    def test_counts_only_heavy_volume_down_days_in_window(self):
        closes = [100.0] * 30
        for i in (3, 8, 20):
            closes[i] = closes[i] - 2.0
        closes[15] = 99.5
        volumes = [1_000_000] * 30
        for i in (3, 8, 20, 25):
            volumes[i] = 2_000_000
        frame = make_ohlcv(closes, volumes)
        assert distribution_days(frame) == 2

    def test_no_down_days_in_steady_uptrend(self):
        assert distribution_days(make_ohlcv(rising_closes())) == 0

    def test_per_day_baseline_adapts_to_volume_regime_change(self):
        """A day is judged against ITS OWN trailing regime, not the latest shared window."""
        n = 60
        closes = [100.0] * n
        closes[52] = 98.0   # down day just after the volume regime steps up
        volumes = [1_000_000] * 50 + [4_000_000] * 10
        volumes[52] = 1_500_000  # 1.5x its own still-mostly-1M baseline -> distribution
        frame = make_ohlcv(closes, volumes)
        # a shared last-25-session baseline (all 4M regime) would NOT flag day 52;
        # the per-day baseline (48x1M + 2x4M) does
        assert distribution_days(frame, volume_mult=1.25) == 1

    def test_scored_day_excluded_from_own_baseline(self):
        n = 60
        closes = [100.0] * n
        closes[-1] = 98.0
        volumes = [1_000_000] * n
        volumes[-1] = 1_300_000  # 1.3x a baseline that EXCLUDES itself
        frame = make_ohlcv(closes, volumes)
        assert distribution_days(frame, volume_mult=1.25) == 1


class TestMarketVerdict:
    def test_pass_with_rising_ma_and_few_distribution_days(self):
        closes = rising_closes()
        passed, reason = market_verdict(closes, make_ohlcv(closes))
        assert passed is True
        assert reason == ""

    def test_below_50ma_fails_first(self):
        closes = rising_closes()[:100] + [rising_closes(130)[99] - 2.0 * i for i in range(1, 31)]
        passed, reason = market_verdict(closes, make_ohlcv(closes))
        assert passed is False
        assert reason == "below_50ma"

    def test_ma_not_rising_though_close_above(self):
        closes = [100.0] * 74 + [120.0] * 6 + [100.0] * 44 + [110.0] * 6
        passed, reason = market_verdict(closes, make_ohlcv(closes))
        assert passed is False
        assert reason == "ma_not_rising"

    def test_too_many_distribution_days_fails(self):
        closes = rising_closes(130)
        for i in (108, 112, 116, 120, 124):
            closes[i] = closes[i - 1] - 1.0
        volumes = [1_000_000] * 130
        for i in (108, 112, 116, 120, 124):
            volumes[i] = 2_000_000
        passed, reason = market_verdict(closes, make_ohlcv(closes, volumes))
        assert passed is False
        assert reason == "distribution_days_high"


class TestMarketDirection:
    def test_pass_when_price_above_rising_ma_with_few_distribution_days(self):
        closes = rising_closes()
        result = market_direction("^GSPC", fetch=lambda index: make_ohlcv(closes))
        assert result.passed is True
        assert result.reason == ""
        assert result.close == closes[-1]
        assert result.sma50 == pytest.approx(pd.Series(closes).rolling(50).mean().iloc[-1])
        assert result.ma_rising is True
        assert result.distribution_days == 0

    def test_fail_when_price_below_50_day_sma(self):
        closes = rising_closes()[:100] + [rising_closes(130)[99] - 2.0 * i for i in range(1, 31)]
        result = market_direction("^GSPC", fetch=lambda index: make_ohlcv(closes))
        assert result.passed is False
        assert result.reason == "below_50ma"
        assert result.close == closes[-1]
        assert result.sma50 is not None

    def test_fail_when_ma_not_rising(self):
        closes = [100.0] * 74 + [120.0] * 6 + [100.0] * 44 + [110.0] * 6
        result = market_direction("^GSPC", fetch=lambda index: make_ohlcv(closes))
        assert result.passed is False
        assert result.reason == "ma_not_rising"
        assert result.ma_rising is False

    def test_fail_when_too_many_distribution_days(self):
        closes = rising_closes(130)
        for i in (108, 112, 116, 120, 124):
            closes[i] = closes[i - 1] - 1.0
        volumes = [1_000_000] * 130
        for i in (108, 112, 116, 120, 124):
            volumes[i] = 2_000_000
        result = market_direction("^GSPC", fetch=lambda index: make_ohlcv(closes, volumes))
        assert result.passed is False
        assert result.reason == "distribution_days_high"
        assert result.distribution_days == 5

    def test_fail_honestly_when_no_price_history(self):
        result = market_direction("^GSPC", fetch=lambda index: pd.DataFrame())
        assert result.passed is False
        assert result.reason == "no_price_history"
        assert result.close is None
        assert result.sma50 is None
        assert result.ma_rising is False
        assert result.distribution_days == 0

    def test_fail_honestly_when_volume_column_missing(self):
        frame = pd.DataFrame({"Close": rising_closes()})
        result = market_direction("^GSPC", fetch=lambda index: frame)
        assert result.passed is False
        assert result.reason == "no_price_history"


class TestDualMarketDirection:
    def rising_fetch(self, index):
        return make_ohlcv(rising_closes())

    def falling_fetch(self, index):
        closes = rising_closes()[:100] + [rising_closes(130)[99] - 2.0 * i for i in range(1, 31)]
        return make_ohlcv(closes)

    def test_pass_requires_both_legs(self):
        result = dual_market_direction("^GSPC", "XBI", fetch=self.rising_fetch)
        assert result.passed is True
        assert result.reason == ""
        assert result.market.passed and result.sector.passed

    def test_market_leg_failure_fails_the_gate(self):
        result = dual_market_direction("^GSPC", "XBI", fetch=self.falling_fetch)
        assert result.passed is False
        assert result.reason == "market:below_50ma sector:below_50ma"

    def test_sector_leg_failure_alone_fails_the_gate(self):
        def fetch(index):
            return make_ohlcv(rising_closes() if index == "^GSPC" else [
                c - 5.0 for c in rising_closes()[:100]
            ] + [rising_closes(130)[99] - 2.0 * i for i in range(1, 31)])
        result = dual_market_direction("^GSPC", "XBI", fetch=fetch)
        assert result.passed is False
        assert result.market.passed is True
        assert result.sector.passed is False
        assert result.reason == "sector:below_50ma"


class TestMarketDirectionCli:
    def test_main_returns_zero_and_prints_verdict_on_pass(self, capsys):
        frame = make_ohlcv(rising_closes())
        with patch("fentu.canslim.market_direction.fetch_index_history", return_value=frame):
            code = main(["--market", "^GSPC", "--sector", "XBI"])
        assert code == 0
        out = capsys.readouterr().out
        assert "PASS" in out and "^GSPC" in out and "XBI" in out

    def test_main_returns_one_on_fail(self, capsys):
        closes = rising_closes(130)
        for i in (108, 112, 116, 120, 124):
            closes[i] = closes[i - 1] - 1.0
        volumes = [1_000_000] * 130
        for i in (108, 112, 116, 120, 124):
            volumes[i] = 2_000_000
        frame = make_ohlcv(closes, volumes)
        with patch("fentu.canslim.market_direction.fetch_index_history", return_value=frame):
            code = main(["--max-distribution-days", "4"])
        assert code == 1
        out = capsys.readouterr().out
        assert "FAIL" in out
