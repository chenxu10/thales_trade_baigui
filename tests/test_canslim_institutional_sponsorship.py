"""CANSLIM criterion I — institutional sponsorship (O'Neil, two-sided rule).

Covered behaviors:
    1. Pass: moderate institutional backing (held_pct ~0.5, 30 holders).
    2. Fail: no institutional holders at all.
    3. Fail: excessive sponsorship (held_pct > 0.85) — crowded ownership is a
       source of selling when anything goes wrong.
    4. Fail: sparse sponsorship (held_pct below the 0.01 floor, or fewer
       holders than min_holders).
    5. Missing data -> "no_sponsorship_data", never a fabricated pass.
    6. fetch_holdings extracts raw numbers from yfinance DataFrames.

Mock seams: screen_institutional_sponsorship takes an injectable `fetch`
callable (same pattern as screen_universe's `score` param); fetch_holdings
itself is tested by patching yfinance.Ticker (same pattern as
test_canslim_screen).
"""
from datetime import datetime
from unittest.mock import patch

import pandas as pd
import pytest

from fentu.canslim.institutional_sponsorship import (
    HoldingsData,
    fetch_holdings,
    main,
    screen_institutional_sponsorship,
    sponsorship_verdict,
)


def fake_holdings(held_pct=None, holder_count=None):
    return HoldingsData(held_pct=held_pct, holder_count=holder_count)


def make_major_holders(held_pct=0.50, count=30):
    return pd.DataFrame(
        {"Breakdown": ["institutionsCount", "institutionsPercentHeld"], "Value": [count, held_pct]}
    )


def make_institutional_holders(n):
    return pd.DataFrame(
        {
            "Date Reported": [datetime(2025, 3, 31)] * n,
            "Holder": [f"Institution {i}" for i in range(n)],
            "Shares": [1000.0 + i for i in range(n)],
            "Value": [100000.0 + i for i in range(n)],
        }
    )


class TestSponsorshipVerdict:
    def test_moderate_backing_passes(self):
        result = sponsorship_verdict(held_pct=0.50, holder_count=30)
        assert result.passed is True
        assert result.reason == ""

    def test_boundary_floor_and_cap_pass(self):
        assert sponsorship_verdict(held_pct=0.01, holder_count=1).passed is True
        assert sponsorship_verdict(held_pct=0.85, holder_count=1).passed is True

    def test_zero_holders_fails_no_institutional_holders(self):
        result = sponsorship_verdict(held_pct=0.50, holder_count=0)
        assert result.passed is False
        assert result.reason == "no_institutional_holders"

    def test_zero_held_pct_fails_no_institutional_holders(self):
        result = sponsorship_verdict(held_pct=0.0, holder_count=12)
        assert result.passed is False
        assert result.reason == "no_institutional_holders"

    def test_excessive_sponsorship_fails(self):
        result = sponsorship_verdict(held_pct=0.95, holder_count=800)
        assert result.passed is False
        assert result.reason == "excessive_sponsorship"

    def test_sparse_held_pct_below_floor_fails(self):
        result = sponsorship_verdict(held_pct=0.005, holder_count=5)
        assert result.passed is False
        assert result.reason == "sparse_sponsorship"

    def test_too_few_holders_fails_sparse(self):
        result = sponsorship_verdict(held_pct=0.50, holder_count=2, min_holders=5)
        assert result.passed is False
        assert result.reason == "sparse_sponsorship"

    def test_missing_data_fails_no_sponsorship_data(self):
        result = sponsorship_verdict(held_pct=None, holder_count=None)
        assert result.passed is False
        assert result.reason == "no_sponsorship_data"


class TestScreenInstitutionalSponsorship:
    def test_screen_pass_with_injected_fetch(self):
        result = screen_institutional_sponsorship("VRTX", fetch=lambda t: fake_holdings(0.50, 30))
        assert result.ticker == "VRTX"
        assert result.passed is True
        assert result.held_pct == pytest.approx(0.50)
        assert result.holder_count == 30
        assert result.reason == ""

    def test_screen_excessive_sponsorship_fails(self):
        result = screen_institutional_sponsorship("PFE", fetch=lambda t: fake_holdings(0.95, 800))
        assert result.passed is False
        assert result.reason == "excessive_sponsorship"

    def test_screen_missing_data_filtered(self):
        result = screen_institutional_sponsorship("XFOR", fetch=lambda t: fake_holdings())
        assert result.passed is False
        assert result.held_pct is None
        assert result.holder_count == 0
        assert result.reason == "no_sponsorship_data"


class TestFetchHoldings:
    def test_extracts_from_yfinance_dataframes(self):
        with patch("yfinance.Ticker") as ticker:
            ticker.return_value.major_holders = make_major_holders(held_pct=0.61, count=30)
            ticker.return_value.institutional_holders = make_institutional_holders(30)
            data = fetch_holdings("VRTX")
        assert data.held_pct == pytest.approx(0.61)
        assert data.holder_count == 30

    def test_reads_breakdown_as_index_name(self):
        major = make_major_holders(held_pct=0.45, count=9).set_index("Breakdown")
        with patch("yfinance.Ticker") as ticker:
            ticker.return_value.major_holders = major
            ticker.return_value.institutional_holders = make_institutional_holders(9)
            data = fetch_holdings("VRTX")
        assert data.held_pct == pytest.approx(0.45)
        assert data.holder_count == 9

    def test_falls_back_to_institutions_count_when_no_holder_rows(self):
        with patch("yfinance.Ticker") as ticker:
            ticker.return_value.major_holders = make_major_holders(held_pct=0.61, count=17)
            ticker.return_value.institutional_holders = pd.DataFrame()
            data = fetch_holdings("VRTX")
        assert data.held_pct == pytest.approx(0.61)
        assert data.holder_count == 17

    def test_missing_data_is_none(self):
        with patch("yfinance.Ticker") as ticker:
            ticker.return_value.major_holders = pd.DataFrame()
            ticker.return_value.institutional_holders = None
            data = fetch_holdings("VRTX")
        assert data.held_pct is None
        assert data.holder_count is None


class TestMain:
    def test_exit_code_zero_on_pass(self, capsys):
        with patch("yfinance.Ticker") as ticker:
            ticker.return_value.major_holders = make_major_holders(held_pct=0.50, count=30)
            ticker.return_value.institutional_holders = make_institutional_holders(30)
            assert main(["VRTX"]) == 0
        out = capsys.readouterr().out
        assert "PASS" in out and "criterion I" in out and "50.0%" in out

    def test_exit_code_one_on_fail(self, capsys):
        with patch("yfinance.Ticker") as ticker:
            ticker.return_value.major_holders = make_major_holders(held_pct=0.95, count=800)
            ticker.return_value.institutional_holders = make_institutional_holders(800)
            assert main(["VRTX"]) == 1
        out = capsys.readouterr().out
        assert "FAIL" in out and "excessive_sponsorship" in out
