from datetime import date, datetime, timedelta

import pandas as pd

from fentu.pricingservices.option_quotes import (
    atm_strike,
    call_iv,
    days_to_expiry,
    mid,
    nearest_strike,
    otm_put_mid,
    otm_strike,
    pick_expiry,
    put_iv,
    straddle_mid,
)


def fake_chain():
    strikes = [710.0, 715.0, 720.0, 725.0, 730.0]
    rows = pd.DataFrame(
        {
            "strike": strikes,
            "bid": [1.0, 2.0, 3.0, 4.0, 5.0],
            "ask": [2.0, 3.0, 4.0, 5.0, 6.0],
            "impliedVolatility": [0.15, 0.16, 0.17, 0.18, 0.19],
        }
    )
    return type("Chain", (), {"calls": rows, "puts": rows})()


def test_mid_averages_bid_ask():
    assert mid({"bid": 4.0, "ask": 6.0}) == 5.0


def test_atm_strike_nearest_to_spot():
    assert atm_strike(fake_chain(), 723.03) == 725.0


def test_straddle_mid_sums_call_and_put():
    assert straddle_mid(fake_chain(), 725.0) == 9.0


def test_otm_strike_rounded_to_step():
    assert otm_strike(723.03, 0.25) == 540.0
    assert otm_strike(723.03, 0.20) == 580.0


def test_otm_put_mid_at_strike():
    assert otm_put_mid(fake_chain(), 720.0) == 3.5


def test_nearest_strike_falls_back_when_exact_absent():
    # A target strike outside the chain range must NOT raise (the old exact
    # `.iloc[0]` raised IndexError); it resolves to the nearest present strike.
    assert nearest_strike(fake_chain().puts, 600.0) == 710.0
    assert nearest_strike(fake_chain().puts, 760.0) == 730.0


def test_nearest_strike_empty_side_returns_none():
    empty = fake_chain().puts.iloc[0:0]
    assert nearest_strike(empty, 720.0) is None


def test_otm_put_mid_out_of_range_uses_nearest_strike():
    # With the 30% wing missing from the chain, fall back to the nearest edge
    # instead of raising.
    assert otm_put_mid(fake_chain(), 600.0) == 1.5
    assert otm_put_mid(fake_chain(), 760.0) == 5.5
    empty_chain = type("Chain", (), {"puts": fake_chain().puts.iloc[0:0],
                                     "calls": fake_chain().calls})()
    assert otm_put_mid(empty_chain, 720.0) == 0.0


def test_iv_getters_fall_back_to_nearest_strike():
    assert put_iv(fake_chain(), 600.0) == 0.15
    assert call_iv(fake_chain(), 760.0) == 0.19
    empty_chain = type("Chain", (), {"puts": fake_chain().puts.iloc[0:0],
                                     "calls": fake_chain().calls})()
    assert put_iv(empty_chain, 720.0) == 0.0


def test_pick_expiry_nearest_to_target():
    today = datetime.now().date()
    expirations = [
        (today + timedelta(days=40)).isoformat(),
        (today + timedelta(days=70)).isoformat(),
        (today + timedelta(days=105)).isoformat(),
    ]
    ticker = type("Ticker", (), {"options": expirations})()
    assert pick_expiry(ticker, 63) == (today + timedelta(days=70)).isoformat()


def test_pick_expiry_none_when_too_far():
    today = datetime.now().date()
    expirations = [(today + timedelta(days=200)).isoformat()]
    ticker = type("Ticker", (), {"options": expirations})()
    assert pick_expiry(ticker, 63) is None


def test_days_to_expiry():
    today = datetime.now().date()
    expiry = (today + timedelta(days=70)).isoformat()
    assert days_to_expiry(expiry) == 70
