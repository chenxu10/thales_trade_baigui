"""CANSLIM criterion N — 52-week new highs / fresh development news verdict.

Covered behaviors:
    1. Pass: latest close within tolerance of the 52-week high (O'Neil:
       "what seems too high usually goes higher").
    2. Pass: below the high but >= 1 fresh development news item in the window
       (new product / industry change / new management proxy).
    3. Fail honestly when below the high and no fresh development news —
       reason "below_high_and_no_news".
    4. Fail honestly when the price history is missing — reason
       "no_price_history", stats all None.

Mock Object seam: `screen_new_highs` takes the two fetch callables as
parameters (same injection pattern as pharma_bio_screen's `score`), so no
network and no patching are needed.
"""
from datetime import datetime

import pandas as pd
import pytest

from fentu.canslim.new_highs import compute_high_stats, score_fresh_news, score_new_high, screen_new_highs

NOW = 1_700_000_000


def fake_history(rows):
    """rows: iterable of ((year, month, day), high, close) -> daily OHLCV DataFrame."""
    index = pd.DatetimeIndex([datetime(*d) for d, _, _ in rows])
    highs = [h for _, h, _ in rows]
    closes = [c for _, _, c in rows]
    return pd.DataFrame(
        {"Open": closes, "High": highs, "Low": closes, "Close": closes, "Volume": [0] * len(rows)},
        index=index,
    )


def fake_news(items):
    """items: iterable of (title, publish_time_unix) -> yfinance-style news dicts."""
    return [{"title": t, "providerPublishTime": ts, "link": "https://example.com"} for t, ts in items]


class TestScoreNewHigh:
    def test_stats_from_history(self):
        history = fake_history([
            ((2026, 1, 2), 100.0, 98.0),
            ((2026, 2, 1), 85.0, 80.0),
        ])
        passed, high, close, distance, days_since_high, reason = score_new_high(history, tolerance=0.05)
        assert passed is False
        assert high == 100.0
        assert close == 80.0
        assert distance == pytest.approx(0.20)
        assert days_since_high == 30
        assert reason == "below_high"

    def test_close_within_tolerance_passes(self):
        history = fake_history([
            ((2026, 1, 2), 100.0, 98.0),
            ((2026, 2, 1), 99.0, 98.5),
        ])
        passed, high, close, distance, _, reason = score_new_high(history, tolerance=0.05)
        assert passed is True
        assert high == 100.0
        assert close == 98.5
        assert distance == pytest.approx(0.015)
        assert reason is None

    def test_missing_history_fails_with_reason(self):
        passed, high, close, distance, days, reason = score_new_high(pd.DataFrame())
        assert passed is False
        assert high is None and close is None and distance is None and days is None
        assert reason == "no_price_history"


class TestComputeHighStats:
    def test_reason_none_for_usable_history(self):
        _, _, _, _, reason = compute_high_stats(fake_history([((2026, 1, 2), 100.0, 98.0)]))
        assert reason is None


class TestScoreFreshNews:
    def test_counts_only_fresh_matching_titles(self):
        news = fake_news([
            ("Vertex announces FDA approval of new therapy", NOW - 1 * 86400),
            ("Phase 3 trial readout for pain candidate", NOW - 3 * 86400),
            ("Earnings call transcript", NOW - 4 * 86400),
            ("Partnership signed last quarter", NOW - 60 * 86400),
        ])
        count, passed, reason = score_fresh_news(news, news_days=30, now=NOW)
        assert count == 2
        assert passed is True
        assert reason is None

    def test_no_fresh_items_fails(self):
        count, passed, reason = score_fresh_news(fake_news([("FDA approval", NOW - 60 * 86400)]), now=NOW)
        assert count == 0
        assert passed is False
        assert reason == "no_fresh_development"

    def test_missing_news_fails(self):
        count, passed, reason = score_fresh_news(None, now=NOW)
        assert count == 0
        assert passed is False
        assert reason == "no_news"


class TestScreenNewHighs:
    def test_pass_via_new_high_within_tolerance(self):
        history = fake_history([
            ((2026, 1, 2), 100.0, 98.0),
            ((2026, 2, 1), 99.0, 98.5),
        ])
        result = screen_new_highs(
            "VRTX",
            fetch_history=lambda t: history,
            fetch_news=lambda t: [],
            now=NOW,
        )
        assert result.passed is True
        assert result.high == 100.0
        assert result.close == 98.5
        assert result.distance_from_high == pytest.approx(0.015)
        assert result.days_since_high == 30
        assert result.fresh_news_count == 0
        assert result.reason == ""

    def test_pass_via_fresh_news_only(self):
        history = fake_history([
            ((2026, 1, 2), 100.0, 98.0),
            ((2026, 2, 1), 85.0, 80.0),
        ])
        news = fake_news([
            ("FDA approves new indication", NOW - 2 * 86400),
            ("Earnings call transcript", NOW - 5 * 86400),
        ])
        result = screen_new_highs(
            "VRTX",
            fetch_history=lambda t: history,
            fetch_news=lambda t: news,
            now=NOW,
        )
        assert result.passed is True
        assert result.distance_from_high == pytest.approx(0.20)
        assert result.fresh_news_count == 1
        assert result.reason == ""

    def test_fail_below_high_and_no_fresh_news(self):
        history = fake_history([
            ((2026, 1, 2), 100.0, 98.0),
            ((2026, 2, 1), 85.0, 80.0),
        ])
        news = fake_news([("Earnings call transcript", NOW - 5 * 86400)])
        result = screen_new_highs(
            "VRTX",
            fetch_history=lambda t: history,
            fetch_news=lambda t: news,
            now=NOW,
        )
        assert result.passed is False
        assert result.distance_from_high == pytest.approx(0.20)
        assert result.fresh_news_count == 0
        assert result.reason == "below_high_and_no_news"

    def test_fail_when_price_history_missing(self):
        result = screen_new_highs(
            "FAKE",
            fetch_history=lambda t: pd.DataFrame(),
            fetch_news=lambda t: [],
            now=NOW,
        )
        assert result.passed is False
        assert result.high is None
        assert result.close is None
        assert result.distance_from_high is None
        assert result.days_since_high is None
        assert result.reason == "no_price_history"

    def test_fail_when_news_missing(self):
        history = fake_history([
            ((2026, 1, 2), 100.0, 98.0),
            ((2026, 2, 1), 85.0, 80.0),
        ])
        result = screen_new_highs(
            "VRTX",
            fetch_history=lambda t: history,
            fetch_news=lambda t: None,
            now=NOW,
        )
        assert result.passed is False
        assert result.reason == "no_news"
