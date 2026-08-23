"""CANSLIM criterion N — fresh new highs + headline confirmation/veto verdict.

Covered behaviors (defaults: tolerance=0.05, max_high_age_days=30,
volume_multiplier=1.5x the prior 10-session mean in these fixtures):
    1. Price gate PASS: e.g. `fresh_breakout_history` — closes drift
       90.0 -> 99.0 for 10 sessions at 1,000,000 shares, then the last day
       prints close=102.9 / high=105.0 on 2,000,000 shares. Close == closing
       high (distance 0.0% <= 5%), high is 0 days old (<= 30), and breakout
       volume 2.0M >= 1.5 x 1.0M baseline -> reason=None.
    2. Price-gate failures carry honest reasons:
       - "below_high": high=100.0, latest close=80.0 -> distance 20% > 5%
         tolerance (test_below_high_fails).
       - "stale_high": close 98.5 vs high 100.0 (1.5% away, within 5%) but
         the high was set ~17 months ago > 30 days -> fails anyway
         (test_stale_high_fails_even_when_close_is_near).
       - "low_volume_breakout": same shape as the passing fixture but the
         breakout day trades only 1.1M < 1.5 x 1.0M -> vetoed
         (test_low_volume_breakout_fails).
       - "no_price_history": empty DataFrame -> all stats None
         (test_missing_history_fails_with_reason). A single row with no
         Volume column leaves volume_confirmed=None and does NOT veto
         (test_unmeasurable_volume_does_not_veto).
    3. News leg over a 30-day lookback ending at NOW=1_700_000_000: a
       "FDA approval" headline 1 day old counts positive; "Phase 3 trial
       fails" 2 days old counts negative and vetoes a passing price gate
       ("negative_news"); a positive headline cannot rescue a stock 20%
       below its high (still "below_high"). Headline matching is
       word-boundary based ("Analysts find value in the industry" must not
       match keyword "ind") and reads both the legacy
       (title/providerPublishTime) and current
       (content.title/content.pubDate, e.g. "2026-08-20T19:50:00Z" ->
       unix 1787255400) yfinance schemas.
    4. Neutral headlines ("Earnings call transcript", a 60-day-old item
       past the cutoff, or news=None) are ignored or count as "no_news" —
       neither rescues nor vetoes.

Mock Object seam: `screen_new_highs` takes the two fetch callables as
parameters (same injection pattern as pharma_bio_screen's `score`), so no
network and no patching are needed.
"""
from datetime import datetime

import pandas as pd
import pytest

from fentu.canslim.new_highs import (
    classify_headline,
    compute_high_stats,
    parse_news_item,
    score_fresh_news,
    score_new_high,
    screen_new_highs,
)

NOW = 1_700_000_000


def fake_history(rows):
    """rows: iterable of ((y, m, d), high, close[, volume]) -> daily OHLCV DataFrame."""
    index = pd.DatetimeIndex([datetime(*d) for d, *_ in rows])
    closes = [r[2] for r in rows]
    volumes = [r[3] if len(r) > 3 else 0 for r in rows]
    return pd.DataFrame(
        {"Open": closes, "High": [h for _, h, *_ in rows], "Low": closes, "Close": closes, "Volume": volumes},
        index=index,
    )


def legacy_news(items):
    """items: (title, unix_ts) -> legacy yfinance news dicts."""
    return [{"title": t, "providerPublishTime": ts} for t, ts in items]


def current_news(items):
    """items: (title, iso_ts) -> current yfinance news dicts (content.pubDate)."""
    return [{"content": {"title": t, "pubDate": ts}} for t, ts in items]


@pytest.fixture
def fresh_breakout_history():
    """Close jumps to a new closing high on the last day, on heavy volume."""
    normal_days = [((2026, 1, 1 + i), 90.0 + i, 90.0 + i, 1_000_000) for i in range(10)]
    return fake_history(normal_days + [((2026, 1, 20), 105.0, 102.9, 2_000_000)])


class TestScoreNewHigh:
    def test_fresh_high_volume_confirmed_passes(self, fresh_breakout_history):
        signal = score_new_high(fresh_breakout_history, tolerance=0.05, max_high_age_days=30)
        assert signal.reason is None
        assert signal.volume_confirmed is True
        assert signal.high == 102.9
        assert signal.close == 102.9
        assert signal.distance_from_high == pytest.approx(0.0)

    def test_below_high_fails(self):
        history = fake_history([
            ((2026, 1, 2), 100.0, 100.0, 1_000_000),
            ((2026, 2, 1), 85.0, 80.0, 1_000_000),
        ])
        signal = score_new_high(history, tolerance=0.05, max_high_age_days=60)
        assert signal.reason == "below_high"
        assert signal.distance_from_high == pytest.approx(0.20)

    def test_stale_high_fails_even_when_close_is_near(self):
        history = fake_history([
            ((2025, 1, 2), 100.0, 100.0, 1_000_000),
            ((2025, 6, 2), 99.0, 98.5, 1_000_000),
            ((2026, 6, 1), 99.0, 98.5, 1_000_000),
        ])
        signal = score_new_high(history, tolerance=0.05, max_high_age_days=30)
        assert signal.reason == "stale_high"
        assert signal.distance_from_high == pytest.approx(0.015)

    def test_low_volume_breakout_fails(self):
        history = fake_history([
            ((2026, 1, 1 + i), 90.0 + i, 90.0 + i, 1_000_000) for i in range(10)
        ] + [((2026, 1, 20), 105.0, 102.9, 1_100_000)])
        signal = score_new_high(history, tolerance=0.05, max_high_age_days=30)
        assert signal.reason == "low_volume_breakout"
        assert signal.volume_confirmed is False

    def test_unmeasurable_volume_does_not_veto(self):
        history = fake_history([((2026, 1, 2), 100.0, 98.0)])
        signal = score_new_high(history, tolerance=0.05, max_high_age_days=30)
        assert signal.reason is None
        assert signal.volume_confirmed is None

    def test_missing_history_fails_with_reason(self):
        signal = score_new_high(pd.DataFrame())
        assert signal.reason == "no_price_history"
        assert signal.high is None and signal.close is None
        assert signal.distance_from_high is None and signal.days_since_high is None


class TestComputeHighStats:
    def test_reason_none_for_usable_history(self):
        stats = compute_high_stats(fake_history([((2026, 1, 2), 100.0, 98.0)]))
        assert stats.reason is None


class TestParseNewsItem:
    def test_legacy_schema(self):
        assert parse_news_item({"title": "FDA approval", "providerPublishTime": 123}) == ("FDA approval", 123.0)

    def test_current_schema(self):
        item = {"content": {"title": "Vertex partnership", "pubDate": "2026-08-20T19:50:00Z"}}
        title, ts = parse_news_item(item)
        assert title == "Vertex partnership"
        assert ts == pytest.approx(1787255400.0, abs=5)

    def test_unknown_shape_returns_none(self):
        assert parse_news_item({"irrelevant": True}) is None
        assert parse_news_item("junk") is None


class TestClassifyHeadline:
    @pytest.mark.parametrize(
        "title,expected",
        [
            ("FDA approves new therapy", "positive"),
            ("Company announces licensing partnership", "positive"),
            ("Phase 3 trial fails to meet endpoint", "negative"),
            ("FDA issues complete response letter", "negative"),
            ("Shares plunge after clinical hold", "negative"),
            ("Earnings call transcript", "neutral"),
            ("Q2 results: revenue in line, guidance unchanged", "neutral"),
            ("Analysts find value in the industry", "neutral"),  # 'ind' must not substring-match
        ],
    )
    def test_classification(self, title, expected):
        assert classify_headline(title) == expected


class TestScoreFreshNews:
    def test_counts_fresh_positive_and_negative_separately(self):
        news = legacy_news([
            ("Vertex announces FDA approval of new therapy", NOW - 1 * 86400),
            ("Phase 3 trial fails primary endpoint", NOW - 3 * 86400),
            ("Earnings call transcript", NOW - 4 * 86400),
            ("Partnership signed last quarter", NOW - 60 * 86400),
        ])
        signal = score_fresh_news(news, news_days=30, now=NOW)
        assert signal.positive == 1
        assert signal.negative == 1

    def test_current_schema_parsed(self):
        news = current_news([("FDA approval of new therapy", "2023-11-14T00:00:00Z")])
        signal = score_fresh_news(news, news_days=30, now=NOW + 0 * 86400)
        assert signal.positive == 1

    def test_no_news(self):
        signal = score_fresh_news(None, now=NOW)
        assert signal.positive == 0 and signal.negative == 0
        assert signal.reason == "no_news"


class TestScreenNewHighs:
    def test_pass_on_fresh_confirmed_high_without_negative_news(self, fresh_breakout_history):
        result = screen_new_highs(
            "VRTX",
            fetch_history=lambda t: fresh_breakout_history,
            fetch_news=lambda t: legacy_news([("FDA approval", NOW - 86400)]),
            now=NOW,
        )
        assert result.passed is True
        assert result.reason == ""
        assert result.fresh_news_count == 1

    def test_positive_news_cannot_rescue_a_failed_price_gate(self):
        history = fake_history([
            ((2026, 1, 2), 100.0, 100.0, 1_000_000),
            ((2026, 2, 1), 85.0, 80.0, 1_000_000),
        ])
        result = screen_new_highs(
            "VRTX",
            fetch_history=lambda t: history,
            fetch_news=lambda t: legacy_news([("FDA approves new indication", NOW - 2 * 86400)]),
            now=NOW,
        )
        assert result.passed is False
        assert result.reason == "below_high"
        assert result.fresh_news_count == 1

    def test_negative_news_vetoes_a_passing_price_gate(self, fresh_breakout_history):
        result = screen_new_highs(
            "VRTX",
            fetch_history=lambda t: fresh_breakout_history,
            fetch_news=lambda t: legacy_news([("Phase 3 trial fails", NOW - 2 * 86400)]),
            now=NOW,
        )
        assert result.passed is False
        assert result.reason == "negative_news"
        assert result.fresh_negative_count == 1

    def test_fail_when_price_history_missing(self):
        result = screen_new_highs(
            "FAKE",
            fetch_history=lambda t: pd.DataFrame(),
            fetch_news=lambda t: [],
            now=NOW,
        )
        assert result.passed is False
        assert result.reason == "no_price_history"
        assert result.high is None

    def test_no_news_does_not_fail_a_valid_breakout(self, fresh_breakout_history):
        result = screen_new_highs(
            "VRTX",
            fetch_history=lambda t: fresh_breakout_history,
            fetch_news=lambda t: None,
            now=NOW,
        )
        assert result.passed is True
        assert result.reason == ""
