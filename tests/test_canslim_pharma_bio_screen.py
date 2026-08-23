"""CANSLIM criterion C universe screen — functional map over pharma/biotech.

Covered behaviors:
    1. fetch_universe_tickers paginates the yfinance screener and dedupes.
    2. screen_universe maps the criterion-C test over tickers, order-preserving.
    3. classify / summarize partition results into PASS / FAIL / FILTERED.
    4. write_screen_ods emits a styled .ods (PASS green, FAIL red, FILTERED yellow).
"""
from unittest.mock import patch

import pytest

from fentu.canslim.pharma_bio_screen import (
    classify,
    fetch_universe_tickers,
    screen_universe,
    summarize,
    write_screen_ods,
)
from fentu.canslim.screen import ScreenResult


def fake_screen_result(ticker, passed, reason="", growth=0.30, eps=0.65):
    return ScreenResult(
        ticker=ticker,
        current_eps=eps,
        prior_year_eps=0.50,
        current_period="2025-03-31",
        prior_period="2024-03-31",
        growth=growth,
        passed=passed,
        reason=reason,
    )


def test_fetch_universe_tickers_paginates_and_dedupes():
    page1 = {"total": 3, "quotes": [
        {"symbol": "VRTX", "shortName": "Vertex Pharmaceuticals"},
        {"symbol": "ZYME", "shortName": "Zymeworks Inc."},
    ]}
    page2 = {"total": 3, "quotes": [
        {"symbol": "VRTX", "shortName": "Vertex Pharmaceuticals"},
        {"symbol": "REGN", "shortName": "Regeneron"},
    ]}
    with patch("yfinance.screener.screen", side_effect=[page1, page2]):
        universe = fetch_universe_tickers()
    assert universe == [
        ("VRTX", "Vertex Pharmaceuticals"),
        ("ZYME", "Zymeworks Inc."),
        ("REGN", "Regeneron"),
    ]


def test_fetch_universe_tickers_respects_limit():
    page = {"total": 3, "quotes": [
        {"symbol": "VRTX", "shortName": "Vertex"},
        {"symbol": "REGN", "shortName": "Regeneron"},
    ]}
    with patch("yfinance.screener.screen", return_value=page):
        universe = fetch_universe_tickers(limit=1)
    assert universe == [("VRTX", "Vertex")]


def test_screen_universe_maps_in_order_with_injected_scorer():
    def fake_score(ticker, min_growth=0.20):
        return fake_screen_result(ticker, passed=ticker in {"PFE", "MRK"})

    results = screen_universe(["PFE", "XFOR", "MRK"], score=fake_score, workers=2)
    assert [r.ticker for r in results] == ["PFE", "XFOR", "MRK"]
    assert [r.passed for r in results] == [True, False, True]


def test_classify_partitions_verdicts():
    assert classify(fake_screen_result("PFE", passed=True)) == "PASS"
    assert classify(fake_screen_result("MRK", passed=False, reason="below_threshold")) == "FAIL"
    assert classify(fake_screen_result("XFOR", passed=False, reason="negative_base")) == "FAIL"
    assert classify(fake_screen_result("XFOR", passed=False, reason="no_quarterly_data")) == "FILTERED"
    assert classify(fake_screen_result("XFOR", passed=False, reason="missing_eps")) == "FILTERED"


def test_summarize_counts_each_verdict():
    results = [
        fake_screen_result("PFE", passed=True),
        fake_screen_result("MRK", passed=False, reason="below_threshold"),
        fake_screen_result("XFOR", passed=False, reason="no_quarterly_data"),
        fake_screen_result("REGN", passed=True),
    ]
    assert summarize(results) == (2, 1, 1)


def test_write_screen_ods_contains_styled_rows(tmp_path):
    results = [
        fake_screen_result("PFE", passed=True),
        fake_screen_result("MRK", passed=False, reason="below_threshold", growth=-0.1),
        fake_screen_result("XFOR", passed=False, reason="no_quarterly_data", eps=None, growth=None),
    ]
    path = str(tmp_path / "screen.ods")
    write_screen_ods(results, {"PFE": "Pfizer", "MRK": "Merck", "XFOR": "X4 Pharma"}, path)

    import zipfile

    with zipfile.ZipFile(path) as archive:
        content = archive.read("content.xml").decode("utf-8")
    assert "CriterionC" in content
    assert "PFE" in content and "XFOR" in content
    assert 'style-name="pass"' in content
    assert 'style-name="fail"' in content
    assert 'style-name="filtered"' in content