"""
Siegfried screen — ROIC and Faustmann ratio derivations (Dao of Capital ch. 10).

Covered behaviors:
    1. ROIC is the rolling 10-year median of annual EBIT / invested
       capital ratios (Dao of Capital ch. 10, Figure 10.1); fewer than ten
       years of history yields None — never a single-year stand-in.
    2. Net worth = invested capital + cash - debt - preferred equity; missing
       cash/debt/preferred lines count as zero, missing invested capital -> None.
    3. Faustmann ratio = market cap / net worth; missing or non-positive
       denominators yield None, never a fabricated number.
    4. build_table assembles one row per ticker from an injected fetch seam
       (network-free, like the CANSLIM screens).
    5. read_tickers finds a ticker/symbol column or falls back to column one.

Mock Object seam: build_table takes the fetch callable as an injected
parameter (like screen_shares_outstanding's `fetch`), defaulting to
fetch_fundamentals.
"""
import pandas as pd
import pytest
from datetime import date

from fentu.siegfried.roic_faustmann import (
    _annual_roic_pairs,
    _merge_roic_histories,
    build_table,
    default_output_path,
    derive_faustmann_ratio,
    derive_net_worth,
    derive_roic,
    read_tickers,
    roic_breakdown,
    write_table,
)

HISTORY_020 = [0.20] * 10


def test_roic_is_rolling_ten_year_median():
    assert derive_roic(HISTORY_020) == pytest.approx(0.20)


def test_roic_median_tames_wild_numbers():
    assert derive_roic([0.20] * 9 + [40.0]) == pytest.approx(0.20)


def test_roic_uses_only_the_ten_most_recent_years():
    assert derive_roic([0.20] * 10 + [9.99] * 5) == pytest.approx(0.20)


def test_roic_none_when_history_shorter_than_ten_years():
    assert derive_roic([0.20] * 9) is None
    assert derive_roic([]) is None
    assert derive_roic(None) is None


def _statement(row: str, values: list) -> pd.DataFrame:
    """A one-row yfinance-style statement: periods as columns, most recent first."""
    return pd.DataFrame(
        [values], columns=[f"Y{i}" for i in range(len(values))], index=[row]
    )


def test_annual_roic_pairs_pair_years_and_skip_incomplete_ones():
    income = _statement("EBIT", [2_000_000, 3_000_000, 1_000_000])
    balance = _statement("Invested Capital", [10_000_000, float("nan"), 0.0])
    pairs = _annual_roic_pairs(income, balance)
    assert len(pairs) == 1
    assert pairs[0][0] == "Y0"
    assert pairs[0][1] == pytest.approx(0.20)


def test_annual_roic_pairs_skip_years_without_both_legs():
    income = _statement("EBIT", [2.0, 4.0])
    balance = _statement("Invested Capital", [10.0])
    pairs = _annual_roic_pairs(income, balance)
    assert len(pairs) == 1
    assert pairs[0][0] == "Y0"
    assert pairs[0][1] == pytest.approx(0.2)


def test_annual_roic_pairs_empty_when_statements_or_rows_missing():
    assert _annual_roic_pairs(None, None) == []
    assert _annual_roic_pairs(pd.DataFrame(), pd.DataFrame()) == []
    assert _annual_roic_pairs(_statement("EBIT", [1.0]), _statement("Total Assets", [1.0])) == []


def test_merge_roic_histories_edgar_wins_and_yfinance_fills_gaps():
    primary = [(date(2023, 12, 31), 0.25), (date(2015, 12, 31), 0.10)]
    secondary = [
        (pd.Timestamp("2023-12-31"), 0.99),  # same fiscal year as EDGAR -> dropped
        (pd.Timestamp("2022-12-31"), 0.20),  # gap -> kept
        (pd.Timestamp("2016-06-30"), 0.15),  # within 185d of 2015-12-31 -> dropped
        (pd.Timestamp("2017-01-31"), 0.05),  # far enough -> kept
    ]
    merged = _merge_roic_histories(primary, secondary)

    assert [value for _, value in merged] == pytest.approx([0.25, 0.20, 0.05, 0.10])
    assert merged[0] == (date(2023, 12, 31), 0.25)  # EDGAR's value wins the overlap
    assert len(merged) == 4


def test_merge_roic_histories_empty_primary_keeps_secondary():
    secondary = [(pd.Timestamp("2022-12-31"), 0.20)]
    assert _merge_roic_histories([], secondary) == secondary


def test_roic_breakdown_windows_to_ten_years_and_medians():
    pairs = [(date(2025, 12, 31), 0.20)] * 10 + [(date(2014, 12, 31), 0.40)]
    windowed, median = roic_breakdown(pairs)
    assert len(windowed) == 10
    assert median == pytest.approx(0.20)


def test_roic_breakdown_placeholder_when_history_short():
    pairs = [(date(2025, 12, 31), 0.20)] * 9
    windowed, median = roic_breakdown(pairs)
    assert len(windowed) == 9
    assert median is None


def test_build_table_reports_progress_per_ticker():
    calls = []
    fetch = lambda t: {"ticker": t, "roic_history": HISTORY_020}
    build_table(["A", "B"], fetch=fetch, progress=lambda done, total, t: calls.append((done, total, t)))
    assert calls == [(1, 2, "A"), (2, 2, "B")]


def test_net_worth_formula():
    net_worth = derive_net_worth(
        invested_capital=10_000_000, cash=500_000, debt=1_000_000, preferred_equity=250_000
    )
    assert net_worth == pytest.approx(9_250_000)


def test_net_worth_treats_absent_lines_as_zero():
    assert derive_net_worth(10_000_000, None, None, None) == pytest.approx(10_000_000)
    assert derive_net_worth(None, 500_000, 1_000_000, None) is None


def test_faustmann_ratio_is_market_cap_over_net_worth():
    assert derive_faustmann_ratio(market_cap=18_500_000, net_worth=9_250_000) == pytest.approx(2.0)


def test_faustmann_ratio_none_on_missing_or_zero_legs():
    assert derive_faustmann_ratio(None, 9_250_000) is None
    assert derive_faustmann_ratio(18_500_000, None) is None
    assert derive_faustmann_ratio(18_500_000, 0) is None


def test_build_table_derives_columns_from_injected_fetch():
    fetch = lambda t: {
        "ticker": t,
        "ebit": 2_000_000,
        "invested_capital": 10_000_000,
        "roic_history": HISTORY_020,
        "market_cap": 18_500_000,
        "cash": 500_000,
        "debt": 1_000_000,
        "preferred_equity": 250_000,
    }
    table = build_table(["VRTX"], fetch=fetch)

    row = table.iloc[0]
    assert row["ticker"] == "VRTX"
    assert row["roic"] == pytest.approx(0.20)
    assert row["net_worth"] == pytest.approx(9_250_000)
    assert row["faustmann_ratio"] == pytest.approx(2.0)


def test_build_table_placeholder_roic_when_history_is_short():
    fetch = lambda t: {
        "ticker": t,
        "ebit": 2_000_000,
        "invested_capital": 10_000_000,
        "roic_history": [0.20] * 9,
        "market_cap": 18_500_000,
        "cash": 500_000,
        "debt": 1_000_000,
        "preferred_equity": 250_000,
    }
    table = build_table(["VRTX"], fetch=fetch)

    row = table.iloc[0]
    assert pd.isna(row["roic"])  # "-" placeholder, never the single-year 0.20
    assert row["net_worth"] == pytest.approx(9_250_000)


def test_build_table_keeps_raw_elements_and_handles_missing_data():
    fetch = lambda t: {
        "ticker": t,
        "ebit": None,
        "invested_capital": None,
        "market_cap": None,
        "cash": None,
        "debt": None,
        "preferred_equity": None,
    }
    table = build_table(["XYZ"], fetch=fetch)

    assert len(table) == 1
    assert pd.isna(table.iloc[0]["roic"])
    assert pd.isna(table.iloc[0]["net_worth"])
    assert pd.isna(table.iloc[0]["faustmann_ratio"])


def test_read_tickers_uses_symbol_column(tmp_path):
    frame = pd.DataFrame({"name": ["Vertex"], "Symbol": ["VRTX"]})
    path = tmp_path / "universe.xlsx"
    frame.to_excel(path, index=False)
    assert read_tickers(str(path)) == ["VRTX"]


def test_read_tickers_falls_back_to_first_column(tmp_path):
    frame = pd.DataFrame({"col": ["AAPL", "MSFT", None]})
    path = tmp_path / "universe.xlsx"
    frame.to_excel(path, index=False)
    assert read_tickers(str(path)) == ["AAPL", "MSFT"]


def _write_ods_with(sheet_name: str, header: list, rows: list, path: str) -> None:
    """Minimal .ods written with odfpy, the way this repo's screens write them (text-only cells)."""
    from odf.opendocument import OpenDocumentSpreadsheet
    from odf.table import Table, TableCell, TableRow
    from odf.text import P

    doc = OpenDocumentSpreadsheet()
    table_el = Table(name=sheet_name)
    for cells in [header] + rows:
        row_el = TableRow()
        for text in cells:
            cell = TableCell()
            cell.addElement(P(text=text))
            row_el.addElement(cell)
        table_el.addElement(row_el)
    doc.spreadsheet.addElement(table_el)
    doc.save(path)


def test_read_tickers_reads_repo_style_ods(tmp_path):
    path = str(tmp_path / "ndx.ods")
    _write_ods_with("Universe", ["Company", "Ticker"], [["Microsoft", "MSFT"], ["Apple", "AAPL"]], path)
    assert read_tickers(path) == ["MSFT", "AAPL"]


def test_read_tickers_ods_without_ticker_header_uses_first_column(tmp_path):
    path = str(tmp_path / "names.ods")
    _write_ods_with("Universe", ["Company"], [["NVDA"], ["TSLA"]], path)
    assert read_tickers(path) == ["NVDA", "TSLA"]


def test_write_table_ods_round_trips_through_reader(tmp_path):
    table = build_table(
        ["VRTX"],
        fetch=lambda t: {
            "ticker": t,
            "ebit": 2_000_000,
            "invested_capital": 10_000_000,
            "roic_history": HISTORY_020,
            "market_cap": 18_500_000,
            "cash": 500_000,
            "debt": 1_000_000,
            "preferred_equity": 250_000,
        },
    )
    path = str(tmp_path / "out.ods")
    write_table(table, path)
    assert read_tickers(path) == ["VRTX"]


def test_write_table_xlsx_round_trips_through_reader(tmp_path):
    table = build_table(["VRTX"], fetch=lambda t: {"ticker": t, "ebit": 2_000_000})
    path = str(tmp_path / "out.xlsx")
    write_table(table, path)
    assert read_tickers(path) == ["VRTX"]


def test_default_output_path_matches_input_extension(tmp_path):
    assert default_output_path(str(tmp_path / "sp500_ticker.ods")) == str(tmp_path / "sp500_ticker_siegfried.ods")
    assert default_output_path(str(tmp_path / "ndx100_ticker.xlsx")) == str(tmp_path / "ndx100_ticker_siegfried.xlsx")
