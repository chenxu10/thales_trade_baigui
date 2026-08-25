"""
Siegfried screen — ROIC and Faustmann ratio derivations (Dao of Capital ch. 10).

Covered behaviors:
    1. ROIC = EBIT / invested capital, exactly as Spitznagel defines it.
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

from fentu.siegfried.roic_faustmann import (
    build_table,
    default_output_path,
    derive_faustmann_ratio,
    derive_net_worth,
    derive_roic,
    read_tickers,
    write_table,
)


def test_roic_is_ebit_over_invested_capital():
    assert derive_roic(ebit=2_000_000, invested_capital=10_000_000) == pytest.approx(0.20)


def test_roic_none_when_ebit_missing_or_capital_zero():
    assert derive_roic(ebit=None, invested_capital=10_000_000) is None
    assert derive_roic(ebit=1_000_000, invested_capital=None) is None
    assert derive_roic(ebit=1_000_000, invested_capital=0) is None


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
