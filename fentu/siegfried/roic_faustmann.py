"""Austrian Investing II screens (Spitznagel, The Dao of Capital, ch. 10).

Definitions from Chapter Ten, "Austrian Investing II: Siegfried":

    ROIC — "best calculated by dividing a company's EBIT (operating earnings
    before interest and tax expenses are deducted) by its invested capital
    (the operating capital required to generate that EBIT)."

        annual ROIC = EBIT / invested_capital

    measured as the rolling 10-year median of that annual ratio —
    "Rolling 10-Year ROIC (Median)" (Figure 10.1), part of a "highly robust
    screen (meaning wild numbers don't have an undue affect)". A firm with
    fewer than ten years of paired EBIT / invested-capital history gets no
    ROIC at all (the "-" placeholder) — never a short-window or single-year
    stand-in.

    Faustmann ratio — "a low market capitalization (of common equity) over
    net worth (or invested capital plus cash minus debt and preferred equity)
    ratio."

        net_worth       = invested_capital + cash - debt - preferred_equity
        faustmann_ratio = market_cap / net_worth

The two-pronged screen: high ROIC flags roundabout, productive firms; a low
Faustmann ratio flags those the market underappreciates. Neither suffices in
isolation. (Spitznagel ignores financials/banks in his own toy screen.)

This module reads a ticker list from an .ods or .xlsx/.xls workbook (e.g. the
repo's ``data/sp500_ticker.ods`` / ``data/ndx100_ticker.ods`` universes, or
any workbook from any folder), pulls the raw elements from yfinance (single
I/O seam: ``fetch_fundamentals``), and derives ROIC, net worth, and the
Faustmann ratio as new columns. Output format follows the input extension.

Usage:
    uv run python -m fentu.siegfried.roic_faustmann data/sp500_ticker.ods
    uv run python -m fentu.siegfried.roic_faustmann data/ndx100_ticker.ods -o out.ods
    uv run python -m fentu.siegfried.roic_faustmann /any/folder/tickers.xlsx
"""
import argparse
import statistics
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd

ROIC_WINDOW_YEARS = 10

TICKER_COLUMNS = ("ticker", "tickers", "symbol", "symbols")

CASH_ROWS = (
    "Cash Cash Equivalents And Short Term Investments",
    "Cash And Cash Equivalents",
    "Cash Equivalents",
    "Cash Financial",
)
PREFERRED_ROWS = (
    "Preferred Stock Equity",
    "Preferred Stock",
    "Preferred Securities Outside Stock Equity",
)

OUTPUT_COLUMNS = [
    "ticker",
    "ebit",
    "invested_capital",
    "roic",
    "market_cap",
    "cash",
    "debt",
    "preferred_equity",
    "net_worth",
    "faustmann_ratio",
]


def _latest(statement: Optional[pd.DataFrame], row: str) -> Optional[float]:
    """Most recent annual value of a yfinance statement row, or None."""
    if statement is None or statement.empty or row not in statement.index:
        return None
    value = statement.loc[row].iloc[0]
    return None if pd.isna(value) else float(value)


def _latest_any(statement: Optional[pd.DataFrame], rows: tuple) -> Optional[float]:
    for row in rows:
        value = _latest(statement, row)
        if value is not None:
            return value
    return None


def _annual_roics(
    income_stmt: Optional[pd.DataFrame],
    balance_sheet: Optional[pd.DataFrame],
) -> List[float]:
    """Annual EBIT / invested capital per period, most recent first.

    Years missing either leg, or with zero invested capital, are skipped —
    they contribute nothing to the rolling window.
    """
    if income_stmt is None or income_stmt.empty or balance_sheet is None or balance_sheet.empty:
        return []
    if "EBIT" not in income_stmt.index or "Invested Capital" not in balance_sheet.index:
        return []
    roics = []
    for period in income_stmt.columns:
        if period not in balance_sheet.columns:
            continue
        ebit = income_stmt.at["EBIT", period]
        invested = balance_sheet.at["Invested Capital", period]
        if pd.isna(ebit) or pd.isna(invested) or not invested:
            continue
        roics.append(float(ebit) / float(invested))
    return roics


def fetch_fundamentals(ticker: str) -> Dict[str, Optional[float]]:
    """Single I/O seam: raw ROIC / Faustmann elements for one ticker.

    Pulls the full annual history from the income statement and balance
    sheet (paired per year into annual ROICs for the rolling window); the
    latest invested capital, cash, debt, and preferred equity from the
    annual balance sheet; and market cap from fast_info (falling back to
    info). Missing pieces come back as None — never fabricated.
    """
    import yfinance as yf

    ticker_obj = yf.Ticker(ticker)
    income_stmt = ticker_obj.income_stmt
    balance_sheet = ticker_obj.balance_sheet

    market_cap = None
    try:
        market_cap = ticker_obj.fast_info.get("market_cap")
    except Exception:
        market_cap = None
    if market_cap is None:
        try:
            market_cap = ticker_obj.info.get("marketCap")
        except Exception:
            market_cap = None

    return {
        "ticker": ticker,
        "ebit": _latest(income_stmt, "EBIT"),
        "invested_capital": _latest(balance_sheet, "Invested Capital"),
        "roic_history": _annual_roics(income_stmt, balance_sheet),
        "market_cap": None if market_cap is None else float(market_cap),
        "cash": _latest_any(balance_sheet, CASH_ROWS),
        "debt": _latest(balance_sheet, "Total Debt"),
        "preferred_equity": _latest_any(balance_sheet, PREFERRED_ROWS),
    }


def derive_roic(roic_history: Optional[List[float]], window: int = ROIC_WINDOW_YEARS) -> Optional[float]:
    """Rolling 10-year ROIC (median): the median of the trailing ``window``
    annual ROICs, history ordered most recent first.

    Fewer than ``window`` years of history is not enough data — None (the
    "-" placeholder), never a short-window or single-year stand-in. The
    median (not the mean) keeps the screen "highly robust", so "wild numbers
    don't have an undue affect" (Spitznagel, ch. 10).
    """
    if not roic_history or len(roic_history) < window:
        return None
    return statistics.median(roic_history[:window])


def derive_net_worth(
    invested_capital: Optional[float],
    cash: Optional[float],
    debt: Optional[float],
    preferred_equity: Optional[float],
) -> Optional[float]:
    """Net worth = invested capital + cash - debt - preferred equity.

    Absent cash/debt/preferred line items are treated as zero (a firm with
    no preferred equity reports nothing); only a missing invested capital
    blocks the derivation.
    """
    if invested_capital is None:
        return None
    return invested_capital + (cash or 0.0) - (debt or 0.0) - (preferred_equity or 0.0)


def derive_faustmann_ratio(market_cap: Optional[float], net_worth: Optional[float]) -> Optional[float]:
    """Faustmann ratio = market cap / net worth; None when either leg is missing or non-positive."""
    if market_cap is None or not net_worth:
        return None
    return market_cap / net_worth


def build_table(
    tickers: List[str],
    fetch: Callable[[str], Dict[str, Optional[float]]] = fetch_fundamentals,
) -> pd.DataFrame:
    """One row per ticker: raw elements plus derived roic / net_worth / faustmann_ratio."""
    rows = []
    for ticker in tickers:
        raw = fetch(ticker)
        net_worth = derive_net_worth(
            raw.get("invested_capital"), raw.get("cash"), raw.get("debt"), raw.get("preferred_equity")
        )
        rows.append(
            {
                "ticker": raw.get("ticker", ticker),
                "ebit": raw.get("ebit"),
                "invested_capital": raw.get("invested_capital"),
                "roic": derive_roic(raw.get("roic_history")),
                "market_cap": raw.get("market_cap"),
                "cash": raw.get("cash"),
                "debt": raw.get("debt"),
                "preferred_equity": raw.get("preferred_equity"),
                "net_worth": net_worth,
                "faustmann_ratio": derive_faustmann_ratio(raw.get("market_cap"), net_worth),
            }
        )
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def _ods_cell_text(cell) -> str:
    """Text of an ODF cell: paragraphs first (this repo's writer), then value attributes (external tools)."""
    from odf.namespaces import OFFICENS
    from odf.text import P

    for paragraph in cell.getElementsByType(P):
        if str(paragraph):
            return str(paragraph)
    for attr in ("value", "string-value"):
        value = cell.getAttrNS(OFFICENS, attr)
        if value is not None:
            return str(value)
    return ""


def _ods_reader(path: str) -> List[str]:
    """Ticker list from an .ods workbook via odfpy (repo writer emits no office:value attrs)."""
    from odf.opendocument import load
    from odf.table import TableCell, TableRow

    doc = load(path)
    rows = doc.spreadsheet.getElementsByType(TableRow)
    if not rows:
        return []
    header = [_ods_cell_text(c) for c in rows[0].getElementsByType(TableCell)]
    lowered = {text.strip().lower(): i for i, text in enumerate(header) if text.strip()}
    column = next((lowered[name] for name in TICKER_COLUMNS if name in lowered), 0)
    out = []
    for row in rows[1:]:
        cells = [_ods_cell_text(c) for c in row.getElementsByType(TableCell)]
        if column < len(cells) and cells[column].strip():
            out.append(cells[column].strip())
    return out


def _xlsx_reader(path: str) -> List[str]:
    """Ticker list from an .xlsx/.xls workbook: ticker/symbol column if present, else column one."""
    frame = pd.read_excel(path)
    lowered = {str(c).strip().lower(): c for c in frame.columns}
    column = next((lowered[name] for name in TICKER_COLUMNS if name in lowered), frame.columns[0])
    return [str(t).strip() for t in frame[column].dropna() if str(t).strip()]


def read_tickers(path: str) -> List[str]:
    """Ticker list from a workbook of any supported format (.ods, .xlsx, .xls), from any folder."""
    suffix = Path(path).suffix.lower()
    if suffix == ".ods":
        return _ods_reader(path)
    return _xlsx_reader(path)


def _ods_table(path: str) -> pd.DataFrame:
    """Full table from an .ods workbook: first row is the header, the rest are records."""
    from odf.opendocument import load
    from odf.table import TableCell, TableRow

    doc = load(path)
    rows = doc.spreadsheet.getElementsByType(TableRow)
    if not rows:
        return pd.DataFrame()
    header = [_ods_cell_text(c) for c in rows[0].getElementsByType(TableCell)]
    records = []
    for row in rows[1:]:
        cells = [_ods_cell_text(c) for c in row.getElementsByType(TableCell)]
        cells = (cells + [""] * len(header))[: len(header)]
        records.append(cells)
    return pd.DataFrame(records, columns=header)


def read_table(path: str) -> pd.DataFrame:
    """A written siegfried workbook (.ods or .xlsx) back into a typed DataFrame.

    Every non-ticker column is coerced to numeric; unparseable entries (the
    "-" placeholders written for missing values) become NaN. Thousands
    separators from ``format_dollars`` are stripped before coercion.
    """
    frame = _ods_table(path) if Path(path).suffix.lower() == ".ods" else pd.read_excel(path)
    for column in frame.columns:
        if str(column).strip().lower() not in TICKER_COLUMNS:
            frame[column] = pd.to_numeric(frame[column].astype(str).str.replace(",", ""), errors="coerce")
    return frame


def _format_cell(value) -> str:
    return "-" if value is None else str(value)


def _write_ods(table: pd.DataFrame, path: str) -> None:
    """Write the table as an .ods with a styled header (same odfpy pattern as the CANSLIM screens)."""
    from odf.opendocument import OpenDocumentSpreadsheet
    from odf.style import Style, TableCellProperties, TableColumnProperties, TextProperties
    from odf.table import Table, TableColumn, TableCell, TableRow
    from odf.text import P

    doc = OpenDocumentSpreadsheet()
    header_style = Style(name="header", family="table-cell")
    header_style.addElement(TableCellProperties(backgroundcolor="#1F4E79"))
    header_style.addElement(TextProperties(color="#FFFFFF", fontweight="bold"))
    doc.automaticstyles.addElement(header_style)
    column_style = Style(name="col", family="table-column")
    column_style.addElement(TableColumnProperties(columnwidth="3.2cm"))
    doc.automaticstyles.addElement(column_style)

    table_el = Table(name="Siegfried")
    table_el.addElement(TableColumn(stylename=column_style))

    header_row = TableRow()
    for column in table.columns:
        cell = TableCell(stylename="header")
        cell.addElement(P(text=str(column)))
        header_row.addElement(cell)
    table_el.addElement(header_row)

    for _, row_data in table.iterrows():
        row_el = TableRow()
        for value in row_data:
            cell = TableCell()
            cell.addElement(P(text=_format_cell(value)))
            row_el.addElement(cell)
        table_el.addElement(row_el)

    doc.spreadsheet.addElement(table_el)
    doc.save(path)


def write_table(table: pd.DataFrame, path: str) -> None:
    """Write the table to .ods or .xlsx/.xls, matching the path extension."""
    suffix = Path(path).suffix.lower()
    if suffix == ".ods":
        _write_ods(table, path)
    else:
        table.to_excel(path, index=False)


def default_output_path(input_path: str) -> str:
    """Default output: same folder and extension as the input, ``_siegfried`` stem suffix."""
    source = Path(input_path)
    return str(source.with_name(f"{source.stem}_siegfried{source.suffix}"))


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Siegfried screen: EBIT, invested capital, ROIC, and Faustmann ratio columns"
    )
    parser.add_argument("excel", help="ticker workbook (.ods, .xlsx, .xls), e.g. data/sp500_ticker.ods")
    parser.add_argument("-o", "--output", help="output Excel path (default: <input stem>_siegfried.xlsx)")
    args = parser.parse_args(argv)

    tickers = read_tickers(args.excel)
    print(f"fetching fundamentals for {len(tickers)} tickers ...")
    table = build_table(tickers)

    output = args.output or default_output_path(args.excel)
    write_table(table, output)
    print(f"wrote {len(table)} rows -> {output}")
    insufficient = int(table["roic"].isna().sum())
    if insufficient:
        print(f"note: {insufficient}/{len(table)} tickers lack the 10-year ROIC history (roic '-'), no single-year stand-in")
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
