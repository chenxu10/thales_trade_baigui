"""Austrian Investing II screens (Spitznagel, The Dao of Capital, ch. 10).

Definitions from Chapter Ten, "Austrian Investing II: Siegfried":

    ROIC — "best calculated by dividing a company's EBIT (operating earnings
    before interest and tax expenses are deducted) by its invested capital
    (the operating capital required to generate that EBIT)."

        annual ROIC = EBIT / invested_capital

    Faustmann ratio — "a low market capitalization (of common equity) over
    net worth (or invested capital plus cash minus debt and preferred equity)
    ratio."

        net_worth       = invested_capital + cash - debt - preferred_equity
        faustmann_ratio = market_cap / net_worth

The two-pronged screen: high ROIC flags roundabout, productive firms; a low
Faustmann ratio flags those the market underappreciates. 

This module reads a ticker list from an .ods or .xlsx/.xls workbook (e.g. the
repo's ``data/sp500_ticker.ods`` / ``data/ndx100_ticker.ods`` universes, or
any workbook from any folder) and derives ROIC, net worth, and the
Faustmann ratio as new columns. Output format follows the input extension.

The ROIC history is backfilled from SEC EDGAR XBRL (see
``edgar_fundamentals``): yfinance annual statements reach only ~4 years back,
while EDGAR companyfacts carry 10+ years of the same 10-K facts.

Effect and output:

  * writes a NEW workbook next to the input, ``<input stem>_siegfried.<ext>``
    (the input file is never modified), one row per ticker, in this column
    order: ``ticker, ebit, invested_capital, roic, market_cap, cash, debt,
    preferred_equity, net_worth, faustmann_ratio``;

  * ``roic`` is the rolling 10-year median of the annual EBIT / invested
    capital ratios; fewer than ten years print "-" (placeholder), never a
    single-year stand-in;

  * EDGAR facts and the CIK map cache under ``data/edgar_cache/`` (fresh
    within 7 / 30 days), so re-runs skip the network for cached tickers;

  * stdout prints a live progress line (done/total, elapsed, ETA) while the
    first uncached run downloads, then the full table, a note counting
    tickers still under ten years of history, and a worked example: the
    per-year ROICs behind one ticker's final ``roic`` column (``--history
    TICKER``, default NVDA) — years inside the 10-year window marked ``*``,
    plus the median.

Usage:
    uv run python -m fentu.siegfried.roic_faustmann data/sp500_ticker.ods
    uv run python -m fentu.siegfried.roic_faustmann data/ndx100_ticker.ods -o out.ods
    uv run python -m fentu.siegfried.roic_faustmann /any/folder/tickers.xlsx
"""
import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd

from fentu.siegfried import edgar_fundamentals

ROIC_WINDOW_YEARS = 10

EDGAR_CACHE_DIR = str(Path(__file__).resolve().parents[2] / "data" / "edgar_cache")

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


def _annual_roic_pairs(
    income_stmt: Optional[pd.DataFrame],
    balance_sheet: Optional[pd.DataFrame],
) -> List[tuple]:
    """(period, annual EBIT / invested capital) pairs, most recent first.

    Years missing either leg, or with zero invested capital, are skipped —
    they contribute nothing to the rolling window.
    """
    if income_stmt is None or income_stmt.empty or balance_sheet is None or balance_sheet.empty:
        return []
    if "EBIT" not in income_stmt.index or "Invested Capital" not in balance_sheet.index:
        return []
    pairs = []
    for period in income_stmt.columns.intersection(balance_sheet.columns):
        ebit = income_stmt.at["EBIT", period]
        invested = balance_sheet.at["Invested Capital", period]
        if not pd.isna(ebit) and not pd.isna(invested) and invested:
            pairs.append((period, float(ebit) / float(invested)))
    return pairs


def _merge_roic_histories(primary: List[tuple], secondary: List[tuple]) -> List[tuple]:
    """EDGAR (primary) wins overlapping years; yfinance fills the gaps. Newest first.

    A secondary year within 185 days of a primary year is the same fiscal
    year (the sources label the same 10-K end date, plus or minus a quarter
    of drift) — EDGAR's value is kept. Distinct fiscal years are ~365 days
    apart, so they never collide.
    """
    merged = list(primary)
    for period, roic in secondary:
        ordinal = pd.Timestamp(period).toordinal()
        if all(abs(ordinal - pd.Timestamp(kept).toordinal()) > 185 for kept, _ in merged):
            merged.append((period, roic))
    merged.sort(key=lambda pair: pd.Timestamp(pair[0]).toordinal(), reverse=True)
    return merged


def fetch_fundamentals(ticker: str) -> Dict[str, Optional[float]]:
    """Single I/O seam: raw ROIC / Faustmann elements for one ticker.

    The ROIC history is the union of the EDGAR 10-K backfill (primary —
    10+ years, fills the rolling window) and the yfinance annual statements
    (secondary — fills any trailing years EDGAR filings lag); the latest
    invested capital, cash, debt, and preferred equity come from the
    yfinance annual balance sheet; market cap from fast_info (falling back
    to info). Missing pieces come back as None — never fabricated.
    """
    import yfinance as yf

    ticker_obj = yf.Ticker(ticker)
    income_stmt = ticker_obj.income_stmt
    balance_sheet = ticker_obj.balance_sheet

    yf_history = _annual_roic_pairs(income_stmt, balance_sheet)
    edgar_history = edgar_fundamentals.edgar_roic_history(ticker, EDGAR_CACHE_DIR)
    merged_pairs = _merge_roic_histories(edgar_history, yf_history)
    roic_history = [roic for _, roic in merged_pairs]

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
        "roic_history": roic_history,
        "roic_history_pairs": merged_pairs,
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


def roic_breakdown(pairs: List[tuple], window: int = ROIC_WINDOW_YEARS):
    """(trailing-window pairs, median) from a merged (period, roic) history.

    Mirrors ``derive_roic``: fewer than ``window`` years yields the whole
    history and None — the "-" placeholder — never a short-window median.
    """
    windowed = pairs[:window]
    return windowed, derive_roic([roic for _, roic in windowed], window)


def print_roic_history(ticker: str, pairs: List[tuple]) -> None:
    """Print the per-year ROICs behind a ticker's final ROIC (worked example).

    Marks the years inside the trailing 10-year window with ``*`` and prints
    the median that lands in the ``roic`` column.
    """
    windowed, median = roic_breakdown(pairs)
    print(f"{ticker} annual ROICs (EDGAR + yfinance, most recent first; * = in the 10-year window):")
    for index, (period, roic) in enumerate(pairs):
        mark = "*" if index < len(windowed) else " "
        print(f"  {mark} {pd.Timestamp(period).strftime('%Y-%m-%d')}  {roic:.3f}")
    if median is None:
        print(f"  fewer than {ROIC_WINDOW_YEARS} years of history -> '-' placeholder (no single-year stand-in)")
    else:
        print(f"  median of the trailing {len(windowed)} annual ROICs = {median:.3f} -> the 'roic' column")


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
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> pd.DataFrame:
    """One row per ticker: raw elements plus derived roic / net_worth / faustmann_ratio.

    ``progress`` (optional) is called as ``progress(done, total, ticker)``
    after each fetch — the CLI uses it for the elapsed/ETA progress line.
    """
    rows = []
    total = len(tickers)
    for index, ticker in enumerate(tickers, start=1):
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
        if progress is not None:
            progress(index, total, ticker)
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


def _ods_rows(path: str) -> tuple:
    """(header cell texts, remaining rows' cell-text lists) from an .ods spreadsheet."""
    from odf.opendocument import load
    from odf.table import TableCell, TableRow

    rows = load(path).spreadsheet.getElementsByType(TableRow)
    if not rows:
        return [], []
    header = [_ods_cell_text(c) for c in rows[0].getElementsByType(TableCell)]
    body = [[_ods_cell_text(c) for c in row.getElementsByType(TableCell)] for row in rows[1:]]
    return header, body


def _ticker_column(lowered, default):
    """``TICKER_COLUMNS`` name whose header key is present, else ``default``."""
    return next((lowered[name] for name in TICKER_COLUMNS if name in lowered), default)


def _ods_reader(path: str) -> List[str]:
    """Ticker list from an .ods workbook via odfpy (repo writer emits no office:value attrs)."""
    header, body = _ods_rows(path)
    lowered = {text.strip().lower(): i for i, text in enumerate(header) if text.strip()}
    column = _ticker_column(lowered, 0)
    return [cells[column].strip() for cells in body if column < len(cells) and cells[column].strip()]


def _xlsx_reader(path: str) -> List[str]:
    """Ticker list from an .xlsx/.xls workbook: ticker/symbol column if present, else column one."""
    frame = pd.read_excel(path)
    lowered = {str(c).strip().lower(): c for c in frame.columns}
    column = _ticker_column(lowered, frame.columns[0])
    return [str(t).strip() for t in frame[column].dropna() if str(t).strip()]


def read_tickers(path: str) -> List[str]:
    """Ticker list from a workbook of any supported format (.ods, .xlsx, .xls), from any folder."""
    suffix = Path(path).suffix.lower()
    if suffix == ".ods":
        return _ods_reader(path)
    return _xlsx_reader(path)


def _ods_table(path: str) -> pd.DataFrame:
    """Full table from an .ods workbook: first row is the header, the rest are records."""
    header, body = _ods_rows(path)
    if not header:
        return pd.DataFrame()
    records = [(cells + [""] * len(header))[: len(header)] for cells in body]
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
    parser.add_argument(
        "--history",
        default="NVDA",
        help="ticker whose per-year ROIC breakdown to print as a worked example (default NVDA)",
    )
    args = parser.parse_args(argv)

    tickers = read_tickers(args.excel)
    print(f"fetching fundamentals for {len(tickers)} tickers ...")
    print(
        "first run: EDGAR 10-K histories + yfinance, throttled to ~0.5s/ticker — allow several minutes;"
        " re-runs hit the data/edgar_cache/ and skip the network for cached tickers"
    )
    start = time.monotonic()

    def progress(done: int, total: int, ticker: str) -> None:
        elapsed = time.monotonic() - start
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (total - done) / rate if rate > 0 else 0.0
        print(f"\r  [{done:>4}/{total}] {ticker:<6} {elapsed:5.0f}s elapsed, ETA {eta:5.0f}s", end="", flush=True)

    table = build_table(tickers, progress=progress)
    print()

    if args.history in table["ticker"].values:
        raw = fetch_fundamentals(args.history)
        print_roic_history(args.history, raw.get("roic_history_pairs") or [])

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
