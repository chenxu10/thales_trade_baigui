"""CANSLIM criterion C screen over the whole US-listed pharma/biotech universe.

Builds the universe from the yfinance screener (Pharmaceuticals + Biotechnology
peer groups, region=us), applies the criterion-C EPS test to every name with
``map`` (thread-pooled, order-preserving), classifies each result as PASS /
FAIL / FILTERED, and writes a styled .ods report.

Usage:
    uv run python -m fentu.canslim.pharma_bio_screen
    uv run python -m fentu.canslim.pharma_bio_screen --min-growth 0.25
    uv run python -m fentu.canslim.pharma_bio_screen --limit 50 --workers 4
"""
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Callable, Iterable, List, Optional, Tuple

from odf.opendocument import OpenDocumentSpreadsheet
from odf.style import Style, TableCellProperties, TableColumnProperties, TextProperties
from odf.table import Table, TableCell, TableColumn, TableRow
from odf.text import P

from fentu.canslim.screen import ScreenResult, screen_current_eps

SCREEN_PAGE_SIZE = 250
PHARMA_BIO_QUERY = ("AND", [
    ("EQ", ["region", "us"]),
    ("OR", [
        ("EQ", ["peer_group", "Pharmaceuticals"]),
        ("EQ", ["peer_group", "EAA CE Sector Equity Biotechnology"]),
    ]),
])

FILTER_REASONS = frozenset({"no_quarterly_data", "no_prior_year_quarter", "missing_eps"})


def _build_query() -> "EquityQuery":
    from yfinance.screener import EquityQuery

    def build(node):
        operator, operand = node
        return EquityQuery(operator, [build(e) if isinstance(e, tuple) else e for e in operand])

    return build(PHARMA_BIO_QUERY)


def fetch_universe_tickers(limit: Optional[int] = None) -> List[Tuple[str, str]]:
    """(ticker, short name) pairs from the yfinance screener, deduped in order."""
    from yfinance.screener import screen

    query = _build_query()
    tickers: List[Tuple[str, str]] = []
    offset = 0
    while True:
        res = screen(query, offset=offset, size=SCREEN_PAGE_SIZE)
        quotes = res.get("quotes", [])
        if not quotes:
            break
        tickers.extend((q["symbol"], q.get("shortName") or q.get("displayName") or "") for q in quotes)
        offset += len(quotes)
        if offset >= res.get("total", 0):
            break
        if limit is not None and len(tickers) >= limit:
            break
    seen = set()
    ordered: List[Tuple[str, str]] = []
    for ticker, name in tickers:
        if ticker not in seen:
            seen.add(ticker)
            ordered.append((ticker, name))
    return ordered[:limit] if limit is not None else ordered


def screen_universe(
    tickers: Iterable[str],
    min_growth: float = 0.20,
    workers: int = 8,
    score: Callable[[str, float], ScreenResult] = screen_current_eps,
) -> List[ScreenResult]:
    """Order-preserving ``map`` of the criterion-C test over every ticker."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(partial(score, min_growth=min_growth), tickers))


def classify(result: ScreenResult) -> str:
    if result.passed:
        return "PASS"
    return "FILTERED" if result.reason in FILTER_REASONS else "FAIL"


def summarize(results: List[ScreenResult]) -> Tuple[int, int, int]:
    return (
        sum(1 for r in results if classify(r) == "PASS"),
        sum(1 for r in results if classify(r) == "FAIL"),
        sum(1 for r in results if classify(r) == "FILTERED"),
    )


def _add_style(doc: OpenDocumentSpreadsheet, name: str, background: str, color: str, bold: bool = False) -> None:
    style = Style(name=name, family="table-cell")
    style.addElement(TableCellProperties(backgroundcolor=background))
    props = TextProperties(color=color, fontweight="bold" if bold else "normal")
    style.addElement(props)
    doc.automaticstyles.addElement(style)


def _cell(text: str, style_name: Optional[str] = None) -> TableCell:
    cell = TableCell(stylename=style_name) if style_name else TableCell()
    cell.addElement(P(text=text))
    return cell


def _format_growth(growth: Optional[float]) -> str:
    return "-" if growth is None else f"{growth * 100:+.1f}%"


def _format_eps(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.3f}"


def _row_cells(result: ScreenResult, name: str) -> List[TableCell]:
    verdict = classify(result)
    style = {"PASS": "pass", "FAIL": "fail", "FILTERED": "filtered"}[verdict]
    return [
        _cell(result.ticker, style),
        _cell(name, style),
        _cell(_format_eps(result.current_eps), style),
        _cell(_format_eps(result.prior_year_eps), style),
        _cell(result.current_period or "-", style),
        _cell(result.prior_period or "-", style),
        _cell(_format_growth(result.growth), style),
        _cell(verdict, style),
        _cell(result.reason or "meets threshold", style),
    ]


def write_screen_ods(results: List[ScreenResult], names: dict, path: str) -> str:
    """Write the styled .ods report: PASS green, FAIL red, FILTERED yellow."""
    doc = OpenDocumentSpreadsheet()
    for style_name, background, color, bold in (
        ("header", "#1F4E79", "#FFFFFF", True),
        ("pass", "#C6EFCE", "#006100", True),
        ("fail", "#FFC7CE", "#9C0006", True),
        ("filtered", "#FFEB9C", "#9C6500", True),
    ):
        _add_style(doc, style_name, background, color, bold)

    table = Table(name="CriterionC")
    for column, width in enumerate(("2.4cm", "5.5cm", "2.6cm", "2.6cm", "3.2cm", "3.2cm", "2.6cm", "2.2cm", "4.5cm")):
        col_style = Style(name=f"col{column}", family="table-column")
        col_style.addElement(TableColumnProperties(columnwidth=width))
        doc.automaticstyles.addElement(col_style)
        table.addElement(TableColumn(stylename=col_style))

    header = TableRow()
    for text in ("Ticker", "Company", "Current EPS", "Prior EPS", "Current Q", "Prior Q", "Growth", "Verdict", "Reason"):
        header.addElement(_cell(text, "header"))
    table.addElement(header)

    for result in results:
        row = TableRow()
        for cell in _row_cells(result, names.get(result.ticker, "")):
            row.addElement(cell)
        table.addElement(row)

    doc.spreadsheet.addElement(table)
    doc.save(path)
    return path


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="CANSLIM criterion C screen over the US pharma/biotech universe")
    parser.add_argument("--min-growth", type=float, default=0.20, help="YoY EPS growth threshold (default 0.20)")
    parser.add_argument("--workers", type=int, default=8, help="threads for the map over tickers (default 8)")
    parser.add_argument("--limit", type=int, default=None, help="cap the universe size (default: all)")
    parser.add_argument("--output", default="data/pharma_bio_canslim_c_screen.ods", help="output .ods path")
    args = parser.parse_args(argv)

    print("fetching US pharma/biotech universe from yfinance screener ...")
    universe = fetch_universe_tickers(limit=args.limit)
    print(f"universe: {len(universe)} tickers")
    tickers, names = zip(*universe) if universe else ((), {})
    name_map = dict(universe)

    results = screen_universe(tickers, min_growth=args.min_growth, workers=args.workers)
    passed, failed, filtered = summarize(results)
    path = write_screen_ods(results, name_map, args.output)

    print(f"PASS {passed} | FAIL {failed} | FILTERED {filtered}")
    print(f"report: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())