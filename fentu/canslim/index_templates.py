"""CANSLIM criterion C screen templates for the S&P 500 and Nasdaq-100.

Fetches the index constituents (ticker, company) from Wikipedia and writes
.ods templates that mimic ``data/pharma_bio_canslim_c_screen.ods``: the same
21-column CANSLIM header and column layout, but only columns A (Ticker) and
B (Company) are populated — every other column is left blank for later fills.

Usage:
    uv run python -m fentu.canslim.index_templates
    uv run python -m fentu.canslim.index_templates --output-dir data
"""
import argparse
import os
import sys
import urllib.request
from typing import Iterable, List, Optional, Tuple

from bs4 import BeautifulSoup
from odf.opendocument import OpenDocumentSpreadsheet
from odf.style import Style, TableCellProperties, TableColumnProperties, TextProperties
from odf.table import Table, TableCell, TableColumn, TableRow
from odf.text import P

USER_AGENT = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
NASDAQ100_URL = "https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies"

HEADERS = (
    "Ticker", "Company", "Current EPS", "Prior EPS", "Current Q", "Prior Q",
    "Growth", "Verdict", "Reason",
    "A: 5y EPS CAGR", "A",
    "N: dist from 52w high", "N",
    "S: shares out", "S",
    "L: RS rank", "L",
    "I: inst held", "I",
    "M: dist days GSPC/XBI", "M",
    "O'Neil letters",
)

COLUMN_WIDTHS = (
    "0.9453in", "2.1654in", "1.0236in", "1.2598in", "0.8661in",
    "1.7717in", "1.1811in", "0.6299in", "0.889in",
)
COLUMN_STYLE_ORDER = (0, 1, 2, 3, 2, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 4)


def _normalize(text: str) -> str:
    return " ".join(text.replace("\xa0", " ").split())


def fetch_index_pairs(url: str, ticker_col: str, name_col: str) -> List[Tuple[str, str]]:
    """(ticker, company) pairs from the first Wikipedia table with both columns."""
    request = urllib.request.Request(url, headers=USER_AGENT)
    with urllib.request.urlopen(request, timeout=30) as response:
        soup = BeautifulSoup(response.read().decode("utf-8", "replace"), "html.parser")

    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        if rows:
            header = [_normalize(th.get_text(" ", strip=True)) for th in rows[0].find_all("th")]
            if ticker_col in header and name_col in header:
                ticker_index = header.index(ticker_col)
                name_index = header.index(name_col)
                pairs: List[Tuple[str, str]] = []
                seen = set()
                for row in rows[1:]:
                    cells = [_normalize(td.get_text(" ", strip=True)) for td in row.find_all(["td", "th"])]
                    if len(cells) > max(ticker_index, name_index):
                        ticker = cells[ticker_index]
                        if ticker and ticker not in seen:
                            seen.add(ticker)
                            pairs.append((ticker, cells[name_index]))
                if pairs:
                    return pairs
    raise ValueError(f"no table with columns {ticker_col!r}/{name_col!r} at {url}")


def _cell(text: str, style_name: Optional[str] = None) -> TableCell:
    cell = TableCell(stylename=style_name) if style_name else TableCell()
    cell.addElement(P(text=text))
    return cell


def write_template_ods(pairs: Iterable[Tuple[str, str]], path: str) -> str:
    """Write an .ods with the CANSLIM header but only columns A/B populated."""
    doc = OpenDocumentSpreadsheet()
    header_style = Style(name="header", family="table-cell")
    header_style.addElement(TableCellProperties(backgroundcolor="#1F4E79"))
    header_style.addElement(TextProperties(color="#FFFFFF", fontweight="bold"))
    doc.automaticstyles.addElement(header_style)

    for index, width in enumerate(COLUMN_WIDTHS):
        col_style = Style(name=f"col{index}", family="table-column")
        col_style.addElement(TableColumnProperties(columnwidth=width))
        doc.automaticstyles.addElement(col_style)

    table = Table(name="CriterionC")
    for style_index in COLUMN_STYLE_ORDER:
        table.addElement(TableColumn(stylename=f"col{style_index}"))

    header = TableRow()
    for text in HEADERS:
        header.addElement(_cell(text, "header"))
    table.addElement(header)

    for ticker, name in pairs:
        row = TableRow()
        row.addElement(_cell(ticker))
        row.addElement(_cell(name))
        for _ in range(len(HEADERS) - 2):
            row.addElement(TableCell())
        table.addElement(row)

    doc.spreadsheet.addElement(table)
    doc.save(path)
    return path


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="CANSLIM templates for S&P 500 and Nasdaq-100 constituents")
    parser.add_argument("--output-dir", default="data", help="directory for the .ods templates")
    args = parser.parse_args(argv)

    for label, url, ticker_col, name_col, filename in (
        ("S&P 500", SP500_URL, "Symbol", "Security", "sp500_canslim_c_screen.ods"),
        ("Nasdaq-100", NASDAQ100_URL, "Ticker", "Company", "ndx100_canslim_c_screen.ods"),
    ):
        pairs = fetch_index_pairs(url, ticker_col, name_col)
        path = write_template_ods(pairs, os.path.join(args.output_dir, filename))
        print(f"{label}: {len(pairs)} constituents -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())