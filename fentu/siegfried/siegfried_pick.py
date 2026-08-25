"""Pick the top slice of an index with Spitznagel's Chapter Ten screen.

From "Austrian Investing II: Siegfried" (The Dao of Capital, ch. 10), the
two-pronged screen — "neither of these screens in isolation is enough":

    1. High ROIC — Siegfrieds are operationally defined as firms realizing
       75 percent or higher ROIC (his own monthly toy screen even used ROIC
       above 100 percent). High ROIC flags roundabout, productive firms that
       keep reinvesting earnings.
    2. Low Faustmann ratio — among those survivors, the market-cap-to-net-
       worth laggards are the ones the market underappreciates ("the lowest
       Faustmann ratio firms among those with high ROIC").

Negative or zero Faustmann ratios (negative net worth) are excluded: a low
ratio must mean underpricing, not a gutted balance sheet. Spitznagel ignores
financials/banks in his own screen — do that upstream when building the
universe, as this table has no sector column.

Reads a siegfried workbook produced by ``roic_faustmann`` (.ods or .xlsx,
from any folder), ranks survivors by lowest Faustmann ratio, picks the top
``--top-n`` (default 10), and updates the workbook **in place** — no new
file: a "Siegfried rank" / "Siegfried pick" column pair, gold highlighting
on picked rows, and rows re-sorted picks-first (same pattern as the CANSLIM
seven-traits report). Running it twice is safe: existing rank/pick columns
are reused, not duplicated.

Usage:
    uv run python -m fentu.siegfried.siegfried_pick data/ndx100_ticker_siegfried.ods
    uv run python -m fentu.siegfried.siegfried_pick data/ndx100_ticker_siegfried.ods --top-n 10 --min-roic 0.75
"""
import argparse
import sys
from typing import Optional

import pandas as pd

from fentu.siegfried.roic_faustmann import read_table

DEFAULT_TOP_N = 10
DEFAULT_MIN_ROIC = 0.75
RANK_COLUMN = "Siegfried rank"
PICK_COLUMN = "Siegfried pick"
GOLD_STYLE = "siegfried_pick"
GOLD_BACKGROUND = "#FFD966"
GOLD_COLOR = "#7F6000"


def rank_siegfrieds(table: pd.DataFrame, min_roic: float = DEFAULT_MIN_ROIC) -> pd.DataFrame:
    """Survivors (ROIC >= min_roic, positive Faustmann ratio) ranked by lowest ratio.

    Rank 1 is the cheapest Siegfried — the best pick under the ch. 10 screen.
    Non-positive Faustmann ratios are excluded: low must mean underpriced,
    not a gutted balance sheet.
    """
    survivors = table[(table["roic"] >= min_roic) & (table["faustmann_ratio"] > 0)]
    ranked = survivors.sort_values("faustmann_ratio").reset_index(drop=True)
    ranked["rank"] = range(1, len(ranked) + 1)
    return ranked


def pick_siegfrieds(
    table: pd.DataFrame,
    top_n: int = DEFAULT_TOP_N,
    min_roic: float = DEFAULT_MIN_ROIC,
) -> pd.DataFrame:
    """Top ``top_n`` picks: the first ``top_n`` ranks of the Siegfried ranking.

    Fewer survivors than ``top_n`` yields just the survivors — the bar is
    never lowered to fill the slate.
    """
    return rank_siegfrieds(table, min_roic=min_roic).head(max(0, top_n))


def _cell(text: str, style_name: Optional[str] = None):
    from odf.table import TableCell
    from odf.text import P

    cell = TableCell(stylename=style_name) if style_name else TableCell()
    cell.addElement(P(text=text))
    return cell


def _row_texts(row) -> list:
    from odf.table import TableCell
    from odf.text import P

    return [str(p) for c in row.getElementsByType(TableCell) for p in c.getElementsByType(P)]


def _set_text(cell, text: str) -> None:
    from odf.text import P

    for paragraph in cell.getElementsByType(P):
        cell.removeChild(paragraph)
    cell.addElement(P(text=text))


def update_ods(path: str, ranks: dict, picks: set) -> None:
    """In-place .ods update: rank/pick columns, gold pick rows, picks-first row order."""
    from odf.opendocument import load
    from odf.style import Style, TableCellProperties, TextProperties
    from odf.table import Table, TableCell, TableRow

    doc = load(path)
    table = doc.spreadsheet.getElementsByType(Table)[0]
    rows = table.getElementsByType(TableRow)

    existing = [s for s in doc.automaticstyles.getElementsByType(Style)]
    if not any(s.getAttribute("name") == GOLD_STYLE for s in existing):
        gold = Style(name=GOLD_STYLE, family="table-cell")
        gold.addElement(TableCellProperties(backgroundcolor=GOLD_BACKGROUND))
        gold.addElement(TextProperties(color=GOLD_COLOR, fontweight="bold"))
        doc.automaticstyles.addElement(gold)

    header = _row_texts(rows[0])
    if RANK_COLUMN in header:
        rank_idx = header.index(RANK_COLUMN)
    else:
        rank_idx = len(header)
        rows[0].addElement(_cell(RANK_COLUMN, "header"))
        rows[0].addElement(_cell(PICK_COLUMN, "header"))

    annotated = []
    for position, row in enumerate(rows[1:]):
        texts = _row_texts(row)
        ticker = texts[0] if texts else ""
        rank = ranks.get(ticker)
        cells = row.getElementsByType(TableCell)
        while len(cells) <= rank_idx + 1:
            row.addElement(_cell("-"))
            cells = row.getElementsByType(TableCell)
        _set_text(cells[rank_idx], str(rank) if rank else "-")
        _set_text(cells[rank_idx + 1], "YES" if ticker in picks else "-")
        if ticker in picks:
            for cell in row.getElementsByType(TableCell):
                cell.setAttribute("stylename", GOLD_STYLE)
        annotated.append((rank, position, row))

    ordered = sorted(annotated, key=lambda item: (item[0] is None, item[0] or 0, item[1]))
    for row in rows[1:]:
        table.removeChild(row)
    for _, _, row in ordered:
        table.addElement(row)
    doc.save(path)


def update_xlsx(path: str, ranks: dict, picks: set) -> None:
    """In-place .xlsx update: rank/pick columns, gold pick rows, picks-first row order."""
    from copy import copy

    import openpyxl
    from openpyxl.styles import Font, PatternFill

    wb = openpyxl.load_workbook(path)
    ws = wb.active
    headers = {ws.cell(row=1, column=c).value: c for c in range(1, ws.max_column + 1)}
    if RANK_COLUMN in headers:
        rank_col, pick_col = headers[RANK_COLUMN], headers[PICK_COLUMN]
    else:
        rank_col, pick_col = ws.max_column + 1, ws.max_column + 2
        ws.cell(row=1, column=rank_col, value=RANK_COLUMN)
        ws.cell(row=1, column=pick_col, value=PICK_COLUMN)

    gold = PatternFill("solid", fgColor="FFD966")
    snapshots = []
    for row_idx in range(2, ws.max_row + 1):
        ticker = str(ws.cell(row=row_idx, column=1).value or "").strip()
        rank = ranks.get(ticker)
        ws.cell(row=row_idx, column=rank_col, value=rank if rank else "-")
        ws.cell(row=row_idx, column=pick_col, value="YES" if ticker in picks else "-")
        if ticker in picks:
            for col_idx in range(1, pick_col + 1):
                ws.cell(row=row_idx, column=col_idx).fill = gold
            ws.cell(row=row_idx, column=1).font = Font(bold=True)
        cells = [
            (ws.cell(row=row_idx, column=c).value, copy(ws.cell(row=row_idx, column=c)._style))
            for c in range(1, pick_col + 1)
        ]
        snapshots.append((rank, row_idx, cells))

    ordered = sorted(snapshots, key=lambda snap: (snap[0] is None, snap[0] or 0, snap[1]))
    for new_row, (_, _, cells) in enumerate(ordered, start=2):
        for col_idx, (value, style) in enumerate(cells, start=1):
            cell = ws.cell(row=new_row, column=col_idx, value=value)
            cell._style = style
    wb.save(path)


def update_workbook(path: str, ranks: dict, picks: set) -> None:
    """In-place update of a siegfried workbook (.ods or .xlsx), by extension."""
    if path.lower().endswith(".ods"):
        update_ods(path, ranks, picks)
    else:
        update_xlsx(path, ranks, picks)


def _fmt_pct(value: Optional[float]) -> str:
    return "-" if value is None or pd.isna(value) else f"{value * 100:.1f}%"


def _fmt_ratio(value: Optional[float]) -> str:
    return "-" if value is None or pd.isna(value) else f"{value:.2f}"


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Siegfried picker: high-ROIC, low-Faustmann-ratio top slice (Dao of Capital ch. 10)"
    )
    parser.add_argument("workbook", help="siegfried workbook (.ods or .xlsx) from roic_faustmann")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N, help="number of picks (default 10)")
    parser.add_argument("--min-roic", type=float, default=DEFAULT_MIN_ROIC, help="minimum ROIC bar (default 0.75)")
    args = parser.parse_args(argv)

    table = read_table(args.workbook)
    ranked = rank_siegfrieds(table, min_roic=args.min_roic)
    picks = pick_siegfrieds(table, top_n=args.top_n, min_roic=args.min_roic)

    print(f"universe : {len(table)} tickers")
    print(f"survivors: {len(ranked)} with ROIC >= {args.min_roic:.0%} and positive net worth")
    if ranked.empty:
        print("no Siegfrieds at this bar — workbook left untouched")
        return 1

    print(f"picks    : top {args.top_n} by lowest Faustmann ratio")
    for _, row in picks.iterrows():
        print(
            f"  {int(row['rank']):>2}. {row['ticker']:<6} ROIC {_fmt_pct(row['roic']):>7}"
            f"  Faustmann {_fmt_ratio(row['faustmann_ratio']):>8}"
        )

    ranks = {row["ticker"]: int(row["rank"]) for _, row in ranked.iterrows()}
    update_workbook(args.workbook, ranks, set(picks["ticker"]))
    print(f"updated  : {args.workbook} (rank/pick columns, gold highlights, picks first)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
