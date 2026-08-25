"""
Siegfried picker — Spitznagel's ch-10 two-pronged screen over a siegfried workbook.

Covered behaviors:
    1. Whole-file ranking: positive net worth first, then ROIC descending
       (higher better), then Faustmann ratio ascending (lower better);
       rank 1 = best candidate. Every row gets a rank.
    2. Picking: the top-N ranks are the picks; the highlight never extends
       into the negative-net-worth tier.
    3. Reasons: one-liners state why each row stands where it does — above
       or below the ROIC bar, low Faustmann, or non-positive net worth.
    4. In-place update: rank/pick/reason columns appear, picked rows are
       re-sorted to the top, and running the update twice reuses the columns.
    5. Live formulas: ROIC (D) and Faustmann ratio (J) become ``of:=B/C`` /
       ``of:=E/I`` in .ods when the divisor is a finite number; .xlsx keeps
       literal values so read_table still parses numerics.
    6. read_table round-trips roic_faustmann output for both .ods and .xlsx.

Network-free: tables are built inline or round-tripped through tmp files.
"""
import pandas as pd
import pytest

from fentu.siegfried.roic_faustmann import OUTPUT_COLUMNS, read_table, write_table
from fentu.siegfried.siegfried_pick import (
    PICK_COLUMN,
    RANK_COLUMN,
    REASON_COLUMN,
    annotate,
    pick_siegfrieds,
    rank_siegfrieds,
    reason_for,
    update_workbook,
)


def _row(ticker, roic, faustmann):
    return {
        "ticker": ticker,
        "ebit": None,
        "invested_capital": None,
        "roic": roic,
        "market_cap": None,
        "cash": None,
        "debt": None,
        "preferred_equity": None,
        "net_worth": None,
        "faustmann_ratio": faustmann,
    }


def _universe():
    return pd.DataFrame(
        [
            _row("CHEAP", 0.80, 1.5),
            _row("MID", 0.90, 3.0),
            _row("DEAR", 0.85, 8.0),
            _row("WEAK", 0.20, 0.5),  # low ROIC, but positive net worth
            _row("NEG", 0.95, -2.0),  # negative net worth — bottom tier despite high ROIC
        ]
        + [_row(f"FILL{i}", 0.10, 10.0 + i) for i in range(95)]
    )


def test_rank_whole_file_by_criteria_descending():
    ranked = rank_siegfrieds(_universe(), min_roic=0.75)
    assert len(ranked) == 100
    assert ranked["rank"].tolist() == list(range(1, 101))
    assert list(ranked["ticker"])[:4] == ["MID", "DEAR", "CHEAP", "WEAK"]  # ROIC desc, Faustmann asc
    assert list(ranked["ticker"])[4] == "FILL0"  # positive net worth rows all rank before NEG
    assert ranked.loc[ranked["ticker"] == "NEG", "rank"].iloc[0] == 100  # high ROIC, negative net worth -> last


def test_pick_top_n_excludes_negative_net_worth():
    picks = pick_siegfrieds(_universe(), top_n=3)
    assert list(picks["ticker"]) == ["MID", "DEAR", "CHEAP"]
    assert "NEG" not in picks["ticker"].values


def test_pick_top_ten_covers_positive_net_worth_rows_first():
    picks = pick_siegfrieds(_universe(), top_n=10)
    assert len(picks) == 10
    assert list(picks["ticker"])[:4] == ["MID", "DEAR", "CHEAP", "WEAK"]
    assert "NEG" not in picks["ticker"].values  # NEG ranks 5, but is in the negative tier


def test_no_data_rows_rank_last():
    table = pd.DataFrame([_row("KNOWN", 0.9, 2.0), _row("UNKNOWN", None, None)])
    ranked = rank_siegfrieds(table)
    assert list(ranked["ticker"]) == ["KNOWN", "UNKNOWN"]


def test_reason_for_above_and_below_bar():
    assert reason_for(0.80, 1.5, 0.75, picked=True) == "ROIC 80% above 75% bar, low Faustmann 1.5"
    assert reason_for(0.60, 1.5, 0.75, picked=False) == "ROIC 60% below 75% bar"
    assert reason_for(0.95, -2.0, 0.75, picked=False) == "non-positive net worth — not a low-Faustmann pick"
    assert reason_for(None, 1.5, 0.75, picked=False) == "missing ROIC data"


def test_annotate_orders_and_reasons_every_row():
    annotations = annotate(_universe(), top_n=2, min_roic=0.75)
    assert [a["ticker"] for a in annotations[:3]] == ["MID", "DEAR", "CHEAP"]
    assert [a["rank"] for a in annotations[:3]] == [1, 2, 3]
    assert [a["picked"] for a in annotations[:2]] == [True, True]
    assert annotations[2]["picked"] is False
    assert annotations[2]["reason"].startswith("ROIC 80% above 75% bar")
    assert annotations[4]["reason"] == "ROIC 10% below 75% bar"
    assert annotations[-1]["ticker"] == "NEG"
    assert annotations[-1]["reason"] == "non-positive net worth — not a low-Faustmann pick"


def _apply(path, table, top_n=2):
    """Rank, pick, and update the workbook in place; returns the annotations."""
    annotations = annotate(table, top_n=top_n, min_roic=0.75)
    update_workbook(path, annotations, top_n)
    return annotations


@pytest.mark.parametrize("suffix", [".ods", ".xlsx"])
def test_update_workbook_in_place(tmp_path, suffix):
    table = _universe().reindex(columns=OUTPUT_COLUMNS)
    path = str(tmp_path / f"siegfried{suffix}")
    write_table(table, path)

    _apply(path, table, top_n=2)

    loaded = read_table(path)
    assert RANK_COLUMN in loaded.columns and PICK_COLUMN in loaded.columns and REASON_COLUMN in loaded.columns
    assert loaded["ticker"].tolist()[:3] == ["MID", "DEAR", "CHEAP"]  # picks first, then the rest ranked
    assert loaded[RANK_COLUMN].tolist()[:3] == [1.0, 2.0, 3.0]
    if suffix == ".xlsx":
        raw = pd.read_excel(path)
        assert raw[PICK_COLUMN].tolist()[:2] == ["YES", "YES"]
        assert raw[PICK_COLUMN].tolist()[2] == "-"
        assert raw[REASON_COLUMN].tolist()[0] == "ROIC 90% above 75% bar, low Faustmann 3.0"


@pytest.mark.parametrize("suffix", [".ods", ".xlsx"])
def test_update_is_idempotent(tmp_path, suffix):
    table = _universe().reindex(columns=OUTPUT_COLUMNS)
    path = str(tmp_path / f"siegfried{suffix}")
    write_table(table, path)

    _apply(path, table, top_n=2)
    _apply(path, table, top_n=2)

    loaded = read_table(path)
    assert list(loaded.columns).count(RANK_COLUMN) == 1
    assert list(loaded.columns).count(PICK_COLUMN) == 1
    assert list(loaded.columns).count(REASON_COLUMN) == 1
    assert loaded["ticker"].tolist()[:3] == ["MID", "DEAR", "CHEAP"]


@pytest.mark.parametrize("suffix", [".ods", ".xlsx"])
def test_update_adds_missing_reason_header_when_rank_pick_exist(tmp_path, suffix):
    """Regression: a workbook updated by an older version (rank/pick but no Reason)
    must gain the Reason header — never an unlabeled orphan column."""
    table = _universe().reindex(columns=OUTPUT_COLUMNS)
    path = str(tmp_path / f"siegfried{suffix}")
    write_table(table, path)

    annotations = annotate(table, top_n=2, min_roic=0.75)
    update_workbook(path, annotations, 2)  # first run: rank/pick/reason all appended
    table = read_table(path).drop(columns=[REASON_COLUMN])  # simulate the old-version state
    path2 = str(tmp_path / f"oldstyle{suffix}")
    write_table(table, path2)

    _apply(path2, table, top_n=2)

    loaded = read_table(path2)
    assert list(loaded.columns).count(REASON_COLUMN) == 1
    assert loaded["ticker"].tolist()[:3] == ["MID", "DEAR", "CHEAP"]


def _metric_table():
    return pd.DataFrame(
        [
            {
                "ticker": "ABC",
                "ebit": 2_000_000,
                "invested_capital": 10_000_000,
                "roic": 0.2,
                "market_cap": 18_500_000,
                "cash": 500_000,
                "debt": 1_000_000,
                "preferred_equity": 250_000,
                "net_worth": 9_250_000,
                "faustmann_ratio": 2.0,
            },
            {
                "ticker": "DEF",
                "ebit": 5_000_000,
                "invested_capital": 20_000_000,
                "roic": 0.25,
                "market_cap": 30_000_000,
                "cash": 1_000_000,
                "debt": 2_000_000,
                "preferred_equity": 0.0,
                "net_worth": 19_000_000,
                "faustmann_ratio": 1.58,
            },
        ]
    ).reindex(columns=OUTPUT_COLUMNS)


def test_update_ods_writes_live_formulas_for_roic_and_faustmann(tmp_path):
    from odf.table import TableCell

    table = _metric_table()
    path = str(tmp_path / "siegfried.ods")
    write_table(table, path)
    _apply(path, table, top_n=1)

    from odf.opendocument import load
    from odf.table import TableRow

    doc = load(path)
    rows = doc.spreadsheet.getElementsByType(TableRow)
    first = rows[1].getElementsByType(TableCell)  # DEF: rank 1 (ROIC 0.25)
    second = rows[2].getElementsByType(TableCell)

    assert first[3].getAttribute("formula") == "of:=B2/C2"  # D = roic
    assert first[9].getAttribute("formula") == "of:=E2/I2"  # J = faustmann ratio
    assert second[3].getAttribute("formula") == "of:=B3/C3"
    assert second[9].getAttribute("formula") == "of:=E3/I3"


def test_update_ods_skips_formula_when_divisor_missing(tmp_path):
    from odf.opendocument import load
    from odf.table import TableRow

    table = _universe().reindex(columns=OUTPUT_COLUMNS)  # invested_capital / net_worth are NaN
    path = str(tmp_path / "siegfried.ods")
    write_table(table, path)
    _apply(path, table, top_n=2)

    doc = load(path)
    rows = doc.spreadsheet.getElementsByType(TableRow)
    cells = rows[1].getElementsByType(__import__("odf.table", fromlist=["TableCell"]).TableCell)
    assert cells[3].getAttribute("formula") is None  # divisor "nan" -> no formula


@pytest.mark.parametrize("suffix", [".ods", ".xlsx"])
def test_read_table_round_trip(tmp_path, suffix):
    table = _universe().reindex(columns=OUTPUT_COLUMNS)
    path = str(tmp_path / f"siegfried{suffix}")
    write_table(table, path)

    loaded = read_table(path)
    assert list(loaded.columns) == OUTPUT_COLUMNS
    assert loaded["ticker"].tolist() == table["ticker"].tolist()
    assert loaded["roic"].tolist() == pytest.approx(table["roic"].tolist())
    assert pd.isna(loaded["ebit"]).all()  # "-"/"nan" placeholders -> NaN

    picks = pick_siegfrieds(loaded, top_n=3)
    assert list(picks["ticker"]) == ["MID", "DEAR", "CHEAP"]