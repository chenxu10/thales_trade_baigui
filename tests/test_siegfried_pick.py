"""
Siegfried picker — Spitznagel's ch-10 two-pronged screen over a siegfried workbook.

Covered behaviors:
    1. Ranking: survivors (ROIC >= bar, positive Faustmann ratio) ranked by
       lowest ratio; rank 1 is the best pick.
    2. Picking: top-N ranks; fewer survivors than N yields just the
       survivors (bar never lowered).
    3. Honesty: negative/zero Faustmann ratios (negative net worth) are
       excluded — a low ratio must mean underpricing, not a gutted balance
       sheet; low-ROIC firms never pass however cheap.
    4. In-place update: the workbook gains "Siegfried rank"/"Siegfried pick"
       columns, picked rows are re-sorted to the top, and running the update
       twice reuses the columns instead of duplicating them.
    5. read_table round-trips roic_faustmann output for both .ods and .xlsx,
       coercing the "-" placeholders back to NaN.

Network-free: tables are built inline or round-tripped through tmp files.
"""
import pandas as pd
import pytest

from fentu.siegfried.roic_faustmann import OUTPUT_COLUMNS, read_table, write_table
from fentu.siegfried.siegfried_pick import (
    PICK_COLUMN,
    RANK_COLUMN,
    pick_siegfrieds,
    rank_siegfrieds,
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
            _row("WEAK", 0.20, 0.5),  # cheapest of all, but fails the ROIC bar
            _row("NEG", 0.95, -2.0),  # negative net worth — not underpriced
        ]
        + [_row(f"FILL{i}", 0.10, 10.0 + i) for i in range(95)]
    )


def test_rank_sorts_survivors_by_lowest_faustmann():
    ranked = rank_siegfrieds(_universe(), min_roic=0.75)
    assert list(ranked["ticker"]) == ["CHEAP", "MID", "DEAR"]
    assert ranked["rank"].tolist() == [1, 2, 3]


def test_low_roic_firms_never_pass_regardless_of_price():
    ranked = rank_siegfrieds(_universe(), min_roic=0.75)
    assert "WEAK" not in ranked["ticker"].values
    assert "NEG" not in ranked["ticker"].values


def test_pick_top_n_selects_cheapest_survivors():
    picks = pick_siegfrieds(_universe(), top_n=2, min_roic=0.75)
    assert list(picks["ticker"]) == ["CHEAP", "MID"]


def test_fewer_survivors_than_top_n_means_fewer_picks():
    picks = pick_siegfrieds(_universe(), top_n=10, min_roic=0.75)
    assert list(picks["ticker"]) == ["CHEAP", "MID", "DEAR"]


def test_no_survivors_returns_empty():
    assert pick_siegfrieds(_universe(), top_n=10, min_roic=0.99).empty


def _apply(path, table, top_n=2):
    """Rank, pick, and update the workbook in place; returns (ranked, picks)."""
    ranked = rank_siegfrieds(table, min_roic=0.75)
    picks = pick_siegfrieds(table, top_n=top_n, min_roic=0.75)
    ranks = {row["ticker"]: int(row["rank"]) for _, row in ranked.iterrows()}
    update_workbook(path, ranks, set(picks["ticker"]))
    return ranked, picks


@pytest.mark.parametrize("suffix", [".ods", ".xlsx"])
def test_update_workbook_in_place(tmp_path, suffix):
    table = _universe().reindex(columns=OUTPUT_COLUMNS)
    path = str(tmp_path / f"siegfried{suffix}")
    write_table(table, path)

    _apply(path, table, top_n=2)

    loaded = read_table(path)
    assert RANK_COLUMN in loaded.columns and PICK_COLUMN in loaded.columns
    assert loaded["ticker"].tolist()[:3] == ["CHEAP", "MID", "DEAR"]  # picks first, then survivor
    assert loaded[RANK_COLUMN].tolist()[:3] == [1.0, 2.0, 3.0]
    assert loaded["ticker"].tolist()[3] == "WEAK"  # unranked rows keep original order
    if suffix == ".xlsx":
        raw = pd.read_excel(path)
        assert raw[PICK_COLUMN].tolist()[:3] == ["YES", "YES", "-"]


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
    assert loaded["ticker"].tolist()[:3] == ["CHEAP", "MID", "DEAR"]


@pytest.mark.parametrize("suffix", [".ods", ".xlsx"])
def test_read_table_round_trip(tmp_path, suffix):
    table = _universe().reindex(columns=OUTPUT_COLUMNS)
    path = str(tmp_path / f"siegfried{suffix}")
    write_table(table, path)

    loaded = read_table(path)
    assert list(loaded.columns) == OUTPUT_COLUMNS
    assert loaded["ticker"].tolist() == table["ticker"].tolist()
    assert loaded["roic"].tolist() == pytest.approx(table["roic"].tolist())
    assert pd.isna(loaded["ebit"]).all()  # "-" placeholders -> NaN

    picks = pick_siegfrieds(loaded, top_n=2, min_roic=0.75)
    assert list(picks["ticker"]) == ["CHEAP", "MID"]
