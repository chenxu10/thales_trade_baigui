"""
SEC EDGAR XBRL backfill — EBIT / invested-capital derivation (Plan A).

Covered behaviors (all network-free):
    1. Annual fact selection: 10-K forms only, keyed by fiscal year end,
       latest filing wins over amendments.
    2. EBIT = OperatingIncomeLoss, falling back to EarningsBeforeInterest
       AndTaxes, then to NetIncome + IncomeTax + InterestExpense.
    3. Invested capital = total debt + equity, with the debt parts-sum
       fallback; a year counts only when EBIT and positive capital exist.
    4. company_roics yields (fiscal_year_end, roic) pairs, newest first.
    5. edgar_roic_history serves a cached facts file without touching the
       network, and degrades to [] on unknown tickers or network failure —
       never fabricated numbers.
"""
import json
from datetime import date

import pytest

from fentu.siegfried import edgar_fundamentals as edgar


def _fact(end="2023-12-31", val=100.0, form="10-K", filed="2024-02-01", start=None):
    fact = {"end": end, "val": val, "form": form, "filed": filed}
    if start is not None:
        fact["start"] = start
    return fact


def _concept(name, values):
    return {"facts": {"us-gaap": {name: {"units": {"USD": values}}}}}


def _facts(**concepts):
    payload = {"facts": {"us-gaap": {name: {"units": {"USD": values}} for name, values in concepts.items()}}}
    return payload


def _us_gaap(payload):
    return payload["facts"]["us-gaap"]


def test_annual_values_keep_only_10k_forms_and_latest_filing():
    values = [
        _fact("2023-12-31", 100.0, filed="2024-02-01"),
        _fact("2023-12-31", 110.0, form="10-K/A", filed="2024-03-15"),  # amendment wins
        _fact("2023-03-31", 5.0, form="10-Q"),  # quarterly — dropped
        _fact("2022-12-31", 90.0),
    ]
    by_year = edgar._annual_values(values)
    assert by_year == {date(2023, 12, 31): 110.0, date(2022, 12, 31): 90.0}


def test_annual_values_ignore_garbage_facts():
    values = [_fact("2023-12-31", 100.0), {"end": "not-a-date", "val": 1.0, "form": "10-K"},
              {"end": "2022-12-31", "val": None, "form": "10-K"}]
    assert edgar._annual_values(values) == {date(2023, 12, 31): 100.0}


def test_ebit_falls_back_to_net_income_reconstruction():
    facts = _facts(
        NetIncomeLoss=[_fact("2023-12-31", 60.0)],
        IncomeTaxExpenseBenefit=[_fact("2023-12-31", 15.0)],
        InterestExpense=[_fact("2023-12-31", 5.0)],
    )
    by_year = edgar._ebit_by_year(_us_gaap(facts))
    assert by_year == {date(2023, 12, 31): 80.0}


def test_ebit_prefers_operating_income_over_fallbacks():
    facts = _facts(
        OperatingIncomeLoss=[_fact("2023-12-31", 42.0)],
        NetIncomeLoss=[_fact("2023-12-31", 60.0)],
        IncomeTaxExpenseBenefit=[_fact("2023-12-31", 15.0)],
        InterestExpense=[_fact("2023-12-31", 5.0)],
    )
    assert edgar._ebit_by_year(_us_gaap(facts)) == {date(2023, 12, 31): 42.0}


def test_debt_sums_parts_when_monolithic_concept_absent():
    facts = _facts(
        LongTermDebtNoncurrent=[_fact("2023-12-31", 400.0)],
        LongTermDebtCurrent=[_fact("2023-12-31", 50.0)],
        FinanceLeaseLiabilityNoncurrent=[_fact("2023-12-31", 10.0)],
    )
    assert edgar._debt_by_year(_us_gaap(facts)) == {date(2023, 12, 31): 460.0}


def test_debt_prefers_monolithic_concept():
    facts = _facts(
        TotalDebtAndCapitalLeaseObligations=[_fact("2023-12-31", 460.0)],
        LongTermDebtNoncurrent=[_fact("2023-12-31", 400.0)],
    )
    assert edgar._debt_by_year(_us_gaap(facts)) == {date(2023, 12, 31): 460.0}


def test_chain_merge_survives_a_tagging_switch():
    """A filer that stopped tagging the first concept keeps its recent years from the second."""
    facts = _facts(
        StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest=[
            _fact("2023-12-31", 900.0),
            _fact("2021-12-31", 800.0),
        ],
        StockholdersEquity=[
            _fact("2025-12-31", 1000.0),
            _fact("2021-12-31", 700.0),  # overlap — the first concept wins
        ],
    )
    equity = edgar._equity_by_year(_us_gaap(facts))
    assert equity == {
        date(2025, 12, 31): 1000.0,
        date(2023, 12, 31): 900.0,
        date(2021, 12, 31): 800.0,
    }


def test_company_roics_pairs_ebit_over_invested_capital_newest_first():
    facts = _facts(
        OperatingIncomeLoss=[
            _fact("2023-12-31", 100.0),
            _fact("2022-12-31", 80.0, filed="2023-02-01"),
            _fact("2021-12-31", 60.0, filed="2022-02-01"),
        ],
        LongTermDebtNoncurrent=[_fact("2023-12-31", 400.0), _fact("2022-12-31", 400.0)],
        StockholdersEquity=[_fact("2023-12-31", 600.0), _fact("2022-12-31", 600.0)],
    )
    pairs = edgar.company_roics(facts)
    assert pairs == [
        (date(2023, 12, 31), pytest.approx(0.10)),
        (date(2022, 12, 31), pytest.approx(0.08)),
    ]
    # 2021 has no capital -> skipped, never fabricated


def test_company_roics_skips_non_positive_invested_capital():
    facts = _facts(
        OperatingIncomeLoss=[_fact("2023-12-31", 100.0)],
        LongTermDebtNoncurrent=[_fact("2023-12-31", -300.0)],
        StockholdersEquity=[_fact("2023-12-31", 300.0)],
    )
    assert edgar.company_roics(facts) == []


def test_edgar_roic_history_serves_cache_without_network(tmp_path, monkeypatch):
    monkeypatch.setattr(edgar, "cik_map", lambda cache_dir: {"AAPL": "0000320193"})
    payload = _facts(
        OperatingIncomeLoss=[_fact("2023-12-31", 100.0)],
        LongTermDebtNoncurrent=[_fact("2023-12-31", 400.0)],
        StockholdersEquity=[_fact("2023-12-31", 600.0)],
    )
    with (tmp_path / "facts_0000320193.json").open("w") as handle:
        json.dump(payload, handle)

    pairs = edgar.edgar_roic_history("AAPL", str(tmp_path))
    assert pairs == [(date(2023, 12, 31), pytest.approx(0.10))]


def test_edgar_roic_history_normalizes_tickers(tmp_path, monkeypatch):
    monkeypatch.setattr(edgar, "cik_map", lambda cache_dir: {"BRK-B": "0001067983"})
    with (tmp_path / "facts_0001067983.json").open("w") as handle:
        json.dump(_facts(OperatingIncomeLoss=[_fact("2023-12-31", 100.0)],
                        StockholdersEquity=[_fact("2023-12-31", 1000.0)]), handle)
    pairs = edgar.edgar_roic_history("BRK.B", str(tmp_path))
    assert pairs == [(date(2023, 12, 31), pytest.approx(0.10))]


def test_edgar_roic_history_empty_for_unknown_ticker(tmp_path, monkeypatch):
    monkeypatch.setattr(edgar, "cik_map", lambda cache_dir: {"AAPL": "0000320193"})
    assert edgar.edgar_roic_history("NOPE", str(tmp_path)) == []


def test_edgar_roic_history_empty_on_network_failure(tmp_path, monkeypatch):
    def boom(url, cache_path=None, ttl_seconds=None):
        raise OSError("no network")

    monkeypatch.setattr(edgar, "_http_get_json", boom)
    assert edgar.edgar_roic_history("AAPL", str(tmp_path)) == []


def test_normalize_ticker_dot_and_case():
    assert edgar._normalize_ticker("brk.b") == "BRK-B"
    assert edgar._normalize_ticker(" BRK-B ") == "BRK-B"
