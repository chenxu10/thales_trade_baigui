"""SEC EDGAR XBRL backfill for 10-year ROIC histories (Plan A).

yfinance annual statements only reach ~4 years back, so the ch-10 rolling
10-year ROIC can never fill from yfinance alone. EDGAR companyfacts is the
free, regulator-published version of the same 10-K facts Spitznagel took
from Compustat: one JSON per filer holds every us-gaap fact since ~2009.

This module:

  * maps ticker -> CIK from the SEC's company_tickers.json (disk-cached),
  * pulls a filer's companyfacts JSON (disk-cached, TTL'd, throttled well
    under the SEC's 10 requests/second),
  * derives per-fiscal-year EBIT and invested capital from us-gaap facts:

        EBIT            = OperatingIncomeLoss
                          (fallback EarningsBeforeInterestAndTaxes,
                          fallback NetIncome + IncomeTax + InterestExpense)

        invested capital = total debt + total equity
                          (debt: TotalDebtAndCapitalLeaseObligations,
                          fallback the sum of long-term debt, current
                          long-term debt, short-term borrowings, finance
                          lease liabilities, commercial paper, and notes
                          payable; equity: StockholdersEquity incl. NCI,
                          fallback StockholdersEquity)

  * returns annual (fiscal_year_end, roic) pairs, most recent first.

Every failure mode degrades to [] — the screen then falls back to the
yfinance-only history and its "-" placeholder, never fabricated numbers.
"""
import gzip
import json
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

EDGAR_USER_AGENT = "thales-trade-baigui research contact example@example.com"
EDGAR_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
EDGAR_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

TICKERS_TTL_SECONDS = 30 * 86400
FACTS_TTL_SECONDS = 7 * 86400
REQUEST_PAUSE_SECONDS = 0.2
MAX_ATTEMPTS = 3

ANNUAL_FORMS = ("10-K", "10-K/A", "20-F", "20-F/A", "40-F", "40-F/A")

EBIT_CHAIN = ("OperatingIncomeLoss", "EarningsBeforeInterestAndTaxes")
EBIT_PARTS = ("NetIncomeLoss", "IncomeTaxExpenseBenefit", "InterestExpense")

DEBT_CHAIN = ("TotalDebtAndCapitalLeaseObligations",)
DEBT_PARTS = (
    "LongTermDebtNoncurrent",
    "LongTermDebtCurrent",
    "ShortTermBorrowings",
    "FinanceLeaseLiabilityNoncurrent",
    "FinanceLeaseLiabilityCurrent",
    "CommercialPaper",
    "NotesPayableCurrent",
)
EQUITY_CHAIN = (
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    "StockholdersEquity",
)


def _normalize_ticker(ticker: str) -> str:
    return ticker.strip().upper().replace(".", "-")


def _fetch_json(url: str) -> dict:
    """One throttled, retrying GET with the SEC-required headers (gzip aware)."""
    for attempt in range(MAX_ATTEMPTS):
        time.sleep(REQUEST_PAUSE_SECONDS)
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": EDGAR_USER_AGENT, "Accept-Encoding": "gzip"},
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = response.read()
                if response.headers.get("Content-Encoding") == "gzip":
                    payload = gzip.decompress(payload)
                return json.loads(payload.decode("utf-8"))
        except urllib.error.HTTPError as error:
            if error.code in (403, 429, 500, 502, 503) and attempt < MAX_ATTEMPTS - 1:
                time.sleep(2 * (attempt + 1))
            else:
                raise
    raise RuntimeError("unreachable")


def _http_get_json(url: str, cache_path: Optional[str] = None, ttl_seconds: Optional[int] = None) -> dict:
    """GET a JSON URL, honoring a fresh on-disk cache when present."""
    path = Path(cache_path) if cache_path else None
    if path is not None and path.exists():
        age = time.time() - path.stat().st_mtime
        if ttl_seconds is None or age < ttl_seconds:
            with path.open() as handle:
                return json.load(handle)
    data = _fetch_json(url)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w") as handle:
            json.dump(data, handle)
        tmp.replace(path)
    return data


def cik_map(cache_dir: Optional[str] = None) -> Dict[str, str]:
    """{normalized ticker: 10-digit CIK} from the SEC's company_tickers.json."""
    payload = _http_get_json(
        EDGAR_TICKERS_URL,
        str(Path(cache_dir) / "company_tickers.json") if cache_dir else None,
        ttl_seconds=TICKERS_TTL_SECONDS,
    )
    mapping = {}
    for _, row in payload.items():
        ticker = _normalize_ticker(str(row.get("ticker", "")))
        if ticker:
            mapping[ticker] = str(row.get("cik_str", "")).zfill(10)
    return mapping


def _parse_end(value) -> Optional[date]:
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def _parse_val(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _annual_values(values: List[dict]) -> Dict[date, float]:
    """{fiscal_year_end: value} over annual (10-K style) facts; latest filed wins."""
    by_end: Dict[date, Tuple[float, str]] = {}
    for fact in values:
        end = _parse_end(fact.get("end"))
        value = _parse_val(fact.get("val"))
        if end is not None and value is not None and fact.get("form") in ANNUAL_FORMS:
            filed = str(fact.get("filed", ""))
            existing = by_end.get(end)
            if existing is None or filed > existing[1]:
                by_end[end] = (value, filed)
    return {end: pair[0] for end, pair in by_end.items()}


def _units(us_gaap: dict, concept: str) -> List[dict]:
    node = us_gaap.get(concept)
    if not node:
        return []
    return node.get("units", {}).get("USD", [])


def _sum_parts(parts: List[Dict[date, float]]) -> Dict[date, float]:
    """Year-wise sum of part series; a year exists if any part has a value."""
    years = set().union(*[set(part) for part in parts])
    return {end: sum(part.get(end, 0.0) for part in parts) for end in years}


def _chain_merged(us_gaap: dict, chain: tuple) -> Dict[date, float]:
    """Union of the chain's concepts, year-wise; earlier concepts win overlaps.

    Filers sometimes switch tagging mid-stream (e.g. from the NCI-inclusive
    equity concept to the plain one), leaving one concept truncated at some
    year — the union keeps both sides of the switch. Chain order gives the
    precedence on shared years.
    """
    merged: Dict[date, float] = {}
    for concept in reversed(chain):
        merged.update(_annual_values(_units(us_gaap, concept)))
    return merged


def _chain_or_parts(us_gaap: dict, chain: tuple, parts: tuple) -> Dict[date, float]:
    """The chain concepts' union, else the year-wise sum of the parts."""
    by_year = _chain_merged(us_gaap, chain)
    if by_year:
        return by_year
    return _sum_parts([_annual_values(_units(us_gaap, name)) for name in parts])


def _ebit_by_year(us_gaap: dict) -> Dict[date, float]:
    """Annual EBIT per fiscal year end, walking the fallback chain."""
    return _chain_or_parts(us_gaap, EBIT_CHAIN, EBIT_PARTS)


def _debt_by_year(us_gaap: dict) -> Dict[date, float]:
    """Annual total debt per fiscal year end: monolithic concept, else the parts sum."""
    return _chain_or_parts(us_gaap, DEBT_CHAIN, DEBT_PARTS)


def _equity_by_year(us_gaap: dict) -> Dict[date, float]:
    return _chain_merged(us_gaap, EQUITY_CHAIN)


def company_roics(facts: dict) -> List[Tuple[date, float]]:
    """Annual (fiscal_year_end, ROIC) pairs from one companyfacts payload, newest first.

    A year contributes only when both EBIT and a positive invested capital
    exist; anything else is skipped, never fabricated.
    """
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    ebit = _ebit_by_year(us_gaap)
    debt = _debt_by_year(us_gaap)
    equity = _equity_by_year(us_gaap)
    pairs = []
    for end, operating in ebit.items():
        invested = equity.get(end, 0.0) + debt.get(end, 0.0)
        if invested > 0:
            pairs.append((end, operating / invested))
    pairs.sort(key=lambda pair: pair[0], reverse=True)
    return pairs


def edgar_roic_history(ticker: str, cache_dir: str) -> List[Tuple[date, float]]:
    """Annual (fiscal_year_end, ROIC) pairs for one ticker from EDGAR; [] on any failure."""
    try:
        cik = cik_map(cache_dir).get(_normalize_ticker(ticker))
        if cik is None:
            return []
        facts = _http_get_json(
            EDGAR_FACTS_URL.format(cik=cik),
            str(Path(cache_dir) / f"facts_{cik}.json"),
            ttl_seconds=FACTS_TTL_SECONDS,
        )
        return company_roics(facts)
    except Exception:
        return []
