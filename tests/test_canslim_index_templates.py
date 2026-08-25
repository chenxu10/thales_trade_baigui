"""CANSLIM index templates — S&P 500 / Nasdaq-100 constituent .ods stubs.

Covered behaviors:
    1. fetch_index_pairs extracts (ticker, company) pairs from the first
       Wikipedia table exposing both columns.
    2. write_template_ods emits a CriterionC .ods with the full CANSLIM
       header but only columns A (Ticker) and B (Company) populated.
"""
import zipfile
from unittest.mock import patch

from fentu.canslim.index_templates import HEADERS, fetch_index_pairs, write_template_ods

SP500_HTML = """
<html><body>
<table>
  <tr><th>Symbol</th><th>Security</th><th>GICS Sector</th></tr>
  <tr><td>MMM</td><td>3M</td><td>Industrials</td></tr>
  <tr><td>AOS</td><td>A. O. Smith</td><td>Industrials</td></tr>
  <tr><td>MMM</td><td>3M duplicate</td><td>Industrials</td></tr>
</table>
<table>
  <tr><th>Other</th></tr>
  <tr><td>ignored</td></tr>
</table>
</body></html>
"""

NDX_HTML = """
<html><body>
<table>
  <tr><th>Ticker</th><th>Company</th><th>ICB Industry</th></tr>
  <tr><td>ADBE</td><td>Adobe Inc.</td><td>Technology</td></tr>
  <tr><td>AMD</td><td>Advanced Micro Devices</td><td>Technology</td></tr>
</table>
</body></html>
"""


class FakeResponse:
    def __init__(self, html):
        self._html = html

    def read(self):
        return self._html.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_fetch_index_pairs_picks_the_matching_table_and_dedupes():
    with patch("urllib.request.urlopen", return_value=FakeResponse(SP500_HTML)):
        pairs = fetch_index_pairs("http://example/sp500", "Symbol", "Security")
    assert pairs == [("MMM", "3M"), ("AOS", "A. O. Smith")]


def test_fetch_index_pairs_uses_named_columns():
    with patch("urllib.request.urlopen", return_value=FakeResponse(NDX_HTML)):
        pairs = fetch_index_pairs("http://example/ndx", "Ticker", "Company")
    assert pairs == [("ADBE", "Adobe Inc."), ("AMD", "Advanced Micro Devices")]


def test_fetch_index_pairs_normalizes_whitespace():
    html = """
    <html><body>
    <table>
      <tr><th>Ticker</th><th>Company</th></tr>
      <tr><td>GOOGL</td><td>Alphabet&nbsp;Inc.\n  Class A</td></tr>
    </table>
    </body></html>
    """
    with patch("urllib.request.urlopen", return_value=FakeResponse(html)):
        pairs = fetch_index_pairs("http://example/ndx", "Ticker", "Company")
    assert pairs == [("GOOGL", "Alphabet Inc. Class A")]


def test_fetch_index_pairs_raises_when_no_matching_table():
    with patch("urllib.request.urlopen", return_value=FakeResponse(SP500_HTML)):
        try:
            fetch_index_pairs("http://example/ndx", "Ticker", "Company")
        except ValueError as exc:
            assert "no table" in str(exc)
        else:
            raise AssertionError("expected ValueError")


def test_write_template_ods_fills_only_ticker_and_company(tmp_path):
    path = str(tmp_path / "template.ods")
    write_template_ods([("MMM", "3M"), ("AOS", "A. O. Smith")], path)

    with zipfile.ZipFile(path) as archive:
        content = archive.read("content.xml").decode("utf-8")

    assert "CriterionC" in content
    assert "MMM" in content and "3M" in content
    assert "AOS" in content and "A. O. Smith" in content
    for header in HEADERS:
        assert header in content
    assert 'style-name="header"' in content
    assert "pass" not in content and "fail" not in content and "filtered" not in content
    assert content.count("<text:p>") == len(HEADERS) + 2 * 2


def test_write_template_ods_blank_columns_are_empty_cells(tmp_path):
    path = str(tmp_path / "template.ods")
    write_template_ods([("MMM", "3M")], path)

    with zipfile.ZipFile(path) as archive:
        content = archive.read("content.xml").decode("utf-8")

    data_rows = content.split("<table:table-row>")[2:]
    assert len(data_rows) == 1
    row = data_rows[0]
    assert row.count("<table:table-cell") == len(HEADERS)
    assert row.count("<text:p>") == 2