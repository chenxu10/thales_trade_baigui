"""RED test: `see_change daily` dies on a non-finite return from an unfinalized bar.

Reproduces the user-visible traceback:

    uv run fentu/explatoryservices/seechange.py daily QQQ
      -> volcalculator.visualize_percentage_change
      -> _plot_percentage_change
      -> ps.histgram_plot
      -> plotting_service.prepare_histogram_data
      -> fit_normal_distribution
      -> scipy.stats.norm.fit
    ValueError: The data contains non-finite values.

Root cause
----------
``ReturnsRepository.get_returns`` (volcalculator.py:250) strips the *leading*
NaN created by ``prices.shift(period_length)`` with a positional slice, but
never drops a NaN sitting at the END of the series:

    return np.log(prices / prices.shift(period_length))[period_length:]

yfinance ``period="max"`` routinely returns the current session as an
*unfinalized* bar whose ``Close`` is NaN (observed live: QQQ 2026-09-24, one
non-finite close out of 6929 rows), so the last daily log return is NaN and
survives into ``scipy.stats.norm.fit``, which rejects non-finite input.

Weekly/monthly/yearly are unaffected: ``non_overlapping_period_returns`` ends in
``.dropna()``. Only the ``daily`` branch is exposed -- which is why
`see_change daily` fails while `see_change weekly` works.

This test is intentionally RED (Kent Beck, TDD By Example): it pins the bug
before the fix.
"""
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

from fentu.explatoryservices.volcalculator import (
    ReturnsRepository,
    VolatilityFacade,
)
from fentu.explatoryservices import plotting_service as ps


def test_daily_histogram_survives_an_unfinalized_final_bar():
    """Live shape: ..., 741.21, NaN -- the 2026-09-24 QQQ bar, session not yet finalized."""
    index = pd.bdate_range("2026-09-14", periods=5)
    prices = pd.Series([740.00, 741.00, 742.00, 741.21, np.nan],
                       index=index, name="Close")
    repo = ReturnsRepository()
    repo.get_prices = lambda instrument: prices  # replace the network seam
    facade = VolatilityFacade("QQQ", repository=repo)

    # ValueError: The data contains non-finite values.
    # (scipy/stats/_continuous_distns.py:416)
    view_model = ps.prepare_histogram_data(facade.daily_returns)

    assert np.isfinite(np.asarray(view_model["data"], dtype=float)).all()
