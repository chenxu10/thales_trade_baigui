"""CANSLIM criterion C: current quarterly earnings per share (O'Neil).

Scores a single company's most recent quarter EPS against the same quarter
one year earlier. PASS requires YoY growth at or above the threshold with a
positive base year; a loss-making base year (prior EPS <= 0) is always a FAIL
with reason "negative_base", since loss-narrowing is not earnings
acceleration.
"""
from dataclasses import dataclass
from typing import Optional

DEFAULT_MIN_GROWTH = 0.20
_EPSILON = 1e-9


@dataclass(frozen=True)
class CurrentEpsResult:
    passed: bool
    growth: float
    reason: Optional[str] = None


def score_current_eps(
    current_eps: float, prior_year_eps: float, min_growth: float = DEFAULT_MIN_GROWTH
) -> CurrentEpsResult:
    """Pure function: score one quarter's EPS YoY growth against the threshold."""
    if prior_year_eps <= 0:
        return CurrentEpsResult(passed=False, growth=float("nan"), reason="negative_base")
    growth = (current_eps - prior_year_eps) / prior_year_eps
    return CurrentEpsResult(passed=growth >= min_growth - _EPSILON, growth=growth)