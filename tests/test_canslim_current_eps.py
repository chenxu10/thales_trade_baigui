"""
INVEST story — CANSLIM criterion C (Current Quarterly EPS) for biotech screening.

Story (I.N.V.E.S.T):
    As a biotech investor running a candidate list of
    [Insilico Medicine, XtalPi, BeOne Medicine, Zai Lab, WuXi Biologics,
    Everest Medicines], I want each name scored against CANSLIM criterion C
    (current quarterly earnings per share, O'Neil: EPS up 18-25%+ YoY in the
    most recent quarter), so that I only hold names with genuine current
    earnings acceleration and never buy a name whose "growth" is an artifact
    of a loss-making base year.

Acceptance criteria (criterion C, current quarter only):
    AC1. Given current-quarter EPS and same-quarter-last-year EPS with a
         positive base, PASS iff YoY growth (cur - prior) / prior >= 20%.
    AC2. Flat or declining EPS (growth < 20%) => FAIL.
    AC3. Prior-year EPS <= 0 (loss-making base) => FAIL with reason
         "negative_base" — loss-narrowing is not earnings acceleration, the
         classic biotech trap (clinical-stage names report GAAP losses).

"""
import pytest

from fentu.canslim.current_eps import score_current_eps


class TestCurrentEpsCriterionPass:
    def test_30_percent_yoy_growth_passes(self):
        result = score_current_eps(current_eps=0.65, prior_year_eps=0.50)
        assert result.passed is True
        assert result.growth == pytest.approx(0.30)
        assert result.reason is None

    def test_exactly_threshold_20_percent_passes(self):
        result = score_current_eps(current_eps=0.60, prior_year_eps=0.50)
        assert result.passed is True
        assert result.growth == pytest.approx(0.20)


class TestCurrentEpsCriterionFail:
    def test_declining_eps_fails(self):
        result = score_current_eps(current_eps=0.55, prior_year_eps=0.60)
        assert result.passed is False
        assert result.growth == pytest.approx(-0.083333333, rel=1e-6)

    def test_flat_eps_fails(self):
        result = score_current_eps(current_eps=0.50, prior_year_eps=0.50)
        assert result.passed is False
        assert result.growth == pytest.approx(0.0)


class TestCurrentEpsCriterionNegativeBase:
    def test_loss_narrowing_from_negative_base_fails(self):
        result = score_current_eps(current_eps=-0.10, prior_year_eps=-0.20)
        assert result.passed is False
        assert result.reason == "negative_base"