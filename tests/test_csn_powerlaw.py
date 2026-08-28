"""Continuous Clauset-Shalizi-Newman (2009) power-law fitter tests.

The money decision (tail alpha on every see_change chart) must come from
MLE + KS-selected x_min — not from OLS on a log-binned histogram, which
CSN 2009 discredit. Synthetic Pareto samples with known alpha pin the
estimator's accuracy.
"""
import numpy as np
import pytest

from fentu.explatoryservices import csn_powerlaw as csn


def pareto_samples(alpha_pdf, n, x_min=0.1, seed=0):
    """Inverse-CDF samples from p(x) = (a-1)/x_min * (x/x_min)^-a."""
    rng = np.random.default_rng(seed)
    u = rng.random(n)
    return x_min * (1.0 - u) ** (-1.0 / (alpha_pdf - 1.0))


class TestMleAlphaContinuous:
    def test_recovers_true_alpha(self):
        x = pareto_samples(alpha_pdf=4.0, n=5000, x_min=0.1)
        tail = x[x >= 0.1]
        a = csn.mle_alpha_continuous(tail, 0.1)
        assert a == pytest.approx(4.0, abs=0.15)

    def test_formula_matches_definition(self):
        tail = np.array([0.2, 0.5, 1.0, 2.0])
        x_min = 0.1
        expected = 1.0 + len(tail) / np.sum(np.log(tail / x_min))
        assert csn.mle_alpha_continuous(tail, x_min) == pytest.approx(expected)

    def test_degenerate_returns_nan(self):
        assert np.isnan(csn.mle_alpha_continuous(np.array([1.0]), 0.5))
        assert np.isnan(csn.mle_alpha_continuous(np.array([1.0, 1.0]), 1.0))


class TestKsDistanceContinuous:
    def test_zero_for_perfect_fit(self):
        x_min, alpha = 0.1, 3.0
        # quantiles of the exact model -> empirical CDF ~ model CDF
        p = (np.arange(1, 501) - 0.5) / 500
        tail = x_min * (1.0 - p) ** (-1.0 / (alpha - 1.0))
        d = csn.ks_distance_continuous(tail, x_min, alpha)
        assert d < 0.02

    def test_large_for_wrong_alpha(self):
        x = pareto_samples(alpha_pdf=4.0, n=2000, x_min=0.1)
        d_right = csn.ks_distance_continuous(x, 0.1, 4.0)
        d_wrong = csn.ks_distance_continuous(x, 0.1, 1.5)
        assert d_wrong > 5 * d_right

    def test_invalid_alpha_returns_inf(self):
        tail = np.array([0.2, 0.3, 0.4])
        assert csn.ks_distance_continuous(tail, 0.1, 1.0) == np.inf
        assert csn.ks_distance_continuous(tail, 0.1, np.nan) == np.inf


class TestSelectXmin:
    def test_fits_pure_pareto_at_lower_edge(self):
        x = pareto_samples(alpha_pdf=4.0, n=5000, x_min=0.1)
        x_min, alpha, d, n_tail = csn.select_xmin_continuous(x)
        assert x_min == pytest.approx(0.1, rel=0.5)
        assert alpha == pytest.approx(4.0, abs=0.3)
        assert n_tail >= 0.8 * len(x)

    def test_recovers_threshold_above_lowervarying_body(self):
        """Body below the threshold is not power-law; x_min must exclude it."""
        rng = np.random.default_rng(1)
        body = rng.uniform(0.01, 0.1, 2000)          # flat body, not power-law
        tail = pareto_samples(alpha_pdf=3.0, n=1000, x_min=0.1, seed=2)
        data = np.concatenate([body, tail])
        x_min, alpha, _, n_tail = csn.select_xmin_continuous(data)
        assert x_min == pytest.approx(0.1, rel=0.6)
        assert alpha == pytest.approx(3.0, abs=0.4)

    def test_too_few_points_returns_none(self):
        x_min, alpha, d, n_tail = csn.select_xmin_continuous(np.array([0.5]))
        assert x_min is None and alpha is None and n_tail == 0


class TestFitPowerLawContinuous:
    def test_result_schema_and_values(self):
        x = pareto_samples(alpha_pdf=4.0, n=5000, x_min=0.1)
        fit = csn.fit_power_law_continuous(x)
        assert set(fit) == {"alpha", "alpha_std", "x_min", "n_tail",
                            "n_total", "ks", "p"}
        assert fit["alpha"] == pytest.approx(4.0, abs=0.3)
        assert fit["n_total"] == 5000
        assert fit["p"] is None  # bootstrap off by default
        assert fit["alpha_std"] == pytest.approx(
            (fit["alpha"] - 1.0) / np.sqrt(fit["n_tail"]))

    def test_bootstrap_pvalue_high_for_true_pareto(self):
        x = pareto_samples(alpha_pdf=3.5, n=2000, x_min=0.05, seed=7)
        fit = csn.fit_power_law_continuous(x, n_bootstrap=50, seed=7)
        assert fit["p"] is not None
        assert 0.0 <= fit["p"] <= 1.0
        assert fit["p"] > 0.1  # should NOT reject a true power law

    def test_filters_nonpositive_and_nan(self):
        x = np.array([0.1, 0.2, 0.3, -1.0, 0.0, np.nan, 0.5, 0.4])
        fit = csn.fit_power_law_continuous(x)
        assert fit["n_total"] == 5

    def test_degenerate_input(self):
        fit = csn.fit_power_law_continuous(np.array([]))
        assert fit["alpha"] is None
        assert fit["n_total"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
