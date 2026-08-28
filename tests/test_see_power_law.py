"""see_power_law tests: histogram display helpers + the CSN fit path.

Regression guard: the same synthetic Pareto sample (true density alpha = 4.0)
that the old binned-OLS estimator pinned at alpha = 1.06 must now come out
near 4.0 via the CSN fitter.
"""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

from fentu.explatoryservices import see_power_law as spl


@pytest.fixture
def synthetic_powerlaw_samples():
    """Same sample the discredited binned-OLS baseline was pinned on."""
    np.random.seed(0)
    return (np.random.pareto(3, 20000) + 1) * 0.1  # true density alpha = 4.0


class TestHistogramHelpers:
    def test_create_log_space_bins_span(self, synthetic_powerlaw_samples):
        samples = synthetic_powerlaw_samples
        bins = spl.create_log_space_bins(np.min(samples), samples)
        assert bins[0] == pytest.approx(np.min(samples))
        assert bins[-1] == pytest.approx(np.max(samples))
        assert len(bins) == 100

    def test_manual_density_integrates_to_one(self, synthetic_powerlaw_samples):
        samples = synthetic_powerlaw_samples
        bins = spl.create_log_space_bins(np.min(samples), samples)
        density, _, edges = spl.compute_histogram_with_bins(
            samples, bins, method='manual_density'
        )
        assert np.sum(density * np.diff(edges)) == pytest.approx(1.0)

    def test_unknown_method_raises(self, synthetic_powerlaw_samples):
        bins = spl.create_log_space_bins(
            np.min(synthetic_powerlaw_samples), synthetic_powerlaw_samples
        )
        with pytest.raises(ValueError):
            spl.compute_histogram_with_bins(
                synthetic_powerlaw_samples, bins, method='bogus'
            )

    def test_positive_density_mask(self):
        mask = spl._positive_density_mask(np.array([0.0, 1.0, 0.0, 2.0]))
        assert mask.dtype == bool
        assert mask.tolist() == [False, True, False, True]


class TestParetoDensityLoglogParams:
    def test_slope_is_minus_alpha(self):
        slope, _ = spl._pareto_density_loglog_params(alpha=4.0, x_min=0.1)
        assert slope == pytest.approx(-4.0)

    def test_line_passes_through_pareto_pdf(self):
        alpha, x_min = 3.5, 0.2
        slope, intercept = spl._pareto_density_loglog_params(alpha, x_min)
        for x in (x_min, 1.0, 10.0):
            pdf = (alpha - 1.0) / x_min * (x / x_min) ** (-alpha)
            assert slope * np.log10(x) + intercept == pytest.approx(np.log10(pdf))


class TestLoglogFitData:
    def test_alpha_recovers_true_exponent(self, synthetic_powerlaw_samples):
        """The money number on the chart: was 1.06 under binned-OLS, must be
        ~4.0 under CSN (true density exponent of the fixture)."""
        data = spl._loglog_fit_data(
            synthetic_powerlaw_samples, np.min(synthetic_powerlaw_samples)
        )
        assert data['alpha'] == pytest.approx(4.0, abs=0.3)

    def test_tmask_marks_only_bins_at_or_above_fit_xmin(
        self, synthetic_powerlaw_samples
    ):
        data = spl._loglog_fit_data(
            synthetic_powerlaw_samples, np.min(synthetic_powerlaw_samples)
        )
        centers = data['valid_centers']
        assert data['tmask'].dtype == bool
        assert centers[data['tmask']].min() >= data['fit']['x_min']
        body = centers[~data['tmask']]
        if len(body):  # pure Pareto may put every bin in the tail
            assert body.max() < data['fit']['x_min']

    def test_slope_intercept_consistent_with_fit(self, synthetic_powerlaw_samples):
        data = spl._loglog_fit_data(
            synthetic_powerlaw_samples, np.min(synthetic_powerlaw_samples)
        )
        expected_slope, expected_intercept = spl._pareto_density_loglog_params(
            data['fit']['alpha'], data['fit']['x_min']
        )
        assert data['slope'] == pytest.approx(expected_slope)
        assert data['intercept'] == pytest.approx(expected_intercept)

    def test_degenerate_input_marks_no_tail(self):
        data = spl._loglog_fit_data(np.array([1.0, 1.0, 1.0]), 1.0)
        assert data['alpha'] is None
        assert not data['tmask'].any()


class TestPlotLoglogWithFit:
    def test_returns_axes_and_alpha(self, synthetic_powerlaw_samples):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        out_ax, alpha = spl.plot_loglog_with_fit(
            synthetic_powerlaw_samples, np.min(synthetic_powerlaw_samples),
            ax=ax, title='Test Tail'
        )
        assert out_ax is ax
        assert alpha == pytest.approx(4.0, abs=0.3)
        assert '4.' in ax.get_title()  # alpha shown on the chart
        plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
