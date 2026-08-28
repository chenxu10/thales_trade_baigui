"""
This script provides tools to plot on a sample of data set on log-log scale
with power law fitting.

It helps you to see whether underlying data set has a power law potential.
The tail exponent alpha is estimated with the continuous
Clauset-Shalizi-Newman (2009) procedure (MLE + KS-selected x_min, see
fentu.explatoryservices.csn_powerlaw) — NOT by OLS on the log-binned
histogram, which CSN 2009 show is biased by whole units of alpha. The
histogram is drawn for display only; the fitted line is the Pareto density
p(x) = (alpha-1)/x_min * (x/x_min)^(-alpha) for x >= x_min.

Author: Xu.Shen<xs286@cornell.edu>
"""

import numpy as np
import matplotlib.pyplot as plt

from fentu.explatoryservices import csn_powerlaw as csn


def create_log_space_bins(x_min, samples) -> np.ndarray:
    """
    Creates an array of numbers that are evenly distrbuted on log space
    """
    bins = np.logspace(np.log10(x_min), np.log10(np.max(samples)), 100)
    return bins


def compute_histogram_with_bins(samples, bins, method='numpy_density'):
    """
    Unified histogram computation pipeline with different density calculation methods

    This function centralizes histogram computation logic to eliminate duplication
    between different plotting methods.

    Parameters:
    samples: Array of samples to create histogram from
    bins: Array of bin edges
    method: Density calculation method
        - 'numpy_density': Use np.histogram with density=True (faster, standard)
        - 'manual_density': Manual normalization by bin width (more control, reduces noise)

    Returns:
    tuple: (values, bin_centers, bin_edges)
        - values: Density values for each bin
        - bin_centers: Center position of each bin
        - bin_edges: Original bin edges array
    """
    if method == 'numpy_density':
        # Standard numpy density calculation
        values, bin_edges = np.histogram(samples, bins=bins, density=True)
        # Use arithmetic mean for bin centers
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    elif method == 'manual_density':
        # Manual density normalization for better control
        counts, bin_edges = np.histogram(samples, bins=bins, density=False)

        # Calculate bin widths and normalize manually
        bin_widths = bin_edges[1:] - bin_edges[:-1]
        values = counts / (bin_widths * len(samples))

        # Use geometric mean for bin centers (correct for log-scale bins)
        bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])

    else:
        raise ValueError(f"Unknown method: {method}. Use 'numpy_density' or 'manual_density'")

    return values, bin_centers, bin_edges


def _positive_density_mask(density):
    """Boolean mask of bins with positive density (log-log needs log10(y) > -inf)."""
    return density > 0


def _pareto_density_loglog_params(alpha, x_min):
    """Slope/intercept of the fitted Pareto density in log10-log10 space.

    p(x) = (alpha-1)/x_min * (x/x_min)^(-alpha)
    => log10(p) = -alpha * log10(x) + log10((alpha-1) * x_min^(alpha-1))
    """
    slope = -alpha
    intercept = np.log10((alpha - 1.0) * x_min ** (alpha - 1.0))
    return slope, intercept


def _loglog_fit_data(samples, x_min):
    """Histogram + CSN tail fit for the log-log plot (pure computation, no axes).

    Returns a dict with the positive-density bins, the CSN fit result, the
    boolean mask marking bins at/above the fitted x_min, and the
    slope/intercept of the fitted Pareto density line.
    """
    bins = create_log_space_bins(x_min, samples)
    density, bin_centers, _ = compute_histogram_with_bins(
        samples, bins, method='manual_density'
    )

    mask = density > 0
    valid_centers = bin_centers[mask]
    valid_density = density[mask]

    fit = csn.fit_power_law_continuous(samples)

    out = {
        'valid_centers': valid_centers,
        'valid_density': valid_density,
        'fit': fit,
        'tmask': None,
        'slope': None,
        'intercept': None,
        'alpha': fit['alpha'],
    }
    if fit['alpha'] is not None:
        out['tmask'] = valid_centers >= fit['x_min']
        out['slope'], out['intercept'] = _pareto_density_loglog_params(
            fit['alpha'], fit['x_min']
        )
    else:
        out['tmask'] = np.zeros(len(valid_centers), dtype=bool)
    return out


def _draw_loglog_series(ax, data):
    """Render body points, tail points, and the fitted line onto the axes."""
    valid_centers = data['valid_centers']
    valid_density = data['valid_density']
    tmask = data['tmask']
    fit = data['fit']

    ax.loglog(
        valid_centers[~tmask], valid_density[~tmask],
        'o', alpha=0.4, color='gray', label='Data (not fitted)'
    )
    if fit['alpha'] is None:
        return

    ax.loglog(
        valid_centers[tmask], valid_density[tmask],
        'o', alpha=0.7, color='blue',
        label=f"Tail (x ≥ x_min={fit['x_min']:.3g}, n={fit['n_tail']})"
    )

    fit_x = valid_centers[tmask]
    fit_y = 10 ** (data['slope'] * np.log10(fit_x) + data['intercept'])
    ax.loglog(fit_x, fit_y, 'r-', linewidth=2, label=f"Fit (α={data['alpha']:.2f})")


def _decorate_loglog_axes(ax, data, title):
    """Axis labels, title, legend, and grid for the log-log fit plot."""
    ax.set_xlabel('x (log scale)')
    ax.set_ylabel('Probability density (log scale)')
    if data['alpha'] is not None:
        fit_text = f"α={data['alpha']:.2f}"
    else:
        fit_text = "power-law fit unavailable"
    if title:
        ax.set_title(f'{title}: {fit_text}')
    else:
        ax.set_title(f'Power-law fit: {fit_text}')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, which='both')


def plot_loglog_with_fit(samples, x_min, ax=None, title=None):
    """
    Plot log-log histogram with the CSN (2009) power-law tail fit.

    The histogram is display only; alpha and the fitted line come from the
    continuous CSN estimator (MLE + KS-selected x_min) on the raw samples.

    Parameters:
    samples: Power-law distributed samples (positive, continuous)
    x_min: Lower edge of the histogram bins (display only — the fit's x_min
           is selected by KS minimization, not taken from this argument)
    ax: Matplotlib axes object. If None, uses current axes
    title: Optional custom title for the plot

    Returns:
    ax: The axes object used for plotting
    alpha: Estimated power law exponent (None if the fit is unavailable)
    """
    if ax is None:
        ax = plt.gca()

    data = _loglog_fit_data(samples, x_min)
    _draw_loglog_series(ax, data)
    _decorate_loglog_axes(ax, data, title)

    return ax, data['alpha']
