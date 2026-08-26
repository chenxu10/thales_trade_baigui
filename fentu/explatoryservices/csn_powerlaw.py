"""
Continuous Clauset-Shalizi-Newman (2009) power-law fit for financial data.

Estimates the tail exponent ``alpha`` of a continuous, positive sample
(e.g. absolute returns) the way CSN 2009 (arXiv:0706.1062, SIAM Rev. 2009)
prescribe — the same procedure the repo already uses for function lengths in
``fentu.metaprogramming.function_length_powerlaw``, but for **continuous**
data (the discrete variant's Hurwitz-zeta machinery does not apply to
real-valued returns):

1. MLE of the exponent  ``alpha = 1 + n / sum( ln(x_i / x_min) )``
2. ``x_min`` chosen by Kolmogorov-Smirnov minimization against the fitted
   Pareto CDF ``F(x) = 1 - (x / x_min)^(1 - alpha)``
3. goodness-of-fit p-value by parametric bootstrap (re-fitting ``x_min``)
4. standard error  ``sigma = (alpha - 1) / sqrt(n_tail)``

This replaces the biased binned-OLS-on-histogram estimator: CSN show that
binning + least-squares on the log-log density can be off by whole units of
alpha — the difference between "hedge is cheap" and "tail is thin".
"""
from __future__ import annotations

import numpy as np

MAX_XMIN_CANDIDATES = 200  # quantile grid cap keeps select_xmin O(n * 200)
MIN_TAIL_POINTS = 10       # CSN: do not trust x_min fitted on a handful of points


def mle_alpha_continuous(tail: np.ndarray, x_min: float) -> float:
    """Continuous MLE: alpha = 1 + n / sum( ln(x_i / x_min) )."""
    tail = np.asarray(tail, dtype=float)
    n = len(tail)
    if n < 2 or x_min <= 0:
        return np.nan
    s = float(np.sum(np.log(tail / x_min)))
    if s <= 0:
        return np.nan
    return 1.0 + n / s


def ks_distance_continuous(tail: np.ndarray, x_min: float, alpha: float) -> float:
    """KS distance between the empirical tail CDF and the fitted continuous
    Pareto CDF ``1 - (x / x_min)^(1 - alpha)``."""
    tail = np.sort(np.asarray(tail, dtype=float))
    n = len(tail)
    if n < 2:
        return 0.0
    if alpha is None or not np.isfinite(alpha) or alpha <= 1.0:
        return np.inf
    with np.errstate(all="ignore"):
        model = 1.0 - (tail / x_min) ** (1.0 - alpha)
        if not np.all(np.isfinite(model)):
            return np.inf
        emp_hi = np.arange(1, n + 1) / n
        emp_lo = np.arange(0, n) / n
        d = max(float(np.max(emp_hi - model)), float(np.max(model - emp_lo)))
    return d if np.isfinite(d) else np.inf


def _xmin_candidates(data: np.ndarray) -> np.ndarray:
    """Candidate x_min values: unique data points, capped at a quantile grid
    and required to leave at least MIN_TAIL_POINTS in the tail."""
    candidates = np.unique(data)
    if len(candidates) > MAX_XMIN_CANDIDATES:
        qs = np.linspace(0.0, 1.0, MAX_XMIN_CANDIDATES)
        candidates = np.unique(np.quantile(data, qs))
    n = len(data)
    return candidates[(n - np.searchsorted(np.sort(data), candidates)) >= MIN_TAIL_POINTS]


def select_xmin_continuous(data: np.ndarray):
    """KS-minimizing x_min selection. Returns (x_min, alpha, D, n_tail)."""
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data) & (data > 0)]
    if len(data) < MIN_TAIL_POINTS + 1:
        return None, None, None, 0
    best = (np.inf, None, None, 0)
    for xm in _xmin_candidates(data):
        tail = data[data >= xm]
        a = mle_alpha_continuous(tail, float(xm))
        if not np.isfinite(a) or a <= 1.0:
            continue
        d = ks_distance_continuous(tail, float(xm), a)
        if d < best[0]:
            best = (d, float(xm), float(a), len(tail))
    if best[1] is None:
        return None, None, None, 0
    return best[1], best[2], best[0], best[3]


def _continuous_pl_samples(alpha: float, x_min: float, n: int, rng) -> np.ndarray:
    """Inverse-CDF sampler for p(x) = (a-1)/x_min * (x/x_min)^-a."""
    u = rng.random(n)
    return x_min * (1.0 - u) ** (-1.0 / (alpha - 1.0))


def _bootstrap_pvalue(d_obs, x_min, alpha, n_tail, B, rng) -> float:
    """Parametric bootstrap goodness-of-fit p-value (re-fits x_min each draw)."""
    count = 0
    for _ in range(B):
        synth = _continuous_pl_samples(alpha, x_min, n_tail, rng)
        _, _, D_s, _ = select_xmin_continuous(synth)
        if D_s is None:
            continue
        if D_s >= d_obs:
            count += 1
    return (count + 1) / (B + 1)


def fit_power_law_continuous(data, n_bootstrap: int = 0, seed: int = 0) -> dict:
    """Full continuous CSN fit. Returns a dict with alpha, alpha_std, x_min,
    n_tail, n_total, ks, p (p is None unless n_bootstrap > 0)."""
    data = np.asarray(data, dtype=float)
    if data.size:
        data = data[np.isfinite(data) & (data > 0)]
    n_total = len(data)
    xm, alpha, D, n_tail = select_xmin_continuous(data)
    out = {"alpha": alpha, "alpha_std": None, "x_min": xm,
           "n_tail": n_tail, "n_total": n_total, "ks": D, "p": None}
    if alpha is None or not np.isfinite(alpha):
        return out
    out["alpha_std"] = (alpha - 1.0) / np.sqrt(n_tail)
    if n_bootstrap and n_bootstrap > 0 and n_tail >= 10:
        rng = np.random.default_rng(seed)
        out["p"] = _bootstrap_pvalue(D, xm, alpha, n_tail, n_bootstrap, rng)
    return out
