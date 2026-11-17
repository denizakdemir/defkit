"""
defkit.distances
================

Probability distances shared across the experiment-comparison packages.

The Gaussian total-variation routine here is the one that matters most. Its
predecessor computed the density crossing points in closed form by solving a
quadratic, which loses precision and silently returns near-0 instead of near-1
when both standard deviations are very small relative to the mean separation
(a near-degenerate point-mass regime, where the discriminant cancels). That bug
was found in the R implementation in August 2026, then found again in a Python
port that had been written independently from the same closed form. This module
exists so there is only one copy left to get wrong.

The replacement integrates ``|f1 - f2|`` over breakpoints scaled to each
component's own standard deviation, which keeps the quadrature from stepping
over a narrow density spike inside a wide, mostly flat range.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

__all__ = [
    "tv_discrete",
    "tv_histogram",
    "tv_gaussian",
    "wasserstein1_normalized",
    "rbf_kernel",
    "mmd_rbf",
    "normalize_weights",
]


def normalize_weights(weights: Optional[np.ndarray], n: int) -> np.ndarray:
    """Weights as a probability vector of length ``n``; uniform if unusable."""
    if weights is None:
        return np.full(n, 1.0 / n, dtype=float)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if weights.shape[0] != n:
        raise ValueError(
            f"weights has length {weights.shape[0]}, expected {n}."
        )
    weights = np.clip(weights, 0.0, np.inf)
    total = float(np.sum(weights))
    if total <= 0:
        return np.full(n, 1.0 / n, dtype=float)
    return weights / total


# ---------------------------------------------------------------------------
# Total variation
# ---------------------------------------------------------------------------

def tv_discrete(p: np.ndarray, q: np.ndarray) -> float:
    """``TV(p, q) = 0.5 * ||p - q||_1`` for two probability vectors."""
    p = np.asarray(p, dtype=float).reshape(-1)
    q = np.asarray(q, dtype=float).reshape(-1)
    if p.shape != q.shape:
        raise ValueError(f"p and q must have the same shape, got {p.shape} and {q.shape}.")
    return float(np.clip(0.5 * np.sum(np.abs(p - q)), 0.0, 1.0))


def tv_histogram(
    values_a: np.ndarray,
    values_b: np.ndarray,
    *,
    bins: int = 50,
    value_range: Tuple[float, float] = (0.0, 1.0),
    weights_a: Optional[np.ndarray] = None,
    weights_b: Optional[np.ndarray] = None,
) -> float:
    """TV between two samples after binning both on a common grid.

    Returns NaN for an empty input or a bin range that captures no mass, rather
    than raising, so streaming callers can carry the gap forward as missing.
    """
    values_a = np.asarray(values_a, dtype=float).reshape(-1)
    values_b = np.asarray(values_b, dtype=float).reshape(-1)
    if values_a.size == 0 or values_b.size == 0:
        return float("nan")

    hist_a, _ = np.histogram(values_a, bins=bins, range=value_range, weights=weights_a)
    hist_b, _ = np.histogram(values_b, bins=bins, range=value_range, weights=weights_b)

    total_a = float(np.sum(hist_a))
    total_b = float(np.sum(hist_b))
    if total_a <= 0 or total_b <= 0:
        return float("nan")

    return tv_discrete(hist_a / total_a, hist_b / total_b)


def tv_gaussian(
    mu1: float,
    sd1: float,
    mu2: float,
    sd2: float,
    *,
    n_points: int = 2001,
) -> float:
    """TV between two univariate normals, by piecewise Simpson quadrature.

    Breakpoints are placed at +/-5, 15 and 30 standard deviations around each
    mean, using that component's own sd, so neither density can hide a narrow
    spike inside a wide integration panel.

    Accurate to roughly 1e-7 across standard deviations spanning 1e-6 to 1e3
    (see ``tests/test_distances.py``). Returns NaN for a non-positive sd.
    """
    mu1, sd1, mu2, sd2 = float(mu1), float(sd1), float(mu2), float(sd2)
    if not np.isfinite([mu1, sd1, mu2, sd2]).all() or sd1 <= 0 or sd2 <= 0:
        return float("nan")
    if mu1 == mu2 and sd1 == sd2:
        return 0.0

    if n_points < 3 or n_points % 2 == 0:
        raise ValueError("n_points must be an odd integer >= 3 for Simpson's rule.")

    offsets = (-30, -15, -5, 0, 5, 15, 30)
    breakpoints = np.array(sorted({mu1 + k * sd1 for k in offsets}
                                  | {mu2 + k * sd2 for k in offsets}))

    def abs_diff(x: np.ndarray) -> np.ndarray:
        f1 = np.exp(-0.5 * ((x - mu1) / sd1) ** 2) / (sd1 * np.sqrt(2.0 * np.pi))
        f2 = np.exp(-0.5 * ((x - mu2) / sd2) ** 2) / (sd2 * np.sqrt(2.0 * np.pi))
        return np.abs(f1 - f2)

    total = 0.0
    for lo, hi in zip(breakpoints[:-1], breakpoints[1:]):
        if hi <= lo:
            continue
        xs = np.linspace(lo, hi, n_points)
        ys = abs_diff(xs)
        h = (hi - lo) / (n_points - 1)
        total += h / 3.0 * (
            ys[0] + ys[-1] + 4.0 * np.sum(ys[1:-1:2]) + 2.0 * np.sum(ys[2:-2:2])
        )

    return float(np.clip(0.5 * total, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Wasserstein
# ---------------------------------------------------------------------------

def wasserstein1_normalized(
    values_a: np.ndarray,
    values_b: np.ndarray,
    *,
    weights_a: Optional[np.ndarray] = None,
    weights_b: Optional[np.ndarray] = None,
    norm_range: Optional[Tuple[float, float]] = None,
) -> float:
    """Weighted 1-D Wasserstein-1 distance, rescaled to [0, 1] by value range.

    Rescaling makes the value comparable to a TV reading on the same data; the
    unnormalised distance is in the units of the input.
    """
    values_a = np.asarray(values_a, dtype=float).reshape(-1)
    values_b = np.asarray(values_b, dtype=float).reshape(-1)
    if values_a.size == 0 or values_b.size == 0:
        return float("nan")

    wa = normalize_weights(weights_a, values_a.size)
    wb = normalize_weights(weights_b, values_b.size)

    order_a = np.argsort(values_a)
    order_b = np.argsort(values_b)
    xa, wa = values_a[order_a], wa[order_a]
    xb, wb = values_b[order_b], wb[order_b]

    cwa = np.cumsum(wa)
    cwb = np.cumsum(wb)

    grid = np.sort(np.unique(np.concatenate([xa, xb])))
    if grid.size <= 1:
        return 0.0

    idx_a = np.searchsorted(xa, grid, side="right") - 1
    idx_b = np.searchsorted(xb, grid, side="right") - 1
    fa = np.where(idx_a >= 0, cwa[np.clip(idx_a, 0, cwa.size - 1)], 0.0)
    fb = np.where(idx_b >= 0, cwb[np.clip(idx_b, 0, cwb.size - 1)], 0.0)

    w1 = float(np.sum(np.abs(fa[:-1] - fb[:-1]) * np.diff(grid)))

    if norm_range is None:
        lo = float(min(xa[0], xb[0]))
        hi = float(max(xa[-1], xb[-1]))
    else:
        lo, hi = float(norm_range[0]), float(norm_range[1])
    return float(np.clip(w1 / max(hi - lo, 1e-12), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Maximum mean discrepancy
# ---------------------------------------------------------------------------

def _as_2d(X: np.ndarray, name: str) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError(f"{name} must be 1- or 2-dimensional, got shape {X.shape}.")
    return X


def rbf_kernel(X: np.ndarray, Y: np.ndarray, gamma: float) -> np.ndarray:
    """Gaussian RBF kernel matrix ``exp(-gamma * ||x - y||^2)``."""
    X = _as_2d(X, "X")
    Y = _as_2d(Y, "Y")
    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"X and Y must share the feature axis, got {X.shape[1]} and {Y.shape[1]}."
        )
    sq = (
        np.sum(X * X, axis=1)[:, None]
        + np.sum(Y * Y, axis=1)[None, :]
        - 2.0 * (X @ Y.T)
    )
    return np.exp(-float(gamma) * np.clip(sq, 0.0, None))


def median_heuristic_gamma(X: np.ndarray, Y: np.ndarray) -> float:
    """Bandwidth from the median pairwise distance of the pooled sample."""
    pooled = np.vstack([_as_2d(X, "X"), _as_2d(Y, "Y")])
    n = pooled.shape[0]
    if n <= 1:
        return 1.0
    if n > 512:
        idx = np.random.default_rng(0).choice(n, size=512, replace=False)
        pooled = pooled[idx]
    sq = (
        np.sum(pooled * pooled, axis=1)[:, None]
        + np.sum(pooled * pooled, axis=1)[None, :]
        - 2.0 * (pooled @ pooled.T)
    )
    sq = np.clip(sq, 0.0, None)
    upper = sq[np.triu_indices_from(sq, k=1)]
    med = float(np.median(upper)) if upper.size else 0.0
    return 1.0 / max(med, 1e-12)


def mmd_rbf(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    gamma: Optional[float] = None,
    weights_x: Optional[np.ndarray] = None,
    weights_y: Optional[np.ndarray] = None,
    unbiased: bool = False,
    normalize: bool = False,
    max_samples: Optional[int] = 512,
    random_state: int = 7,
) -> float:
    """Maximum mean discrepancy under an RBF kernel.

    Parameters
    ----------
    gamma:
        Kernel bandwidth. ``None`` uses the median pairwise-distance heuristic.
    unbiased:
        Use the U-statistic (diagonal removed). Requires equal weights and at
        least two points per sample; the biased V-statistic is the default
        because it is the one that stays valid under weighting.
    normalize:
        Divide by ``sqrt(2)``, the supremum of the RBF MMD, to land in [0, 1].
    max_samples:
        Subsample larger inputs for tractability. ``None`` disables.

    Returns the MMD itself, not its square.
    """
    X = _as_2d(X, "X")
    Y = _as_2d(Y, "Y")
    if X.shape[0] == 0 or Y.shape[0] == 0:
        return float("nan")

    wx = normalize_weights(weights_x, X.shape[0])
    wy = normalize_weights(weights_y, Y.shape[0])

    if max_samples is not None:
        rng = np.random.default_rng(random_state)
        if X.shape[0] > max_samples:
            idx = rng.choice(X.shape[0], size=max_samples, replace=False, p=wx)
            X, wx = X[idx], wx[idx] / np.sum(wx[idx])
        if Y.shape[0] > max_samples:
            idx = rng.choice(Y.shape[0], size=max_samples, replace=False, p=wy)
            Y, wy = Y[idx], wy[idx] / np.sum(wy[idx])

    if gamma is None:
        gamma = median_heuristic_gamma(X, Y)

    kxx = rbf_kernel(X, X, gamma)
    kyy = rbf_kernel(Y, Y, gamma)
    kxy = rbf_kernel(X, Y, gamma)

    if unbiased:
        n, m = X.shape[0], Y.shape[0]
        if n < 2 or m < 2:
            raise ValueError("unbiased MMD needs at least 2 points in each sample.")
        if weights_x is not None or weights_y is not None:
            raise ValueError(
                "unbiased MMD is only defined here for unweighted samples; "
                "pass unbiased=False to use the weighted V-statistic."
            )
        sum_xx = (kxx.sum() - np.trace(kxx)) / (n * (n - 1))
        sum_yy = (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
        mmd2 = float(sum_xx + sum_yy - 2.0 * kxy.mean())
    else:
        mmd2 = float(wx @ kxx @ wx + wy @ kyy @ wy - 2.0 * (wx @ kxy @ wy))

    mmd = float(np.sqrt(max(mmd2, 0.0)))
    if normalize:
        return float(np.clip(mmd / np.sqrt(2.0), 0.0, 1.0))
    return mmd
