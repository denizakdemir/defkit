"""
defkit.gaussian
===============

Closed-form results for Gaussian experiments.

Every public name here says what it is. Three different things had all been
called "the Gaussian deficiency" across the packages this module replaces:

* an exact dominance test (theorem-backed, gives delta = 0 or "positive"),
* a rigorous lower bound from a two-point construction,
* an unvalidated closed-form guess inherited from a design document.

Reporting the third as though it were the first is the failure mode this
module is shaped to prevent, so the third carries ``_proxy`` in its name and
returns a value documented as having no proven relationship to delta.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from .distances import tv_gaussian

__all__ = [
    "gaussian_location_dominates",
    "gaussian_location_deficiency_proxy",
    "gaussian_scalar_location_deficiency",
    "two_point_confounding_lower_bound",
    "expfam_deficiency_proxy",
]

Direction = Literal["forward", "backward", "symmetric"]


def _symmetric(M: np.ndarray, name: str) -> np.ndarray:
    M = np.asarray(M, dtype=float)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"{name} must be a square matrix, got shape {M.shape}.")
    if not np.allclose(M, M.T, atol=1e-10):
        raise ValueError(f"{name} must be symmetric.")
    return 0.5 * (M + M.T)


# ---------------------------------------------------------------------------
# Exact: the Blackwell ordering for Gaussian location families
# ---------------------------------------------------------------------------

def gaussian_location_dominates(
    Sigma1: np.ndarray,
    Sigma2: np.ndarray,
    *,
    tol: float = 1e-10,
) -> bool:
    """Does ``E1 = N(theta, Sigma1)`` dominate ``E2 = N(theta, Sigma2)``?

    True exactly when ``Sigma2 - Sigma1`` is positive semidefinite, in which
    case ``E2`` is reproduced from ``E1`` by adding independent noise of
    covariance ``Sigma2 - Sigma1`` and ``delta(E1, E2) = 0``.

    This direction is exact and needs no numerical bound. A False result means
    only that delta is strictly positive, not how large it is.
    """
    S1 = _symmetric(Sigma1, "Sigma1")
    S2 = _symmetric(Sigma2, "Sigma2")
    if S1.shape != S2.shape:
        raise ValueError(
            f"Sigma1 and Sigma2 must have the same shape, got {S1.shape} and {S2.shape}."
        )
    return bool(np.all(np.linalg.eigvalsh(S2 - S1) >= -tol))


def gaussian_scalar_location_deficiency(sd_source: float, sd_target: float) -> float:
    """Upper bound on the deficiency between scalar Gaussian location experiments.

    ``N(theta, sd_source^2)`` against ``N(theta, sd_target^2)``.

    * ``sd_target >= sd_source``: returns exactly ``0.0``, and this is exact.
      The target is reproduced by adding independent noise of variance
      ``sd_target^2 - sd_source^2``, so delta really is 0.
    * ``sd_source > sd_target``: returns ``TV(N(0, sd_source), N(0, sd_target))``.

    The second case is a **rigorous upper bound**, not a heuristic. Taking the
    identity kernel gives ``TV(N(theta, sd_source^2), N(theta, sd_target^2))``
    at every theta, and by translation invariance that does not depend on
    theta, so the supremum over theta equals the constant above and delta is at
    most that. It is not tight: checked against the exact LP on discretised
    versions of these experiments it runs roughly a factor of two high (for
    example LP 0.149 against bound 0.323 at sds 2.0 and 1.0).

    Use :func:`defkit.solvers.deficiency_exact` on a discretisation when the
    factor of two matters.
    """
    sd_source, sd_target = float(sd_source), float(sd_target)
    if sd_source <= 0 or sd_target <= 0:
        raise ValueError("standard deviations must be positive.")
    if sd_target >= sd_source:
        return 0.0
    return tv_gaussian(0.0, sd_source, 0.0, sd_target)


# ---------------------------------------------------------------------------
# Proxy: the closed form inherited from the package blueprints
# ---------------------------------------------------------------------------

def gaussian_location_deficiency_proxy(
    Sigma1: np.ndarray,
    Sigma2: np.ndarray,
    *,
    direction: Direction = "forward",
    tol: float = 1e-10,
) -> float:
    """A severity proxy for non-dominated Gaussian location experiments.

    Returns 0 exactly when :func:`gaussian_location_dominates` holds, which is
    the case where delta is genuinely 0. Otherwise it returns

        ``2 * Phi(sqrt(-lambda_min) / 2) - 1``

    where ``lambda_min`` is the smallest eigenvalue of ``Sigma2 - Sigma1``.

    This closed form comes from a package design document. It is monotone in
    the dominance violation and lies in [0, 1], which makes it usable as a
    severity readout, but **no proof relates it to delta** and it should not be
    quoted as the deficiency. For a number with a proof behind it, use
    :func:`gaussian_location_dominates` for the zero case, or solve the finite
    problem with :func:`defkit.solvers.deficiency_exact` after discretising.
    """
    from scipy.stats import norm

    S1 = _symmetric(Sigma1, "Sigma1")
    S2 = _symmetric(Sigma2, "Sigma2")
    if S1.shape != S2.shape:
        raise ValueError(
            f"Sigma1 and Sigma2 must have the same shape, got {S1.shape} and {S2.shape}."
        )

    def one_way(A: np.ndarray, B: np.ndarray) -> float:
        eig = np.linalg.eigvalsh(B - A)
        lambda_min = float(eig.min())
        if lambda_min >= -tol:
            return 0.0
        val = 2.0 * float(norm.cdf(np.sqrt(-lambda_min) / 2.0)) - 1.0
        return float(np.clip(val, 0.0, 1.0))

    if direction == "forward":
        return one_way(S1, S2)
    if direction == "backward":
        return one_way(S2, S1)
    if direction == "symmetric":
        return max(one_way(S1, S2), one_way(S2, S1))
    raise ValueError(
        'direction must be "forward", "backward" or "symmetric", '
        f"got {direction!r}."
    )


def expfam_deficiency_proxy(I1: np.ndarray, I2: np.ndarray) -> float:
    """Fisher-information severity proxy ``0.5 * sqrt(tr(I1^-1 (I1 - I2)^2))``.

    Another blueprint closed form, carried over so call sites keep working.
    Like :func:`gaussian_location_deficiency_proxy` it is a severity readout
    with no proven relationship to delta, and unlike that one it is not even
    bounded by 1.
    """
    I1 = np.asarray(I1, dtype=float)
    I2 = np.asarray(I2, dtype=float)
    if I1.shape != I2.shape:
        raise ValueError(
            f"I1 and I2 must have the same shape, got {I1.shape} and {I2.shape}."
        )
    diff = I1 - I2
    val = 0.5 * float(np.sqrt(np.trace(np.linalg.pinv(I1) @ diff @ diff)))
    return float(max(val, 0.0))


# ---------------------------------------------------------------------------
# Rigorous lower bound: two-point construction
# ---------------------------------------------------------------------------

def two_point_confounding_lower_bound(
    alpha: float,
    gamma: float,
    sigma_a: float = 1.0,
    sigma_y: float = 1.0,
    a: float = 1.0,
) -> float:
    """Lower bound on causal deficiency under unmeasured confounding.

    For the linear-Gaussian structural model with confounder ``U`` acting on
    treatment through ``alpha`` and on outcome through ``gamma``, a two-point
    construction gives ``delta_causal(a) >= 0.5 * TV(P_0, P_1)`` where the two
    observationally matched laws are

        ``P_0 = N(0, gamma^2 + sigma_y^2)``
        ``P_1 = N(Delta, gamma^2 sigma_a^2 / (alpha^2 + sigma_a^2) + sigma_y^2)``

    with mean separation ``Delta = |a| |alpha gamma| / (alpha^2 + sigma_a^2)``.

    This is a lower bound, so a small value does not license a claim that
    confounding is mild. It is the Python counterpart of the R implementation
    in ``causaldef::confounding_frontier()``; the two agree to quadrature
    tolerance and a shared fixture pins them together.
    """
    alpha, gamma = float(alpha), float(gamma)
    sigma_a, sigma_y, a = float(sigma_a), float(sigma_y), float(a)
    if sigma_a < 0 or sigma_y < 0:
        raise ValueError("sigma_a and sigma_y must be non-negative.")

    denom = alpha * alpha + sigma_a * sigma_a
    if denom <= 0:
        return 0.0

    delta_mu = abs(a * alpha * gamma) / denom
    gamma_prime_sq = (gamma * gamma) * (sigma_a * sigma_a) / denom

    sd1 = np.sqrt(gamma * gamma + sigma_y * sigma_y)
    sd2 = np.sqrt(gamma_prime_sq + sigma_y * sigma_y)

    tv = tv_gaussian(0.0, sd1, delta_mu, sd2)
    if not np.isfinite(tv):
        return float("nan")
    return float(0.5 * tv)
