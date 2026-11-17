"""Tests for the Gaussian closed forms.

These also pin the exact/proxy distinction: the proxy functions are allowed to
be wrong about delta, but they are not allowed to disagree with the exact
dominance test about when delta is zero.
"""

import numpy as np
import pytest

from defkit.gaussian import (
    expfam_deficiency_proxy,
    gaussian_location_deficiency_proxy,
    gaussian_location_dominates,
    gaussian_scalar_location_deficiency,
    two_point_confounding_lower_bound,
)


# ---------------------------------------------------------------------------
# Exact dominance
# ---------------------------------------------------------------------------

def test_equal_covariances_dominate_each_other():
    S = np.array([[2.0, 0.3], [0.3, 1.0]])
    assert gaussian_location_dominates(S, S)


def test_smaller_covariance_dominates_larger():
    """Less noise is more informative: N(theta, I) dominates N(theta, 2I)."""
    S1 = np.eye(2)
    S2 = 2.0 * np.eye(2)
    assert gaussian_location_dominates(S1, S2)
    assert not gaussian_location_dominates(S2, S1)


def test_incomparable_covariances_dominate_in_neither_direction():
    S1 = np.diag([1.0, 5.0])
    S2 = np.diag([5.0, 1.0])
    assert not gaussian_location_dominates(S1, S2)
    assert not gaussian_location_dominates(S2, S1)


def test_dominance_rejects_asymmetric_input():
    with pytest.raises(ValueError, match="symmetric"):
        gaussian_location_dominates(np.array([[1.0, 2.0], [0.0, 1.0]]), np.eye(2))


def test_dominance_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        gaussian_location_dominates(np.eye(2), np.eye(3))


# ---------------------------------------------------------------------------
# The proxy agrees with the exact test about the zero set
# ---------------------------------------------------------------------------

def test_proxy_is_zero_exactly_when_dominance_holds():
    """Both branches are constructed on purpose.

    Two independently drawn covariance matrices are essentially never
    comparable in the positive-semidefinite order, so sampling alone exercises
    only the non-dominated branch. Dominated pairs are built by adding noise.
    """
    rng = np.random.default_rng(0)
    checked_zero = checked_positive = 0

    for _ in range(20):
        A = rng.normal(size=(3, 3))
        S1 = A @ A.T + 0.1 * np.eye(3)

        # Dominated by construction: S2 = S1 + (something PSD).
        B = rng.normal(size=(3, 3))
        S2 = S1 + B @ B.T + 1e-6 * np.eye(3)
        assert gaussian_location_dominates(S1, S2)
        assert gaussian_location_deficiency_proxy(S1, S2) == 0.0
        checked_zero += 1

        # Not dominated: the reverse direction of the same pair.
        assert not gaussian_location_dominates(S2, S1)
        assert gaussian_location_deficiency_proxy(S2, S1) > 0.0
        checked_positive += 1

    assert checked_zero > 0 and checked_positive > 0


def test_proxy_is_in_unit_interval():
    rng = np.random.default_rng(1)
    for _ in range(20):
        A = rng.normal(size=(2, 2))
        S1 = A @ A.T + 0.1 * np.eye(2)
        B = rng.normal(size=(2, 2))
        S2 = B @ B.T + 0.1 * np.eye(2)
        assert 0.0 <= gaussian_location_deficiency_proxy(S1, S2) <= 1.0


def test_proxy_symmetric_direction_is_the_max_of_both_ways():
    S1 = np.diag([1.0, 5.0])
    S2 = np.diag([5.0, 1.0])
    fwd = gaussian_location_deficiency_proxy(S1, S2, direction="forward")
    bwd = gaussian_location_deficiency_proxy(S1, S2, direction="backward")
    sym = gaussian_location_deficiency_proxy(S1, S2, direction="symmetric")
    assert sym == pytest.approx(max(fwd, bwd))


def test_proxy_rejects_unknown_direction():
    with pytest.raises(ValueError, match="direction must be"):
        gaussian_location_deficiency_proxy(np.eye(2), np.eye(2), direction="sideways")


# ---------------------------------------------------------------------------
# Scalar location
# ---------------------------------------------------------------------------

def test_scalar_deficiency_is_zero_when_target_is_noisier():
    assert gaussian_scalar_location_deficiency(1.0, 2.0) == 0.0
    assert gaussian_scalar_location_deficiency(1.0, 1.0) == 0.0


def test_scalar_deficiency_is_positive_when_source_is_noisier():
    assert gaussian_scalar_location_deficiency(2.0, 1.0) > 0.0


def test_scalar_deficiency_grows_as_the_gap_widens():
    a = gaussian_scalar_location_deficiency(2.0, 1.5)
    b = gaussian_scalar_location_deficiency(4.0, 1.5)
    assert b > a


def test_scalar_deficiency_rejects_nonpositive_sd():
    with pytest.raises(ValueError, match="positive"):
        gaussian_scalar_location_deficiency(0.0, 1.0)


# ---------------------------------------------------------------------------
# Two-point confounding bound
# ---------------------------------------------------------------------------

def test_confounding_bound_is_zero_without_confounding():
    """No path from U to A, or none from U to Y, leaves nothing to bound."""
    assert two_point_confounding_lower_bound(0.0, 1.0) == pytest.approx(0.0, abs=1e-12)
    assert two_point_confounding_lower_bound(1.0, 0.0) == pytest.approx(0.0, abs=1e-12)


def test_confounding_bound_grows_with_confounding_strength():
    weak = two_point_confounding_lower_bound(0.3, 0.3)
    strong = two_point_confounding_lower_bound(2.0, 2.0)
    assert strong > weak


def test_confounding_bound_is_in_unit_interval():
    rng = np.random.default_rng(2)
    for _ in range(30):
        val = two_point_confounding_lower_bound(
            alpha=rng.uniform(-3, 3),
            gamma=rng.uniform(-3, 3),
            sigma_a=rng.uniform(0.1, 3),
            sigma_y=rng.uniform(0.1, 3),
        )
        assert 0.0 <= val <= 1.0


def test_confounding_bound_depends_only_on_the_magnitude_of_the_product():
    """Flipping both signs leaves the confounding path unchanged."""
    a = two_point_confounding_lower_bound(1.5, 2.0)
    b = two_point_confounding_lower_bound(-1.5, -2.0)
    assert a == pytest.approx(b, abs=1e-12)


def test_confounding_bound_rejects_negative_noise():
    with pytest.raises(ValueError, match="non-negative"):
        two_point_confounding_lower_bound(1.0, 1.0, sigma_a=-1.0)


# ---------------------------------------------------------------------------
# Exponential-family proxy
# ---------------------------------------------------------------------------

def test_expfam_proxy_is_zero_for_identical_information():
    I = np.diag([2.0, 3.0])
    assert expfam_deficiency_proxy(I, I) == pytest.approx(0.0, abs=1e-12)


def test_expfam_proxy_is_nonnegative():
    rng = np.random.default_rng(3)
    for _ in range(10):
        A = rng.normal(size=(2, 2))
        I1 = A @ A.T + np.eye(2)
        B = rng.normal(size=(2, 2))
        I2 = B @ B.T + np.eye(2)
        assert expfam_deficiency_proxy(I1, I2) >= 0.0


def test_expfam_proxy_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        expfam_deficiency_proxy(np.eye(2), np.eye(3))


def test_scalar_deficiency_is_a_valid_upper_bound_on_the_exact_lp():
    """The identity-kernel argument, checked against the LP on a discretisation.

    Taking the identity kernel gives TV(N(t, sd_s^2), N(t, sd_t^2)) at every t,
    which by translation invariance is constant in t. So the value returned is
    an upper bound on delta. It is not tight -- roughly a factor of two high --
    and this test pins both facts.
    """
    from scipy.stats import norm

    from defkit.solvers import deficiency_exact

    thetas = np.linspace(-1.0, 1.0, 5)
    grid = np.linspace(-12.0, 12.0, 81)
    edges = np.concatenate([[-np.inf], (grid[:-1] + grid[1:]) / 2, [np.inf]])

    def discretize(sd):
        M = np.column_stack([
            np.diff(norm.cdf(edges, loc=t, scale=sd)) for t in thetas
        ])
        return M / M.sum(axis=0, keepdims=True)

    saw_slack = False
    for sd_source, sd_target in [(2.0, 1.0), (3.0, 1.0), (1.5, 1.0)]:
        exact = deficiency_exact(discretize(sd_source), discretize(sd_target))
        bound = gaussian_scalar_location_deficiency(sd_source, sd_target)
        assert exact <= bound + 1e-6, f"bound {bound} below exact {exact}"
        if bound > 1.5 * exact:
            saw_slack = True
    assert saw_slack, "expected the bound to be materially loose"
