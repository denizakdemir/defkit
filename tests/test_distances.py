"""Tests for the shared probability distances.

The Gaussian TV tests are the important ones: they encode the bug that was
fixed in ``causaldef`` in August 2026 and then found again, unfixed, in an
independently written Python port of the same closed form.
"""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from defkit.distances import (
    mmd_rbf,
    normalize_weights,
    rbf_kernel,
    tv_discrete,
    tv_gaussian,
    tv_histogram,
    wasserstein1_normalized,
)


def tv_gaussian_reference(mu1, sd1, mu2, sd2):
    """Independent ground truth, integrated piecewise around both densities.

    Deliberately not the implementation under test: this splits on a fine grid
    of panels around each mean and sums ``quad`` over each. A single wide
    ``quad`` call is what produced the original misdiagnosis, because adaptive
    quadrature can step over a narrow spike in a wide, mostly flat range.
    """
    def f(x):
        return abs(norm.pdf(x, mu1, sd1) - norm.pdf(x, mu2, sd2))

    edges = sorted(
        {mu1 + k * sd1 for k in range(-40, 41, 2)}
        | {mu2 + k * sd2 for k in range(-40, 41, 2)}
    )
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi > lo:
            total += quad(f, lo, hi, limit=200)[0]
    return min(1.0, max(0.0, 0.5 * total))


# ---------------------------------------------------------------------------
# Discrete TV
# ---------------------------------------------------------------------------

def test_tv_discrete_known_values():
    assert tv_discrete([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0)
    assert tv_discrete([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.0)
    assert tv_discrete([0.7, 0.3], [0.5, 0.5]) == pytest.approx(0.2)


def test_tv_discrete_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        tv_discrete([0.5, 0.5], [1 / 3, 1 / 3, 1 / 3])


# ---------------------------------------------------------------------------
# Gaussian TV
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "mu1,sd1,mu2,sd2",
    [
        (0.0, 1.0, 0.0, 1.0),
        (0.0, 1.0, 1.0, 1.0),
        (0.0, 1.0, 0.0, 2.0),
        (-1.5, 0.7, 2.0, 3.1),
        (0.0, 1.0, 5.0, 1.0),
    ],
)
def test_tv_gaussian_matches_reference_on_ordinary_parameters(mu1, sd1, mu2, sd2):
    got = tv_gaussian(mu1, sd1, mu2, sd2)
    want = tv_gaussian_reference(mu1, sd1, mu2, sd2)
    assert got == pytest.approx(want, abs=1e-7)


def test_tv_gaussian_near_degenerate_regime():
    """The regression test for the actual bug.

    Both sds tiny relative to the mean separation. The closed-form
    crossing-point algorithm returned ~0 here, when the true value is ~1: two
    nearly disjoint point masses. That is the maximum possible error.
    """
    cases = [
        (0.0, 1e-6, 1.0, 1e-6),
        (0.0, 1e-8, 0.5, 1e-8),
        (0.0, 1e-5, 0.001, 1e-5),
    ]
    for mu1, sd1, mu2, sd2 in cases:
        got = tv_gaussian(mu1, sd1, mu2, sd2)
        assert got == pytest.approx(1.0, abs=1e-6), (
            f"TV for N({mu1},{sd1}) vs N({mu2},{sd2}) should be ~1, got {got}"
        )


def test_tv_gaussian_accurate_across_many_orders_of_magnitude():
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(60):
        sd1 = 10.0 ** rng.uniform(-6, 3)
        sd2 = 10.0 ** rng.uniform(-6, 3)
        mu1 = rng.normal(0, max(sd1, sd2))
        mu2 = rng.normal(0, max(sd1, sd2))
        got = tv_gaussian(mu1, sd1, mu2, sd2)
        want = tv_gaussian_reference(mu1, sd1, mu2, sd2)
        worst = max(worst, abs(got - want))
    assert worst < 1e-6, f"worst discrepancy {worst:.2e} across 6+ orders of magnitude"


def test_tv_gaussian_identical_distributions_is_exactly_zero():
    assert tv_gaussian(2.0, 3.0, 2.0, 3.0) == 0.0


def test_tv_gaussian_rejects_nonpositive_sd():
    assert np.isnan(tv_gaussian(0.0, 0.0, 1.0, 1.0))
    assert np.isnan(tv_gaussian(0.0, -1.0, 1.0, 1.0))


def test_tv_gaussian_rejects_even_n_points():
    with pytest.raises(ValueError, match="odd integer"):
        tv_gaussian(0.0, 1.0, 1.0, 1.0, n_points=2000)


def test_tv_gaussian_is_symmetric():
    a = tv_gaussian(0.0, 1.0, 2.0, 3.0)
    b = tv_gaussian(2.0, 3.0, 0.0, 1.0)
    assert a == pytest.approx(b, abs=1e-12)


# ---------------------------------------------------------------------------
# Histogram TV
# ---------------------------------------------------------------------------

def test_tv_histogram_disjoint_samples():
    a = np.full(100, 0.1)
    b = np.full(100, 0.9)
    assert tv_histogram(a, b, bins=10) == pytest.approx(1.0)


def test_tv_histogram_identical_samples():
    rng = np.random.default_rng(0)
    x = rng.random(500)
    assert tv_histogram(x, x, bins=20) == pytest.approx(0.0)


def test_tv_histogram_returns_nan_on_empty_input():
    assert np.isnan(tv_histogram(np.array([]), np.array([0.5])))


def test_tv_histogram_returns_nan_when_range_captures_nothing():
    a = np.full(10, 5.0)
    b = np.full(10, 6.0)
    assert np.isnan(tv_histogram(a, b, bins=10, value_range=(0.0, 1.0)))


# ---------------------------------------------------------------------------
# Wasserstein
# ---------------------------------------------------------------------------

def test_wasserstein_identical_samples_is_zero():
    rng = np.random.default_rng(1)
    x = rng.normal(size=200)
    assert wasserstein1_normalized(x, x) == pytest.approx(0.0, abs=1e-12)


def test_wasserstein_shifted_samples_is_positive():
    rng = np.random.default_rng(2)
    x = rng.normal(size=300)
    y = x + 3.0
    assert wasserstein1_normalized(x, y) > 0.1


def test_wasserstein_is_in_unit_interval():
    rng = np.random.default_rng(3)
    for _ in range(10):
        x = rng.normal(size=100)
        y = rng.normal(loc=rng.uniform(-3, 3), size=100)
        assert 0.0 <= wasserstein1_normalized(x, y) <= 1.0


# ---------------------------------------------------------------------------
# MMD
# ---------------------------------------------------------------------------

def test_mmd_identical_samples_is_near_zero():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(200, 2))
    assert mmd_rbf(x, x, gamma=1.0) == pytest.approx(0.0, abs=1e-9)


def test_mmd_separated_samples_is_larger_than_overlapping_ones():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(200, 2))
    near = rng.normal(loc=0.1, size=(200, 2))
    far = rng.normal(loc=6.0, size=(200, 2))
    assert mmd_rbf(x, far, gamma=1.0) > mmd_rbf(x, near, gamma=1.0)


def test_mmd_is_nonnegative():
    rng = np.random.default_rng(6)
    for _ in range(10):
        x = rng.normal(size=(60, 3))
        y = rng.normal(loc=rng.uniform(-2, 2), size=(60, 3))
        assert mmd_rbf(x, y) >= 0.0


def test_mmd_accepts_1d_input():
    rng = np.random.default_rng(7)
    x = rng.normal(size=100)
    y = rng.normal(loc=2.0, size=100)
    assert mmd_rbf(x, y) > 0.0


def test_mmd_normalized_is_in_unit_interval():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(100, 2))
    y = rng.normal(loc=10.0, size=(100, 2))
    assert 0.0 <= mmd_rbf(x, y, gamma=1.0, normalize=True) <= 1.0


def test_unbiased_mmd_rejects_weights():
    rng = np.random.default_rng(9)
    x = rng.normal(size=(20, 2))
    y = rng.normal(size=(20, 2))
    with pytest.raises(ValueError, match="unweighted"):
        mmd_rbf(x, y, unbiased=True, weights_x=np.ones(20) / 20)


def test_rbf_kernel_diagonal_is_one():
    rng = np.random.default_rng(10)
    x = rng.normal(size=(10, 3))
    K = rbf_kernel(x, x, gamma=0.5)
    np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-12)


def test_rbf_kernel_rejects_feature_mismatch():
    with pytest.raises(ValueError, match="feature axis"):
        rbf_kernel(np.zeros((4, 2)), np.zeros((4, 3)), gamma=1.0)


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------

def test_normalize_weights_defaults_to_uniform():
    np.testing.assert_allclose(normalize_weights(None, 4), 0.25)


def test_normalize_weights_falls_back_to_uniform_on_zero_total():
    np.testing.assert_allclose(normalize_weights(np.zeros(4), 4), 0.25)


def test_normalize_weights_rejects_wrong_length():
    with pytest.raises(ValueError, match="expected 4"):
        normalize_weights(np.ones(3), 4)
