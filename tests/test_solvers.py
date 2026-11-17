"""Tests for the deficiency solvers.

Several of these pin down properties that the predecessor implementations
either got wrong or left untested, so they are written as regression tests
with the reason stated rather than as bare assertions.
"""

import numpy as np
import pytest

from defkit.solvers import (
    compute_deficiency,
    deficiency_exact,
    deficiency_upper_bound,
    evaluate_kernel,
    project_column_stochastic,
)


def rand_experiment(dim, n_thetas, rng):
    M = rng.random((dim, n_thetas)) + 0.05
    return M / M.sum(axis=0, keepdims=True)


# ---------------------------------------------------------------------------
# Correctness of the exact solver
# ---------------------------------------------------------------------------

def test_identical_experiments_have_zero_deficiency():
    rng = np.random.default_rng(0)
    P = rand_experiment(4, 3, rng)
    assert deficiency_exact(P, P) == pytest.approx(0.0, abs=1e-9)


def test_deficiency_to_a_constant_experiment_is_zero():
    """Any experiment can simulate one whose law does not depend on theta."""
    rng = np.random.default_rng(1)
    P = rand_experiment(4, 3, rng)
    q = rng.random(5)
    q /= q.sum()
    Q = np.tile(q[:, None], (1, 3))
    assert deficiency_exact(P, Q) == pytest.approx(0.0, abs=1e-8)


def test_two_point_deficiency_matches_closed_form():
    """For two perfectly separated targets from a useless source, delta = 1/2.

    A source whose law is the same under both parameter values carries no
    information, so the best any kernel can do against two disjoint targets is
    to split the difference: delta = TV/2 with TV = 1.
    """
    P = np.array([[0.5, 0.5], [0.5, 0.5]])
    Q = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert deficiency_exact(P, Q) == pytest.approx(0.5, abs=1e-7)


def test_deficiency_is_in_unit_interval():
    rng = np.random.default_rng(2)
    for _ in range(10):
        P = rand_experiment(3, 4, rng)
        Q = rand_experiment(4, 4, rng)
        d = deficiency_exact(P, Q)
        assert 0.0 <= d <= 1.0


def test_exact_solver_returns_a_feasible_optimal_kernel():
    rng = np.random.default_rng(3)
    P = rand_experiment(3, 4, rng)
    Q = rand_experiment(3, 4, rng)
    delta, K = deficiency_exact(P, Q, return_kernel=True)

    assert np.all(K >= -1e-9)
    np.testing.assert_allclose(K.sum(axis=0), 1.0, atol=1e-7)
    # The returned kernel must actually achieve the reported value.
    assert evaluate_kernel(K, P, Q) == pytest.approx(delta, abs=1e-6)


# ---------------------------------------------------------------------------
# The upper bound is an upper bound, and is loose
# ---------------------------------------------------------------------------

def test_upper_bound_never_falls_below_exact():
    """The contract that makes the fast path safe to use as a screen."""
    rng = np.random.default_rng(4)
    for _ in range(25):
        P = rand_experiment(4, 5, rng)
        Q = rand_experiment(4, 5, rng)
        exact = deficiency_exact(P, Q)
        bound = deficiency_upper_bound(P, Q)
        assert bound >= exact - 1e-7, f"bound {bound} below exact {exact}"


def test_upper_bound_is_loose():
    """Regression test for the reason the two functions have separate names.

    The predecessor API defaulted to this path under the name
    "compute_deficiency", so callers received these errors believing they had
    delta. On random dense experiments the gap reaches ~0.1 absolute.
    """
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(25):
        P = rand_experiment(4, 5, rng)
        Q = rand_experiment(4, 5, rng)
        worst = max(worst, deficiency_upper_bound(P, Q) - deficiency_exact(P, Q))
    assert worst > 0.02, (
        "expected the projected-gradient bound to be materially loose; if this "
        "now fails the solver improved and the docstrings should be revisited"
    )


# ---------------------------------------------------------------------------
# Orientation handling
# ---------------------------------------------------------------------------

def test_theta_rows_orientation_matches_transposed_input():
    rng = np.random.default_rng(5)
    P = rand_experiment(3, 4, rng)
    Q = rand_experiment(3, 4, rng)
    a = deficiency_exact(P, Q)
    b = deficiency_exact(P.T, Q.T, orientation="theta_rows")
    assert a == pytest.approx(b, abs=1e-7)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_non_normalized_columns_are_rejected_with_an_actionable_message():
    P = np.array([[0.5, 0.5], [0.2, 0.2]])
    Q = np.array([[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="columns must each sum to 1"):
        deficiency_exact(P, Q)


def test_mismatched_parameter_axis_is_rejected():
    rng = np.random.default_rng(6)
    P = rand_experiment(3, 4, rng)
    Q = rand_experiment(3, 5, rng)
    with pytest.raises(ValueError, match="share the parameter axis"):
        deficiency_exact(P, Q)


def test_negative_entries_are_rejected():
    P = np.array([[1.2, 0.5], [-0.2, 0.5]])
    Q = np.array([[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="negative"):
        deficiency_exact(P, Q)


# ---------------------------------------------------------------------------
# Dispatch shim
# ---------------------------------------------------------------------------

def test_auto_solver_is_refused_with_an_explanation():
    """The old default silently returned a loose bound; it must not come back."""
    rng = np.random.default_rng(8)
    P = rand_experiment(3, 3, rng)
    Q = rand_experiment(3, 3, rng)
    with pytest.raises(ValueError, match='solver="auto" is no longer accepted'):
        compute_deficiency(P, Q, solver="auto")


@pytest.mark.parametrize("alias", ["exact", "lp", "scipy", "cvxpy"])
def test_exact_aliases_dispatch_to_the_lp(alias):
    rng = np.random.default_rng(9)
    P = rand_experiment(3, 3, rng)
    Q = rand_experiment(3, 3, rng)
    assert compute_deficiency(P, Q, solver=alias) == pytest.approx(
        deficiency_exact(P, Q), abs=1e-9
    )


@pytest.mark.parametrize("alias", ["upper_bound", "sinkhorn", "gradient"])
def test_bound_aliases_dispatch_to_the_gradient_solver(alias):
    rng = np.random.default_rng(10)
    P = rand_experiment(3, 3, rng)
    Q = rand_experiment(3, 3, rng)
    assert compute_deficiency(P, Q, solver=alias) == pytest.approx(
        deficiency_upper_bound(P, Q), abs=1e-9
    )


def test_unknown_solver_is_rejected():
    rng = np.random.default_rng(11)
    P = rand_experiment(3, 3, rng)
    Q = rand_experiment(3, 3, rng)
    with pytest.raises(ValueError, match="unknown solver"):
        compute_deficiency(P, Q, solver="nonesuch")


# ---------------------------------------------------------------------------
# Projection helper
# ---------------------------------------------------------------------------

def test_projection_produces_column_stochastic_matrices():
    rng = np.random.default_rng(12)
    K = rng.normal(size=(4, 3))
    Kp = project_column_stochastic(K)
    assert np.all(Kp >= 0)
    np.testing.assert_allclose(Kp.sum(axis=0), 1.0, atol=1e-12)


def test_projection_handles_an_all_negative_column():
    """The predecessor divided by a guarded zero, leaving the column at zero.

    A zero column is not a probability distribution, so downstream TV readings
    silently gained mass deficit. Uniform is the right fallback.
    """
    K = np.array([[-1.0, 0.5], [-2.0, 0.5]])
    Kp = project_column_stochastic(K)
    np.testing.assert_allclose(Kp.sum(axis=0), 1.0, atol=1e-12)
    np.testing.assert_allclose(Kp[:, 0], [0.5, 0.5], atol=1e-12)


def test_upper_bound_considers_the_warm_start_itself():
    """The warm start must be scored, not just the post-gradient iterates.

    The predecessor loops initialised ``best = inf`` and only evaluated after a
    gradient step, discarding a warm start that beat everything after it. This
    pins the fix: the returned bound is never worse than the warm start's own
    value.
    """
    rng = np.random.default_rng(20)
    for _ in range(30):
        P = rand_experiment(rng.integers(2, 6), rng.integers(2, 5), rng)
        Q = rand_experiment(rng.integers(2, 6), P.shape[1], rng)

        dim_x, dim_y = P.shape[0], Q.shape[0]
        ppt = P @ P.T
        if np.linalg.cond(ppt) < 1e10:
            K0 = Q @ P.T @ np.linalg.inv(ppt + 1e-6 * np.eye(dim_x))
        else:
            K0 = Q @ np.linalg.pinv(P)
        warm_value = evaluate_kernel(project_column_stochastic(K0), P, Q)

        assert deficiency_upper_bound(P, Q) <= warm_value + 1e-12


def test_exact_solver_falls_back_when_simplex_reports_unknown():
    """Large instances break HiGHS's simplex paths; the IPM path handles them.

    A 121-bin discretisation against 5 parameter values makes HiGHS return an
    unrecognised status (model Unknown, primal Feasible) from both "highs" and
    "highs-ds", which previously surfaced as a spurious RuntimeError. That size
    is well inside what a real DeficiencyMonitor call produces.

    Here the source is strictly less noisy than the target, so it dominates and
    the true deficiency is 0.
    """
    from scipy.stats import norm

    thetas = np.linspace(-1.0, 1.0, 5)
    grid = np.linspace(-12.0, 12.0, 121)
    edges = np.concatenate([[-np.inf], (grid[:-1] + grid[1:]) / 2, [np.inf]])

    def discretize(sd):
        M = np.column_stack([
            np.diff(norm.cdf(edges, loc=t, scale=sd)) for t in thetas
        ])
        return M / M.sum(axis=0, keepdims=True)

    delta = deficiency_exact(discretize(1.0), discretize(2.0))
    assert delta == pytest.approx(0.0, abs=1e-6)
