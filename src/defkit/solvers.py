"""
defkit.solvers
==============

Deficiency between two finite statistical experiments.

An experiment is a matrix whose columns are probability distributions indexed
by a parameter theta. For a source ``P`` of shape ``(n_outcomes_P, n_thetas)``
and a target ``Q`` of shape ``(n_outcomes_Q, n_thetas)``, the deficiency is

    delta(P, Q) = min_K  max_theta  TV(K P_theta, Q_theta)

over column-stochastic kernels ``K`` of shape ``(n_outcomes_Q, n_outcomes_P)``,
with ``TV(p, q) = 0.5 * ||p - q||_1``.

Two solvers are exposed, and they answer different questions:

``deficiency_exact``
    Solves the linear program. Returns delta itself, to solver tolerance.

``deficiency_upper_bound``
    Projected-gradient descent on the least-squares surrogate
    ``||K P - Q||_F^2``, returning the best feasible kernel it finds. Because
    every feasible kernel witnesses an upper bound, the result is always
    ``>= delta``, never below it. It is fast but it is *not* delta, and on
    random dense experiments it has been measured up to ~0.1 absolute above
    the exact value (see ``tests/test_solvers.py::test_upper_bound_is_loose``).

The naming is deliberate. An earlier generation of this code exposed a single
``compute_deficiency(..., solver="auto")`` whose default silently took the
projected-gradient path, so callers asking for "the deficiency" received a
loose upper bound with no indication in the return value. ``compute_deficiency``
is kept below for source compatibility, but it requires an explicit choice.

Orientation
-----------
The canonical layout is parameters-in-columns, ``(n_outcomes, n_thetas)``.
Callers that store experiments the other way round (parameters in rows) can
pass ``orientation="theta_rows"`` instead of transposing at every call site.
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csc_matrix

Orientation = Literal["theta_columns", "theta_rows"]

__all__ = [
    "deficiency_exact",
    "deficiency_upper_bound",
    "compute_deficiency",
    "evaluate_kernel",
    "project_column_stochastic",
]


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------

def _as_experiment(
    M: np.ndarray,
    *,
    name: str,
    orientation: Orientation,
    tol: float = 1e-8,
) -> np.ndarray:
    """Validate an experiment matrix and return it parameters-in-columns."""
    M = np.asarray(M, dtype=float)
    if M.ndim != 2:
        raise ValueError(f"{name} must be 2-dimensional, got shape {M.shape}.")
    if orientation == "theta_rows":
        M = M.T
    elif orientation != "theta_columns":
        raise ValueError(
            'orientation must be "theta_columns" or "theta_rows", '
            f"got {orientation!r}."
        )
    if M.size == 0:
        raise ValueError(f"{name} is empty.")
    if not np.all(np.isfinite(M)):
        raise ValueError(f"{name} contains non-finite entries.")
    if np.any(M < -tol):
        raise ValueError(f"{name} contains negative entries.")
    col_sums = M.sum(axis=0)
    if not np.allclose(col_sums, 1.0, atol=1e-6):
        worst = float(np.max(np.abs(col_sums - 1.0)))
        raise ValueError(
            f"{name}'s columns must each sum to 1 (worst deviation {worst:.2e}). "
            "Each column is a probability distribution over outcomes; if your "
            'matrix stores parameters in rows, pass orientation="theta_rows".'
        )
    return np.clip(M, 0.0, None)


def _check_pair(P: np.ndarray, Q: np.ndarray) -> None:
    if P.shape[1] != Q.shape[1]:
        raise ValueError(
            f"P and Q must share the parameter axis: P has {P.shape[1]} "
            f"parameter values, Q has {Q.shape[1]}."
        )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def evaluate_kernel(K: np.ndarray, P: np.ndarray, Q: np.ndarray) -> float:
    """``max_theta TV(K P_theta, Q_theta)`` for a given kernel.

    Any column-stochastic ``K`` gives a valid upper bound on the deficiency.
    """
    K = np.asarray(K, dtype=float)
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    residual = K @ P - Q
    return float(np.max(0.5 * np.sum(np.abs(residual), axis=0)))


def project_column_stochastic(K: np.ndarray) -> np.ndarray:
    """Project onto nonnegative matrices whose columns sum to 1.

    Columns that clip to all-zero are replaced by the uniform distribution
    rather than being normalised by zero.
    """
    K = np.clip(np.asarray(K, dtype=float), 0.0, None)
    col_sums = K.sum(axis=0, keepdims=True)
    degenerate = col_sums <= 1e-12
    if np.any(degenerate):
        K = np.where(degenerate, 1.0 / K.shape[0], K)
        col_sums = K.sum(axis=0, keepdims=True)
    return K / col_sums


# ---------------------------------------------------------------------------
# Exact solver (linear program)
# ---------------------------------------------------------------------------

def _build_lp(
    P: np.ndarray,
    Q: np.ndarray,
) -> tuple:
    """Build the deficiency LP in sparse form.

    Returns (c, A_ub, b_ub, A_eq, b_eq, bounds, n_k, n_vars) where A_ub and
    A_eq are scipy.sparse.csc_matrix.  The LP is 99.6 %+ sparse for the
    histogram sizes produced by DeficiencyMonitor, so dense construction
    allocates gigabytes and stalls; sparse construction uses ~76 MB for the
    same (240 × 20) problem.
    """
    dim_x, n_thetas = P.shape
    dim_y = Q.shape[0]

    n_k = dim_y * dim_x
    n_z = dim_y * n_thetas
    n_vars = n_k + n_z + 1
    idx_eps = n_vars - 1

    c = np.zeros(n_vars)
    c[idx_eps] = 1.0

    # ------------------------------------------------------------------
    # A_ub  (sparse COO → CSC)
    # ------------------------------------------------------------------
    # Row blocks:
    #   [0,   2*n_thetas*dim_y)  — abs-value linearisation (2 per (t,y) pair)
    #   [2*n_thetas*dim_y, ...)  — TV ≤ 2ε  (one per t)
    #
    # Nonzero counts per block:
    #   abs rows: each touches dim_x K-vars + 1 z-var  → (dim_x+1) per row
    #   TV  rows: each touches dim_y z-vars + 1 eps-var → (dim_y+1) per row

    n_abs_rows = 2 * n_thetas * dim_y
    n_tv_rows  = n_thetas
    n_ub_rows  = n_abs_rows + n_tv_rows

    nnz_abs = n_abs_rows * (dim_x + 1)   # K-block + z-block
    nnz_tv  = n_tv_rows  * (dim_y + 1)   # z-block + eps
    ub_rows = np.empty(nnz_abs + nnz_tv, dtype=np.int32)
    ub_cols = np.empty(nnz_abs + nnz_tv, dtype=np.int32)
    ub_vals = np.empty(nnz_abs + nnz_tv, dtype=np.float64)

    # --- abs-value rows ---
    # For each (t, y): row_pos = 2*(t*dim_y + y), row_neg = row_pos+1
    # K-block columns for row y: y*dim_x + x  for x in 0..dim_x-1
    # z-block column for (y,t):  n_k + y*n_thetas + t

    t_idx = np.repeat(np.arange(n_thetas), dim_y)   # (n_thetas*dim_y,)
    y_idx = np.tile(np.arange(dim_y), n_thetas)     # (n_thetas*dim_y,)
    pair  = t_idx * dim_y + y_idx                   # linear index of (t,y)

    row_pos = 2 * pair
    row_neg = 2 * pair + 1

    # K-block: dim_x entries per (t,y) pair
    x_idx = np.arange(dim_x)                        # (dim_x,)
    # broadcast: (n_thetas*dim_y, dim_x)
    k_rows_pos = np.repeat(row_pos, dim_x)
    k_rows_neg = np.repeat(row_neg, dim_x)
    k_cols     = (np.repeat(y_idx, dim_x) * dim_x
                  + np.tile(x_idx, n_thetas * dim_y))
    # P[x, t] for each (t, y, x) — value is the same for pos and neg
    k_vals_pos = P[np.tile(x_idx, n_thetas * dim_y),
                   np.repeat(t_idx, dim_x)]
    k_vals_neg = -k_vals_pos

    # z-block: 1 entry per (t,y) pair
    z_cols = n_k + y_idx * n_thetas + t_idx
    z_vals = np.full(len(pair), -1.0)

    ptr = 0
    n_k_entries = len(k_rows_pos)
    ub_rows[ptr:ptr + n_k_entries] = k_rows_pos
    ub_cols[ptr:ptr + n_k_entries] = k_cols
    ub_vals[ptr:ptr + n_k_entries] = k_vals_pos
    ptr += n_k_entries

    ub_rows[ptr:ptr + n_k_entries] = k_rows_neg
    ub_cols[ptr:ptr + n_k_entries] = k_cols
    ub_vals[ptr:ptr + n_k_entries] = k_vals_neg
    ptr += n_k_entries

    n_z_entries = len(pair)
    ub_rows[ptr:ptr + n_z_entries] = row_pos
    ub_cols[ptr:ptr + n_z_entries] = z_cols
    ub_vals[ptr:ptr + n_z_entries] = z_vals
    ptr += n_z_entries

    ub_rows[ptr:ptr + n_z_entries] = row_neg
    ub_cols[ptr:ptr + n_z_entries] = z_cols
    ub_vals[ptr:ptr + n_z_entries] = z_vals
    ptr += n_z_entries

    # --- TV rows ---
    # Row n_abs_rows + t: sum_y z[y,t] - 2*eps <= 0
    tv_t   = np.repeat(np.arange(n_thetas), dim_y)
    tv_y   = np.tile(np.arange(dim_y), n_thetas)
    tv_row = n_abs_rows + tv_t
    tv_col = n_k + tv_y * n_thetas + tv_t
    tv_val = np.ones(n_thetas * dim_y)

    eps_row = n_abs_rows + np.arange(n_thetas)
    eps_col = np.full(n_thetas, idx_eps)
    eps_val = np.full(n_thetas, -2.0)

    n_tv_z   = len(tv_row)
    n_tv_eps = len(eps_row)
    ub_rows[ptr:ptr + n_tv_z] = tv_row
    ub_cols[ptr:ptr + n_tv_z] = tv_col
    ub_vals[ptr:ptr + n_tv_z] = tv_val
    ptr += n_tv_z

    ub_rows[ptr:ptr + n_tv_eps] = eps_row
    ub_cols[ptr:ptr + n_tv_eps] = eps_col
    ub_vals[ptr:ptr + n_tv_eps] = eps_val
    ptr += n_tv_eps

    A_ub = csc_matrix(
        (ub_vals[:ptr], (ub_rows[:ptr], ub_cols[:ptr])),
        shape=(n_ub_rows, n_vars),
    )

    b_ub = np.empty(n_ub_rows)
    b_ub[:n_abs_rows:2]  = Q[y_idx, t_idx]   # pos rows: Q[y,t]
    b_ub[1:n_abs_rows:2] = -Q[y_idx, t_idx]  # neg rows: -Q[y,t]
    b_ub[n_abs_rows:]    = 0.0

    # ------------------------------------------------------------------
    # A_eq  (sparse COO → CSC)
    # ------------------------------------------------------------------
    # dim_x rows: sum_y K[y,x] = 1  for each x
    eq_x   = np.repeat(np.arange(dim_x), dim_y)
    eq_y   = np.tile(np.arange(dim_y), dim_x)
    eq_col = eq_y * dim_x + eq_x
    eq_row = eq_x
    A_eq = csc_matrix(
        (np.ones(dim_x * dim_y), (eq_row, eq_col)),
        shape=(dim_x, n_vars),
    )
    b_eq = np.ones(dim_x)

    bounds = [(0.0, 1.0)] * n_k + [(0.0, None)] * n_z + [(0.0, 1.0)]

    return c, A_ub, b_ub, A_eq, b_eq, bounds, n_k, n_vars


def deficiency_exact(
    P: np.ndarray,
    Q: np.ndarray,
    *,
    orientation: Orientation = "theta_columns",
    return_kernel: bool = False,
    method: str = "highs",
) -> float | Tuple[float, np.ndarray]:
    """Exact deficiency ``delta(P, Q)`` by linear programming.

    The program minimises ``epsilon`` over variables ``(K, z, epsilon)``:

        minimise    epsilon
        subject to   (K P)[y, t] - z[y, t] <= Q[y, t]
                    -(K P)[y, t] - z[y, t] <= -Q[y, t]
                     sum_y z[y, t] <= 2 * epsilon      for every t
                     sum_y K[y, x] = 1                 for every x
                     K >= 0,  z >= 0

    The first two constraint families force ``z[y, t] >= |(K P - Q)[y, t]|``,
    so the third reads ``0.5 * ||K P_t - Q_t||_1 <= epsilon``, which is exactly
    ``TV(K P_t, Q_t) <= epsilon``.

    Constraint matrices are built as sparse (CSC) regardless of problem size.
    For the histogram dimensions produced by DeficiencyMonitor the LP is
    99.6 %+ sparse; dense construction allocates gigabytes and stalls.

    Parameters
    ----------
    P, Q:
        Experiment matrices. See module docstring for the layout.
    orientation:
        ``"theta_columns"`` (default) or ``"theta_rows"``.
    return_kernel:
        If True, also return the optimal kernel, shape ``(n_out_Q, n_out_P)``.
    method:
        Passed through to :func:`scipy.optimize.linprog`.

    Returns
    -------
    float, or ``(float, ndarray)`` when ``return_kernel`` is set.

    Raises
    ------
    RuntimeError
        If the LP does not solve. Earlier versions of this code returned
        ``None`` on failure, which silently propagated into arithmetic; a
        failed solve is now an error.
    """
    P = _as_experiment(P, name="P", orientation=orientation)
    Q = _as_experiment(Q, name="Q", orientation=orientation)
    _check_pair(P, Q)

    c, A_ub, b_ub, A_eq, b_eq, bounds, n_k, n_vars = _build_lp(P, Q)
    dim_x = P.shape[0]
    dim_y = Q.shape[0]
    idx_eps = n_vars - 1

    # HiGHS's simplex paths ("highs" and "highs-ds") return an unrecognised
    # status on larger instances of this program -- the model status comes back
    # Unknown even though the primal is Feasible -- while the interior-point
    # path solves the same instance cleanly. A 121-bin histogram against 5
    # parameter values is already enough to trigger it, which is well inside
    # the range a real DeficiencyMonitor call produces. Skip the doomed
    # simplex attempt and go straight to IPM when the problem is large.
    _n_vars = len(c)
    _effective_method = "highs-ipm" if (_n_vars > 5000 and method == "highs") else method

    res = linprog(
        c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
        bounds=bounds, method=_effective_method,
    )

    if not res.success and _effective_method in ("highs", "highs-ds"):
        res = linprog(
            c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
            bounds=bounds, method="highs-ipm",
        )

    if not res.success or res.x is None:
        raise RuntimeError(f"deficiency LP failed to solve: {res.message}")

    delta = float(np.clip(res.x[idx_eps], 0.0, 1.0))
    if not return_kernel:
        return delta

    K = project_column_stochastic(res.x[:n_k].reshape(dim_y, dim_x))
    return delta, K


# ---------------------------------------------------------------------------
# Fast upper bound (projected gradient)
# ---------------------------------------------------------------------------

def deficiency_upper_bound(
    P: np.ndarray,
    Q: np.ndarray,
    *,
    orientation: Orientation = "theta_columns",
    step_scale: float = 0.1,
    max_iter: int = 100,
    tol: float = 1e-6,
    return_kernel: bool = False,
) -> float | Tuple[float, np.ndarray]:
    """A fast upper bound on ``delta(P, Q)``. **Not** the deficiency.

    Runs projected gradient descent on ``||K P - Q||_F^2`` from a least-squares
    warm start, tracking the best (lowest) max-theta TV seen at any feasible
    iterate. Since every feasible kernel witnesses an upper bound, the returned
    value satisfies ``result >= delta(P, Q)``.

    Use it when you need a conservative screen cheaply and a value above the
    truth is safe. Use :func:`deficiency_exact` when you need delta.

    ``step_scale`` was called ``sinkhorn_reg`` in the predecessor code, which
    was misleading: no entropic regularisation or Sinkhorn scaling is involved.

    One deliberate difference from the predecessors: the least-squares warm
    start is itself evaluated as a candidate before the first gradient step.
    The old loops began with ``best = inf`` and only scored post-step iterates,
    so a warm start better than everything that followed was discarded. Across
    120 random problems this changes nothing in 113 and tightens the bound in
    7, never loosens it, and the result remains a valid upper bound throughout.
    """
    P = _as_experiment(P, name="P", orientation=orientation)
    Q = _as_experiment(Q, name="Q", orientation=orientation)
    _check_pair(P, Q)

    dim_x = P.shape[0]
    dim_y = Q.shape[0]

    ppt = P @ P.T
    try:
        if np.linalg.cond(ppt) < 1e10:
            K = Q @ P.T @ np.linalg.inv(ppt + 1e-6 * np.eye(dim_x))
        else:
            K = Q @ np.linalg.pinv(P)
    except np.linalg.LinAlgError:
        K = np.ones((dim_y, dim_x)) / dim_y

    K = project_column_stochastic(K)

    step = step_scale / max(1.0, float(np.linalg.norm(ppt)))
    best = evaluate_kernel(K, P, Q)
    best_K = K.copy()

    for iteration in range(max_iter):
        K_old = K
        K = project_column_stochastic(K - step * ((K @ P - Q) @ P.T))

        current = evaluate_kernel(K, P, Q)
        if current < best:
            best = current
            best_K = K.copy()

        if np.max(np.abs(K - K_old)) < tol:
            break
        if iteration > 0 and iteration % 20 == 0:
            step *= 0.9

    best = float(np.clip(best, 0.0, 1.0))
    return (best, best_K) if return_kernel else best


# ---------------------------------------------------------------------------
# Source-compatible entry point
# ---------------------------------------------------------------------------

def compute_deficiency(
    P: np.ndarray,
    Q: np.ndarray,
    solver: str,
    *,
    orientation: Orientation = "theta_columns",
    **kwargs,
) -> float:
    """Dispatch to :func:`deficiency_exact` or :func:`deficiency_upper_bound`.

    ``solver`` is required and has no default, on purpose. Pass ``"exact"``
    (equivalently ``"lp"``, ``"scipy"``, ``"cvxpy"``) for the deficiency, or
    ``"upper_bound"`` (equivalently ``"sinkhorn"``, the old name) for the fast
    bound. The legacy aliases exist so migrated call sites keep working; new
    code should call the two functions directly.
    """
    exact_names = {"exact", "lp", "scipy", "cvxpy"}
    bound_names = {"upper_bound", "upper", "sinkhorn", "gradient"}

    if solver in exact_names:
        return deficiency_exact(P, Q, orientation=orientation, **kwargs)
    if solver in bound_names:
        return deficiency_upper_bound(P, Q, orientation=orientation, **kwargs)
    if solver == "auto":
        raise ValueError(
            'solver="auto" is no longer accepted. It used to resolve to the '
            "projected-gradient path, so callers asking for the deficiency "
            "silently received a loose upper bound. Choose explicitly: "
            '"exact" for delta, "upper_bound" for the fast conservative screen.'
        )
    raise ValueError(
        f"unknown solver {solver!r}; expected one of "
        f"{sorted(exact_names | bound_names)}."
    )
