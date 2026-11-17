# defkit

Primitives for comparing statistical experiments: the Le Cam deficiency between
two finite experiments, the probability distances it is built on, and the
Gaussian closed forms.

Used by [ReliabilityLayer](https://github.com/denizakdemir/ReliabilityLayer)
(conformal prediction under distribution shift). Parity tests pin it against the
independent implementations in [AEGIS](https://github.com/denizakdemir/AEGIS)
(Python) and [causaldef](https://github.com/denizakdemir/causaldef) (R).

## Why one package

The deficiency linear program and the Gaussian total-variation closed form are
short enough that they tend to get re-implemented wherever they are needed.
That is not harmless: the closed-form crossing-point algorithm for Gaussian TV
loses precision and returns near 0 instead of near 1 when both standard
deviations are small relative to the mean separation, and the same bug was found
and fixed in one implementation while sitting unfixed in another written from
the same formula. One implementation, one test suite, one place a fix has to land.

## Naming rule

**A function that returns the deficiency says so. A function that returns a
bound or a heuristic says which, in its name.**

```python
from defkit import deficiency_exact, deficiency_upper_bound

deficiency_exact(P, Q)         # delta, by linear programming (HiGHS via scipy)
deficiency_upper_bound(P, Q)   # >= delta, fast, and NOT delta
```

`compute_deficiency(P, Q, solver=...)` exists for older call sites, but `solver`
is required and `solver="auto"` raises: on dense random problems the upper
bound sits up to 0.1 absolute above the true deficiency, so nobody should get it
by default.

The same rule governs `defkit.gaussian`:

```python
gaussian_location_dominates(S1, S2)          # exact: delta == 0 iff S2 - S1 is PSD
two_point_confounding_lower_bound(...)       # rigorous lower bound
gaussian_location_deficiency_proxy(S1, S2)   # heuristic, no proof relates it to delta
```

## Install

```bash
pip install git+https://github.com/denizakdemir/defkit.git
```

Depends only on `numpy` and `scipy`. `cvxpy` is optional and not required by
any code path.

## Conventions

An experiment is a matrix whose **columns** are probability distributions
indexed by a parameter:

```
P.shape == (n_outcomes, n_thetas)     # each column sums to 1
```

Callers that store parameters in rows pass `orientation="theta_rows"`.

Deficiency is

```
delta(P, Q) = min_K max_theta TV(K P_theta, Q_theta)
```

over column-stochastic kernels `K`, with `TV(p, q) = 0.5 * ||p - q||_1`
(the sup convention; values in [0, 1]).

The exact solver builds the LP in sparse form (`scipy.sparse`); histogram-sized
problems are more than 99% sparse, and dense construction allocates gigabytes.
Problems with more than 5000 variables go straight to the HiGHS interior-point
method, whose simplex paths return an unknown status on such instances.

## Tests

```bash
python -m pytest
```

Tests worth knowing about:

- `test_upper_bound_never_falls_below_exact` — the contract that makes the
  fast path safe as a conservative screen.
- `test_upper_bound_is_loose` — pins the reason the two solvers have separate
  names, and fails if someone quietly makes them interchangeable.
- `test_tv_gaussian_near_degenerate_regime` — the bug described above.
- `test_tv_gaussian_accurate_across_many_orders_of_magnitude` — agreement with
  independent quadrature to better than 1e-6 for standard deviations spanning
  1e-6 to 1e3.
- `test_parity.py` — agreement with AEGIS (Python) and causaldef (R, via
  `Rscript`). These skip unless the other side is available; set
  `DEFKIT_PARITY_ROOT` to a directory containing an `AEGIS` checkout to enable
  the Python half.
- `test_exact_solver_falls_back_when_simplex_reports_unknown` — the HiGHS
  simplex-to-IPM fallback.

## License

MIT
