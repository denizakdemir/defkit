"""
defkit — primitives for comparing statistical experiments.

One implementation of each primitive: the Le Cam deficiency between two finite
experiments, the probability distances it is built on, and the Gaussian closed
forms. Short routines like these tend to be re-implemented wherever they are
needed, and a numerical bug fixed in one copy of the Gaussian total-variation
closed form was later found unfixed in an independently written copy; keeping a
single implementation with a single test suite is the point of this package.

Naming rule
-----------
A function that returns the deficiency says so. A function that returns a
bound or a heuristic says which, in its name. See :mod:`defkit.solvers` and
:mod:`defkit.gaussian`.
"""

from ._version import __version__
from .distances import (
    mmd_rbf,
    normalize_weights,
    rbf_kernel,
    tv_discrete,
    tv_gaussian,
    tv_histogram,
    wasserstein1_normalized,
)
from .gaussian import (
    expfam_deficiency_proxy,
    gaussian_location_deficiency_proxy,
    gaussian_location_dominates,
    gaussian_scalar_location_deficiency,
    two_point_confounding_lower_bound,
)
from .solvers import (
    compute_deficiency,
    deficiency_exact,
    deficiency_upper_bound,
    evaluate_kernel,
    project_column_stochastic,
)

__all__ = [
    "__version__",
    # solvers
    "deficiency_exact",
    "deficiency_upper_bound",
    "compute_deficiency",
    "evaluate_kernel",
    "project_column_stochastic",
    # distances
    "tv_discrete",
    "tv_histogram",
    "tv_gaussian",
    "wasserstein1_normalized",
    "rbf_kernel",
    "mmd_rbf",
    "normalize_weights",
    # gaussian
    "gaussian_location_dominates",
    "gaussian_location_deficiency_proxy",
    "gaussian_scalar_location_deficiency",
    "two_point_confounding_lower_bound",
    "expfam_deficiency_proxy",
]
