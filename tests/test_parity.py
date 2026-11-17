"""Drift catchers for implementations that deliberately stayed separate.

Two related packages keep their own copies of primitives that defkit
also provides, for good reasons:

* ``AEGIS`` advertises minimal dependencies and is published with a DOI, so it
  vendors what it needs rather than taking a dependency on defkit.
* ``causaldef`` is an R package on CRAN and cannot import Python at all.

Both agree with defkit. The point of these tests is that if any of the three
drifts, something fails; a numerical bug once fixed in causaldef sat unfixed in
the AEGIS port of the same closed form because nothing connected them.

Every test here skips cleanly when the other side is not installed, so the
suite stays green on a machine that only has defkit.
"""

import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

from defkit.distances import tv_gaussian
from defkit.gaussian import two_point_confounding_lower_bound

# Directory containing an AEGIS checkout; unset means the Python parity tests skip.
CORPUS = os.environ.get("DEFKIT_PARITY_ROOT", "")
AEGIS_SRC = os.path.join(CORPUS, "AEGIS", "src") if CORPUS else ""

# Parameter grid shared by both parity checks, including the near-degenerate
# regime that broke the original closed-form algorithm.
TV_CASES = [
    (0.0, 1.0, 0.0, 1.0),
    (0.0, 1.0, 1.0, 1.0),
    (0.0, 1.0, 0.0, 2.0),
    (-1.5, 0.7, 2.0, 3.1),
    (0.0, 1.0, 5.0, 1.0),
    (0.0, 1e-6, 1.0, 1e-6),
    (0.0, 1e-4, 0.01, 1e-4),
    (3.0, 100.0, -3.0, 0.5),
]

CONFOUNDING_CASES = [
    (0.5, 0.5, 1.0, 1.0),
    (1.0, 2.0, 1.0, 1.0),
    (2.0, 0.3, 0.5, 2.0),
    (0.1, 0.1, 1.0, 1.0),
    (3.0, 3.0, 0.2, 0.2),
]


# ---------------------------------------------------------------------------
# AEGIS (Python, vendored)
# ---------------------------------------------------------------------------

def _load_aegis_theory():
    if not os.path.isdir(AEGIS_SRC):
        pytest.skip("AEGIS source tree not present")
    if AEGIS_SRC not in sys.path:
        sys.path.insert(0, AEGIS_SRC)
    try:
        from aegis import theory
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"AEGIS not importable: {exc}")
    return theory


@pytest.mark.parametrize("mu1,sd1,mu2,sd2", TV_CASES)
def test_aegis_tv_gaussian_matches_defkit(mu1, sd1, mu2, sd2):
    theory = _load_aegis_theory()
    mine = tv_gaussian(mu1, sd1, mu2, sd2)
    theirs = theory._tv_distance_normal(mu1, sd1, mu2, sd2)
    assert theirs == pytest.approx(mine, abs=1e-6), (
        "AEGIS's vendored _tv_distance_normal has drifted from defkit.tv_gaussian"
    )


@pytest.mark.parametrize("alpha,gamma,sigma_a,sigma_y", CONFOUNDING_CASES)
def test_aegis_confounding_bound_matches_defkit(alpha, gamma, sigma_a, sigma_y):
    theory = _load_aegis_theory()
    mine = two_point_confounding_lower_bound(alpha, gamma, sigma_a, sigma_y, a=1.0)
    theirs = theory._deficiency_gaussian(alpha, gamma, sigma_a, sigma_y, 1.0)
    assert theirs == pytest.approx(mine, abs=1e-6), (
        "AEGIS's vendored _deficiency_gaussian has drifted from defkit"
    )


# ---------------------------------------------------------------------------
# causaldef (R, on CRAN)
# ---------------------------------------------------------------------------

R_SCRIPT = r"""
suppressMessages(library(causaldef))
args <- commandArgs(trailingOnly = TRUE)
cases <- jsonlite::fromJSON(args[[1]])
out <- args[[2]]
tv <- apply(cases$tv, 1, function(r)
  causaldef:::.tv_distance_normal(r[[1]], r[[2]], r[[3]], r[[4]]))
cf <- apply(cases$confounding, 1, function(r)
  causaldef:::.deficiency_gaussian(r[[1]], r[[2]], r[[3]], r[[4]]))
writeLines(jsonlite::toJSON(list(tv = tv, confounding = cf), digits = 15), out)
"""


def _run_r_reference(tmp_dir):
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript not available")

    probe = subprocess.run(
        ["Rscript", "-e",
         'if (!requireNamespace("causaldef", quietly=TRUE) || '
         '!requireNamespace("jsonlite", quietly=TRUE)) quit(status=1)'],
        capture_output=True,
    )
    if probe.returncode != 0:
        pytest.skip("causaldef and/or jsonlite not installed in R")

    # Everything goes through files: passing JSON as an inline -e argument runs
    # into Rscript's own --args handling and its shell quoting.
    script_path = os.path.join(tmp_dir, "parity.R")
    input_path = os.path.join(tmp_dir, "cases.json")
    output_path = os.path.join(tmp_dir, "out.json")

    with open(script_path, "w") as fh:
        fh.write(R_SCRIPT)
    with open(input_path, "w") as fh:
        json.dump({
            "tv": [list(c) for c in TV_CASES],
            "confounding": [list(c) for c in CONFOUNDING_CASES],
        }, fh)

    res = subprocess.run(
        ["Rscript", script_path, input_path, output_path],
        capture_output=True, text=True, timeout=300,
    )
    if res.returncode != 0 or not os.path.exists(output_path):
        pytest.skip(f"R reference did not run: {res.stderr.strip()[:400]}")
    with open(output_path) as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def r_reference(tmp_path_factory):
    return _run_r_reference(str(tmp_path_factory.mktemp("rparity")))


def test_causaldef_tv_gaussian_matches_defkit(r_reference):
    theirs = np.atleast_1d(np.asarray(r_reference["tv"], dtype=float))
    mine = np.array([tv_gaussian(*c) for c in TV_CASES])
    worst = float(np.max(np.abs(theirs - mine)))
    assert worst < 1e-6, (
        f"causaldef:::.tv_distance_normal disagrees with defkit.tv_gaussian "
        f"by {worst:.2e}; the R and Python sides have drifted"
    )


def test_causaldef_confounding_bound_matches_defkit(r_reference):
    theirs = np.atleast_1d(np.asarray(r_reference["confounding"], dtype=float))
    mine = np.array([
        two_point_confounding_lower_bound(a, g, sa, sy, a=1.0)
        for (a, g, sa, sy) in CONFOUNDING_CASES
    ])
    worst = float(np.max(np.abs(theirs - mine)))
    assert worst < 1e-6, (
        f"causaldef:::.deficiency_gaussian disagrees with defkit by {worst:.2e}"
    )
