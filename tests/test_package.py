"""Package import and configuration sanity."""

from __future__ import annotations

from pathlib import Path

import lcdm_plus_s
from lcdm_plus_s import BackgroundParams, solve_background


def test_package_version():
    assert lcdm_plus_s.__version__ == "2.0.0"


def test_public_solve_background_is_bayesian_implementation():
    sol = solve_background(BackgroundParams(), nsteps=400)
    assert sol.t0 > 0.0
    assert len(sol.H) > 10


def test_default_yaml_exists_and_has_fiducial_keys():
    root = Path(__file__).resolve().parents[1]
    text = (root / "configs" / "default.yaml").read_text(encoding="utf-8")
    for key in ("H0", "Omega_Lambda", "k", "t_crit", "lcdm_limit"):
        assert key in text
