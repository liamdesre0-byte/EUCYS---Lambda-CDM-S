"""Analytical / coupled background solver tests."""

from __future__ import annotations

import math

import numpy as np
import pytest

from lcdm_plus_s.analytical_solver import (
    LCDMSParams,
    relative_H_error,
    rhs_N,
    solve_background_N,
    test_lcdm_recovery as analytical_lcdm_recovery,
)
from lcdm_plus_s.bayesian_validation import (
    BackgroundParams,
    KMSMPC_TO_INVGYR,
    lcdm_hubble,
    lcdm_recovery_test,
    solve_background,
)


def test_lcdm_limit_matches_analytic_hubble():
    result = lcdm_recovery_test(nsteps=1200)
    assert result["passed"], result
    assert result["max_fractional_residual"] < 1e-4


def test_present_day_hubble_equals_h0():
    params = BackgroundParams()
    sol = solve_background(params, nsteps=2000)
    H0_gyr = params.H0_kms_mpc * KMSMPC_TO_INVGYR
    H_today = sol.hubble_of_z(0.0)
    assert math.isfinite(H_today)
    assert abs(H_today - H0_gyr) / H0_gyr < 5e-4


def test_hubble_positive_and_finite_on_grid():
    sol = solve_background(BackgroundParams(), nsteps=1500)
    assert all(math.isfinite(h) and h > 0.0 for h in sol.H)
    assert all(math.isfinite(a) and a > 0.0 for a in sol.a)
    assert sol.t0 > 0.0


def test_lcdm_hubble_increases_with_redshift():
    params = BackgroundParams(lcdm_limit=True)
    h0 = lcdm_hubble(params, 0.0)
    h1 = lcdm_hubble(params, 1.0)
    h2 = lcdm_hubble(params, 2.0)
    assert h0 < h1 < h2


def test_analytical_lcdm_recovery_function():
    payload = analytical_lcdm_recovery(z_max=10.0)
    assert payload["lcdm_flag"]["pass"], payload["lcdm_flag"]
    assert payload["gamma0"]["pass"], payload["gamma0"]


def test_analytical_friedmann_residual_small():
    p = LCDMSParams(lcdm_limit=True)
    sol = solve_background_N(p, z_max=10.0, n_eval=400, rtol=1e-8, atol=1e-10)
    e2 = sol.E ** 2
    rtot = sol.r_r + sol.r_m + sol.r_S
    rel = np.max(np.abs(e2 - rtot) / np.maximum(e2, 1e-30))
    assert float(rel) < 1e-8


def test_rhs_n_lcdm_freezes_chi():
    p = LCDMSParams(lcdm_limit=True, gamma=0.35)
    y = [p.Omega_r0, p.Omega_m0, p.Omega_S0, p.chi0]
    dchi = rhs_N(0.0, y, p)[3]
    assert dchi == 0.0


def test_relative_h_error_lcdm_limit_near_zero():
    p = LCDMSParams(lcdm_limit=True)
    sol = solve_background_N(p, z_max=5.0, n_eval=300)
    err = relative_H_error(sol, p)
    assert err["max_rel_H"] < 1e-6
