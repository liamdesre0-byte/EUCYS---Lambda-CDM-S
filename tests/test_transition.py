"""Logistic transition χ(t) properties required by the Γ-class closure."""

from __future__ import annotations

import math

from lcdm_plus_s.transition import (
    LOGISTIC_TRANSITION,
    logistic_rate,
    logistic_weight,
    solve_logistic_transition_ode,
)


def test_midpoint_is_one_half():
    k, t_crit = 0.37, 15.8
    assert math.isclose(logistic_weight(t_crit, k, t_crit), 0.5, rel_tol=1e-12)


def test_early_and_late_limits():
    k, t_crit = 0.37, 15.8
    assert logistic_weight(-40.0, k, t_crit) < 1e-6
    assert logistic_weight(80.0, k, t_crit) > 1.0 - 1e-6
    assert LOGISTIC_TRANSITION.limiting_value("early") == 0.0
    assert LOGISTIC_TRANSITION.limiting_value("late") == 1.0


def test_monotonic_increasing():
    k, t_crit = 0.37, 15.8
    ts = [t * 0.5 for t in range(0, 80)]
    chi = [logistic_weight(t, k, t_crit) for t in ts]
    assert all(chi[i] <= chi[i + 1] + 1e-15 for i in range(len(chi) - 1))


def test_ode_rate_identity():
    k = 0.37
    for chi in (0.1, 0.5, 0.9):
        assert math.isclose(logistic_rate(chi, k), k * chi * (1.0 - chi), rel_tol=0.0, abs_tol=1e-15)
    assert logistic_rate(0.0, k) == 0.0
    assert math.isclose(logistic_rate(1.0, k), 0.0, abs_tol=1e-15)


def test_max_slope_at_midpoint():
    k, t_crit = 0.37, 15.8
    chi_mid = logistic_weight(t_crit, k, t_crit)
    assert math.isclose(logistic_rate(chi_mid, k), k / 4.0, rel_tol=1e-12)


def test_rk4_matches_analytic_sigmoid():
    k, t_crit = 0.37, 15.8
    ts, ws = solve_logistic_transition_ode(k, t_crit, t_span=(0.0, 30.0), nsteps=4000)
    analytic = [logistic_weight(t, k, t_crit) for t in ts]
    err = max(abs(a - b) for a, b in zip(ws, analytic))
    assert err < 2e-4


def test_k_zero_is_frozen():
    assert logistic_weight(0.0, 0.0, 15.8) == logistic_weight(20.0, 0.0, 15.8)
