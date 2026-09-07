"""Horizon entropy identities from the original geometric-unit closures."""

from __future__ import annotations

import math

from lcdm_plus_s.entropy import (
    entropy_time_derivative,
    gsl_satisfied,
    horizon_perturbation_chain,
    horizon_state,
    p_S_horizon,
    rho_S_horizon,
    w_S_thermodynamic,
)
from lcdm_plus_s.bayesian_validation import BackgroundParams, _w_S_analytical


def test_horizon_entropy_is_pi_over_h_squared():
    H = 0.07
    state = horizon_state(H)
    assert math.isclose(state.entropy, math.pi / (H * H), rel_tol=1e-12)
    assert math.isclose(state.radius, 1.0 / H, rel_tol=1e-12)
    assert math.isclose(state.temperature, H / (2.0 * math.pi), rel_tol=1e-12)


def test_horizon_state_rejects_nonpositive_h():
    try:
        horizon_state(0.0)
    except ValueError:
        return
    raise AssertionError("H <= 0 must raise")


def test_gsl_equivalent_to_nonpositive_hdot():
    H = 0.07
    assert gsl_satisfied(H, Hdot=-1e-4)
    assert not gsl_satisfied(H, Hdot=+1e-4)
    dS = entropy_time_derivative(H, Hdot=-1e-4)
    assert dS > 0.0


def test_thermodynamic_eos_de_sitter_is_minus_one():
    assert math.isclose(w_S_thermodynamic(H=0.07, Hdot=0.0), -1.0, rel_tol=0.0, abs_tol=1e-12)


def test_horizon_density_pressure_ratio_when_w_minus_one_branch():
    H = 0.07
    # p_S = H²/8π, ρ_S = 3H²/8π  ⇒  p/ρ = 1/3 in this geometric closure
    # (distinct from the logistic Layer-1 EOS used in the background).
    rho = rho_S_horizon(H)
    p = p_S_horizon(H)
    assert math.isclose(p / rho, 1.0 / 3.0, rel_tol=1e-12)


def test_layer1_eos_lcdm_limit_is_minus_one():
    p = BackgroundParams(lcdm_limit=True)
    w = _w_S_analytical(chi=0.4, H_gyr=0.07, p=p)
    assert w == -1.0


def test_perturbation_chain_signs():
    chain = horizon_perturbation_chain(0.02)
    assert chain["delta_SH_over_SH"] == -0.02
    assert chain["delta_H_over_H"] == 0.5 * 0.02
