"""Background cosmology API.

Re-exports the two background implementations that already exist in the
research code:

* ``bayesian_validation.solve_background`` — four-parameter θ used for
  inference: (H0, Ω_Λ ≡ Ω_S,0, k, t_crit) with C=1 logistic χ(t).
* ``analytical_solver.solve_background`` — coupled radiation/matter/entropy
  ODE system parameterized by (H0, Ω_m0, γ, χ0).

This module does not introduce a third Friedmann equation.
"""

from __future__ import annotations

from .analytical_solver import LCDMSParams
from .analytical_solver import BackgroundSolution as AnalyticalBackgroundSolution
from .analytical_solver import lcdm_E
from .analytical_solver import solve_background as solve_background_analytical
from .analytical_solver import solve_background_N
from .analytical_solver import solve_background_t_rk4
from .bayesian_validation import (
    BackgroundParams,
    BackgroundSolution,
    KMSMPC_TO_INVGYR,
    comoving_distance,
    compute_background_observables,
    deceleration_parameter,
    distance_modulus,
    lcdm_hubble,
    luminosity_distance,
    solve_background,
)

__all__ = [
    "AnalyticalBackgroundSolution",
    "BackgroundParams",
    "BackgroundSolution",
    "KMSMPC_TO_INVGYR",
    "LCDMSParams",
    "comoving_distance",
    "compute_background_observables",
    "deceleration_parameter",
    "distance_modulus",
    "lcdm_E",
    "lcdm_hubble",
    "luminosity_distance",
    "solve_background",
    "solve_background_N",
    "solve_background_analytical",
    "solve_background_t_rk4",
]
