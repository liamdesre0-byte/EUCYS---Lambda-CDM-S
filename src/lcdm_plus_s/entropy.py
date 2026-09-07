"""Horizon entropy and entropy-sector closures.

All formulas are those already implemented in ``bayesian_validation``
(geometric-unit horizon state, GSL test, thermodynamic w_S) and the
Layer-1 thermodynamic EOS used by the background solver.
"""

from __future__ import annotations

from .bayesian_validation import (
    ANALYTICAL_S_EARLY,
    ANALYTICAL_S_MAX,
    ENTROPY_SECTOR,
    HorizonState,
    continuity_rhs,
    entropy_time_derivative,
    gsl_satisfied,
    horizon_perturbation_chain,
    horizon_state,
    p_S_horizon,
    rho_S_horizon,
    w_S_from_continuity,
    w_S_thermodynamic,
)
from .bayesian_validation import _w_S_analytical as w_S_analytical

__all__ = [
    "ANALYTICAL_S_EARLY",
    "ANALYTICAL_S_MAX",
    "ENTROPY_SECTOR",
    "HorizonState",
    "continuity_rhs",
    "entropy_time_derivative",
    "gsl_satisfied",
    "horizon_perturbation_chain",
    "horizon_state",
    "p_S_horizon",
    "rho_S_horizon",
    "w_S_analytical",
    "w_S_from_continuity",
    "w_S_thermodynamic",
]
