"""Logistic / Γ-class entropy transition χ(t).

Implements the same closure as ``analytical_solver`` and
``bayesian_validation``:

    χ̇ = k χ (1 − χ)
    χ(t) = 1 / (1 + exp[−k(t − t_crit)])    with χ(t_crit) = 1/2
"""

from __future__ import annotations

from .bayesian_validation import (
    LOGISTIC_TRANSITION,
    logistic_rate,
    logistic_transition_diagnostics,
    logistic_weight,
    solve_logistic_transition_ode,
    t_crit_to_z_crit,
    transition_epoch_map,
)

__all__ = [
    "LOGISTIC_TRANSITION",
    "logistic_rate",
    "logistic_transition_diagnostics",
    "logistic_weight",
    "solve_logistic_transition_ode",
    "t_crit_to_z_crit",
    "transition_epoch_map",
]
