"""ΛCDM+S — entropy-extended cosmology research software.

Public names re-export the original scientific implementations in
``analytical_solver`` (Layer-1 derivation + coupled background ODEs) and
``bayesian_validation`` (four-parameter background + Layer-3 inference).
Equations are not reimplemented in this package init.
"""

from __future__ import annotations

from .bayesian_validation import (
    ANALYTICAL_S_EARLY,
    ANALYTICAL_S_MAX,
    BackgroundParams,
    BackgroundSolution,
    FIDUCIAL_H0,
    FIDUCIAL_K_GYR,
    FIDUCIAL_OMEGA_LAMBDA,
    FIDUCIAL_T_CRIT_GYR,
    JointLikelihood,
    ModifiedCLASS,
    POSTULATES,
    TABLE1_PRIORS,
    __version__,
    build_production_posterior,
    lcdm_hubble,
    lcdm_recovery_test,
    lcdm_s_core_registry,
    logistic_rate,
    logistic_weight,
    solve_background,
)

__all__ = [
    "__version__",
    "ANALYTICAL_S_EARLY",
    "ANALYTICAL_S_MAX",
    "BackgroundParams",
    "BackgroundSolution",
    "FIDUCIAL_H0",
    "FIDUCIAL_K_GYR",
    "FIDUCIAL_OMEGA_LAMBDA",
    "FIDUCIAL_T_CRIT_GYR",
    "JointLikelihood",
    "ModifiedCLASS",
    "POSTULATES",
    "TABLE1_PRIORS",
    "build_production_posterior",
    "lcdm_hubble",
    "lcdm_recovery_test",
    "lcdm_s_core_registry",
    "logistic_rate",
    "logistic_weight",
    "solve_background",
]
