"""MCMC diagnostics and goodness-of-fit statistics.

Gelman–Rubin R̂, ESS, Geweke z, AIC/BIC/AICc, and the Layer-3
``diagnose_mcmc`` / ``compute_statistics`` reports are re-exported from
``bayesian_validation`` without changing the formulas.
"""

from __future__ import annotations

from .bayesian_validation import (
    DIAG_ESS_MIN,
    DIAG_RHAT_OK,
    DiagnosticsReport,
    MCMCDiagnostics,
    StatsReport,
    aic,
    aicc,
    bic,
    compute_statistics,
    diagnose_mcmc,
    effective_sample_size,
    gelman_rubin,
    integrated_autocorr_time,
    reduced_chi_squared,
    split_rhat,
)

__all__ = [
    "DIAG_ESS_MIN",
    "DIAG_RHAT_OK",
    "DiagnosticsReport",
    "MCMCDiagnostics",
    "StatsReport",
    "aic",
    "aicc",
    "bic",
    "compute_statistics",
    "diagnose_mcmc",
    "effective_sample_size",
    "gelman_rubin",
    "integrated_autocorr_time",
    "reduced_chi_squared",
    "split_rhat",
]
