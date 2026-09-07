"""MCMC, nested sampling, and posterior analysis.

Samplers live in ``bayesian_validation`` (Metropolis–Hastings, affine
ensemble, differential evolution, builtin nested sampling, optional
dynesty / UltraNest / PolyChord / Cobaya backends).
"""

from __future__ import annotations

from .bayesian_validation import (
    MCMCResult,
    MCMCRunner,
    NestedSamplingResult,
    NestedSamplingRunner,
    Posterior,
    PosteriorAnalyzer,
    PosteriorSummary,
    analyze_posterior,
    bayes_factor,
    interpret_bayes_factor,
    lcdm_s_core_registry,
    log_bayes_factor,
    run_mcmc,
    run_nested_sampling,
    table1_gaussian_priors,
)

__all__ = [
    "MCMCResult",
    "MCMCRunner",
    "NestedSamplingResult",
    "NestedSamplingRunner",
    "Posterior",
    "PosteriorAnalyzer",
    "PosteriorSummary",
    "analyze_posterior",
    "bayes_factor",
    "interpret_bayes_factor",
    "lcdm_s_core_registry",
    "log_bayes_factor",
    "run_mcmc",
    "run_nested_sampling",
    "table1_gaussian_priors",
]
