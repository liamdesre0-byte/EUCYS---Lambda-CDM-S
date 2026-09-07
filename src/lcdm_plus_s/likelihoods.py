"""Gaussian and joint likelihoods used by Layer-3 inference.

Likelihood evaluation is delegated to ``bayesian_validation``: χ² is
computed from dataset residuals and (diagonal or dense) covariance;
production MCMC uses the Table-2 joint of DESI DR2, BOSS DR12, Planck
compressed distance priors, and SH0ES (plus SN CSVs if provided).
"""

from __future__ import annotations

from .bayesian_validation import (
    BAODataset,
    BOSSDR12Likelihood,
    CMBDataset,
    ChronometerLikelihood,
    DESIDR2Likelihood,
    DESLikelihood,
    Dataset,
    DatasetManager,
    GaussianLikelihood,
    GrowthLikelihood,
    JointLikelihood,
    Likelihood,
    PantheonLikelihood,
    PlanckLikelihood,
    Posterior,
    SHOESDataset,
    SHOESLikelihood,
    SupernovaDataset,
    TABLE2_DATASETS,
    audit_likelihoods,
    build_production_posterior,
    evidence_based_comparison,
)

__all__ = [
    "BAODataset",
    "BOSSDR12Likelihood",
    "CMBDataset",
    "ChronometerLikelihood",
    "DESIDR2Likelihood",
    "DESLikelihood",
    "Dataset",
    "DatasetManager",
    "GaussianLikelihood",
    "GrowthLikelihood",
    "JointLikelihood",
    "Likelihood",
    "PantheonLikelihood",
    "PlanckLikelihood",
    "Posterior",
    "SHOESDataset",
    "SHOESLikelihood",
    "SupernovaDataset",
    "TABLE2_DATASETS",
    "audit_likelihoods",
    "build_production_posterior",
    "evidence_based_comparison",
]
