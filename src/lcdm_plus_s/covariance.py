"""Covariance algebra used by Gaussian likelihoods.

The original code implements stdlib matrix inverse and log-determinant
for small dense covariances (Planck compressed 3×3, optional SN
covariance files). Dataset objects expose ``cov_matrix`` / ``inv_cov``.
"""

from __future__ import annotations

from .bayesian_validation import (
    PLANCK2018_COMPRESSED_COV,
    PLANCK2018_COMPRESSED_MEAN,
    Dataset,
    _invert_symmetric,
    _logdet_from_cholesky_diag,
    correlation_matrix,
    covariance_matrix,
)

__all__ = [
    "PLANCK2018_COMPRESSED_COV",
    "PLANCK2018_COMPRESSED_MEAN",
    "Dataset",
    "correlation_matrix",
    "covariance_matrix",
    "invert_symmetric",
    "logdet_spd",
]

invert_symmetric = _invert_symmetric
logdet_spd = _logdet_from_cholesky_diag
