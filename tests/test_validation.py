"""Likelihood, covariance, prior, and validation-path tests."""

from __future__ import annotations

import math
import random

from lcdm_plus_s.bayesian_validation import (
    BAODataset,
    CMBDataset,
    GaussianPrior,
    SHOESDataset,
    TABLE1_PRIORS,
    _invert_symmetric,
    _logdet_from_cholesky_diag,
    aic,
    bic,
    gelman_rubin,
    lcdm_s_core_registry,
    log_factorized_gaussian_prior,
    table1_gaussian_priors,
    validate_theta,
)
from lcdm_plus_s.covariance import PLANCK2018_COMPRESSED_COV, PLANCK2018_COMPRESSED_MEAN


def test_table1_prior_names_and_units():
    names = [row.name for row in TABLE1_PRIORS]
    assert names == ["H0", "Omega_Lambda", "k", "t_crit"]
    by_name = {row.name: row for row in TABLE1_PRIORS}
    assert by_name["H0"].units == "km s^{-1} Mpc^{-1}"
    assert by_name["k"].units == "Gyr^{-1}"
    assert by_name["t_crit"].units == "Gyr"


def test_gaussian_prior_logpdf_at_mean():
    prior = GaussianPrior(10.0, 2.0)
    expected = -math.log(2.0 * math.sqrt(2.0 * math.pi))
    assert math.isclose(prior.logpdf(10.0), expected, rel_tol=1e-12)


def test_factorized_prior_finite_at_fiducial():
    reg = lcdm_s_core_registry()
    theta = {p.name: p.fiducial for p in reg.sampled()}
    logp = log_factorized_gaussian_prior(theta, table1_gaussian_priors())
    assert math.isfinite(logp)


def test_theta_validation_rejects_out_of_bounds():
    reg = lcdm_s_core_registry()
    bad = [-10.0, 0.685, 0.37, 15.8]
    errs = validate_theta(reg, bad)
    assert errs


def test_planck_covariance_is_spd_and_invertible():
    cov = [list(row) for row in PLANCK2018_COMPRESSED_COV]
    logdet = _logdet_from_cholesky_diag(cov)
    inv = _invert_symmetric(cov)
    assert math.isfinite(logdet)
    # C C^{-1} ≈ I
    n = 3
    for i in range(n):
        for j in range(n):
            s = sum(cov[i][k] * inv[k][j] for k in range(n))
            target = 1.0 if i == j else 0.0
            assert abs(s - target) < 1e-8


def test_compressed_catalogs_load_without_scaffold_flag():
    shoes = SHOESDataset.load()
    assert shoes.data == [73.04]
    assert shoes.sigma == [1.04]
    bao = BAODataset.load_desi_dr2()
    assert bao.n_data() >= 10
    cmb = CMBDataset.load()
    assert cmb.data == list(PLANCK2018_COMPRESSED_MEAN)
    assert cmb.covariance is not None


def test_aic_bic_penalty_grows_with_parameters():
    logL = -100.0
    assert aic(logL, 4) > aic(logL, 2)
    assert bic(logL, 4, 50) > bic(logL, 2, 50)


def test_gelman_rubin_identical_chains_near_one():
    rng = random.Random(8)
    chains = [[rng.gauss(0.0, 1.0) for _ in range(80)] for _ in range(4)]
    rhat = gelman_rubin(chains)
    assert 0.9 < rhat < 1.3


def test_registry_expand_flatness():
    reg = lcdm_s_core_registry()
    theta = reg.fiducial_vector().values
    expanded = reg.expand(theta)
    assert math.isclose(
        expanded["omega_m0"] + expanded["Omega_Lambda"] + expanded["omega_r0"],
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
