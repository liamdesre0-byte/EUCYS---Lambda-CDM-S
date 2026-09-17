"""Published EUCYS MCMC defaults must not silently fall back to diagnostic sizes."""

from __future__ import annotations

from pathlib import Path

import yaml

from lcdm_plus_s.bayesian_validation import (
    EUCYS_BURN,
    EUCYS_CHAINS,
    EUCYS_ESS_MIN,
    EUCYS_METHOD,
    EUCYS_POSTERIOR_SAMPLES,
    EUCYS_PROD_STEPS,
    EUCYS_WALKERS,
    RunConfig,
    parse_cli_args,
)


ROOT = Path(__file__).resolve().parents[1]


def test_eucys_sample_count_is_7_2_million():
    assert EUCYS_WALKERS * EUCYS_PROD_STEPS * EUCYS_CHAINS == 7_200_000
    assert EUCYS_POSTERIOR_SAMPLES == 7_200_000
    assert EUCYS_BURN == 2_000
    assert EUCYS_ESS_MIN == 75_000.0
    assert EUCYS_METHOD == "affine"


def test_default_yaml_is_published_eucys_posterior():
    data = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    mcmc = data["mcmc"]
    assert mcmc["walkers"] == 48
    assert mcmc["steps"] == 50_000
    assert mcmc["burn"] == 2_000
    assert mcmc["chains"] == 3
    assert mcmc["ess_min"] == 75_000.0
    assert mcmc["method"] == "affine"
    assert mcmc["lcdm_limit"] is False
    assert mcmc["walkers"] * mcmc["steps"] * mcmc["chains"] == 7_200_000


def test_diagnostic_yaml_is_not_the_paper():
    data = yaml.safe_load(
        (ROOT / "configs" / "diagnostic.yaml").read_text(encoding="utf-8")
    )
    mcmc = data["mcmc"]
    assert mcmc["steps"] < 1_000
    assert mcmc["walkers"] < 48 or mcmc["chains"] < 3


def test_cli_defaults_match_published_eucys():
    cfg = parse_cli_args([])
    assert cfg.walkers == 48
    assert cfg.steps == 50_000
    assert cfg.burn == 2_000
    assert cfg.chains == 3
    assert cfg.ess_min == 75_000.0
    assert cfg.mcmc_method == "affine"
    assert cfg.lcdm_limit is False
    assert cfg.resolved_nsteps() == 52_000
    assert cfg.expected_posterior_samples() == 7_200_000
    assert cfg.matches_eucys_published() is True


def test_lcdm_limit_flag_is_not_the_paper():
    cfg = parse_cli_args(["--lcdm-limit"])
    assert cfg.lcdm_limit is True
    assert cfg.matches_eucys_published() is False


def test_runconfig_banner_mentions_seven_point_two_million():
    banner = RunConfig().eucys_banner()
    assert "7,200,000" in banner
    assert "published EUCYS match: YES" in banner
    assert "entropy sector: ON" in banner
