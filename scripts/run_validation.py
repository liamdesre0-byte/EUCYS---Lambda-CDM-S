#!/usr/bin/env python3
"""Run Layer-3 Bayesian validation (bayesian_validation.main).

Examples
--------
python scripts/run_validation.py
python scripts/run_validation.py --quick --skip-validation --no-artifacts
python scripts/run_validation.py --entropy-sector --steps 200 --seed 8
python scripts/run_validation.py --config configs/default.yaml --output results
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _cli_utils import add_common_args, default_config_path, load_yaml


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ΛCDM+S Bayesian validation (Layer 3, Phases 0–15)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_args(parser)
    parser.add_argument("--steps", "--prod-steps", dest="steps", type=int, default=None)
    parser.add_argument("--chains", type=int, default=None)
    parser.add_argument("--burn", type=int, default=None)
    parser.add_argument("--walkers", type=int, default=None)
    parser.add_argument("--nlive", type=int, default=None)
    parser.add_argument("--ns-iter", dest="ns_iter", type=int, default=None)
    parser.add_argument("--mcmc-method", dest="mcmc_method", type=str, default=None)
    parser.add_argument("--data-dir", dest="data_dir", type=str, default=None)
    parser.add_argument("--require-sn", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--no-artifacts", action="store_true")
    parser.add_argument("--skip-prior-predictive", action="store_true")
    parser.add_argument("--skip-cmb", action="store_true")
    parser.add_argument("--no-getdist", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--ppc-runs", dest="ppc_runs", type=int, default=None)
    parser.add_argument("--theory-nsteps", dest="theory_nsteps", type=int, default=None)
    parser.add_argument(
        "--entropy-sector",
        action="store_true",
        help="Sample logistic ΛCDM+S (overrides original lcdm_limit=True)",
    )
    parser.add_argument(
        "--lcdm-limit",
        action="store_true",
        help="Freeze entropy sector (original production default)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Small MCMC + skip end-of-run unit suite and prior PPC",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg_path = args.config
    if cfg_path is None and default_config_path().is_file():
        cfg_path = default_config_path()
    cfg = load_yaml(cfg_path)
    mcmc = cfg.get("mcmc") or {}
    paths = cfg.get("paths") or {}

    def pick(cli, key, default):
        if cli is not None:
            return cli
        if key in mcmc and mcmc[key] is not None:
            return mcmc[key]
        return default

    steps = pick(args.steps, "steps", 300)
    if args.quick:
        steps = min(int(steps), 40)

    forwarded: list[str] = [
        "--steps", str(int(steps)),
        "--chains", str(int(pick(args.chains, "chains", 2))),
        "--walkers", str(int(pick(args.walkers, "walkers", 10))),
        "--nlive", str(int(pick(args.nlive, "nlive", 25))),
        "--ns-iter", str(int(pick(args.ns_iter, "ns_iter", 100))),
        "--seed", str(int(pick(args.seed, "seed", 8))),
        "--mcmc-method", str(pick(args.mcmc_method, "method", "metropolis")),
        "--theory-nsteps", str(int(pick(args.theory_nsteps, "theory_nsteps", 400))),
        "--ppc-runs", str(int(pick(args.ppc_runs, "ppc_runs", 200))),
    ]
    burn = args.burn if args.burn is not None else mcmc.get("burn")
    if burn is not None:
        forwarded.extend(["--burn", str(int(burn))])

    outdir = args.output or paths.get("outdir") or "results"
    forwarded.extend(["--outdir", str(Path(outdir))])

    data_dir = args.data_dir if args.data_dir is not None else paths.get("data_dir")
    if data_dir:
        forwarded.extend(["--data-dir", str(data_dir)])
    if args.require_sn:
        forwarded.append("--require-sn")

    skip_val = args.skip_validation or bool(mcmc.get("skip_validation")) or args.quick
    no_art = args.no_artifacts or bool(mcmc.get("no_artifacts"))
    skip_ppc = (
        args.skip_prior_predictive
        or bool(mcmc.get("skip_prior_predictive"))
        or args.quick
    )
    if skip_val:
        forwarded.append("--skip-validation")
    if no_art:
        forwarded.append("--no-artifacts")
    if skip_ppc:
        forwarded.append("--skip-prior-predictive")
    if args.skip_cmb or bool(mcmc.get("skip_cmb")):
        forwarded.append("--skip-cmb")
    if args.no_getdist or bool(mcmc.get("no_getdist")):
        forwarded.append("--no-getdist")
    if args.no_progress:
        forwarded.append("--no-progress")

    lcdm_limit = bool(mcmc.get("lcdm_limit", True))
    if args.entropy_sector:
        lcdm_limit = False
    if args.lcdm_limit:
        lcdm_limit = True
    if lcdm_limit:
        forwarded.append("--lcdm-limit")
    else:
        forwarded.append("--entropy-sector")

    if args.verbose:
        print("run_validation forwarding:", " ".join(forwarded), flush=True)

    from lcdm_plus_s.bayesian_validation import main as validation_main

    return validation_main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
