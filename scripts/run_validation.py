#!/usr/bin/env python3
"""Run Layer-3 Bayesian validation (bayesian_validation.main).

The packaged default is the **published EUCYS posterior**, not a smoke test:

    48 walkers × 50,000 production × 3 affine ensembles
      = 7,200,000 posterior samples
    burn-in 2,000 · ESS target 75,000 · entropy sector ON

Examples
--------
python scripts/run_validation.py
python scripts/run_validation.py --quick
python scripts/run_validation.py --config configs/diagnostic.yaml
python scripts/run_validation.py --config configs/default.yaml --output results
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _cli_utils import add_common_args, default_config_path, load_yaml

EUCYS_WALKERS = 48
EUCYS_PROD_STEPS = 50_000
EUCYS_BURN = 2_000
EUCYS_CHAINS = 3
EUCYS_ESS_MIN = 75_000.0
EUCYS_POSTERIOR_SAMPLES = EUCYS_WALKERS * EUCYS_PROD_STEPS * EUCYS_CHAINS


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "LCDM+S Bayesian validation (Layer 3, Phases 0-15). "
            "Default MCMC is the published EUCYS run: 48 walkers x "
            "50,000 production x 3 chains = 7,200,000 samples, entropy "
            "sector ON. Pass --quick for a package diagnostic that does "
            "NOT reproduce the paper."
        ),
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
    parser.add_argument("--ess-min", dest="ess_min", type=float, default=None)
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
        help="Sample logistic LCDM+S (published EUCYS path; default)",
    )
    parser.add_argument(
        "--lcdm-limit",
        action="store_true",
        help="Freeze entropy sector (original script path; NOT the paper)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Diagnostic MCMC only - does NOT reproduce the published posterior",
    )
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="Load configs/diagnostic.yaml (alias for a short non-paper run)",
    )
    return parser


def _print_banner(
    *,
    walkers: int,
    steps: int,
    chains: int,
    burn: int,
    ess_min: float,
    method: str,
    lcdm_limit: bool,
    quick: bool,
) -> None:
    samples = int(walkers) * int(steps) * int(chains)
    published = (
        not quick
        and not lcdm_limit
        and int(walkers) == EUCYS_WALKERS
        and int(steps) == EUCYS_PROD_STEPS
        and int(chains) == EUCYS_CHAINS
        and int(burn) == EUCYS_BURN
        and abs(float(ess_min) - EUCYS_ESS_MIN) < 1.0
        and str(method) == "affine"
    )
    print("=" * 72, flush=True)
    if published:
        print("EUCYS published reproduction MCMC", flush=True)
        print(
            f"  {EUCYS_WALKERS} walkers x {EUCYS_PROD_STEPS:,} production "
            f"x {EUCYS_CHAINS} chains = {EUCYS_POSTERIOR_SAMPLES:,} "
            "posterior samples",
            flush=True,
        )
        print(
            f"  burn-in {EUCYS_BURN:,}  |  ESS target {EUCYS_ESS_MIN:,.0f}  "
            "|  affine  |  entropy sector ON",
            flush=True,
        )
        print(
            "  This matches the written-report chain. "
            "Smoke test only: python scripts/run_validation.py --quick",
            flush=True,
        )
    else:
        print("NOT the published EUCYS posterior", flush=True)
        if lcdm_limit:
            print(
                "  entropy sector FROZEN (lcdm_limit=True) - chi(t) does not vary",
                flush=True,
            )
        if quick:
            print("  --quick package diagnostic (short chains)", flush=True)
        print(
            f"  this run: {walkers} walkers x {steps:,} production x "
            f"{chains} chains = {samples:,} samples; burn-in {burn:,}; "
            f"ESS min {ess_min:,.0f}; method {method}",
            flush=True,
        )
        print(
            f"  published target: {EUCYS_WALKERS} walkers x "
            f"{EUCYS_PROD_STEPS:,} production x {EUCYS_CHAINS} chains "
            f"= {EUCYS_POSTERIOR_SAMPLES:,}; burn-in {EUCYS_BURN:,}; "
            f"ESS {EUCYS_ESS_MIN:,.0f}; affine; entropy sector ON",
            flush=True,
        )
        print(
            "  reproduce the paper: python scripts/run_validation.py",
            flush=True,
        )
    print("=" * 72, flush=True)


def main(argv: list[str] | None = None) -> int:
    _configure_stdio()
    args = build_parser().parse_args(argv)
    cfg_path = args.config
    if args.diagnostic and cfg_path is None:
        cfg_path = Path(__file__).resolve().parents[1] / "configs" / "diagnostic.yaml"
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

    steps = int(pick(args.steps, "steps", EUCYS_PROD_STEPS))
    chains = int(pick(args.chains, "chains", EUCYS_CHAINS))
    walkers = int(pick(args.walkers, "walkers", EUCYS_WALKERS))
    burn = args.burn if args.burn is not None else mcmc.get("burn")
    if burn is None:
        burn = EUCYS_BURN
    burn = int(burn)
    ess_min = float(pick(args.ess_min, "ess_min", EUCYS_ESS_MIN))
    method = str(pick(args.mcmc_method, "method", "affine"))

    if args.quick:
        steps = min(steps, 40)
        chains = min(chains, 2)
        walkers = min(walkers, 10)
        burn = min(burn, 8)
        ess_min = min(ess_min, 50.0)
        method = "metropolis"

    lcdm_limit = bool(mcmc.get("lcdm_limit", False))
    if args.entropy_sector:
        lcdm_limit = False
    if args.lcdm_limit:
        lcdm_limit = True

    _print_banner(
        walkers=walkers,
        steps=steps,
        chains=chains,
        burn=burn,
        ess_min=ess_min,
        method=method,
        lcdm_limit=lcdm_limit,
        quick=bool(args.quick or args.diagnostic),
    )

    forwarded: list[str] = [
        "--steps", str(int(steps)),
        "--chains", str(int(chains)),
        "--burn", str(int(burn)),
        "--walkers", str(int(walkers)),
        "--nlive", str(int(pick(args.nlive, "nlive", 25))),
        "--ns-iter", str(int(pick(args.ns_iter, "ns_iter", 100))),
        "--seed", str(int(pick(args.seed, "seed", 8))),
        "--mcmc-method", method,
        "--theory-nsteps", str(int(pick(args.theory_nsteps, "theory_nsteps", 400))),
        "--ppc-runs", str(int(pick(args.ppc_runs, "ppc_runs", 200))),
        "--ess-min", str(ess_min),
    ]

    outdir = args.output or paths.get("outdir") or "results"
    forwarded.extend(["--outdir", str(Path(outdir))])

    data_dir = args.data_dir if args.data_dir is not None else paths.get("data_dir")
    if data_dir:
        forwarded.extend(["--data-dir", str(data_dir)])
    if args.require_sn:
        forwarded.append("--require-sn")

    skip_val = args.skip_validation or bool(mcmc.get("skip_validation")) or args.quick
    no_art = args.no_artifacts or bool(mcmc.get("no_artifacts")) or args.quick
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
    if args.no_getdist or bool(mcmc.get("no_getdist")) or args.quick:
        forwarded.append("--no-getdist")
    if args.no_progress:
        forwarded.append("--no-progress")

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
