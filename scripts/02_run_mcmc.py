#!/usr/bin/env python3
"""02_run_mcmc.py — production / preliminary MCMC on the real joint likelihood.

Training joint: Pantheon+ SNe + DESI DR2 + BOSS DR12 + SH0ES + Planck
compressed CMB (parameter-dependent r_s).  DES-SN Y5 is HELD OUT and never
enters this likelihood (see 03_ppc_des_sn5.py).

Sampler: affine-invariant ensemble (Goodman & Weaver stretch move) from
lcdm_plus_s.bayesian_validation.  Independent runs use independent seeds;
Gelman-Rubin R-hat is computed ACROSS the independent runs.

Full written-report production size:
    python scripts/02_run_mcmc.py --walkers 48 --steps 50000 --burn 2000 --runs 3

Preliminary (hours-scale) size:
    python scripts/02_run_mcmc.py --walkers 48 --steps 300 --burn 90 --runs 2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import lcdm_plus_s.bayesian_validation as bv  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--walkers", type=int, default=48)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--burn", type=int, default=90)
    ap.add_argument("--runs", type=int, default=2,
                    help="independent ensembles (independent seeds)")
    ap.add_argument("--seed", type=int, default=20260920)
    ap.add_argument("--data-dir", type=str, default=str(ROOT / "data"))
    ap.add_argument("--skip-cmb", action="store_true")
    ap.add_argument("--theory-nsteps", type=int, default=300)
    ap.add_argument("--outdir", type=str, default=str(ROOT / "output"))
    args = ap.parse_args()

    outdir = Path(args.outdir)
    (outdir / "chains").mkdir(parents=True, exist_ok=True)
    (outdir / "tables").mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    reg, theory, joint, post = bv.build_production_posterior(
        data_dir=args.data_dir, skip_cmb=args.skip_cmb,
        theory_nsteps=args.theory_nsteps, lcdm_limit=False)
    names = list(reg.names())
    comps = [getattr(lk, "table2_key", getattr(lk, "name", "?"))
             for lk in joint.likelihoods]
    holdout = list(getattr(joint, "holdout_likelihoods", {}).keys())
    print(f"joint likelihood components : {comps}", flush=True)
    print(f"held-out (never in training): {holdout}", flush=True)

    flats: list[np.ndarray] = []
    for r in range(args.runs):
        seed = args.seed + 1000 * r
        print(f"[run {r + 1}/{args.runs}] affine, {args.walkers} walkers x "
              f"{args.steps} steps, burn {args.burn}, seed {seed}", flush=True)
        res = bv.run_mcmc(post, nchains=args.walkers, nsteps=args.steps,
                          burn=args.burn, seed=seed, method="affine",
                          step_frac=0.3, nwalkers=args.walkers)
        arr = np.vstack([np.asarray(ch, dtype=float) for ch in res.chains])
        flats.append(arr)
        np.savetxt(outdir / "chains" / f"chain_run{r + 1}.csv", arr,
                   delimiter=",", header=",".join(names), comments="")
        print(f"  samples: {arr.shape[0]}  elapsed: "
              f"{time.time() - t0:.0f}s", flush=True)

    # Gelman-Rubin across INDEPENDENT runs (not across walkers of one run)
    rhat = {}
    if len(flats) >= 2:
        m = len(flats)
        n = min(f.shape[0] for f in flats)
        for i, nm in enumerate(names):
            seqs = np.vstack([f[:n, i] for f in flats])
            mean_c = seqs.mean(axis=1)
            var_c = seqs.var(axis=1, ddof=1)
            B = n * mean_c.var(ddof=1)
            W = var_c.mean()
            rhat[nm] = float(np.sqrt(((n - 1) / n * W + B / n) / W))

    allf = np.vstack(flats)
    priors = bv.table1_gaussian_priors()
    summary = {"parameters": {}, "rhat_across_runs": rhat,
               "n_samples_total": int(allf.shape[0]),
               "config": vars(args), "joint_components": comps,
               "holdout": holdout,
               "label": ("PRELIMINARY chain — regenerate at full production "
                         "size before quoting" if args.steps < 50000
                         else "production")}
    for i, nm in enumerate(names):
        v = allf[:, i]
        p2, p16, p50, p84, p97 = np.percentile(v, [2.5, 16, 50, 84, 97.5])
        pr = priors[nm]
        summary["parameters"][nm] = {
            "median": float(p50), "p16": float(p16), "p84": float(p84),
            "p2.5": float(p2), "p97.5": float(p97),
            "prior_mean": pr.mu, "prior_sigma": pr.sigma,
            "posterior_over_prior_width": float((p84 - p16) / 2 / pr.sigma),
            "shift_from_prior_sigma": float((p50 - pr.mu) / pr.sigma),
        }
    corr = np.corrcoef(allf.T)
    summary["correlations"] = {
        f"{names[i]}__{names[j]}": float(corr[i, j])
        for i in range(len(names)) for j in range(i + 1, len(names))}

    out = outdir / "tables" / "posterior_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["parameters"], indent=2), flush=True)
    print("R-hat:", rhat, flush=True)
    print("wrote", out, flush=True)
    print("TOTAL", round(time.time() - t0, 1), "s", flush=True)


if __name__ == "__main__":
    main()
