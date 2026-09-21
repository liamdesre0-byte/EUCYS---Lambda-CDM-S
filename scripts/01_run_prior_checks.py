#!/usr/bin/env python3
"""01_run_prior_checks.py — priors, provenance, and prior-predictive ensemble.

Runs BEFORE any observational likelihood (anti-circularity: this stage never
reads posterior output).  Produces:

  output/tables/prior_table.json        declared Table-1 priors + provenance
  output/tables/prior_predictive.json   fRMSE stats (horizon-AREA metric)
  output/chains/prior_ensemble.csv      the 500 prior draws + per-draw fRMSE
  output/figures/prior_predictive_area.png

Metric note: the fractional RMSE is computed on the Hubble-horizon AREA
A_H ∝ 1/H², exactly the formula printed in the written report.  Do not
compare against the older R_H-based 14.9% / 93.4% numbers.
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import lcdm_plus_s.bayesian_validation as bv  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    t0 = time.time()
    out_fig = ROOT / "output" / "figures"
    out_tab = ROOT / "output" / "tables"
    out_ch = ROOT / "output" / "chains"
    for d in (out_fig, out_tab, out_ch):
        d.mkdir(parents=True, exist_ok=True)

    # ---- declared priors + provenance -----------------------------------
    priors = bv.table1_gaussian_priors()
    mix = bv.derive_hubble_omega_priors_from_mixing()
    ode = bv.derive_transition_priors_from_odes()
    prior_table = {
        "declared_priors": {n: {"mean": p.mu, "sigma": p.sigma}
                            for n, p in priors.items()},
        "provenance": {
            "H0_planck_anchored_raw": mix.H0_planck_anchored_raw,
            "H0_derivation_reproduces_declared":
                mix.derivation_reproduces_declared,
            "H0_note": mix.notes,
            "k_from_tau": {"tau_gyr": ode.tau_tr, "k_mean": ode.k_mean},
            "t_crit_crossover_gyr": ode.crossover_target_gyr,
        },
        "stage": "prior (no posterior/likelihood information used)",
    }
    (out_tab / "prior_table.json").write_text(
        json.dumps(prior_table, indent=2), encoding="utf-8")

    # ---- prior-predictive ensemble (500 draws, AREA metric) -------------
    print("prior predictive: 500 draws, metric = horizon AREA...", flush=True)
    ppc = bv.prior_predictive_horizon_check(n_runs=500, seed=20260728,
                                            nsteps=500, metric="A_H")
    frmse = np.asarray(ppc.fractional_rmse, dtype=float)
    ok = np.isfinite(frmse)
    stats = {
        "stage": "prior_predictive",
        "metric": "A_H (horizon area, written-report formula)",
        "n_draws": int(ppc.n_runs),
        "seed": 20260728,
        "mean_frmse": float(frmse[ok].mean()),
        "sd_frmse": float(frmse[ok].std(ddof=1)),
        "sem_frmse": float(frmse[ok].std(ddof=1) / np.sqrt(ok.sum())),
        "mean_similarity": float(1.0 - frmse[ok].mean()),
        "best_similarity": float(ppc.best_similarity),
        "best_theta": ppc.best_theta,
        "ks_pvalues": ppc.ks_p,
        "labeling_note": ("best_similarity is the BEST prior-predictive "
                          "realization, a model-to-model comparison vs "
                          "flat LCDM sharing (H0, Omega_L) — it is NOT "
                          "observational accuracy."),
    }
    (out_tab / "prior_predictive.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8")

    # ---- ensemble figure: A_H(z)/A_H,LCDM ratio --------------------------
    print("drawing ensemble figure...", flush=True)
    reg = bv.lcdm_s_core_registry()
    rng = random.Random(20260728)
    zg = np.linspace(0.0, 2.0, 60)
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11, 4.2), constrained_layout=True)
    rows = []
    for i in range(120):
        vec = reg.sample_prior(rng)
        th = vec.as_dict()
        bg = reg.to_background_params(vec.values, lcdm_limit=False)
        sol = bv.solve_background(bg, nsteps=400, normalization_iterations=2)
        ref = bv.BackgroundParams(
            H0_kms_mpc=th["H0"], Omega_Lambda=th["Omega_Lambda"],
            k_gyr=th["k"], t_crit_gyr=th["t_crit"], lcdm_limit=True)
        ratio = [(1.0 / max(sol.hubble_of_z(z), 1e-30) ** 2)
                 / (1.0 / max(bv.lcdm_hubble(ref, z), 1e-30) ** 2)
                 for z in zg]
        ax1.plot(zg, ratio, color="C0", alpha=0.10, lw=0.8)
        rows.append([th["H0"], th["Omega_Lambda"], th["k"], th["t_crit"]])
    ax1.axhline(1.0, color="k", lw=1.2, label=r"flat $\Lambda$CDM")
    ax1.set_xlabel("z")
    ax1.set_ylabel(r"$A_H^{\Lambda\mathrm{CDM}+S}\,/\,A_H^{\Lambda\mathrm{CDM}}$")
    ax1.set_title("Prior-predictive horizon-area ratio (120 of 500 draws)")
    ax1.legend()

    ax2.hist(100 * frmse[ok], bins=40, color="C0", alpha=0.8)
    ax2.axvline(100 * frmse[ok].mean(), color="k", lw=1.5,
                label=f"mean {100 * frmse[ok].mean():.1f}%")
    ax2.set_xlabel("fractional RMSE on horizon area [%]")
    ax2.set_ylabel("draws")
    ax2.set_title("Prior-predictive fRMSE (N=500, area metric)")
    ax2.legend()
    fig.savefig(out_fig / "prior_predictive_area.png", dpi=180)

    np.savetxt(out_ch / "prior_ensemble_draws.csv", np.asarray(rows),
               delimiter=",", header="H0,Omega_Lambda,k,t_crit", comments="")
    np.savetxt(out_ch / "prior_ensemble_frmse.csv", frmse,
               delimiter=",", header="frmse_area", comments="")

    print(json.dumps(stats, indent=2)[:800], flush=True)
    print("DONE", round(time.time() - t0, 1), "s", flush=True)


if __name__ == "__main__":
    main()
