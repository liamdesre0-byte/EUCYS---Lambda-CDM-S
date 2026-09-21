#!/usr/bin/env python3
"""03_ppc_des_sn5.py — held-out DES-SN Y5 posterior-predictive comparison.

Requires: output/tables/posterior_summary.json from 02_run_mcmc.py.

Design (written-report hold-out):
  * Both models are conditioned ONLY on the training joint
    (Pantheon+ + DESI + BOSS + SH0ES + Planck compressed).
  * Parameters are then FROZEN and evaluated on the held-out DES vector
    (1820 unique SNe = 1623 DES + 197 external low-z).
  * LCDM baseline is refit on the SAME training joint (lcdm_limit=True,
    entropy sector frozen), same data, same covariance treatment.

Outputs: output/figures/des_sn5_ppc.png, output/tables/model_comparison.json
and a LaTeX table.  Statistics are labeled by dataset — held-out numbers are
never mixed with training AIC/BIC (anti-circularity spec §N.38).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

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
    out_fig.mkdir(parents=True, exist_ok=True)
    out_tab.mkdir(parents=True, exist_ok=True)

    summ = json.loads((out_tab / "posterior_summary.json").read_text())
    names = list(summ["parameters"].keys())
    theta_s = [summ["parameters"][n]["median"] for n in names]
    print("LCDM+S posterior median:", dict(zip(names, np.round(theta_s, 4))),
          flush=True)

    # --- LCDM+S: training joint + held-out DES ---------------------------
    reg_s, th_s, joint_s, post_s = bv.build_production_posterior(
        data_dir=str(ROOT / "data"), lcdm_limit=False, theory_nsteps=300)
    des_s = joint_s.holdout_likelihoods["des_sny5"]
    train_logl_s = joint_s.log_likelihood(theta_s)
    des_chi2_s = des_s.chi2(theta_s)
    des_logl_s = des_s.log_likelihood(theta_s)

    # --- LCDM baseline: SAME training joint, entropy frozen --------------
    reg_l, th_l, joint_l, post_l = bv.build_production_posterior(
        data_dir=str(ROOT / "data"), lcdm_limit=True, theory_nsteps=300)
    des_l = joint_l.holdout_likelihoods["des_sny5"]
    priors = bv.table1_gaussian_priors()
    kf, tf = priors["k"].mu, priors["t_crit"].mu

    def neg_logl_lcdm(x):
        theta = [float(x[0]), float(x[1]), kf, tf]
        try:
            v = joint_l.log_likelihood(theta)
        except Exception:
            return 1e12
        return -v if np.isfinite(v) else 1e12

    best = None
    for x0 in ([67.4, 0.685], [70.0, 0.70], [73.0, 0.688]):
        r = minimize(neg_logl_lcdm, x0, method="Nelder-Mead",
                     options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 400})
        if best is None or r.fun < best.fun:
            best = r
    theta_l = [float(best.x[0]), float(best.x[1]), kf, tf]
    train_logl_l = -float(best.fun)
    des_chi2_l = des_l.chi2(theta_l)
    des_logl_l = des_l.log_likelihood(theta_l)
    print(f"LCDM baseline fit: H0={theta_l[0]:.3f} OmL={theta_l[1]:.4f} "
          f"train lnL={train_logl_l:.2f}", flush=True)

    # --- statistics, kept separate by dataset ----------------------------
    n_train = sum(lk.n_data() for lk in joint_s.likelihoods)
    k_s, k_l = 4, 2
    comparison = {
        "training": {
            "datasets": [getattr(lk, "table2_key", "?")
                         for lk in joint_s.likelihoods],
            "n_data": int(n_train),
            "logL_LCDMS_at_posterior_median": float(train_logl_s),
            "logL_LCDM_at_bestfit": float(train_logl_l),
            "AIC_LCDMS": float(2 * k_s - 2 * train_logl_s),
            "AIC_LCDM": float(2 * k_l - 2 * train_logl_l),
            "BIC_LCDMS": float(k_s * np.log(n_train) - 2 * train_logl_s),
            "BIC_LCDM": float(k_l * np.log(n_train) - 2 * train_logl_l),
            "note": ("AIC/BIC computed on the SAME training joint for both "
                     "models; LCDM+S value uses the posterior median, not "
                     "the MAP, so its AIC/BIC is slightly conservative."),
        },
        "holdout_des_sn5": {
            "n_unique_sn": 1820, "n_des": 1623, "n_lowz": 197,
            "chi2_LCDMS": float(des_chi2_s),
            "chi2_LCDM": float(des_chi2_l),
            "delta_chi2_holdout": float(des_chi2_s - des_chi2_l),
            "delta_lnL_holdout": float(des_logl_s - des_logl_l),
            "note": ("held-out statistics; NEVER subtract these from "
                     "training/joint evidence numbers"),
        },
        "theta_LCDMS_median": dict(zip(names, map(float, theta_s))),
        "theta_LCDM_bestfit": {"H0": theta_l[0], "Omega_Lambda": theta_l[1]},
        "label": summ.get("label", ""),
    }
    (out_tab / "model_comparison.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8")

    # --- LaTeX table -------------------------------------------------------
    tr, ho = comparison["training"], comparison["holdout_des_sn5"]
    latex = (
        "\\begin{tabular}{lcc}\n\\toprule\n"
        " & $\\Lambda$CDM & $\\Lambda$CDM$+$S \\\\\n\\midrule\n"
        f"training $\\ln L$ & {tr['logL_LCDM_at_bestfit']:.1f} & "
        f"{tr['logL_LCDMS_at_posterior_median']:.1f} \\\\\n"
        f"training AIC & {tr['AIC_LCDM']:.1f} & {tr['AIC_LCDMS']:.1f} \\\\\n"
        f"training BIC & {tr['BIC_LCDM']:.1f} & {tr['BIC_LCDMS']:.1f} \\\\\n"
        f"held-out DES $\\chi^2$ & {ho['chi2_LCDM']:.1f} & "
        f"{ho['chi2_LCDMS']:.1f} \\\\\n\\bottomrule\n\\end{tabular}\n")
    (out_tab / "model_comparison.tex").write_text(latex, encoding="utf-8")

    # --- figure: DES Hubble diagram + residuals ---------------------------
    ds = des_s.dataset
    z = np.asarray(ds.z, float)
    mu = np.asarray(ds.data, float)
    sig = np.asarray(ds.sigma, float)
    zs = np.argsort(z)

    # use the likelihood's own prediction vector for exactness
    pred_s = np.asarray(des_s._model(theta_s), float)
    pred_l = np.asarray(des_l._model(theta_l), float)

    fig, (a1, a2) = plt.subplots(
        2, 1, figsize=(9, 7), sharex=True, constrained_layout=True,
        height_ratios=[2.2, 1.0])
    a1.errorbar(z, mu, yerr=sig, fmt=".", ms=2.5, alpha=0.25, color="0.4",
                label="DES-SN Y5 (held out, 1820 SNe)")
    a1.plot(z[zs], pred_l[zs], lw=1.8, color="C1",
            label=r"$\Lambda$CDM (refit on training)")
    a1.plot(z[zs], pred_s[zs], lw=1.8, color="C0", ls="--",
            label=r"$\Lambda$CDM$+$S (posterior median)")
    a1.set_ylabel(r"$\mu(z)$")
    a1.set_xscale("log")
    a1.legend(fontsize=9)
    a1.set_title(
        f"Held-out DES-SN Y5: "
        f"$\\chi^2_{{\\Lambda CDM+S}}$={des_chi2_s:.1f}, "
        f"$\\chi^2_{{\\Lambda CDM}}$={des_chi2_l:.1f} "
        f"({comparison['label'].split(chr(8212))[0].strip() or 'chain'})")
    a2.axhline(0, color="k", lw=1)
    a2.plot(z[zs], (mu - pred_l)[zs], ".", ms=2.5, alpha=0.3, color="C1")
    a2.plot(z[zs], (pred_s - pred_l)[zs], lw=1.8, color="C0", ls="--",
            label=r"$\mu_{\Lambda CDM+S}-\mu_{\Lambda CDM}$")
    a2.set_xlabel("z")
    a2.set_ylabel(r"$\Delta\mu$ [mag]")
    a2.set_ylim(-0.6, 0.6)
    a2.legend(fontsize=9)
    fig.savefig(out_fig / "des_sn5_ppc.png", dpi=180)

    print(json.dumps(comparison["holdout_des_sn5"], indent=2), flush=True)
    print("DONE", round(time.time() - t0, 1), "s", flush=True)


if __name__ == "__main__":
    main()
