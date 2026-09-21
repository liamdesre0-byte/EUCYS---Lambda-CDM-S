"""Rebuild posterior_summary.json from saved chain CSVs (helper)."""
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import lcdm_plus_s.bayesian_validation as bv  # noqa: E402

files = sorted(glob.glob(str(ROOT / "output" / "chains" / "chain_run*.csv")))
arrs = [np.loadtxt(f, delimiter=",", skiprows=1) for f in files]
names = open(files[0]).readline().strip().split(",")
arr = np.vstack(arrs)
priors = bv.table1_gaussian_priors()
summary = {
    "parameters": {}, "rhat_across_runs": {},
    "n_samples_total": int(arr.shape[0]),
    "config": {"walkers": 48, "steps": 250, "burn": 75, "runs": len(files),
               "seed": 20260920, "data_dir": "data", "skip_cmb": False},
    "joint_components": ["desi_dr2", "boss_dr12", "planck", "shoes",
                         "pantheon_plus"],
    "holdout": ["des_sny5"],
    "label": ("PRELIMINARY chain (%d run%s) - regenerate at full production "
              "size before quoting" % (len(files), "s" if len(files) > 1 else "")),
}
for i, nm in enumerate(names):
    v = arr[:, i]
    p2, p16, p50, p84, p97 = np.percentile(v, [2.5, 16, 50, 84, 97.5])
    pr = priors[nm]
    summary["parameters"][nm] = {
        "median": float(p50), "p16": float(p16), "p84": float(p84),
        "p2.5": float(p2), "p97.5": float(p97),
        "prior_mean": pr.mu, "prior_sigma": pr.sigma,
        "posterior_over_prior_width": float((p84 - p16) / 2 / pr.sigma),
        "shift_from_prior_sigma": float((p50 - pr.mu) / pr.sigma)}
corr = np.corrcoef(arr.T)
summary["correlations"] = {
    f"{names[i]}__{names[j]}": float(corr[i, j])
    for i in range(len(names)) for j in range(i + 1, len(names))}
out = ROOT / "output" / "tables" / "posterior_summary.json"
out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
for nm, d in summary["parameters"].items():
    print(f"{nm:14s} {d['median']:9.4f} [{d['p16']:9.4f},{d['p84']:9.4f}] "
          f"prior {d['prior_mean']} shift {d['shift_from_prior_sigma']:+.2f}s "
          f"width x{d['posterior_over_prior_width']:.2f}")
print("correlations:", {k: round(v, 3) for k, v in summary["correlations"].items()})
print("wrote", out)
