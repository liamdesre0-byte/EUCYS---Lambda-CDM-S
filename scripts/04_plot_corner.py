#!/usr/bin/env python3
"""04_plot_corner.py — corner plot + LaTeX tables from the ACTUAL chain.

Everything here is generated from output/chains/chain_run*.csv — nothing is
manually entered (anti-circularity spec §12: the chain is the source of
truth; the printed table and the plot use the same samples).
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import lcdm_plus_s.bayesian_validation as bv  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LABELS = {"H0": r"$H_0$", "Omega_Lambda": r"$\Omega_{\Lambda+S}$",
          "k": r"$k$", "t_crit": r"$t_{\rm crit}$"}


def main() -> None:
    files = sorted(glob.glob(str(ROOT / "output" / "chains" / "chain_run*.csv")))
    if not files:
        raise SystemExit("no chains found — run scripts/02_run_mcmc.py first")
    arrs = []
    names = None
    for f in files:
        with open(f, encoding="utf-8") as fh:
            header = fh.readline().strip().split(",")
        names = header
        arrs.append(np.loadtxt(f, delimiter=",", skiprows=1))
    chain = np.vstack(arrs)
    n = len(names)
    priors = bv.table1_gaussian_priors()

    fig, axes = plt.subplots(n, n, figsize=(10, 10))
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue
            if i == j:
                ax.hist(chain[:, i], bins=50, color="C0", alpha=0.75,
                        density=True)
                pr = priors[names[i]]
                xx = np.linspace(chain[:, i].min(), chain[:, i].max(), 200)
                ax.plot(xx, np.exp(-0.5 * ((xx - pr.mu) / pr.sigma) ** 2)
                        / (pr.sigma * np.sqrt(2 * np.pi)),
                        color="C3", lw=1.2, label="prior")
                if i == 0:
                    ax.legend(fontsize=7)
            else:
                ax.hist2d(chain[:, j], chain[:, i], bins=45, cmap="Blues")
            if i == n - 1:
                ax.set_xlabel(LABELS.get(names[j], names[j]))
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(LABELS.get(names[i], names[i]))
            else:
                ax.set_yticklabels([])
    fig.suptitle("ΛCDM+S posterior (generated from the shipped chain; "
                 "red = prior)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_fig = ROOT / "output" / "figures"
    out_fig.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig / "corner_posterior.png", dpi=170)

    # LaTeX posterior + correlation tables straight from the chain
    out_tab = ROOT / "output" / "tables"
    out_tab.mkdir(parents=True, exist_ok=True)
    lines = ["\\begin{tabular}{lccc}", "\\toprule",
             "parameter & prior & posterior median & 68\\% CI \\\\",
             "\\midrule"]
    for i, nm in enumerate(names):
        p16, p50, p84 = np.percentile(chain[:, i], [16, 50, 84])
        pr = priors[nm]
        lines.append(
            f"{LABELS.get(nm, nm)} & ${pr.mu} \\pm {pr.sigma}$ & "
            f"${p50:.3f}$ & $[{p16:.3f},\\,{p84:.3f}]$ \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (out_tab / "posterior_table.tex").write_text("\n".join(lines) + "\n",
                                                 encoding="utf-8")

    corr = np.corrcoef(chain.T)
    clines = ["\\begin{tabular}{l" + "c" * n + "}", "\\toprule",
              " & " + " & ".join(LABELS.get(x, x) for x in names) + " \\\\",
              "\\midrule"]
    for i in range(n):
        row = [LABELS.get(names[i], names[i])]
        for j in range(n):
            row.append(f"${corr[i, j]:+.2f}$" if j <= i else "")
        clines.append(" & ".join(row) + " \\\\")
    clines += ["\\bottomrule", "\\end{tabular}"]
    (out_tab / "correlation_table.tex").write_text("\n".join(clines) + "\n",
                                                   encoding="utf-8")

    print("chain samples:", chain.shape[0])
    print("correlations from chain:")
    for i in range(n):
        for j in range(i + 1, n):
            print(f"  {names[i]} vs {names[j]}: {corr[i, j]:+.3f}")
    print("wrote corner_posterior.png, posterior_table.tex, "
          "correlation_table.tex")


if __name__ == "__main__":
    main()
