# 🌌 ΛCDM+S — Entropy-Extended Cosmology

**Can cosmic acceleration emerge from horizon thermodynamics?**

[📄 Paper](paper/README.md)
· [🧮 Mathematics](mathematics/README.md)
· [💻 Source Code](src/lcdm_plus_s/)
· [🧪 Validation](REPRODUCING.md)
· [🚀 Interactive App](app/README.md)

Research software for **ΛCDM+S** (entropy-extended cosmology): a
Layer-1 thermodynamic derivation and a Layer-3 Bayesian validation
pipeline. The science is the content of two existing programs,

1. `analytical_solver.py` — first-principles / symbolic + coupled ODEs
2. `Bayesian_Validationn.py` — four-parameter background + inference

packaged without rewriting the cosmological model.

This repository is a **hub** (paper, math, code, tests, app). A poster
QR code should point here, not only at a Streamlit demo.

---

# ΛCDM+S: Entropy-Extended Cosmology

## 1. Overview

ΛCDM+S extends a flat FLRW background with an **entropy sector**
\(\Omega_S(z)\) motivated by Hubble-horizon thermodynamics. Late-time
acceleration is associated with a de Sitter attractor of that sector,
not with a separately inserted vacuum energy as the *definition* of the
model. Observationally, the code still *compares* predictions to the
same distance and compressed-CMB probes used in standard analyses.

The software has two layers that must not be conflated:

| Layer | Module | Role |
| --- | --- | --- |
| 1 | `lcdm_plus_s.analytical_solver` | Tagged derivation \(S_H\to\) attractor \(\to\Lambda_S\); coupled \((\rho_r,\rho_m,\rho_S,\chi)\) ODEs |
| 3 | `lcdm_plus_s.bayesian_validation` | \(\theta=(H_0,\Omega_\Lambda,k,t_{\mathrm{crit}})\), logistic Friedmann solver, Gaussian likelihoods, MCMC / nested sampling |

Facade modules under `src/lcdm_plus_s/` re-export those implementations.
They do not contain a second copy of the Friedmann equation.

## 2. Research Question

Can a thermodynamically motivated modification of cosmological
dynamics, based on cosmic horizon entropy growth, **reproduce the
observed expansion history** without treating dark energy as a
fundamental cosmological-constant fluid?

The code answers **computational** parts of that question (solve \(H(t)\),
fit compressed likelihoods, report \(\chi^2\), AIC/BIC, optional
\(\ln Z\)). It does **not**, by itself, prove that dark energy is
unnecessary.

## 3. Scientific Motivation

Horizon thermodynamics (Bekenstein–Hawking entropy on the Hubble
sphere, generalized second law) suggests that expansion is constrained
by \(\dot S_H\ge 0\). With a **finite** maximum entropy \(S_{\max}\) and
an assumed completion \(\chi\), the Layer-1 solver obtains a stable de
Sitter end point and an asymptotic \(\Lambda_S=3\pi/(G S_{\max})\).

The inference layer uses a **logistic** \(\chi(t)\) as the
phenomenological closure of that picture (`k ≡ γ`). The analytical
solver states explicitly that \(F(\Pi)\) and logistic \(w(t)\) are
**not** the first-principles engine.

## 4. Model

Sampled parameters (Phase 2) — **only these four**:

\[
\theta = (H_0,\;\Omega_\Lambda,\;k,\;t_{\mathrm{crit}}),
\qquad \Omega_\Lambda\equiv\Omega_{S,0}.
\]

Fixed: \(\omega_{r0}=9\times 10^{-5}\), \(S_{\max}=1\),
\(S_{\mathrm{early}}=10^{-3}\). Derived:
\(\Omega_{m0}=1-\Omega_\Lambda-\omega_{r0}\),
\(\chi_0=\chi(t_0;k,t_{\mathrm{crit}})\).

ΛCDM is recovered when `lcdm_limit=True` or \(k\to 0\) (constant entropy
density, \(w_S=-1\)).

## 5. Mathematical Framework

See [mathematics/mathematical_framework.md](mathematics/mathematical_framework.md)
for equation-by-equation mapping onto source functions. The inference
background is

\[
\dot\chi=k\chi(1-\chi),\quad
\chi(t)=\frac{1}{1+e^{-k(t-t_{\mathrm{crit}})}},
\]

\[
E^2=\Omega_r a^{-4}+\Omega_m a^{-3}+\Omega_\Lambda\frac{\chi(t)}{\chi_0},
\quad
w_S=-1+\frac{\Delta S\,k\,\chi(1-\chi)}{3 H S_H}.
\]

Fiducial \(t_{\mathrm{crit}}\approx 15.8\,\mathrm{Gyr}\) is the
\(\chi=1/2\) midpoint (often **after** today). That is the behaviour of
the code, not a documentation error.

## 6. Computational Method

- **Bayesian background:** RK4 in \(\ln a\), iterative age so \(H(0)=H_0\).
- **Analytical background:** `scipy.integrate.solve_ivp` (RK45) in
  \(N=\ln a\), plus an independent cosmic-time RK4.
- **Theory interface:** `ModifiedCLASS` stacks background → sample-\(k\)
  perturbations → compressed CMB \((R,\ell_A,\omega_b)\) → BBKS \(P(k)\)
  → distances → growth. This is **not** a Boltzmann code.
- **Samplers:** Metropolis–Hastings (default), affine ensemble,
  differential evolution; builtin nested sampling; optional dynesty /
  UltraNest / PolyChord / Cobaya as backends only.

## 7. Bayesian Validation

Production joint likelihood (no SN files required):

- DESI DR2 compressed BAO
- BOSS DR12 BAO/RSD
- Planck compressed distance priors (full 3×3 covariance)
- SH0ES compressed \(H_0=73.04\pm 1.04\)

Pantheon+ / DES-SN Y5 are loaded **only** from `--data-dir` CSVs.
Scaffolded SPARC/WMAP catalogs are refused (`ALLOW_SCAFFOLD_DATASETS=False`).

**Important (original behaviour):** `build_production_posterior` default
is `lcdm_limit=True`, i.e. the entropy sector is frozen during
production sampling. Pass `--entropy-sector` to evaluate logistic
ΛCDM+S while sampling. See `SCIENTIFIC_AUDIT.md`.

Distinguish always:

| Quantity | Meaning in this code |
| --- | --- |
| \(\chi^2\), RMS | goodness of fit of a point θ |
| \(\ln L\) | Gaussian likelihood (includes \(\ln|2\pi C|\)) |
| AIC / BIC | information criteria (parameter penalty) |
| \(\ln Z\) | Bayesian evidence (nested sampling only) |
| Bayes factor | \(Z_i/Z_j\), not \(\Delta\chi^2\) |

Default CLI chains (`steps=300`) are **diagnostic**, not a quoted
posterior.

## 8. Results

This repository **does not commit numerical posteriors**. Table 1
**priors** in the source are

\[
H_0=72.8\pm 2.059,\;
\Omega_\Lambda=0.685\pm 0.012,\;
k=0.37\pm 0.068,\;
t_{\mathrm{crit}}=15.8\pm 1.408
\]

(in the units of §5). Those are prior means/widths, **not** posterior
intervals from a run in this git tree.

Do not read EUCYS dashboard cartoons (synthetic walkers, calibrated
Δχ² scoreboards) as output of `Bayesian_Validationn.py`. The original
computing notes already separate those visualizations from this engine.

## 9. Repository Structure

```text
LambdaCDM-plus-S/
├── README.md
├── LICENSE
├── CITATION.cff
├── REPRODUCING.md
├── SCIENTIFIC_AUDIT.md
├── src/lcdm_plus_s/          # installable package
├── scripts/                  # run_solver / run_validation / reproduce_figures
├── tests/
├── configs/default.yaml
├── app/streamlit_app.py
├── mathematics/              # existing derivation notes + TeX
├── paper/                    # written-report TeX (PDF not bundled)
├── original/                 # unmodified input scripts
├── data/                     # optional user CSVs (not committed)
├── figures/                  # generated locally
└── results/                  # generated locally
```

## 10. Installation

```text
python -m pip install -e ".[app,test]"
```

See `environment.yml` for conda. Python ≥ 3.10.

## 11. Quick Start

```text
python -m pytest
python scripts/run_solver.py --skip-validation --export-dir theory_export
python scripts/reproduce_figures.py --output figures
python -m streamlit run app/streamlit_app.py
```

## 12. Reproducing Results

Full instructions, dataset paths, seeds, and MCMC sizes:
[REPRODUCING.md](REPRODUCING.md).

```text
python scripts/run_solver.py
python scripts/run_validation.py --entropy-sector --output results
```

## 13. Example Usage

```python
from lcdm_plus_s import BackgroundParams, solve_background

lcdms = solve_background(BackgroundParams(lcdm_limit=False), nsteps=2000)
lcdm = solve_background(BackgroundParams(lcdm_limit=True), nsteps=2000)
print(lcdms.t0, lcdms.chi0, lcdm.t0)
```

Notebook: `notebooks/example_analysis.ipynb`.

## 14. Data Sources

Compressed DESI, BOSS, Planck, and SH0ES summary statistics are
embedded with literature citations in `bayesian_validation.py`. Full
supernova and rotation-curve catalogs are **not** distributed.
[data/README.md](data/README.md).

## 15. Reproducibility

- Master seed default: `8`
- Config snapshot / git hash / dataset fingerprints: Phase 15
  `capture_reproducibility`
- Two Hubble conversion constants exist (Bayesian vs analytical); do
  not mix \(H\) arrays blindly (`SCIENTIFIC_AUDIT.md`)

## 16. Limitations

**Model assumptions.** Finite \(S_{\max}\); logistic / Γ-class closure;
flat FLRW; Hubble-horizon identification \(R_H=1/H\); application of
Bekenstein–Hawking entropy to that horizon; \(\Omega_\Lambda\) label
means \(\Omega_{S,0}\).

**Numerical approximations.** RK4 / RK45 backgrounds; compressed CMB
not \(C_\ell\); BBKS \(P(k)\); sample-\(k\) Newtonian entropy
perturbations; fixed \(r_d=147.09\) Mpc nuisance in BAO ratios.

**Dataset limitations.** Production SN/SPARC/WMAP require external
files; builtins for DESI/BOSS/Planck/SH0ES are **compressed** summaries,
not the raw catalogs; some BAO rows are treated as diagonal in \(\sigma\).

**Statistical limitations.** Default MCMC is short; AIC/BIC are not
evidence; production default `lcdm_limit=True` freezes \(\chi(t)\)
unless `--entropy-sector` is set; nested-sampling live points default
to 25. At the code fiducial, a smoke evaluation of the 30-element
production joint returned a very large \(\chi^2\) (see
`SCIENTIFIC_AUDIT.md`); do not treat that number as a validated fit.

**Missing physics / future work.** Full CLASS/CAMB entropy-sector
Boltzmann hierarchy; local Einstein equations from the Hubble horizon
alone (explicitly not claimed); UV completion; rotation-curve dark-matter
replacement (explicitly forbidden by Phase 0).

## 17. Future Work

Items already labelled future in the source: Hamiltonian MCMC, full
\(C_\ell\), WMAP/SPARC real catalogs, Cobaya as a sampler only with
this theory wrapper.

## 18. Citation

See `CITATION.cff`. Cite the software as research code by Liam Desre.
No DOI or journal article is invented here.

## 19. License

MIT — `LICENSE`.
