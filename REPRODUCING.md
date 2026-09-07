# Reproducing ΛCDM+S results

This guide documents **commands that exist in this repository**. It does
not invent datasets, posterior tables, or MCMC runtimes that were not
executed here.

## 1. Python version

Python **3.10 or newer** (the packaged modules use `list[float] | None`
syntax). Developed against CPython 3.13 on Windows.

## 2. Installation

From the repository root:

```text
python -m pip install -e .
python -m pip install -e ".[app,test]"
```

Conda alternative:

```text
conda env create -f environment.yml
conda activate lcdm-plus-s
```

## 3. Dependencies

Required (see `requirements.txt` / `pyproject.toml`):

- `numpy`, `scipy`, `sympy` — analytical solver and coupled ODEs
- `matplotlib` — figure export
- `PyYAML` — `configs/default.yaml`
- `pytest` — tests

Optional:

- `streamlit` — interactive app
- `getdist` — triangle / 1D / 2D posterior plots
- `rich` or `tqdm` — MCMC progress bars
- `cobaya`, `dynesty`, `ultranest`, `pypolychord` — optional sampling backends

Core Bayesian background math is stdlib + the packaged solvers; MCMC
does not require Cobaya.

## 4. Required datasets

Production validation **always** uses literature compressed probes
embedded in `bayesian_validation.py`:

| Probe | What is shipped | Source cited in code |
| --- | --- | --- |
| DESI DR2 BAO | `(z, value, σ, label)` rows | DESI Collaboration 2024/2025 |
| BOSS DR12 BAO/RSD | same | Alam et al. 2017, 2021 |
| Planck compressed | mean `(R, ℓ_A, ω_b)` + 3×3 covariance | Planck 2018 / Chen–Huang–Wang style |
| SH0ES | `H0 = 73.04 ± 1.04` km s⁻¹ Mpc⁻¹ | Riess et al. 2022 |

**Not shipped** (copyright / size): Pantheon+ and DES-SN Y5 supernova
catalogs, SPARC rotation curves, WMAP \(C_\ell\). Place CSVs in a folder
and pass `--data-dir` if you have licensed copies. Expected names:

```text
data/pantheon_plus.csv
data/des_sny5.csv
```

See `data/README.md`.

## 5. Expected file locations

| Path | Role |
| --- | --- |
| `configs/default.yaml` | fiducials, priors, MCMC sizes |
| `src/lcdm_plus_s/` | installable package |
| `original/` | unmodified input scripts |
| `results/` | Bayesian artifacts (created at run time) |
| `theory_export/` | analytical-solver exports (created at run time) |
| `figures/` | fiducial background plots from `reproduce_figures.py` |

## 6. Configuration

```text
python scripts/run_solver.py --config configs/default.yaml
python scripts/run_validation.py --config configs/default.yaml --output results
```

CLI flags override YAML. `--seed` and `--output` are accepted by both
wrappers.

## 7. Running the analytical solver

Full symbolic derivation plus integrated numerics (can take many minutes):

```text
python scripts/run_solver.py
```

Faster integrated analysis:

```text
python scripts/run_solver.py --quick
```

Symbolic derivation only:

```text
python scripts/run_solver.py --skip-validation --export-dir theory_export
```

Numerical validation suite only:

```text
python scripts/run_solver.py --validation-only --validation-quick --validation-out lcdms_validation_out
```

Equivalent package entry point after install:

```text
lcdms-solver --validation-quick --export-dir theory_export
```

## 8. Running Bayesian validation

**Warning.** The original production path builds `ModifiedCLASS(..., lcdm_limit=True)`,
which freezes the entropy sector to constant \(\Omega_\Lambda\). That
default is preserved. To actually evaluate logistic ΛCDM+S while sampling:

```text
python scripts/run_validation.py --entropy-sector --output results
```

Original (frozen-sector) production default:

```text
python scripts/run_validation.py --output results
```

Smoke run (small MCMC, skip the end-of-run unit suite and prior PPC):

```text
python scripts/run_validation.py --quick --no-artifacts --no-progress --output results
```

With supernova CSVs:

```text
python scripts/run_validation.py --data-dir data --require-sn --entropy-sector
```

Direct module invocation (same flags as the original script):

```text
python -m lcdm_plus_s.bayesian_validation --steps 300 --seed 8 --outdir results
```

## 9. Running tests

```text
python -m pip install -e ".[test]"
python -m pytest
```

Tests check limiting behaviour, logistic identities, GSL sign, covariance
SPD, and ΛCDM recovery. They do **not** run production MCMC.

## 10. Generating figures

Fiducial background comparison (no MCMC):

```text
python scripts/reproduce_figures.py --output figures --verbose
```

Publication MCMC figures (trace, corner, GetDist) are written under
`results/figures/` by `run_validation.py` unless `--no-artifacts` is set.
Those files reflect **that run**, not a committed “official” posterior.

## 11. Expected outputs

Analytical solver (when export is on): markdown/TeX/CSV under
`theory_export/` (exact filenames depend on `export_artifacts`).

Bayesian validation (when artifacts are on): `results/` tree including
trace/corner PDFs, diagnostic CSV, and a validation report if the
end-of-run suite is enabled.

## 12. Random seeds

Default master seed is **8** (`RunConfig.seed` / `mcmc.seed` in YAML).
Child seeds for nested sampling, PPC, and diagnostics are offsets of
this master seed inside `bayesian_validation.main` (for example
`seed+11`, `seed+62`). Fix `--seed` for repeatable sampler draws; ODE
integrators are deterministic given θ.

## 13. MCMC configuration

Defaults from the original `RunConfig` (also in `configs/default.yaml`):

| Setting | Default |
| --- | --- |
| method | `metropolis` |
| steps | 300 |
| chains | 2 |
| burn | `steps//6` |
| walkers | 10 |
| nlive / ns_iter | 25 / 100 |
| ess_min / rhat_max | 50 / 1.1 |
| theory_nsteps | 400 |

These defaults are **short diagnostic chains**, not a 50,000-step
publication run. Increase `--steps`, `--chains`, and `--diag-steps`
explicitly for inference you intend to quote.

## 14. Numerical tolerances

| Quantity | Where | Value |
| --- | --- | --- |
| `solve_ivp` rtol / atol | `analytical_solver.solve_background_N` | \(10^{-8}\) / \(10^{-10}\) |
| ΛCDM recovery ε | `LCDM_RECOVERY.tolerance` | \(10^{-4}\) |
| logistic overflow clip | `logistic_weight` | \(\lvert x\rvert > 700\) |
| χ clip in EOS | `_w_S_analytical` | \([10^{-12}, 1-10^{-12}]\) |

## 15. Troubleshooting

- **`FATAL: refusing synthetic/scaffold datasets`** — SPARC/WMAP/SN
  builtin scaffolds are disabled (`ALLOW_SCAFFOLD_DATASETS=False`). Use
  the compressed DESI/BOSS/Planck/SH0ES suite, or provide real CSVs.
- **`--require-sn` without files** — add Pantheon+ / DES-SN Y5 CSVs
  under `--data-dir`.
- **ImportError: sympy / scipy** — install `requirements.txt`. The
  Bayesian background itself is stdlib, but the analytical solver is not.
- **Slow `run_solver.py`** — pass `--quick` or `--skip-validation`.
- **Streamlit cannot import `lcdm_plus_s`** — `pip install -e .` or run
  from the repo so `app/streamlit_app.py` can add `src/` to `sys.path`.
- **Unicode on Windows consoles** — both `main()` functions call
  `sys.stdout.reconfigure(encoding="utf-8")`.
