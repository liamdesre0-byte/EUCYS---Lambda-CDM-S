# Changelog

## 2.0.0 — 2026-09-07

Packaged the existing research scripts `Bayesian_Validationn.py` and
`analytical_solver.py` as the `lcdm-plus-s` Python package.

- Scientific equations, fiducials, Table 1 priors, and likelihood
  definitions are those of the original files.
- Production MCMC still defaults to `lcdm_limit=True` (original
  behaviour). Pass `--entropy-sector` to evaluate logistic ΛCDM+S
  dynamics while sampling.
- Unmodified copies of the input scripts are stored under `original/`.
