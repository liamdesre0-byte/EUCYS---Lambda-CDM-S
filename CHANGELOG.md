# Changelog

## 2.1.0 — 2026-09-17

Packaged CLI / YAML defaults now match the **published EUCYS posterior**
instead of the original-script diagnostic sizes.

- `configs/default.yaml` and `RunConfig`: 48 walkers, 50,000 production
  steps, 2,000 burn-in, 3 affine ensembles, ESS 75,000, entropy sector
  ON → 7,200,000 posterior samples.
- `--steps` / `--prod-steps` are production samples **kept after
  burn-in** (total walker steps = 52,000).
- Affine production runs `chains` independent ensembles of `walkers`.
- `python scripts/run_validation.py --quick` and
  `configs/diagnostic.yaml` remain the short non-paper smoke path.
- `--lcdm-limit` still freezes χ(t); that is **not** the written-report
  posterior. `build_production_posterior(..., lcdm_limit=True)` keeps
  the original function default. `original/` scripts are unchanged.

## 2.0.0 — 2026-09-07

Packaged the existing research scripts `Bayesian_Validationn.py` and
`analytical_solver.py` as the `lcdm-plus-s` Python package.

- Scientific equations, fiducials, Table 1 priors, and likelihood
  definitions are those of the original files.
- Production MCMC still defaults to `lcdm_limit=True` (original
  behaviour). Pass `--entropy-sector` to evaluate logistic ΛCDM+S
  dynamics while sampling.
- Unmodified copies of the input scripts are stored under `original/`.
