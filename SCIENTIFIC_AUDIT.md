# Scientific audit

Comparison of the original research scripts with the packaged
`lcdm-plus-s` implementation. This file records **what was preserved**,
**what was wrapped**, and **what was changed**. It is not a claim that
ΛCDM+S is observationally preferred.

## Original implementation

### `analytical_solver.py` (Layer 1)

- SymPy derivation: Hubble horizon \(R_H=1/H\), \(S_H=\pi/(G H^2)\),
  entropy production, thermodynamic EOM, equilibrium \(H_*=\sqrt{\pi/(G S_{\max})}\),
  linearization \(\lambda=-\gamma<0\), asymptotic
  \(\Lambda_S=3\pi/(G S_{\max})\).
- Explicit tags: AXIOM / DERIVED / ASSUMED / CONDITIONAL / THEOREM /
  PHENOMENOLOGY. Full local Einstein equations are **not** claimed from
  the global Hubble horizon alone.
- Coupled numerical background: `LCDMSParams` with
  \((H_0,\Omega_{m0},\gamma,\chi_0)\), `rhs_N` / `rhs_t`, `solve_ivp`
  (RK45) and independent RK4 in cosmic time.
- Embedded validation V0–V17 (Friedmann residual, dual integrators,
  ΛCDM recovery, synthetic IC scaffolds).

### `Bayesian_Validationn.py` (Layer 3)

- Four sampled parameters \(\theta=(H_0,\Omega_\Lambda,k,t_{\mathrm{crit}})\)
  with \(\Omega_\Lambda\equiv\Omega_{S,0}\).
- C=1 logistic \(\chi(t)=1/(1+e^{-k(t-t_{\mathrm{crit}})})\).
- Background:
  \(E^2=\Omega_r a^{-4}+\Omega_m a^{-3}+\Omega_\Lambda\chi(t)/\chi_0\).
- Table 1 Gaussian priors (means/widths hard-coded as published Table 1
  in the EUCYS written report, according to the source comments).
- Production joint likelihood: DESI DR2, BOSS DR12, Planck compressed
  \((R,\ell_A,\omega_b)\), SH0ES compressed \(H_0\); SN only from disk.
- MCMC (MH, affine, DE), builtin nested sampling, diagnostics, AIC/BIC,
  GetDist optional plots, Phase 0 claims auditor.

## Refactored implementation

- Both files are copied into `src/lcdm_plus_s/` as
  `analytical_solver.py` and `bayesian_validation.py`.
- Facade modules (`cosmology.py`, `entropy.py`, `transition.py`,
  `likelihoods.py`, `inference.py`, `covariance.py`, `diagnostics.py`,
  `plotting.py`) **re-export** those implementations. They do not
  contain a second Friedmann equation.
- `scripts/` wrap the original `main()` functions with `--config`,
  `--seed`, `--output`, `--verbose`.
- Unmodified snapshots: `original/Analytical_Solver.py`,
  `original/Bayesian_Validationn.py`.

## Changes

| Item | Original | Packaged | Why |
| --- | --- | --- | --- |
| Module path | loose scripts in Downloads | `lcdm_plus_s.*` | installable package |
| `build_production_posterior(..., lcdm_limit=)` | hard-coded `True` | argument, **function default still `True`**; packaged CLI/YAML default is `False` (2.1.0) | original script freeze preserved as an API; published EUCYS path is explicit |
| CLI MCMC sizes | 300 steps / 2 chains / 10 walkers / Metropolis / `lcdm_limit=True` | 48 walkers / 50,000 production / 2,000 burn-in / 3 affine ensembles / ESS 75,000 / entropy sector ON (2.1.0) | match the written-report posterior; `--quick` keeps the diagnostic path |
| Docstring headers | single-file mandate | note that the file is still the implementation | packaging without splitting science |

No fiducials, priors, compressed data vectors, or likelihood algebra
were edited.

## Bug fixes

None of the cosmological ODEs were rewritten.

**Documented original behaviour (not silently “fixed” in the snapshot):**
`original/Bayesian_Validationn.py` still builds production
`ModifiedCLASS` with `lcdm_limit=True`, so \(k\) and \(t_{\mathrm{crit}}\)
do not affect that script's sampled background. Treating *that* path as
the published ΛCDM+S posterior would be incorrect.

From **2.1.0** the packaged CLI / `configs/default.yaml` default is the
published EUCYS chain: entropy sector ON, 48 walkers × 50,000
production × 3 ensembles = 7,200,000 samples, burn-in 2,000, ESS
75,000. Pass `--lcdm-limit` to recover the frozen-sector original-script
behaviour. `--quick` is the short diagnostic and is not the paper.

## Numerical differences to expect

1. **Hubble unit conversion**
   - Bayesian: `KMSMPC_TO_INVGYR = GYR_S / MPC_KM` with
     `GYR_S=3.155815e16`, `MPC_KM=3.0856775814913673e19`.
   - Analytical: `KMSMPC_TO_INVGYR = 1.022712165045695e-3`.
   These constants differ at \(\sim 10^{-8}\) relative level. Do not
   mix \(H\) arrays from the two modules without converting.
2. **Two parameterizations of χ**
   - Analytical ODE: present-day `chi0` (default 0.92) is a boundary
     value of the coupled system.
   - Bayesian: `t_crit≈15.8 Gyr` with C=1 logistic, so \(\chi_0<1/2\)
     at \(t_0\approx 13.8\,\mathrm{Gyr}\). The Bayesian file states
     this is intentional and forbids remapping \(t_{\mathrm{crit}}\)
     to force \(\chi_0\approx 0.92\).
3. **Algebraic mixing vs Friedmann solver.**
   `algebraic_mixed_hubble` is used only in prior construction
   (Phase 3.3). The sampled background is the RK logistic Friedmann
   system, not \(H_{\mathrm{mix}}=(1-w)H_m+w H_S\).

## Validation evidence (this packaging)

Executed while packaging (2026-09-07, CPython 3.13, Windows):

- `python -m pip install -e ".[test]"` — success
- `python -m pytest` — **34 passed**
- `python scripts/run_solver.py --skip-validation --no-export --quiet` — exit 0
- `python scripts/reproduce_figures.py --output figures` — wrote H(z), ratio, distances, transition PNGs
- `build_production_posterior(theory_nsteps=80, lcdm_limit=True)` — DESI+BOSS+Planck+SH0ES, 30 likelihood elements
- `run_mcmc(..., nsteps=8, nchains=2)` — 12 kept samples, mean acceptance 0.375
- `python -m py_compile app/streamlit_app.py` — success
- Symbolic Layer-1 checks: geometry derived; EOM derived; \(H_*\) conditional on \(S_{\max}\); linearization \(\lambda=-\gamma<0\)

**Not executed:** full `scripts/run_validation.py` Phase 0–15 (multiple samplers, nested sampling, prior PPC, paper artifacts); full `run_solver.py` without `--skip-validation`; Streamlit server session.

**Observation (original code, not a packaging artifact):** at Table 1 fiducial θ with `lcdm_limit=True`, `Posterior.chi2` on the 30-element production joint is \(\sim 7.4\times 10^4\). That is not a published goodness-of-fit and is **not** quoted as a cosmological result. Diagnose the theory–data vector alignment before treating production \(\chi^2\) as physical.

If a command was not run, this audit does not pretend that it was.

## Dual BackgroundSolution types

Both modules define a class named `BackgroundSolution` with different
fields. Use

- `lcdm_plus_s.cosmology.BackgroundSolution` (Bayesian / inference)
- `lcdm_plus_s.cosmology.AnalyticalBackgroundSolution`

Do not pass one into functions written for the other.
