# Mathematical framework (as implemented)

Every equation below is taken from `analytical_solver.py` and/or
`bayesian_validation.py`. If a popular cosmology formula is not in
those files, it is not listed here.

Units: Bayesian background \(H\) is stored in Gyr\(^{-1}\) and converted
with `KMSMPC_TO_INVGYR`; distances use \(c=299792.458\,\mathrm{km\,s^{-1}}\).
Analytical solver uses the same \(c\) and its own `KMSMPC_TO_INVGYR`
(see `SCIENTIFIC_AUDIT.md`). Entropy in `horizon_state` is geometric
(\(c=k_B=\hbar=1\)): \(S_H=\pi/H^2\).

## 1. Horizon geometry and entropy

**Implemented in** `bayesian_validation.horizon_state`,
`analytical_solver.step_geometry` / `step_entropy`.

\[
R_H=\frac{1}{H},\qquad
A_H=4\pi R_H^2,\qquad
S_H=\frac{\pi}{H^2}\quad(G=1\text{ geometric units}).
\]

The postulate text also writes the dimensionful Bekenstein–Hawking form
\(S_H=k_B c^3 A_H/(4G\hbar)\). The numeric `horizon_state` function uses
the geometric expression above.

Time derivative (Theorem 2.2 / `entropy_time_derivative`):

\[
\dot S_H=-\frac{2\pi\dot H}{H^3}.
\]

GSL in this code: \(\dot S_H\ge 0\) for \(H>0\) is equivalent to
\(\dot H\le 0\) (`gsl_satisfied`).

Temperature used in `horizon_state`: \(T_H=H/(2\pi)\).

## 2. Finite \(S_{\max}\) and the de Sitter attractor

**Implemented in** `analytical_solver.step_solve_H_star`,
`step_linearize_stability`, `step_thermodynamic_Lambda_S`.

Assumed: \(0<S_{\max}<\infty\) and a thermodynamic completion
\(\chi\in[0,1]\) with \(S_H=S_{\mathrm{early}}+(S_{\max}-S_{\mathrm{early}})\chi\).

Equilibrium of the entropy chart:

\[
H_\infty=\sqrt{\frac{\pi}{G S_{\max}}},\qquad
\Lambda_S=3 H_\infty^2=\frac{3\pi}{G S_{\max}}.
\]

The solver tags the **asymptotic** Einstein relation
\(G_{\mu\nu}^{(\infty)}+\Lambda_S g_{\mu\nu}=0\) as not a derivation of
the full local Einstein field equations from the global Hubble horizon.

Logistic / Γ-class closure (assumed phenomenological completion):

\[
\dot\chi=\gamma\chi(1-\chi),\qquad \lambda=-\gamma<0
\]

about \(\chi=1\) (stable de Sitter for \(\gamma>0\)).

## 3. Logistic transition used in inference

**Implemented in** `logistic_weight`, `logistic_rate`,
`LOGISTIC_TRANSITION`. Here \(k\equiv\gamma\).

\[
\dot\chi=k\chi(1-\chi),\qquad
\chi(t)=\frac{1}{1+\exp[-k(t-t_{\mathrm{crit}})]},\qquad
\chi(t_{\mathrm{crit}})=\frac12.
\]

Maximum slope \(\dot\chi(t_{\mathrm{crit}})=k/4\). Timescale
\(\tau_{\mathrm{tr}}=1/k\). 90% width \(\Delta t_{90}=2\ln 9/k\).

Fiducial \(t_{\mathrm{crit}}\approx 15.8\,\mathrm{Gyr}\) (Table 1 in
code). With \(t_0\sim 13.8\,\mathrm{Gyr}\) this implies \(\chi_0<1/2\).
The Bayesian source forbids remapping \(t_{\mathrm{crit}}\) to force
\(\chi_0\approx 0.92\).

## 4. Friedmann equation (Bayesian / inference background)

**Implemented in** `bayesian_validation.solve_background`.

Sampled \(\theta=(H_0,\Omega_\Lambda,k,t_{\mathrm{crit}})\) with
\(\Omega_\Lambda\equiv\Omega_{S,0}\) and
\(\Omega_{m0}=1-\Omega_\Lambda-\omega_{r0}\), \(\omega_{r0}=9\times 10^{-5}\)
fixed.

\[
E^2(a)=\Omega_{r0}a^{-4}+\Omega_{m0}a^{-3}
+\Omega_\Lambda\frac{\chi(t)}{\chi_0},\qquad
H=H_0 E,\quad \chi_0=\chi(t_0).
\]

`lcdm_limit=True` or \(k\le 0\) replaces the last term by constant
\(\Omega_\Lambda\) (flat ΛCDM). Integration is RK4 in \(\ln a\) with
iterative \(t_0\) so that \(H(0)=H_0\).

**Coupled ODE background** (`analytical_solver.rhs_N`): densities
\((r_r,r_m,r_S,\chi)\) with continuity

\[
\frac{\mathrm{d}r_r}{\mathrm{d}N}=-4 r_r,\quad
\frac{\mathrm{d}r_m}{\mathrm{d}N}=-3 r_m,\quad
\frac{\mathrm{d}r_S}{\mathrm{d}N}=-3(1+w_S)r_S,\quad
\frac{\mathrm{d}\chi}{\mathrm{d}N}=\frac{\tilde\gamma\,\chi(1-\chi)}{E},
\]

\(E^2=r_r+r_m+r_S\), \(\tilde\gamma=\gamma/H_0\), \(N=\ln a\).

These two backgrounds are **related but not identical** (χ0 boundary vs
t_crit normalization). See `SCIENTIFIC_AUDIT.md`.

## 5. Entropy-sector equation of state

**Layer-1 thermo EOS** (`_w_S` / `_w_S_analytical`):

\[
w_S=-1+\frac{\Delta S\,k\,\chi(1-\chi)}{3 H S_H},\qquad
S_H=S_{\mathrm{early}}+\Delta S\,\chi,\quad
\Delta S=S_{\max}-S_{\mathrm{early}}.
\]

Default bookkeeping: \(S_{\max}=1\), \(S_{\mathrm{early}}=10^{-3}\).

**Geometric closures** also present (`rho_S_horizon`, `p_S_horizon`,
`w_S_thermodynamic`):

\[
\rho_S=\frac{3H^2}{8\pi},\quad
p_S=\frac{H^2}{8\pi},\quad
w_S=-1-\frac{2}{3}\frac{\dot H}{H^2}.
\]

The Friedmann integrator uses the Layer-1 logistic EOS, not \(p_S=\rho_S/3\).

## 6. Distances

**Implemented in** `comoving_distance`, `luminosity_distance`,
`distance_modulus`, `BackgroundObservables`.

\[
\chi(z)=c\int_0^z\frac{\mathrm{d}z'}{H(z')},\quad
d_L=(1+z)\chi,\quad
\mu=5\log_{10}(d_L/\mathrm{Mpc})+25.
\]

BAO ratios \(D_M/r_d\), \(D_H/r_d\), \(D_V/r_d\) use a fixed Planck
sound-horizon nuisance `PLANCK_RS_MHD_MPC = 147.09` Mpc unless a dataset
overrides `r_d`.

Compressed CMB (`ModifiedCLASS._stage_cmb`):

\[
R=\sqrt{\Omega_m}\,(H_0/c)\,\chi(z_*),\qquad
\ell_A=\pi\chi(z_*)/r_s(z_*).
\]

This is a **background compressed prior**, not a Boltzmann \(C_\ell\) fit
(Phase 0 scope).

## 7. Priors (Table 1 in code)

Factorized Gaussians \(\pi(\theta)=\prod_i\mathcal{N}(\mu_i,\sigma_i^2)\):

| Parameter | μ | σ | Units |
| --- | ---: | ---: | --- |
| \(H_0\) | 72.8 | 2.059 | km s⁻¹ Mpc⁻¹ |
| \(\Omega_\Lambda\equiv\Omega_{S,0}\) | 0.685 | 0.012 | — |
| \(k\) | 0.37 | 0.068 | Gyr⁻¹ |
| \(t_{\mathrm{crit}}\) | 15.8 | 1.408 | Gyr |

Log-prior (`log_factorized_gaussian_prior`):

\[
\ln\pi(\theta)=\sum_i\left[-\frac12\frac{(x_i-\mu_i)^2}{\sigma_i^2}
-\ln(\sigma_i\sqrt{2\pi})\right].
\]

Phase 3 comments derive \((k,t_{\mathrm{crit}})\) from the logistic ODE
and \((H_0,\Omega_\Lambda)\) from algebraic mixing plus Planck
\(\Omega_m h^2=0.1432\), \(r_s=147.09\) Mpc; the **numeric means/widths
are the Table 1 constants above**, not re-solved each run.

## 8. Likelihoods and covariance

Gaussian probe (`Dataset.chi2`, `Dataset.log_likelihood`):

\[
\chi^2=\Delta y^\top C^{-1}\Delta y,\qquad
\ln L=-\frac12\left[\chi^2+\ln|2\pi C|\right].
\]

Joint: \(\ln L_{\mathrm{joint}}=\sum_i\ln L_i\). Posterior:
\(\ln\pi(\theta\mid d)=\ln\pi(\theta)+\ln L(\theta)\).

Planck uses a dense 3×3 covariance (`PLANCK2018_COMPRESSED_COV`).
DESI/BOSS/SH0ES builtins are diagonal in \(\sigma\). Information
criteria (`aic`, `bic`):

\[
\mathrm{AIC}=2k-2\ln\hat L,\qquad
\mathrm{BIC}=k\ln n-2\ln\hat L.
\]

These are **goodness-of-fit / complexity penalties**, not Bayesian
evidence. Evidence \(\ln Z\) is produced only if nested sampling is
run (`run_nested_sampling`).

## 9. Perturbations in this package

`ModifiedCLASS` evolves a **Newtonian sample-\(k\)** entropy
perturbation stage and a BBKS \(P(k)\) normalized to a fiducial
\(\sigma_8\). Phase 0 explicitly labels full CLASS/CAMB \(C_\ell\) as
**future work**. Do not cite those arrays as Boltzmann-accurate CMB
spectra.

## 10. What is not implemented (do not infer)

- A unique derivation “GSL \(\Rightarrow\) de Sitter” without
  \(S_{\max}\) and a closure (see `counterexample_report.md`).
- Local Einstein equations from the Hubble horizon alone.
- Production SN likelihoods without user-supplied CSVs.
- A committed numerical posterior table for \((H_0,k,t_{\mathrm{crit}},\Omega_\Lambda)\).
