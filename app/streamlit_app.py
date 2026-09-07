"""ΛCDM+S interactive explorer — uses the packaged research model.

Run from the repository root:

    python -m streamlit run app/streamlit_app.py

The background integration is ``lcdm_plus_s.bayesian_validation.solve_background``.
This app does not re-code Friedmann equations and does not display unpublished
MCMC posteriors as if they were computed here.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import streamlit as st

from lcdm_plus_s.bayesian_validation import (
    BackgroundParams,
    FIDUCIAL_H0,
    FIDUCIAL_K_GYR,
    FIDUCIAL_OMEGA_LAMBDA,
    FIDUCIAL_T_CRIT_GYR,
    KMSMPC_TO_INVGYR,
    POSTULATES,
    TABLE1_PRIORS,
    distance_modulus,
    logistic_weight,
    solve_background,
)

st.set_page_config(
    page_title="ΛCDM+S — Entropy-Extended Cosmology",
    layout="wide",
)

st.markdown(
    """
# ΛCDM+S — Entropy-Extended Cosmology

Can cosmic acceleration emerge from horizon thermodynamics?

[Paper](../paper/README.md) · [Mathematics](../mathematics/README.md)
· [Source](../src/lcdm_plus_s/) · [Validation](../REPRODUCING.md)
"""
)
st.caption(
    "Live background from `lcdm_plus_s.solve_background`. "
    "ΛCDM is the same integrator with `lcdm_limit=True`. "
    "Curves are the model at the slider values, not posterior draws."
)


@st.cache_data(show_spinner=False)
def integrate(H0: float, Omega_L: float, k: float, t_crit: float,
              lcdm_limit: bool, nsteps: int):
    params = BackgroundParams(
        H0_kms_mpc=H0,
        Omega_Lambda=Omega_L,
        k_gyr=k,
        t_crit_gyr=t_crit,
        lcdm_limit=lcdm_limit,
    )
    sol = solve_background(params, nsteps=nsteps)
    z = [1.0 / a - 1.0 for a in sol.a]
    H_kms = [h / KMSMPC_TO_INVGYR for h in sol.H]
    return {
        "t": sol.t,
        "a": sol.a,
        "z": z,
        "H_kms": H_kms,
        "w_S": sol.w_S,
        "chi": sol.chi,
        "t0": sol.t0,
        "chi0": sol.chi0,
        "omega_m0": params.omega_m0,
    }


def past_branch(bundle: dict) -> dict:
    t0 = bundle["t0"]
    mask = [t <= t0 + 1e-9 for t in bundle["t"]]
    out = {}
    for key, series in bundle.items():
        if isinstance(series, list):
            out[key] = [v for v, keep in zip(series, mask) if keep]
        else:
            out[key] = series
    return out


with st.sidebar:
    st.header("Parameters θ")
    st.write("Table 1 fiducials are the slider defaults.")
    H0 = st.slider("H₀ [km s⁻¹ Mpc⁻¹]", 60.0, 80.0, float(FIDUCIAL_H0), 0.1)
    Omega_L = st.slider("Ω_Λ ≡ Ω_S,0", 0.50, 0.85, float(FIDUCIAL_OMEGA_LAMBDA), 0.001)
    k = st.slider("k ≡ γ [Gyr⁻¹]", 0.05, 1.20, float(FIDUCIAL_K_GYR), 0.01)
    t_crit = st.slider("t_crit [Gyr]", 8.0, 22.0, float(FIDUCIAL_T_CRIT_GYR), 0.1)
    nsteps = st.select_slider("Integrator steps", options=[800, 1500, 3000, 5000], value=1500)
    st.markdown(
        f"Derived Ω_m,0 = `{1.0 - Omega_L - 9.0e-5:.4f}` "
        f"(flatness: Ω_m + Ω_r + Ω_S,0 = 1)."
    )
    st.markdown("**Table 1 Gaussian priors (code)**")
    for row in TABLE1_PRIORS:
        st.write(f"{row.symbol}: μ={row.mu}, σ={row.sigma} [{row.units}]")

col_a, col_b = st.columns(2)
with st.spinner("Integrating ΛCDM+S and ΛCDM backgrounds…"):
    s = integrate(H0, Omega_L, k, t_crit, False, int(nsteps))
    l = integrate(H0, Omega_L, k, t_crit, True, int(nsteps))
s_past = past_branch(s)
l_past = past_branch(l)

m1, m2, m3, m4 = st.columns(4)
m1.metric("t₀ (ΛCDM+S)", f"{s['t0']:.3f} Gyr")
m2.metric("χ₀ = χ(t₀)", f"{s['chi0']:.4f}")
m3.metric("χ(t_crit)", f"{logistic_weight(t_crit, k, t_crit):.3f}")
m4.metric("t₀ (ΛCDM)", f"{l['t0']:.3f} Gyr")
if t_crit > s["t0"]:
    st.info(
        "Fiducial t_crit is after today, so χ₀ < 1/2. The code treats this as "
        "intentional: the χ=1 de Sitter attractor still lies in the future."
    )

tab_h, tab_chi, tab_w, tab_mu, tab_eq = st.tabs(
    ["H(z)", "χ(t)", "w_S(t)", "μ(z)", "Equations"]
)

with tab_h:
    st.subheader("Hubble parameter")
    zs = np.array(s_past["z"])
    order = np.argsort(zs)
    chart = {
        "z": zs[order],
        "ΛCDM+S": np.array(s_past["H_kms"])[order],
        "ΛCDM": np.array(l_past["H_kms"])[order],
    }
    st.line_chart(chart, x="z", y=["ΛCDM+S", "ΛCDM"])
    st.caption("H(z) in km s⁻¹ Mpc⁻¹. Past light cone only (t ≤ t₀).")

with tab_chi:
    st.subheader("Logistic transition")
    chart = {
        "t [Gyr]": s["t"],
        "χ(t) ΛCDM+S": s["chi"],
        "χ = 1/2": [0.5] * len(s["t"]),
    }
    st.line_chart(chart, x="t [Gyr]", y=["χ(t) ΛCDM+S", "χ = 1/2"])
    st.caption("χ(t) = 1 / (1 + exp[−k(t − t_crit)]). Vertical science: χ(t_crit)=1/2.")

with tab_w:
    st.subheader("Entropy-sector equation of state")
    chart = {
        "t [Gyr]": s["t"],
        "w_S ΛCDM+S": s["w_S"],
        "w_Λ = −1": [-1.0] * len(s["t"]),
    }
    st.line_chart(chart, x="t [Gyr]", y=["w_S ΛCDM+S", "w_Λ = −1"])
    st.caption(
        r"Layer-1 EOS: \(w_S = -1 + \Delta S\, k\, \chi(1-\chi)/(3 H S_H)\). "
        "ΛCDM has w = −1 by construction (`lcdm_limit=True`)."
    )

with tab_mu:
    st.subheader("Distance modulus (background prediction)")
    z_grid = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5]
    params_s = BackgroundParams(
        H0_kms_mpc=H0, Omega_Lambda=Omega_L, k_gyr=k, t_crit_gyr=t_crit,
        lcdm_limit=False,
    )
    params_l = BackgroundParams(
        H0_kms_mpc=H0, Omega_Lambda=Omega_L, k_gyr=k, t_crit_gyr=t_crit,
        lcdm_limit=True,
    )
    sol_s = solve_background(params_s, nsteps=int(nsteps))
    sol_l = solve_background(params_l, nsteps=int(nsteps))

    def mu_from_sol(sol, z):
        def Hz(zz: float) -> float:
            return sol.hubble_of_z(zz)
        return distance_modulus(Hz, z)

    chart = {
        "z": z_grid,
        "ΛCDM+S": [mu_from_sol(sol_s, z) for z in z_grid],
        "ΛCDM": [mu_from_sol(sol_l, z) for z in z_grid],
    }
    st.line_chart(chart, x="z", y=["ΛCDM+S", "ΛCDM"])
    st.caption(r"\(\mu(z) = 5\log_{10}(d_L/\mathrm{Mpc}) + 25\).")

with tab_eq:
    st.markdown(
        r"""
The integrator in this app is the packaged Bayesian background, not a
simplified cartoon.

\[
\dot\chi = k\,\chi(1-\chi),\qquad
\chi(t)=\frac{1}{1+e^{-k(t-t_{\mathrm{crit}})}}
\]

\[
E^2(a)=\Omega_{r,0}a^{-4}+\Omega_{m,0}a^{-3}
+\Omega_{\Lambda}\frac{\chi(t)}{\chi_0}
\]

with \(\Omega_{\Lambda}\equiv\Omega_{S,0}\) and \(\chi_0=\chi(t_0)\).
Setting `lcdm_limit=True` freezes \(\Omega_S(z)=\Omega_\Lambda\).

Full MCMC, covariance likelihoods, and nested sampling are CLI tools:

```text
python scripts/run_validation.py --entropy-sector
```
"""
    )
    st.subheader("Postulates (from the research code)")
    for p in POSTULATES:
        st.markdown(f"**{p.number}. {p.name}.** {p.statement}")
        st.code(p.equation)

st.markdown("---")
st.caption(
    "Scope: background-level thermodynamic cosmology. This dashboard does not "
    "claim that the model disproves dark energy, replace Boltzmann codes, or "
    "reproduce a specific unpublished posterior."
)
