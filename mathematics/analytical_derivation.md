# First-Principles Proof Architecture
## Hubble Horizon → de Sitter Attractor → Asymptotic Einstein → On-Shell Action

Version 0 mathematics locked from analytical_solver.py / theory_export (Hubble Horizon -> de Sitter Attractor -> Asymptotic Einstein -> On-Shell Action).

**Wording caution.** Do NOT write '2nd law = de Sitter = acceleration'. Rigorous: GSL AND finite S_max AND thermodynamic closure => stable de Sitter attractor => accelerated expansion.

**Boundary.** NOT DERIVED from Module G (global Hubble-horizon) alone: G_mu_nu + Lambda g_mu_nu = 8 pi G T_mu_nu. Module G yields asymptotic G_mu_nu^(infty) + Lambda_S g_mu_nu = 0 with Lambda_S = 3 pi/(G S_max). Full local EFE is obtained in Module J (Jacobson) as a CONDITIONAL_THEOREM under Clausius, local Rindler equilibrium, and null Raychaudhuri premises.

## Calibrated claims A–F

- **CLAIM A:** S_H=π/(G H²) derived from horizon geometry + BH application
- **CLAIM B:** finite S_max + χ→1 ⇒ H→sqrt(π/(G S_max))>0
- **CLAIM C:** logistic (γ>0): λ=-γ<0 ⇒ stable χ→1
- **CLAIM D:** asymptotic a∼e^{H_∞ t}
- **CLAIM E:** G_μν^(∞)+Λ_S g_μν=0 with Λ_S=3π/(G S_max)
- **CLAIM F:** full local EFE: CONDITIONAL_THEOREM in Module J (Jacobson); NOT derived from Module G alone

## Equation registry (tagged)


### Module `G0_premises`

**(G0.1) [ASSUMED] FLRW_flat**

$$ds^2=-dt^2+a(t)^2 d\mathbf{x}^2\quad(k=0)$$

- Note: Background geometry assumption; NOT derived from horizon thermo.

**(G0.2) [AXIOM] Hubble_def**

$$H\equiv\dot a/a$$

- Note: Definition of the Hubble parameter.

**(G0.3) [JUSTIFIED_ASSUMPTION] R_H_identification**

$$R_H=1/H$$

- Depends on: G0.1, G0.2
- Note: Apparent/Hubble horizon for flat FLRW (c=1). Domain: NOT automatically identical to particle/event horizons. STANDARD_RESULT in flat FLRW apparent-horizon literature; APPLICATION to thermo here is the modeling choice.

**(G0.4) [JUSTIFIED_ASSUMPTION] BH_entropy_application**

$$S_H=A_H/(4G)$$

- Note: Bekenstein–Hawking is a STANDARD_RESULT for black-hole event horizons / some causal horizons. Its APPLICATION to the FLRW apparent horizon is an assumption of this framework (P1). Possible QG area corrections not included.

**(G0.5) [ASSUMED] S_max_finite**

$$0<S_{\max}<\infty$$

- Note: Finite holographic entropy budget. REQUIRED for H_∞>0. Justification: finite de Sitter horizon area ⇒ finite S_H. If S_max=∞ then H_∞→0 and Λ_S→0 (no positive-H attractor).

**(G0.6) [ASSUMED] GSL**

$$\dot S_{\mathrm{tot}}\ge 0$$

- Note: GSL postulate. Constrains thermodynamic arrow; does NOT uniquely determine H(t), χ̇, or a de Sitter endpoint (see counterexamples).


### Module `G1_geometry`

**(G1.1) [AXIOM] R_H**

$$R_H=1/H$$

- Operation: `adopt_horizon_radius`
- Depends on: G0.3

**(G1.2) [DERIVED] A_H_def**

$$A_H=4\pi R_H^2$$

- Operation: `sphere_area`
- Depends on: G1.1
- Note: Do not jump from R_H to S_H — area is intermediate.

**(G1.3) [DERIVED] A_H**

$$A_H=4\pi/H^2$$

- Operation: `substitute_R_H`
- Depends on: G1.1, G1.2

**(G1.4) [DERIVED] S_H_from_A**

$$S_H=A_H/(4G)=(1/(4G))(4\pi/H^2)$$

- Operation: `apply_BH_entropy`
- Depends on: G1.3, G0.4
- Premises: G0.4

**(G1.5) [DERIVED] S_H**

$$S_H=\pi/(G H^2)$$

- Operation: `simplify`
- Depends on: G1.4
- Premises: G0.4
- Note: CLAIM A. π retained. Horizon entropy as geometric function of H.


### Module `G2_monotonicity`

**(G2.1) [DERIVED] S_H_dot**

$$\dot S_H=-(2\pi/G)\,(\dot H/H^3)$$

- Operation: `differentiate_wrt_t_via_chain`
- Depends on: G1.5

**(G2.2) [THEOREM] Theorem_2_2_monotonicity**

$$H>0,G>0\Rightarrow \dot S_H\ge 0\Leftrightarrow \dot H\le 0$$

- Operation: `sign_analysis_of_prefactor`
- Depends on: G2.1
- Note: LEMMA/THEOREM: monotonicity only. Prefactor 2π/(G H^3)>0. Does NOT prove a de Sitter attractor.


### Module `G3_GSL`

**(G3.1) [ASSUMED] S_tot**

$$S_{\mathrm{tot}}=S_H+S_m+S_{\mathrm{fields}}$$

- Note: Additive coarse-graining; microphysics of S_m, S_fields open.

**(G3.2) [DERIVED] P_production**

$$P\equiv\dot S_{\mathrm{tot}}=\dot S_H+\dot S_m+\dot S_{\mathrm{fields}}$$

- Operation: `differentiate`
- Depends on: G3.1

**(G3.3) [ASSUMED] GSL_inequality**

$$P\ge 0$$

- Depends on: G0.6, G3.2
- Note: GSL ⇏ unique χ̇ or unique de Sitter endpoint.

**(G3.4) [LEMMA] GSL_not_unique_closure**

$$\mathrm{GSL}\not\Rightarrow\text{unique }\dot\chi$$

- Operation: `logical_underdetermination`
- Depends on: G3.3
- Note: Proven by counterexample module: many H(t) obey GSL without dS.


### Module `G4_chi`

**(G4.1) [ASSUMED] chi_def**

$$\chi=(S_H-S_{\mathrm{early}})/(S_{\max}-S_{\mathrm{early}})\in[0,1]$$

- Note: Model coordinate — NOT fundamental geometry; NOT equation-of-state w.

**(G4.2) [DERIVED] S_of_chi**

$$S_H=S_{\mathrm{early}}+\chi(S_{\max}-S_{\mathrm{early}})$$

- Operation: `invert_chi_definition`
- Depends on: G4.1


### Module `G5_closure`

**(G5.1) [ASSUMED] Gamma_class**

$$\dot\chi=\Gamma(\chi),\;\Gamma(0)=\Gamma(1)=0,\;\Gamma>0\text{ on }(0,1)$$

- Depends on: G3.4, G4.1
- Note: Allowed closure class. Conclusions H→H_∞ from χ→1 are class-robust; transient details are model-dependent.

**(G5.2) [ASSUMED] Gamma_logistic**

$$\Gamma(\chi)=\gamma\chi(1-\chi),\quad\gamma>0$$

- Depends on: G5.1
- Note: Minimal polynomial member of Γ-class. Motivated by fixed points at 0,1 and Γ>0 on (0,1). NOT derived from GSL.


### Module `G6_EOM`

**(G6.1) [DERIVED] H_of_S**

$$H=\sqrt{\pi/(G S_H)}\quad(H>0)$$

- Operation: `invert_S_H`
- Depends on: G1.5

**(G6.2) [DERIVED] dH_dS**

$$dH/dS_H=-H/(2 S_H)$$

- Operation: `differentiate`
- Depends on: G6.1

**(G6.3) [DERIVED] S_H_dot_chi**

$$\dot S_H=(S_{\max}-S_{\mathrm{early}})\dot\chi$$

- Operation: `chain_rule`
- Depends on: G4.2, G5.2
- Premises: G5.2

**(G6.4) [DERIVED] thermo_EOM**

$$\dot H=-(H/(2S_H))(S_{\max}-S_{\mathrm{early}})\gamma\chi(1-\chi)$$

- Operation: `compose_chain_rule`
- Depends on: G6.2, G6.3, G5.2
- Premises: G5.2, G4.1
- Note: Central thermodynamic background EOM (Version 0).


### Module `G7_equilibrium`

**(G7.1) [DERIVED] fixed_points**

$$\dot\chi=0\Rightarrow\chi=0\text{ or }\chi=1$$

- Operation: `solve_Gamma_equals_zero`
- Depends on: G5.2
- Premises: G5.2

**(G7.2) [CONDITIONAL_THEOREM] H_infty**

$$H_\infty=\sqrt{\pi/(G S_{\max})}>0$$

- Operation: `S_H→S_max then invert`
- Depends on: G1.5, G0.5, G4.2, G5.1
- Premises: G0.5, G5.1
- Note: CLAIM B. Existence of positive H_∞ CONDITIONAL on finite S_max.


### Module `G8_stability`

**(G8.1) [DERIVED] linearization**

$$\chi=1-\varepsilon\;\Rightarrow\;\dot\varepsilon=-\gamma\varepsilon+O(\varepsilon^2)$$

- Operation: `perturb_and_linearize`
- Depends on: G5.2
- Premises: G5.2

**(G8.2) [CONDITIONAL_THEOREM] stability_eigenvalue**

$$\lambda=-\gamma<0\;\Rightarrow\;\chi\to 1\;\Rightarrow\;H\to H_\infty$$

- Operation: `linear_ODE_stability`
- Depends on: G8.1, G7.2, G6.1
- Premises: G5.2, G0.5
- Note: CLAIM C. Stability theorem WITHIN the assumed logistic closure (and more generally for Γ'(1)<0). Numerics do not replace this.

**(G8.3) [CONDITIONAL_THEOREM] chi0_unstable**

$$\chi=0\text{ linearized: }\dot\chi=\gamma\chi\;\Rightarrow\;\lambda=+\gamma>0$$

- Operation: `linearize_about_zero`
- Depends on: G5.2
- Premises: G5.2
- Note: Physical interval [0,1] is forward-invariant for logistic flow.


### Module `G9_expansion`

**(G9.1) [CONDITIONAL_THEOREM] a_deSitter**

$$a(t)=a_0 e^{H_\infty t}$$

- Operation: `integrate_H_equals_H_infty`
- Depends on: G0.2, G7.2, G8.2
- Premises: G0.5, G5.2
- Note: CLAIM D. Accelerated: ä = H_∞² a > 0.


### Module `G10_exact`

**(G10.1) [DERIVED] chi_exact**

$$\chi(t)=\chi_0 e^{\gamma(t-t_0)}/[1-\chi_0+\chi_0 e^{\gamma(t-t_0)}]$$

- Operation: `solve_separable_ODE`
- Depends on: G5.2
- Premises: G5.2
- Note: t_crit is optional reparametrization of C; IC form preferred.

**(G10.2) [CONDITIONAL_THEOREM] late_limits**

$$\lim_{t\to\infty}\chi=1,\;S_H\to S_{\max},\;H\to H_\infty$$

- Operation: `take_limit`
- Depends on: G10.1, G4.2, G6.1, G7.2
- Premises: G5.2, G0.5

**(G10.3) [CONDITIONAL_THEOREM] class_robust_H_infty**

$$(\chi\to 1)\Rightarrow(S_H\to S_{\max})\Rightarrow(H\to\sqrt{\pi/(GS_{\max})})$$

- Operation: `composition_of_continuous_maps`
- Depends on: G5.1, G4.2, G6.1, G0.5
- Premises: G5.1, G0.5
- Note: Independent of logistic specifics: any Γ-class dynamics with χ→1 yields the same H_∞. Logistic is the simplest realization.


### Module `G11_GR_bridge`

**(G11.1) [STANDARD_RESULT] Friedmann**

$$H^2=(8\pi G/3)\rho_{\mathrm{tot}}\quad(k=0)$$

- Note: Imported GR (flat FLRW). NOT derived from global Hubble-horizon entropy. Required for enthalpy/w_S interpretation.

**(G11.2) [STANDARD_RESULT] Raychaudhuri**

$$\dot H=-4\pi G(\rho_{\mathrm{tot}}+P_{\mathrm{tot}})$$

- Note: Imported GR / Raychaudhuri reduction. NOT from global horizon thermo.

**(G11.3) [CONDITIONAL_THEOREM] enthalpy_bridge**

$$\dot H\to 0\Rightarrow\rho_{\mathrm{tot}}+P_{\mathrm{tot}}\to 0$$

- Operation: `substitute_attractor_into_Raychaudhuri`
- Depends on: G11.2, G7.2, G8.2
- Premises: G11.2, G5.2, G0.5

**(G11.4) [CONDITIONAL_THEOREM] w_S_eq**

$$\rho_m,\rho_r\to 0\Rightarrow\rho_S+P_S=0\Rightarrow w_S=-1$$

- Operation: `restrict_to_S_sector`
- Depends on: G11.3
- Premises: G11.2, matter/radiation dilution
- Note: Dilution of ordinary matter/radiation is an additional late-time assumption.


### Module `G12_curvature`

**(G12.1) [STANDARD_RESULT] R_FLRW**

$$R=6(\dot H+2H^2)\quad(k=0)$$

- Note: Flat FLRW Ricci scalar (imported differential geometry).

**(G12.2) [CONDITIONAL_THEOREM] R_inf**

$$R_\infty=12 H_\infty^2=12\pi/(G S_{\max})$$

- Operation: `attractor_limit`
- Depends on: G12.1, G7.2
- Premises: G0.5, G5.1


### Module `G13_Einstein_tensor`

**(G13.1) [STANDARD_RESULT] R_mu_nu_inf**

$$R^{(\infty)}_{\mu\nu}=3 H_\infty^2 g_{\mu\nu}$$

- Note: de Sitter geometry identity (imported), applied at attractor.

**(G13.2) [DERIVED] G_mu_nu_inf**

$$G^{(\infty)}_{\mu\nu}=-3 H_\infty^2 g_{\mu\nu}$$

- Operation: `Einstein_tensor_algebra`
- Depends on: G13.1, G12.2
- Note: G = R_μν - (1/2) R g = 3H²g - 6H²g = -3H²g.


### Module `G14_Lambda`

**(G14.1) [AXIOM] Lambda_S_def**

$$\Lambda_S\equiv 3 H_\infty^2$$

- Note: Definition of thermodynamic effective cosmological constant.

**(G14.2) [CONDITIONAL_THEOREM] Lambda_S_thermo**

$$\Lambda_S=3\pi/(G S_{\max})$$

- Operation: `substitute_H_infty`
- Depends on: G14.1, G7.2
- Premises: G0.5, G0.4
- Note: CLAIM E (half). Model-level consequence, not experimental claim.


### Module `G15_stress`

**(G15.1) [CONDITIONAL_THEOREM] rho_S_inf**

$$\rho_{S,\infty}=3H_\infty^2/(8\pi G)=3/(8 G^2 S_{\max})$$

- Operation: `Friedmann_at_attractor`
- Depends on: G11.1, G7.2, G11.4
- Premises: G11.1, G0.5, matter dilution

**(G15.2) [CONDITIONAL_THEOREM] T_S_inf**

$$T^{(S,\infty)}_{\mu\nu}=-\rho_{S,\infty} g_{\mu\nu}=-(\Lambda_S/(8\pi G))g_{\mu\nu}$$

- Operation: `perfect_fluid_with_w_minus_1`
- Depends on: G15.1, G11.4, G14.1
- Premises: G11.4


### Module `G16_asymptotic_EFE`

**(G16.1) [CONDITIONAL_THEOREM] asymptotic_Einstein**

$$G^{(\infty)}_{\mu\nu}+\Lambda_S g_{\mu\nu}=0$$

- Operation: `combine_Einstein_tensor_and_Lambda_S`
- Depends on: G13.2, G14.1, G15.2
- Premises: G0.5, G5.1, G11.1, G11.2
- Note: CLAIM E. ASYMPTOTIC vacuum Einstein only.

**(G16.2) [UNPROVEN] full_EFE_not_from_Module_G**

$$G_{\mu\nu}+\Lambda g_{\mu\nu}=8\pi G T_{\mu\nu}\quad\text{NOT from Module G; see Module J}$$

- Note: Module G stops at asymptotic Einstein. Full local EFE is derived in Module J as CONDITIONAL_THEOREM J5.1. NOT DERIVED from Module G (global Hubble-horizon) alone: G_mu_nu + Lambda g_mu_nu = 8 pi G T_mu_nu. Module G yields asymptotic G_mu_nu^(infty) + Lambda_S g_mu_nu = 0 with Lambda_S = 3 pi/(G S_max). Full local EFE is obtained in Module J (Jacobson) as a CONDITIONAL_THEOREM under Clausius, local Rindler equilibrium, and null Raychaudhuri premises.


### Module `G17_action`

**(G17.1) [ASSUMED] S_EH**

$$S_{\mathrm{EH}}=\frac{1}{16\pi G}\int\sqrt{-g}(R-2\Lambda)+S_m$$

- Note: ASSUMED dynamical theory whose variation reproduces Einstein eq. NOT reverse-engineered from the attractor.


### Module `G18_Lorentzian`

**(G18.1) [CONDITIONAL_THEOREM] Lorentzian_onshell_density**

$$S_{\mathrm{EH}}^{\mathrm{dS}}/V_4=\Lambda_S/(8\pi G)=3/(8G^2 S_{\max})$$

- Operation: `evaluate_on_shell_dS`
- Depends on: G17.1, G14.2, G12.2
- Premises: G17.1, G0.5
- Note: Flat FLRW V_4 diverges as t→∞; report density / finite region only.


### Module `G19_Euclidean`

**(G19.1) [STANDARD_RESULT] V_S4**

$$V_{S^4}=8\pi^2/(3 H_\infty^4)$$

- Note: Four-volume of round S⁴ with radius 1/H_∞.

**(G19.2) [CONDITIONAL_THEOREM] I_E_dS**

$$I_E^{\mathrm{dS}}=-\pi/(G H_\infty^2)=-S_H(H_\infty)$$

- Operation: `Euclidean_onshell_evaluation`
- Depends on: G17.1, G19.1, G14.1, G1.5
- Premises: G17.1, Euclidean dS continuation

**(G19.3) [CONDITIONAL_THEOREM] I_E_equals_minus_S_max**

$$I_E^{\mathrm{dS}}=-S_{\max}$$

- Operation: `insert_attractor`
- Depends on: G19.2, G7.2
- Premises: G17.1, G0.5, Euclidean dS continuation
- Note: Do NOT say 'the universe chooses de Sitter because QG maximizes Z'. Conditional: under Euclidean dS continuation and EH convention, I_E^{dS}=-S_dS; attractor sets S_dS=S_max => I_E=-S_max. Loop: S_max→H_∞→Λ_S→dS→I_E→-S_max.


### Module `G20_semiclassical`

**(G20.1) [CONJECTURE] Z_dS**

$$Z\sim e^{-I_E}\Rightarrow Z_{\mathrm{dS}}\sim e^{S_{\max}}$$

- Depends on: G19.3
- Premises: semiclassical Euclidean QG, G19.3
- Note: Do NOT say 'the universe chooses de Sitter because QG maximizes Z'. Conditional: under Euclidean dS continuation and EH convention, I_E^{dS}=-S_dS; attractor sets S_dS=S_max => I_E=-S_max.


### Module `G21_extremum`

**(G21.1) [CONDITIONAL_THEOREM] EH_critical_coincides_S_max**

$$S_H=S_{\max}\Leftrightarrow H=H_\infty\Leftrightarrow H^2=\Lambda_S/3$$

- Operation: `compare_critical_points`
- Depends on: G14.2, G7.2, G6.1
- Premises: G14.1, G0.5
- Note: Coincidence of equilibria under shared Λ_S — NOT an independent derivation of EH from entropy alone.


### Module `G_consistency`

**(GC.1) [CONSISTENCY_CHECK] Q_formalism**

$$\dot\rho_m+3H(\rho_m+P_m)=-Q,\;\dot\rho_S+3H(\rho_S+P_S)=+Q$$

- Note: FLAG: paper sourcing with separately conserved matter may break total ∇_μ T^{μν}=0. Use Q-cancellation for Bianchi consistency.


### Module `G_phenomenology`

**(GP.1) [PHENOMENOLOGY] F_Pi**

$$F(\Pi)=F_0+\tfrac12 k\Pi^2$$

- Note: NOT part of attractor proof.

**(GP.2) [PHENOMENOLOGY] w_logistic_pheno**

$$w_S(t)=1/(1+e^{-k(t-t_{\mathrm{crit}})})$$

- Note: χ ≠ w unless an explicit mapping is derived. Observational layer only.


### Module `J0_boundary`

**(J0.1) [AXIOM] module_boundary**

$$\text{Module J $\perp$ Module G (no shared soft premises)}$$

- Note: Do not feed J-results backward into G-theorems as if G alone derived the EFE. Architecture: J => EFE => FLRW => Module G.


### Module `J1_premises`

**(J1.1) [JUSTIFIED_ASSUMPTION] local_Rindler_horizons**

$$\text{At each }p\text{: local Rindler horizon through }p\text{ with approximate Killing vector }\chi^\mu$$

- Depends on: J0.1
- Note: Jacobson: enough local acceleration horizons exist that the null-projection condition for all null k^μ can be imposed.

**(J1.2) [JUSTIFIED_ASSUMPTION] Clausius**

$$\delta Q = T\,dS$$

- Depends on: J1.1
- Note: Clausius relation for heat across the local horizon (equilibrium).

**(J1.3) [JUSTIFIED_ASSUMPTION] entropy_area**

$$dS=\frac{\eta}{4}\,dA\quad(\eta=1/G\;\Rightarrow\;dS=dA/(4G))$$

- Depends on: J1.1
- Note: Horizon entropy proportional to area. Standard BH value eta=1/G. If eta were field-dependent, modified gravity can appear.

**(J1.4) [STANDARD_RESULT] Unruh_temperature**

$$T=\frac{\hbar\kappa}{2\pi k_B}\;\xrightarrow{\mathrm{nat.\,units}}\;T=\frac{\kappa}{2\pi}$$

- Depends on: J1.1
- Note: Unruh temperature for acceleration kappa of the local Rindler horizon.


### Module `J2_geometry`

**(J2.1) [STANDARD_RESULT] Raychaudhuri_null**

$$\frac{d\theta}{d\lambda}=-\frac12\theta^2-\sigma_{\mu\nu}\sigma^{\mu\nu}+\omega_{\mu\nu}\omega^{\mu\nu}-R_{\mu\nu}k^\mu k^\nu$$

- Depends on: J0.1
- Note: k^μ = null generator; λ affine parameter; θ expansion; σ shear; ω vorticity.

**(J2.2) [JUSTIFIED_ASSUMPTION] local_equilibrium**

$$\omega_{\mu\nu}=0,\quad \theta\approx 0,\quad \sigma_{\mu\nu}\approx 0$$

- Depends on: J1.1, J2.1
- Note: At the instant the horizon is chosen through p: vanishing expansion and shear (and vorticity for hypersurface-orthogonal generators).

**(J2.3) [DERIVED] Raychaudhuri_linearized**

$$\frac{d\theta}{d\lambda}\approx -R_{\mu\nu}k^\mu k^\nu$$

- Operation: `drop_quadratic_terms_at_equilibrium`
- Depends on: J2.1, J2.2
- Premises: J2.2

**(J2.4) [STANDARD_RESULT] area_variation**

$$\delta A=\int_H \theta\,d\lambda\,dA$$

- Depends on: J2.1
- Note: First-order area change of a pencil of generators.


### Module `J3_Clausius`

**(J3.1) [JUSTIFIED_ASSUMPTION] heat_flux**

$$\delta Q=\int_H T_{\mu\nu}\chi^\mu d\Sigma^\nu$$

- Depends on: J1.1
- Note: Matter energy flux across the horizon. chi^μ ≈ -κ λ k^μ on the approximate Killing horizon.

**(J3.2) [DERIVED] Clausius_expanded**

$$\delta Q=T\,dS=\frac{\kappa}{2\pi}\cdot\frac{1}{4G}\,\delta A=\frac{\kappa}{8\pi G}\int_H\theta\,d\lambda\,dA$$

- Operation: `substitute_T_and_dS`
- Depends on: J1.2, J1.3, J1.4, J2.4
- Premises: J1.2, J1.3, J1.4

**(J3.3) [DERIVED] integrate_by_parts_theta**

$$\int\theta\,d\lambda\,dA=-\int\lambda\,\frac{d\theta}{d\lambda}\,d\lambda\,dA\approx\int\lambda R_{\mu\nu}k^\mu k^\nu\,d\lambda\,dA$$

- Operation: `integrate_by_parts_plus_Raychaudhuri`
- Depends on: J2.3, J2.4
- Premises: J2.2
- Note: Boundary term vanishes for the local pencil at equilibrium.

**(J3.4) [DERIVED] equate_fluxes**

$$\int_H T_{\mu\nu}k^\mu k^\nu\,\kappa\lambda\,d\lambda\,dA=\frac{\kappa}{8\pi G}\int_H R_{\mu\nu}k^\mu k^\nu\,\lambda\,d\lambda\,dA$$

- Operation: `equate_deltaQ_forms_cancel_kappa`
- Depends on: J3.1, J3.2, J3.3
- Premises: J1.2, J1.3, J1.4, J2.2, J3.1

**(J3.5) [CONDITIONAL_THEOREM] null_projection**

$$T_{\mu\nu}k^\mu k^\nu=\frac{1}{8\pi G}R_{\mu\nu}k^\mu k^\nu\quad\text{for all null }k^\mu$$

- Operation: `localize_integrand_arbitrary_null_k`
- Depends on: J3.4, J1.1
- Premises: J1.1, J1.2, J1.3, J1.4, J2.2, J3.1
- Note: Because local Rindler horizons exist in every null direction at each p, the integrand identity holds for all null k^μ.


### Module `J4_reconstruction`

**(J4.1) [LEMMA] null_tensor_lemma**

$$(S_{\mu\nu}k^\mu k^\nu=0\;\forall\text{ null }k)\;\Rightarrow\;S_{\mu\nu}=f g_{\mu\nu}$$

- Operation: `algebraic_identity_Lorentzian_metric`
- Depends on: J0.1
- Note: Standard GR lemma: a symmetric tensor orthogonal to all null vectors is proportional to the metric.

**(J4.2) [CONDITIONAL_THEOREM] Einstein_up_to_scalar**

$$R_{\mu\nu}-8\pi G\,T_{\mu\nu}=f g_{\mu\nu}$$

- Operation: `apply_null_tensor_lemma_to_S=R-8piG T`
- Depends on: J3.5, J4.1
- Premises: J1.1, J1.2, J1.3, J1.4, J2.2, J3.1

**(J4.3) [STANDARD_RESULT] contracted_Bianchi**

$$\nabla^\mu G_{\mu\nu}=0$$

- Depends on: J0.1
- Note: Contracted Bianchi identity (differential geometry).

**(J4.4) [JUSTIFIED_ASSUMPTION] stress_energy_conservation**

$$\nabla^\mu T_{\mu\nu}=0$$

- Depends on: J0.1
- Note: Local matter conservation. Needed to fix the integration function as a cosmological constant. Interacting sectors must conserve totally.

**(J4.5) [CONDITIONAL_THEOREM] Lambda_integration_constant**

$$\nabla_\nu\bigl(f-\tfrac12 R\bigr)=0\;\Rightarrow\;f-\tfrac12 R=-\Lambda=\mathrm{const}$$

- Operation: `take_divergence_plus_Bianchi`
- Depends on: J4.2, J4.3, J4.4
- Premises: J4.4
- Note: Λ appears as an integration constant of the local thermodynamic derivation — not inserted by hand into the action at this step.


### Module `J5_EFE`

**(J5.1) [CONDITIONAL_THEOREM] Einstein_field_equations**

$$\boxed{G_{\mu\nu}+\Lambda g_{\mu\nu}=8\pi G\,T_{\mu\nu}}$$

- Operation: `rearrange_to_Einstein_form`
- Depends on: J4.2, J4.5
- Premises: J1.1, J1.2, J1.3, J1.4, J2.2, J3.1, J4.4
- Note: CLAIM F (Module J): full local Einstein equation, CONDITIONAL on local Rindler Clausius thermodynamics + Raychaudhuri equilibrium + stress-energy conservation. NOT a Module-G result.

**(J5.2) [CONDITIONAL_THEOREM] Lambda_vs_Lambda_S_pointer**

$$\Lambda\leftrightarrow\Lambda_S:\ \text{see Module K hard-match theorem K5.1}$$

- Operation: `deferred_to_Module_K`
- Depends on: J5.1
- Premises: J5.1, Module K
- Note: Hard match is completed in Module K (K5.1), not left as informal matching.

**(J5.3) [AXIOM] architecture_J_then_G**

$$\text{local thermo}\to\text{EFE}\to\text{FLRW}\to S_H=\pi/(GH^2)\to\text{attractor}\to\Lambda_S$$

- Depends on: J5.1
- Note: Research architecture. Module G remains valid as a standalone asymptotic analysis even if one only imports EFE as STANDARD_RESULT.


### Module `K1_closure_variational`

**(K1.1) [JUSTIFIED_ASSUMPTION] Onsager_mobility**

$$\dot\chi=M(\chi)\,\partial_\chi S_H,\quad M(\chi)\ge 0,\; M(0)=M(1)=0$$

- Note: Bounded entropy-completion coordinate: mobility vanishes at endpoints so chi=0,1 are fixed points. Standard irreversible-thermo structure.

**(K1.2) [DERIVED] dS_dchi**

$$\partial_\chi S_H=S_{\max}-S_{\mathrm{early}}=\Delta S$$

- Operation: `differentiate_S_of_chi`
- Depends on: G4.2

**(K1.3) [JUSTIFIED_ASSUMPTION] minimal_mobility**

$$M_{\mathrm{min}}(\chi)=(\gamma/\Delta S)\,\chi(1-\chi)$$

- Depends on: K1.1
- Note: Minimal nonnegative polynomial mobility with simple zeros at 0,1. Uniqueness within polynomials of degree <=2 with those zeros.

**(K1.4) [CONDITIONAL_THEOREM] Gamma_from_Onsager**

$$\dot\chi=\gamma\chi(1-\chi)$$

- Operation: `M_times_dS_dchi`
- Depends on: K1.1, K1.2, K1.3, G5.2
- Premises: K1.1, K1.3
- Note: Closes Gap 1: logistic is derived from Onsager gradient flow of S_H with minimal endpoint-vanishing mobility — not from GSL alone.

**(K1.5) [CONDITIONAL_THEOREM] Gamma_class_from_mobility**

$$M(0)=M(1)=0,\;M>0\text{ on }(0,1)\;\Rightarrow\;\Gamma=\Delta S\,M\text{ is a }\Gamma\text{-class closure}$$

- Operation: `sign_and_fixed_point_analysis`
- Depends on: K1.1, K1.2
- Premises: K1.1
- Note: Class robustness: attractor H_infty depends only on chi->1, not on M details.


### Module `K2_Smax`

**(K2.1) [STANDARD_RESULT] covariant_entropy_bound**

$$S(\mathcal{L})\le A/(4G)\quad\text{(Bousso / holographic bound)}$$

- Note: Covariant entropy bound: entropy on a lightsheet <= boundary area/4G.

**(K2.2) [CONDITIONAL_THEOREM] apparent_horizon_bound**

$$S_H\le A_H/(4G)=\pi/(G H^2)$$

- Operation: `apply_bound_to_apparent_horizon`
- Depends on: K2.1, G1.3, G0.4
- Premises: K2.1, G0.3
- Note: With equality in the BH/equilibrium case used in Module G.

**(K2.3) [CONDITIONAL_THEOREM] Smax_finite**

$$(0<H_{\mathrm{eq}}<\infty)\;\Rightarrow\;0<S_{\max}=A_{\mathrm{eq}}/(4G)<\infty$$

- Operation: `evaluate_bound_at_equilibrium_horizon`
- Depends on: K2.2, G0.5
- Premises: existence of finite-area causal/apparent equilibrium horizon
- Note: Closes Gap 2: S_max is finite whenever the equilibrium horizon area is finite and positive. Still conditional on such an equilibrium existing (supplied dynamically by Module G closure, or by Module J vacuum+Lambda).


### Module `K3_FLRW`

**(K3.1) [JUSTIFIED_ASSUMPTION] Copernican**

$$\text{spatial homogeneity + isotropy (Copernican principle)}$$

- Note: Cannot be derived from local horizon thermo alone. Standard cosmological symmetry assumption.

**(K3.2) [CONDITIONAL_THEOREM] FLRW_from_EFE_plus_Copernican**

$$(\mathrm{EFE})+\text{homogeneity+isotropy}\;\Rightarrow\;ds^2=-dt^2+a(t)^2 d\mathbf{x}^2_{(k)}$$

- Operation: `symmetry_reduction_of_EFE`
- Depends on: J5.1, K3.1
- Premises: J5.1, K3.1
- Note: Closes Gap 3 at the standard level: FLRW is derived from Module-J EFE plus Copernican symmetry — not from horizon entropy alone.

**(K3.3) [ASSUMED] flatness_choice**

$$k=0\quad\text{(observationally favored / simplest branch)}$$

- Depends on: K3.2
- Note: Spatial flatness remains a branch choice (or inflationary late-time attractor).


### Module `K4_apparent_horizon`

**(K4.1) [STANDARD_RESULT] Hayward_surface_gravity**

$$\kappa_H=\frac{1}{R_H}\bigl(1-\frac{\dot R_H}{2HR_H}\bigr)$$

- Note: Dynamical apparent-horizon surface gravity (Hayward; Cai–Kim).

**(K4.2) [STANDARD_RESULT] unified_first_law**

$$dE_H=T_H dS_H+W dV_H,\quad T_H=|\kappa_H|/(2\pi),\; S_H=A_H/(4G)$$

- Depends on: K4.1, G0.4
- Note: Closes Gap 4: applying S=A/4G to the FLRW apparent horizon is the standard apparent-horizon thermodynamics framework, consistent with Friedmann when the Clausius relation is imposed (Cai–Kim).

**(K4.3) [CONDITIONAL_THEOREM] Clausius_recovers_Friedmann**

$$\delta Q=T_H dS_H\text{ on apparent horizon }\Rightarrow\text{Friedmann eq.}$$

- Operation: `Cai_Kim_argument`
- Depends on: K4.2, J5.1, K3.2
- Premises: K4.2, flat FLRW
- Note: Consistency bridge between Module J local thermo and Module G horizon formulas.


### Module `K5_Lambda_match`

**(K5.1) [CONDITIONAL_THEOREM] hard_match_Lambda_Lambda_S**

$$\bigl(H^2=\Lambda/3\bigr)\wedge\bigl(H\to H_\infty=\sqrt{\pi/(GS_{\max})}\bigr)\;\Rightarrow\;\Lambda=\Lambda_S=\dfrac{3\pi}{GS_{\max}}$$

- Operation: `equate_vacuum_Friedmann_to_attractor_H`
- Depends on: J5.1, G7.2, G11.1, G14.2
- Premises: J5.1, G0.5, G5.1, late-time vacuum
- Note: Closes Gap 5: once Module J supplies EFE with integration constant Lambda and Module G supplies H_infty, vacuum matching FORCES Lambda=Lambda_S. No informal identification step remains.


### Module `K6_action_extremum`

**(K6.1) [DERIVED] reduced_EH_FLRW**

$$S_{\mathrm{EH}}[a]\propto\int dt\,a^3\bigl(6(\dot H+2H^2)-2\Lambda_S\bigr)$$

- Operation: `reduce_EH_on_FLRW`
- Depends on: G17.1, G12.1, G14.1
- Note: Gibbons-Hawking boundary terms omitted in bulk diagnostic.

**(K6.2) [STANDARD_RESULT] EL_recovers_Friedmann**

$$\delta S_{\mathrm{EH}}/\delta a=0\;\Rightarrow\;\text{Friedmann/Raychaudhuri}$$

- Depends on: K6.1
- Note: On-shell continuum of histories includes the de Sitter solution.

**(K6.3) [DERIVED] entropy_potential**

$$\Phi(\chi):=S_H(\chi)=S_{\mathrm{early}}+\chi\Delta S,\quad \Phi'(\chi)=\Delta S>0$$

- Operation: `define_entropy_potential`
- Depends on: G4.2

**(K6.4) [THEOREM] Smax_maximizes_Phi**

$$\max_{\chi\in[0,1]}\Phi(\chi)=\Phi(1)=S_{\max}$$

- Operation: `endpoint_maximum_of_linear_Phi`
- Depends on: K6.3
- Note: Closes Gap 6 (thermodynamic side): S_H=S_max is the unique maximum of the entropy potential on the physical interval.

**(K6.5) [CONDITIONAL_THEOREM] gradient_flow_is_EH_critical_at_endpoint**

$$(\chi\to 1)\Rightarrow(H\to H_\infty)\Rightarrow(H^2=\Lambda_S/3)\text{ is on-shell EH}$$

- Operation: `compose_thermo_flow_with_EH_criticality`
- Depends on: K1.4, K5.1, G21.1, K6.2
- Premises: K1.1, K5.1
- Note: Along the Onsager trajectory, the late-time state is an EH critical point. Full off-shell comparison of S_EH among all a(t) with fixed boundaries still requires GH terms; bulk diagnostic is settled.


### Module `K7_Q_formalism`

**(K7.1) [STANDARD_RESULT] total_conservation**

$$\nabla_\mu T^{\mu\nu}_{\mathrm{tot}}=0\;\Leftrightarrow\;\dot\rho_{\mathrm{tot}}+3H(\rho_{\mathrm{tot}}+P_{\mathrm{tot}})=0$$

- Depends on: J4.4, J5.1
- Note: Follows from Bianchi + EFE (or imposed in Module J premises).

**(K7.2) [DERIVED] split_with_transfer**

$$\dot\rho_m+3H(\rho_m+P_m)=-Q,\quad\dot\rho_S+3H(\rho_S+P_S)=+Q$$

- Operation: `decompose_total_continuity`
- Depends on: K7.1
- Note: Any split into sectors requires Q_m + Q_S = 0 for total conservation.

**(K7.3) [ASSUMED] paper_Q_choice**

$$Q=3H(\rho_m+P_m)\quad\text{(effective full transfer of matter enthalpy)}$$

- Depends on: K7.2, GC.1
- Note: Closes Gap 8 structurally: the paper's source term is reinterpreted as this Q. Matter is then NOT separately conserved; total is.

**(K7.4) [ASSUMED] radiation_passive**

$$\dot\rho_r+4H\rho_r=0\quad(Q_r=0)$$

- Depends on: K7.1


### Module `K8_perturbations`

**(K8.1) [DERIVED] perturbed_conservation**

$$\nabla_\mu T_S^{\mu\nu}=Q^\nu,\quad \nabla_\mu T_m^{\mu\nu}=-Q^\nu$$

- Operation: `linearize_covariant_conservation`
- Depends on: K7.2
- Note: Perturbation equations must descend from this identity — not invented.

**(K8.2) [AXIOM] density_contrast**

$$\delta_S:=\delta\rho_S/\bar\rho_S,\quad \delta_m:=\delta\rho_m/\bar\rho_m$$

- Depends on: K8.1

**(K8.3) [DERIVED] entropy_sector_continuity_linear**

$$\dot{\delta\rho_S}+3H(\delta\rho_S+\delta P_S)+(\bar\rho_S+\bar P_S)(\theta_S/a)=\delta Q$$

- Operation: `FLRW_linearization`
- Depends on: K8.1, K8.2
- Premises: K7.2

**(K8.4) [DERIVED] delta_S_H**

$$\delta S_H=-\frac{2\pi}{G H^3}\delta H\quad(S_H=\pi/(GH^2))$$

- Operation: `linearize_S_H`
- Depends on: G1.5
- Note: Horizon-entropy perturbation tied to Hubble perturbation.

**(K8.5) [DERIVED] isocurvature**

$$S_{mS}:=3(\zeta_m-\zeta_S)\propto \delta_m/\bigl(1+w_m\bigr)-\delta_S/\bigl(1+w_S\bigr)$$

- Operation: `define_relative_entropy_perturbation`
- Depends on: K8.2, K8.3
- Note: Closes Gap 7 at definition+continuity level: isocurvature is derived from sector contrasts. Transfer Q sources relative entropy evolution. Do NOT identify delta S_H with S_mS without gauge-ready map.

**(K8.6) [CONDITIONAL_THEOREM] attractor_isocurvature_decay**

$$w_S\to -1\;\Rightarrow\; c_{a,S}^2\to -1\text{ delicate; }\delta_S\text{ vacuum-like}$$

- Depends on: G11.4, K8.3
- Premises: attractor, no anisotropic stress
- Note: Late-time S-sector behaves as cosmological constant perturbations (smooth).


### Module `K9_pheno_bridge`

**(K9.1) [LEMMA] chi_not_identically_w**

$$\chi\not\equiv w_S\quad\text{as tensorial/EOS quantities}$$

- Operation: `category_mismatch`
- Depends on: G4.1
- Note: Closes Gap 9 (refutation half): chi is an entropy-completion coordinate; w_S:=P_S/rho_S is an equation-of-state parameter. Distinct definitions. Does not depend on the phenomenological sigmoid.

**(K9.2) [CONDITIONAL_THEOREM] asymptotic_agreement**

$$\chi\to 1\text{ and }w_S\to -1\text{ at the attractor (different targets)}$$

- Depends on: G10.2, G11.4
- Premises: G5.1, matter dilution
- Note: Asymptotic diagnostics agree that equilibrium is reached; values differ.

**(K9.3) [PHENOMENOLOGY] mixing_variable_map**

$$\text{If }w_{\mathrm{mix}}:=\chi\text{ is }used\text{ as a regime weight,}\text{ then }w_{\mathrm{mix}}(t)=\chi(t)\text{ by definition (phenomenology)}$$

- Depends on: K9.1, K1.4
- Note: Allowed observational bridge: use chi(t) as the mixing weight replacing the old sigmoid. This is a modeling choice, not a derivation that w_EOS=chi.


### Module `K10_Jacobson_rigor`

**(K10.1) [JUSTIFIED_ASSUMPTION] Clausius_domain**

$$\delta Q=T dS\text{ holds for }instantaneous\text{ local equilibrium pencils}$$

- Depends on: J1.2, J2.2
- Note: Outside equilibrium (large theta, sigma), Clausius may need nonequilibrium entropy production terms.

**(K10.2) [DERIVED] error_control**

$$\frac{d\theta}{d\lambda}+R_{\mu\nu}k^\mu k^\nu=O(\theta^2,\sigma^2)$$

- Operation: `expand_Raychaudhuri`
- Depends on: J2.1, J2.2
- Note: Closes Gap 10 (error control): Einstein null projection holds up to quadratic nonequilibrium corrections that vanish at the equilibrium pencil.

**(K10.3) [STANDARD_RESULT] area_law_corrections**

$$S=\frac{A}{4G}+S_{\mathrm{corr}}\;\Rightarrow\;\text{modified field equations (e.g. }f(R)\text{)}$$

- Depends on: J1.3
- Note: If the area law is corrected, Module J yields modified gravity (Jacobson–type arguments with S≠A/4G). Present EFE assumes S_corr=0.

**(K10.4) [AXIOM] gaps_closure_summary**

$$\text{Gaps 1--10 closed at stated epistemic strength (see notes)}$$

- Note: Residual soft content: Copernican (K3.1), minimal mobility (K1.3), transfer ansatz Q (K7.3), flatness k=0 (K3.3), S_corr=0 (K10.3). These are now explicit — not hidden.


### Module `G_adversarial`

**(GX.1) [LEMMA] GSL_insufficient**

$$\mathrm{GSL}\not\Rightarrow\mathrm{de\ Sitter}$$

- Operation: `counterexample`
- Depends on: G3.3
- Note: GSL ∧ ¬(finite S_max + closure) ⇏ de Sitter. Do NOT write '2nd law = de Sitter = acceleration'. Rigorous: GSL AND finite S_max AND thermodynamic closure => stable de Sitter attractor => accelerated expansion.


### Module `G_limits`

**(GL.1) [DERIVED] S_max_to_infinity**

$$S_{\max}\to\infty\Rightarrow H_\infty\to 0,\;\Lambda_S\to 0$$

- Operation: `limit`
- Depends on: G7.2, G14.2
- Note: No positive-H thermodynamic attractor if entropy budget unbounded.

**(GL.2) [DERIVED] S_max_to_zero**

$$S_{\max}\to 0^+\Rightarrow H_\infty\to\infty$$

- Operation: `limit`
- Depends on: G7.2
- Note: Physically extreme; likely outside trusted EFT regime.
