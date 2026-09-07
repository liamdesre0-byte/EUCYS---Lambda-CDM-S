# Assumption ledger

Version 0 mathematics locked from analytical_solver.py / theory_export (Hubble Horizon -> de Sitter Attractor -> Asymptotic Einstein -> On-Shell Action).

Do NOT write '2nd law = de Sitter = acceleration'. Rigorous: GSL AND finite S_max AND thermodynamic closure => stable de Sitter attractor => accelerated expansion.

| ID | Tag | Statement | Consequences |
|---|---|---|---|
| G0.1 | ASSUMED | flat FLRW | Background geometry assumption; NOT derived from horizon thermo. |
| G0.2 | AXIOM | H = a_dot/a | Definition of the Hubble parameter. |
| G0.3 | JUSTIFIED_ASSUMPTION | R_H = 1/H | Apparent/Hubble horizon for flat FLRW (c=1). Domain: NOT automatically identical to particle/event horizons. STANDARD_RE |
| G0.4 | JUSTIFIED_ASSUMPTION | S_H = A_H/(4G) | Bekenstein–Hawking is a STANDARD_RESULT for black-hole event horizons / some causal horizons. Its APPLICATION to the FLR |
| G0.5 | ASSUMED | 0 < S_max < ∞ | Finite holographic entropy budget. REQUIRED for H_∞>0. Justification: finite de Sitter horizon area ⇒ finite S_H. If S_m |
| G0.6 | ASSUMED | S_tot_dot >= 0 | GSL postulate. Constrains thermodynamic arrow; does NOT uniquely determine H(t), χ̇, or a de Sitter endpoint (see counte |
| G1.1 | AXIOM | R_H = 1/H |  |
| G3.1 | ASSUMED | S_tot = S_H + S_m + S_fields | Additive coarse-graining; microphysics of S_m, S_fields open. |
| G3.3 | ASSUMED | P >= 0 | GSL ⇏ unique χ̇ or unique de Sitter endpoint. |
| G4.1 | ASSUMED | chi = (S_H-S_early)/(S_max-S_early) | Model coordinate — NOT fundamental geometry; NOT equation-of-state w. |
| G5.1 | ASSUMED | chi_dot = Gamma(chi) in fixed-point class | Allowed closure class. Conclusions H→H_∞ from χ→1 are class-robust; transient details are model-dependent. |
| G5.2 | ASSUMED | Gamma = gamma*chi*(1-chi) | Minimal polynomial member of Γ-class. Motivated by fixed points at 0,1 and Γ>0 on (0,1). NOT derived from GSL. |
| G11.1 | STANDARD_RESULT | H^2 = 8πG/3 rho_tot | Imported GR (flat FLRW). NOT derived from global Hubble-horizon entropy. Required for enthalpy/w_S interpretation. |
| G11.2 | STANDARD_RESULT | Hdot = -4πG (rho_tot + P_tot) | Imported GR / Raychaudhuri reduction. NOT from global horizon thermo. |
| G12.1 | STANDARD_RESULT | R = 6(Hdot + 2 H^2) | Flat FLRW Ricci scalar (imported differential geometry). |
| G13.1 | STANDARD_RESULT | R_mu_nu = 3 H_infty^2 g_mu_nu | de Sitter geometry identity (imported), applied at attractor. |
| G14.1 | AXIOM | Lambda_S := 3 H_infty^2 | Definition of thermodynamic effective cosmological constant. |
| G16.2 | UNPROVEN | full EFE not from global Hubble module (see Module J) | Module G stops at asymptotic Einstein. Full local EFE is derived in Module J as CONDITIONAL_THEOREM J5.1. NOT DERIVED fr |
| G17.1 | ASSUMED | Einstein-Hilbert action | ASSUMED dynamical theory whose variation reproduces Einstein eq. NOT reverse-engineered from the attractor. |
| G19.1 | STANDARD_RESULT | V_S4 = 8π²/(3 H_infty^4) | Four-volume of round S⁴ with radius 1/H_∞. |
| G20.1 | CONJECTURE | Z_dS ~ e^{S_max} | Do NOT say 'the universe chooses de Sitter because QG maximizes Z'. Conditional: under Euclidean dS continuation and EH  |
| GP.1 | PHENOMENOLOGY | F(Pi) = F0 + (1/2) k Pi^2 | NOT part of attractor proof. |
| GP.2 | PHENOMENOLOGY | w_S phenomenological sigmoid | χ ≠ w unless an explicit mapping is derived. Observational layer only. |
| J0.1 | AXIOM | Jacobson module SEPARATE from global attractor proof | Do not feed J-results backward into G-theorems as if G alone derived the EFE. Architecture: J => EFE => FLRW => Module G |
| J1.1 | JUSTIFIED_ASSUMPTION | local Rindler horizons at each spacetime point | Jacobson: enough local acceleration horizons exist that the null-projection condition for all null k^μ can be imposed. |
| J1.2 | JUSTIFIED_ASSUMPTION | delta Q = T dS | Clausius relation for heat across the local horizon (equilibrium). |
| J1.3 | JUSTIFIED_ASSUMPTION | dS = dA/(4G) | Horizon entropy proportional to area. Standard BH value eta=1/G. If eta were field-dependent, modified gravity can appea |
| J1.4 | STANDARD_RESULT | T = kappa/(2 pi) | Unruh temperature for acceleration kappa of the local Rindler horizon. |
| J2.1 | STANDARD_RESULT | null Raychaudhuri equation | k^μ = null generator; λ affine parameter; θ expansion; σ shear; ω vorticity. |
| J2.2 | JUSTIFIED_ASSUMPTION | hypersurface-orthogonal + instantaneous equilibrium | At the instant the horizon is chosen through p: vanishing expansion and shear (and vorticity for hypersurface-orthogonal |
| J2.4 | STANDARD_RESULT | delta A = integral theta d lambda dA | First-order area change of a pencil of generators. |
| J3.1 | JUSTIFIED_ASSUMPTION | delta Q = integral T_mu_nu chi^mu dSigma^nu | Matter energy flux across the horizon. chi^μ ≈ -κ λ k^μ on the approximate Killing horizon. |
| J4.3 | STANDARD_RESULT | div G = 0 | Contracted Bianchi identity (differential geometry). |
| J4.4 | JUSTIFIED_ASSUMPTION | div T = 0 | Local matter conservation. Needed to fix the integration function as a cosmological constant. Interacting sectors must c |
| J5.3 | AXIOM | ultimate architecture: J then G | Research architecture. Module G remains valid as a standalone asymptotic analysis even if one only imports EFE as STANDA |
| K1.1 | JUSTIFIED_ASSUMPTION | chi_dot = M(chi) * dS_H/dchi  (Onsager / gradient flow) | Bounded entropy-completion coordinate: mobility vanishes at endpoints so chi=0,1 are fixed points. Standard irreversible |
| K1.3 | JUSTIFIED_ASSUMPTION | M_min = (gamma/DeltaS) chi(1-chi) | Minimal nonnegative polynomial mobility with simple zeros at 0,1. Uniqueness within polynomials of degree <=2 with those |
| K2.1 | STANDARD_RESULT | S <= A/(4G) on lightsheet L | Covariant entropy bound: entropy on a lightsheet <= boundary area/4G. |
| K3.1 | JUSTIFIED_ASSUMPTION | Copernican: homogeneous + isotropic spatial slices | Cannot be derived from local horizon thermo alone. Standard cosmological symmetry assumption. |
| K3.3 | ASSUMED | k=0 selected | Spatial flatness remains a branch choice (or inflationary late-time attractor). |
| K4.1 | STANDARD_RESULT | Hayward/Cai surface gravity on apparent horizon | Dynamical apparent-horizon surface gravity (Hayward; Cai–Kim). |
| K4.2 | STANDARD_RESULT | unified first law on apparent horizon | Closes Gap 4: applying S=A/4G to the FLRW apparent horizon is the standard apparent-horizon thermodynamics framework, co |
| K6.2 | STANDARD_RESULT | Euler-Lagrange of reduced EH => background GR equations | On-shell continuum of histories includes the de Sitter solution. |
| K7.1 | STANDARD_RESULT | total stress-energy conservation | Follows from Bianchi + EFE (or imposed in Module J premises). |
| K7.3 | ASSUMED | Q = 3H(rho_m+P_m) | Closes Gap 8 structurally: the paper's source term is reinterpreted as this Q. Matter is then NOT separately conserved;  |
| K7.4 | ASSUMED | radiation separately conserved unless interaction specified |  |
| K8.2 | AXIOM | density contrasts |  |
| K9.3 | PHENOMENOLOGY | optional: identify mixing weight with chi (PHENO choice) | Allowed observational bridge: use chi(t) as the mixing weight replacing the old sigmoid. This is a modeling choice, not  |
| K10.1 | JUSTIFIED_ASSUMPTION | Clausius domain: local equilibrium horizons | Outside equilibrium (large theta, sigma), Clausius may need nonequilibrium entropy production terms. |
| K10.3 | STANDARD_RESULT | entropy corrections => modified gravity | If the area law is corrected, Module J yields modified gravity (Jacobson–type arguments with S≠A/4G). Present EFE assume |
| K10.4 | AXIOM | Module K gap-closure summary | Residual soft content: Copernican (K3.1), minimal mobility (K1.3), transfer ansatz Q (K7.3), flatness k=0 (K3.3), S_corr |
