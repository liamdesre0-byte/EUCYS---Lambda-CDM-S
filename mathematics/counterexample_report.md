# Counterexample report (adversarial module)

Purpose: prevent circular reasoning and slogan inflation.

## GSL_without_de_Sitter

**Statement.** GSL alone does not force a de Sitter endpoint.

**Construction.** Take Ḣ = -α H² with 0<α<2 (power-law a∝t^{2/α} style). Then Ṡ_H = -(2π/G) Ḣ/H³ = (2π/G) α / H > 0, so horizon entropy increases, yet H(t)→0 (not H→H_∞>0). Choose S_m, S_fields so that Ṡ_tot≥0.

**Implication.** GSL ∧ ¬(finite S_max + closure) ⇏ de Sitter. Do NOT write '2nd law = de Sitter = acceleration'. Rigorous: GSL AND finite S_max AND thermodynamic closure => stable de Sitter attractor => accelerated expansion.

## finite_S_max_without_reaching

**Statement.** Finite S_max can exist without the system reaching it.

**Construction.** Set Γ(χ)=0 for all χ (no thermodynamic completion dynamics), or trap χ at χ_trap∈(0,1) with Γ(χ_trap)=0 and Γ'(χ_trap)<0. Then S_H never reaches S_max.

**Implication.** Finite S_max is necessary but not sufficient; need dynamics with χ→1 (Γ-class + basin).

## alternate_closure_same_endpoint

**Statement.** Different closures share H_∞ but differ in transients.

**Construction.** Γ₁=γ χ(1-χ) vs Γ₂=γ χ²(1-χ). Both have χ=1 attracting (under suitable IC) ⇒ same H_∞=sqrt(π/(G S_max)), different χ(t) and H(t) histories.

**Implication.** Asymptotic scale is S_max-controlled; transients are closure-dependent (phenomenology-adjacent).
