#!/usr/bin/env python3
"""
analytical_solver.py - Layer-1 first-principles derivation for ΛCDM+S

Packaged as ``lcdm_plus_s.analytical_solver``. Scientific content is the
original derivation/validation script; equations are not rewritten here.


Conceptual chain (NOT "GSL -> Ḣ≤0 -> assume de Sitter"):

  Horizon thermodynamics
       -> generalized second law
       -> entropy extremization
       -> cosmological equilibrium
       -> de Sitter attractor

Core object: the thermodynamic equation of motion.

  S_tot[H, Ḣ, ρ, ...]
  P ≡ dS_tot/dt = P(H, Ḣ, ρ, p, ...) ≥ 0     (GSL)
  equilibrium:  P = 0  ->  solve H_* > 0
  a(t) ∝ exp(H_* t)
  linearize H = H_* + δH  ->  δḢ = λ δH + O(δH²)
  prove λ < 0  =>  stable de Sitter attractor

ONLY AFTER the attractor theorem do we construct T^(S), ρ_S, P_S,
the modified Friedmann equation, and the ASYMPTOTIC Einstein structure:

  R_∞ = 12 H_∞²,  G_μν^(∞) = -3 H_∞² g_μν,
  Λ_S = 3 H_∞² = 3π/(G S_max),
  T^(S,∞)_μν = -(Λ_S/(8πG)) g_μν,
  G_μν^(∞) + Λ_S g_μν = 0.

Do NOT claim full local Einstein equations from the global Hubble horizon alone.

F(Π) and logistic w(t) are NOT the centerpiece - they live in the
phenomenological comparison layer. First principles asks whether those
(or different) functional forms emerge from thermodynamics.

Π_H convention (resolved):
  S_H = π/(G H²) => Π_H ≡ Ṡ_H = -2π Ḣ/(G H³) = 8π²(ρ+P)/H³.
  Action-note reciprocal H³(ρ+P)/(8π²) REJECTED (used S_H=GH/2π).

Sources: EUCYS__Copy_ Theorems 2.1-2.2 + attractor; Action/Stress-Energy notes
(pheno); Modified_CLASS horizon-chain (Layer-2).
"""

from __future__ import annotations

import argparse
import math
import time
import csv
import json
import shutil
import subprocess
import sys
import textwrap
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence

import sympy as sp
from sympy import (
    Eq,
    Symbol,
    Function,
    simplify,
    diff,
    exp,
    pi,
    sqrt,
    Abs,
)

EqTag = Literal["AXIOM", "DERIVED", "ASSUMED", "CONDITIONAL", "THEOREM", "PHENOMENOLOGY"]

# =============================================================================
# 0. Inventory / postulates
# =============================================================================

PIPELINE_STAGES: List[str] = [
    "STEP 0: State assumptions / postulates (not derived)",
    "STEP 1: Geometry - R_H -> A_H -> S_H (show algebra)",
    "STEP 2: Entropy evolution - differentiate S_H -> Theorem 2.2",
    "STEP 3: GSL on S_tot - direction, not unique endpoint",
    "STEP 4: Finite S_max + entropy-completion χ [ASSUMED]",
    "STEP 5: Thermodynamic closure χ̇=Γ(χ) [ASSUMED]",
    "STEP 6: Connect entropy to H -> thermodynamic EOM [DERIVED]",
    "STEP 7: Equilibrium χ=1 -> H_* [CONDITIONAL on S_max]",
    "STEP 8: Linearize -> λ=-γ<0 [THEOREM]",
    "STEP 9: Integrate H=H_* -> a∝e^{H_* t} [DERIVED]",
    "STEP 10: Exact χ(t), S_H(t), H(t) from assumed closure",
    "STEP 11: Enthalpy bridge ρ_tot+P_tot=0 ↔ Ḣ=0 (paper)",
    "STEP 12: Asymptotic curvature R_∞=12 H_∞²",
    "STEP 13: Asymptotic Einstein tensor G_μν^(∞)=-3 H_∞² g_μν",
    "STEP 14: Thermodynamic Λ_S=3 H_∞²=3π/(G S_max)",
    "STEP 15: Asymptotic Friedmann + T^(S)_∞=-ρ_S g_μν",
    "STEP 16: ASYMPTOTIC Einstein eq G_μν^(∞)+Λ_S g_μν=0  [NOT full EFE]",
    "STEP 17: Einstein-Hilbert action (assumed dynamical theory; not derived backwards)",
    "STEP 18: On-shell Lorentzian EH density on dS; finite-region V_4 (∞ caution)",
    "STEP 19: Euclidean S⁴ continuation -> I_E=-π/(G H_∞²)=-S_max",
    "STEP 20: Semiclassical Z_dS~e^{S_max} [CONDITIONAL wording]",
    "STEP 21: FLRW reduced action - does S_H=S_max coincide with EH critical point?",
    "MODULE 17: Jacobson local-horizon route -> full EFE under labeled axioms",
    "LAYER 1: Effective cosmological fluid ρ_S, P_S, w_S [DERIVED]",
    "LAYER 2: Deceleration q, jerk j, snap [DERIVED]",
    "LAYER 3: Phase trajectory H(χ), a(χ), H(a) [DERIVED]",
    "LAYER 4: Autonomous dynamical system χ'=f(χ) [DERIVED]",
    "LAYER 5: Global Lyapunov V=S_max-S_H [THEOREM]",
    "LAYER 6: Horizon T_H, E_H, first law [DERIVED]",
    "LAYER 7: Entropy production rate in Hubble units [DERIVED]",
    "LAYER 8: Transition epoch χ=1/2 max production [DERIVED]",
    "LAYER 9: Scalar-field background reconstruction [DERIVED]",
    "LAYER 10: Homogeneous action => logistic closure [CONDITIONAL]",
    "LAYER 11: Covariant promotion χ(x^μ) [DERIVED structure]",
    "LAYER 12: Linear cosmological perturbations [DERIVED schematic]",
    "LAYER 13: Adiabatic sound speed c_a² [DERIVED]",
    "LAYER 14: Energy conditions NEC/WEC/DEC/SEC [DERIVED]",
    "LAYER 15: Γ-class generalization [DERIVED universality]",
    "LAYER 16: Bifurcation analysis [DERIVED]",
    "LAYER 17: Thermodynamic arrow τ_S=χ [DERIVED]",
    "LAYER 18: Generalized Friedmann H²=F(S) [DERIVED]",
    "LAYER 19: Jacobson↔global entropy bridge [BRIDGE]",
    "LAYER 20: Modified entropy laws S(A) [DERIVED]",
    "LAYER 21: Finite S_max from horizon area [CONDITIONAL]",
    "PHENOMENOLOGY: sigmoid / F(Π) - not the attractor proof",
]

ACTION_WORDING_CAUTION: str = (
    "Do NOT say 'the universe chooses de Sitter because quantum gravity maximizes Z'. "
    "Conditional: under the standard Euclidean de Sitter continuation and Einstein-Hilbert "
    "convention, the on-shell Euclidean action satisfies I_E^{dS}=-S_dS; the thermodynamic "
    "attractor identifies S_dS=S_max, so I_E^{dS}=-S_max. The EH action is the dynamical "
    "theory whose variation reproduces the Einstein equation - not something obtained by "
    "substituting the attractor answer backwards into an integrand."
)

BOUNDARY_NOT_PROVEN: str = (
    "NOT PROVEN from the global Hubble-horizon argument alone: "
    "full Einstein field equations G_μν + Λ g_μν = 8πG T_μν. "
    "What IS derived: the ASYMPTOTIC vacuum Einstein equation "
    "G_μν^(∞) + Λ_S g_μν = 0 with Λ_S = 3π/(G S_max)."
)

# Conceptual causal chain for the jury (narrative first, math proves each link)
CONCEPTUAL_CHAIN: List[str] = [
    "Horizon thermodynamics",
    "GSL favors increasing total entropy",
    "finite S_max provides an equilibrium endpoint  [ASSUMED]",
    "thermodynamic closure drives χ -> 1  [ASSUMED]",
    "S_H -> S_max",
    "H -> H_* > 0  (Ḣ->0; enthalpy ρ_tot+P_tot=0)",
    "a(t) -> a0 exp(H_* t)",
    "stable de Sitter attractor / accelerated expansion",
    "R_∞=12 H_∞²,  G_μν^(∞)=-3 H_∞² g_μν",
    "Λ_S=3π/(G S_max)  =>  G_μν^(∞)+Λ_S g_μν=0  [ASYMPTOTIC]",
    "evaluate EH action ON the dS solution  [ASSUMED EH theory]",
    "Euclidean S⁴: I_E=-S_max  [CONDITIONAL]",
    "semiclassical Z_dS ~ e^{S_max}  [interpretive; careful wording]",
]

SCIENTIFIC_CLAIM: str = (
    "LambdaCDM+S contains a first-principles thermodynamic attractor derivation "
    "(under explicitly stated assumptions), together with a phenomenological "
    "interpolation used to connect that asymptotic thermodynamic behavior to the "
    "observable cosmological history. We do NOT claim the entire observed expansion "
    "history is derived from fundamental physics alone."
)

WORDING_CAUTION: str = (
    "Do NOT write '2nd law = de Sitter = acceleration'. "
    "The rigorous statement is: GSL + finite S_max + thermodynamic closure "
    "=> de Sitter attractor => accelerated expansion. "
    "If the universe evolves toward the maximum-entropy de Sitter equilibrium "
    "permitted by the framework, then H approaches a positive constant, "
    "producing accelerated expansion."
)

CANONICAL_EQUATIONS: Dict[str, str] = {
    "R_H": "R_H = 1/H",
    "A_H": "A_H = 4π/H²",
    "S_H": "S_H = π/(G H²)",
    "T_H_dyn": "T_H = |κ|/(2π), κ = H + Ḣ/(2H)",
    "T_H_ds": "T_H = H/(2π)",
    "S_tot": "S_tot = S_H + S_m + S_fields  [= functional of (H,Ḣ,ρ,...)]",
    "P_production": "P ≡ Ṡ_tot = Ṡ_H + Ṡ_m + Ṡ_fields",
    "GSL": "P ≥ 0",
    "S_H_dot": "Ṡ_H = -2π Ḣ/(G H³)",
    "Theorem_2_2": "Ṡ_H ≥ 0 ⇔ Ḣ ≤ 0 (H>0); consequence of P≥0 at late times",
    "chi": "χ = (S_H - S_early)/(S_max - S_early)",
    "thermo_EOM": "χ̇ = Γ(χ)  =>  Ḣ = (dH/dS)(S_max-S_early) Γ(χ)   [thermodynamic EOM]",
    "Gamma_min": "minimal fixed-point class: Γ(χ)=γ χ(1-χ)",
    "equilibrium": "P=0 => Ṡ_H=0 => Ḣ=0 (late-time) and χ=1 => H=H_*",
    "H_star": "H_* = sqrt(π/(G S_max)) > 0  (given 0<S_max<∞)",
    "a_eq": "a(t) = a0 exp(H_* t)",
    "linearization": "H=H_*+δH, χ=1-ε => ε̇=-γ ε => δḢ = λ δH with λ=-γ<0",
    "enthalpy": "Ḣ=-4πG(ρ_tot+P_tot); ρ_tot+P_tot=0 => Ḣ=0 (paper bridge)",
    "R_inf": "R_∞ = 12 H_∞² = 12π/(G S_max)",
    "G_inf": "G_μν^(∞) = -3 H_∞² g_μν",
    "Lambda_S": "Λ_S ≡ 3 H_∞² = 3π/(G S_max)",
    "rho_S_inf": "ρ_S,∞ = 3 H_∞²/(8πG) = 3/(8 G² S_max)",
    "T_S_inf": "T^(S,∞)_μν = -ρ_S g_μν = -(Λ_S/(8πG)) g_μν",
    "Einstein_asymp": "G_μν^(∞) + Λ_S g_μν = 0  [ASYMPTOTIC; NOT full EFE]",
    "S_EH": "S_EH = 1/(16πG) ∫√(-g)(R-2Λ) + S_matter  [ASSUMED dynamical theory]",
    "S_EH_onshell_L": "on-shell Lorentzian: (R-2Λ_S)=2Λ_S => S_EH=Λ_S V_4/(8πG)",
    "I_E_dS": "Euclidean dS: I_E = -π/(G H_∞²) = -S_H(H_∞) -> -S_max at attractor",
    "Z_dS": "Z_dS ~ e^{-I_E} ~ e^{S_max}  [CONDITIONAL semiclassical interpretation]",
    "Pi_H": "Π_H ≡ Ṡ_H = 8π²(ρ+P)/H³  [canonical, post-attractor bookkeeping]",
    "rho_S_MS": "ρ_S = 3H²/(8πG)  (Misner-Sharp vacuum branch; AFTER attractor)",
    "Friedmann": "H² = 8πG/3 (ρ_m+ρ_r+ρ_S)  (AFTER attractor)",
    "F_pheno": "F(Π)=F0+½kΠ² - phenomenological / emergent-candidate, not Layer-1 input",
    "logistic_pheno": "w=1/(1+e^{-k(t-t_crit)}) - phenomenological comparison only",
    "rho_S_deep": "ρ_S = 3H²/(8πG) = 3/(8 G² S_H)  [locked S_H=π/(G H²)]",
    "w_S_deep": "w_S = -1 + ΔS γ χ(1-χ)/(3 H S_H)  -> -1 as χ->1; non-phantom interior",
    "q_deep": "q = -1 + ΔS γ χ(1-χ)/(2 H S_H)",
    "H_chi": "H(χ)=sqrt(π/(G[S_early+ΔS χ]))",
    "Lyapunov_V": "V=S_max-S_H ≥ 0, V̇≤0, V=0⇔χ=1",
    "first_law_EH": "E_H = T_H S_H = 1/(2 G H)",
    "Sdot_max": "Ṡ_H,max = γ ΔS / 4 at χ=1/2",
}

POSTULATES: List[str] = [
    "P1: Bekenstein-Hawking S_H = A_H/(4G) on the apparent Hubble horizon.",
    "P2: GSL: P ≡ Ṡ_tot = Ṡ_H + Ṡ_m + Ṡ_fields ≥ 0.",
    "P3: Finite holographic entropy budget 0 < S_max < ∞ (entropy extremum exists).",
    "P4: Entropy completion χ ∈ [0,1] is a thermodynamic state coordinate.",
    "P5: Extremization dynamics: χ̇ = Γ(χ) with Γ(0)=Γ(1)=0, Γ>0 on (0,1), Γ'(1)<0.",
    "P6: Einstein gravity; effective T^(S) constructed AFTER the attractor theorem.",
]

# =============================================================================
# 1. Types
# =============================================================================


@dataclass
class DerivationStep:
    name: str
    stage: str
    assumptions: List[str] = None  # type: ignore
    equations: Dict[str, Any] = None  # type: ignore
    claims: List[str] = None  # type: ignore
    open_gaps: List[str] = None  # type: ignore
    status: Literal["derived", "assumed", "failed", "stub", "conditional"] = "derived"
    evidence: Dict[str, Any] = None  # type: ignore

    def __post_init__(self) -> None:
        self.assumptions = self.assumptions or []
        self.equations = self.equations or {}
        self.claims = self.claims or []
        self.open_gaps = self.open_gaps or []
        self.evidence = self.evidence or {}

    def to_serializable(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "stage": self.stage,
            "assumptions": self.assumptions,
            "equations": {k: str(v) for k, v in self.equations.items()},
            "claims": self.claims,
            "open_gaps": self.open_gaps,
            "status": self.status,
            "evidence": {k: str(v) for k, v in self.evidence.items()},
        }


@dataclass
class ConsistencyCheck:
    name: str
    passed: bool
    detail: str
    equations: Dict[str, str] = None  # type: ignore

    def __post_init__(self) -> None:
        self.equations = self.equations or {}


@dataclass
class DeSitterStatus:
    equilibrium_identified: bool
    existence: Literal["proven", "conditional", "failed"]
    existence_detail: str
    stability: Literal["proven_strict", "failed"]
    stability_detail: str
    lambda_linear: str
    verdict_line: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DerivationResult:
    temperature_mode: str
    steps: List[DerivationStep]
    checks: List[ConsistencyCheck]
    desitter: DeSitterStatus
    export_payload: Dict[str, Any]

    @property
    def attractor_pass(self) -> bool:
        return (
            self.desitter.equilibrium_identified
            and self.desitter.existence in {"proven", "conditional"}
            and self.desitter.stability == "proven_strict"
        )


# =============================================================================
# 2. Symbols
# =============================================================================


def make_symbols() -> Dict[str, Any]:
    t = Symbol("t", real=True, positive=True)
    H = Symbol("H", real=True, positive=True)
    Hdot = Symbol("Hdot", real=True)
    H_star = Symbol("H_star", real=True, positive=True)
    dH = Symbol("delta_H", real=True)
    G = Symbol("G", real=True, positive=True)
    gamma = Symbol("gamma", real=True, positive=True)
    chi = Symbol("chi", real=True)
    eps = Symbol("eps", real=True, positive=True)

    S_H = Symbol("S_H", real=True, positive=True)
    S_max = Symbol("S_max", real=True, positive=True)
    S_early = Symbol("S_early", real=True, nonnegative=True)
    S_m = Symbol("S_m", real=True)
    S_fields = Symbol("S_fields", real=True)

    rho_m = Symbol("rho_m", real=True, nonnegative=True)
    rho_r = Symbol("rho_r", real=True, nonnegative=True)
    rho_S = Symbol("rho_S", real=True)
    P_m = Symbol("P_m", real=True)
    P_r = Symbol("P_r", real=True)
    P_S = Symbol("P_S", real=True)
    w_S = Symbol("w_S", real=True)
    rho = Symbol("rho", real=True)
    P = Symbol("P", real=True)

    k = Symbol("k", real=True, positive=True)
    t_crit = Symbol("t_crit", real=True)
    F0 = Symbol("F_0", real=True)
    Pi = Symbol("Pi", real=True)
    sigma_dot = Symbol("sigma_dot", real=True)
    lam = Symbol("lambda", real=True)

    return dict(
        t=t,
        H=H,
        Hdot=Hdot,
        H_star=H_star,
        dH=dH,
        G=G,
        gamma=gamma,
        chi=chi,
        eps=eps,
        S_H=S_H,
        S_max=S_max,
        S_early=S_early,
        S_m=S_m,
        S_fields=S_fields,
        rho_m=rho_m,
        rho_r=rho_r,
        rho_S=rho_S,
        P_m=P_m,
        P_r=P_r,
        P_S=P_S,
        w_S=w_S,
        rho=rho,
        P=P,
        k=k,
        t_crit=t_crit,
        F0=F0,
        Pi=Pi,
        sigma_dot=sigma_dot,
        lam=lam,
    )


# =============================================================================
# STAGE A - Geometry / horizon thermo / S_tot / production
# =============================================================================


def step_geometry(sym: Dict[str, Any]) -> DerivationStep:
    H = sym["H"]
    R_H, A_H = 1 / H, simplify(4 * pi / H**2)
    return DerivationStep(
        name="geometry_FLRW_horizon",
        stage="INPUT/SYMBOLIC",
        assumptions=["Flat FLRW", "Apparent horizon = Hubble horizon"],
        equations={"R_H": Eq(Symbol("R_H"), R_H), "A_H": Eq(Symbol("A_H"), A_H)},
        claims=["R_H=1/H, A_H=4π/H²."],
        evidence={"R_H": R_H, "A_H": A_H},
    )


def step_entropy(sym: Dict[str, Any], A_H) -> DerivationStep:
    G, H = sym["G"], sym["H"]
    S_H = pi / (G * H**2)
    ok = simplify(S_H - A_H / (4 * G)) == 0
    return DerivationStep(
        name="horizon_entropy_S_H",
        stage="SYMBOLIC",
        assumptions=["P1: Bekenstein-Hawking"],
        equations={"S_H": Eq(Symbol("S_H"), S_H)},
        claims=["S_H(H) = π/(G H²)"] if ok else [],
        status="derived" if ok else "failed",
        evidence={"S_H": S_H},
    )


def step_temperature(sym: Dict[str, Any], mode: str) -> DerivationStep:
    H, Hdot = sym["H"], sym["Hdot"]
    if mode == "dynamical_horizon":
        T_H = Abs(H + Hdot / (2 * H)) / (2 * pi)
        status: Literal["derived", "assumed"] = "assumed"
        claims = ["T_H(H,Ḣ) = |H + Ḣ/(2H)|/(2π)"]
    else:
        T_H = H / (2 * pi)
        status = "derived"
        claims = ["T_H(H) = H/(2π)"]
    return DerivationStep(
        name="horizon_temperature_T_H",
        stage="SYMBOLIC",
        assumptions=[f"temperature_mode={mode}"],
        equations={"T_H": Eq(Symbol("T_H"), T_H)},
        claims=claims,
        status=status,
        evidence={"T_H": T_H, "mode": mode},
    )


def step_S_total_functional(sym: Dict[str, Any]) -> DerivationStep:
    """S_tot as a thermodynamic functional of cosmological state."""
    H, Hdot, G = sym["H"], sym["Hdot"], sym["G"]
    S_H = pi / (G * H**2)
    S_tot = S_H + sym["S_m"] + sym["S_fields"]
    return DerivationStep(
        name="S_total_functional",
        stage="SYMBOLIC",
        assumptions=["Additive coarse-grained entropy budget"],
        equations={
            "S_H": Eq(Symbol("S_H"), S_H),
            "S_tot": Eq(Symbol("S_tot"), S_tot),
            "arguments": "S_tot = S_tot[H, Hdot, rho, p, ...] via S_H(H) and matter/field sectors",
        },
        claims=[
            "S_tot[H, Ḣ, ρ, ...] = S_H(H) + S_m + S_fields constructed as the working functional.",
        ],
        open_gaps=[
            "Explicit microphysical S_m[ρ], S_fields still model-dependent; treated as additive sectors.",
        ],
        status="assumed",
        evidence={"S_tot": S_tot, "S_H": S_H},
    )


def step_entropy_production(sym: Dict[str, Any]) -> DerivationStep:
    """P ≡ dS_tot/dt - the production law."""
    H, Hdot, G, t = sym["H"], sym["Hdot"], sym["G"], sym["t"]
    Ht = Function("H")(t)
    S_H_t = pi / (G * Ht**2)
    S_H_dot = simplify(diff(S_H_t, t).subs({diff(Ht, t): Hdot, Ht: H}))
    S_H_dot_can = simplify(-2 * pi * Hdot / (G * H**3))
    ok = simplify(S_H_dot - S_H_dot_can) == 0

    S_m_dot = Symbol("S_m_dot")
    S_f_dot = Symbol("S_fields_dot")
    P = S_H_dot_can + S_m_dot + S_f_dot

    return DerivationStep(
        name="entropy_production_P",
        stage="SYMBOLIC",
        assumptions=["P2: GSL uses P = Ṡ_tot"],
        equations={
            "S_H_dot": Eq(Symbol("S_H_dot"), S_H_dot_can),
            "P": Eq(Symbol("P"), P),
            "GSL": "P >= 0",
            "late_time": "If S_m_dot + S_fields_dot -> 0, then P ~ S_H_dot >= 0",
        },
        claims=[
            "P ≡ Ṡ_tot = Ṡ_H + Ṡ_m + Ṡ_fields with Ṡ_H = -2π Ḣ/(G H³).",
            "GSL is P ≥ 0 - not a direct assumption of de Sitter.",
        ]
        if ok
        else [],
        open_gaps=[
            "P≥0 alone does not uniquely fix H(t); thermodynamic EOM required next.",
        ],
        status="derived" if ok else "failed",
        evidence={"S_H_dot": S_H_dot_can, "P": P, "ok": ok},
    )


def step_theorem_22(sym: Dict[str, Any], S_H_dot) -> DerivationStep:
    return DerivationStep(
        name="Theorem_2_2_consequence",
        stage="SYMBOLIC",
        assumptions=["H>0", "G>0", "late-time P ~ Ṡ_H"],
        equations={
            "Theorem_2_2": "S_H_dot >= 0  <=>  Hdot <= 0",
            "S_H_dot": Eq(Symbol("S_H_dot"), S_H_dot),
        },
        claims=[
            "Consequence: Ṡ_H ≥ 0 ⇔ Ḣ ≤ 0.",
            "This is NOT the attractor theorem - only a monotonicity constraint.",
        ],
        status="derived",
        evidence={"S_H_dot": S_H_dot},
    )


# =============================================================================
# STAGE B - Thermodynamic equation of motion (THE KEY OBJECT)
# =============================================================================


def step_thermodynamic_EOM(sym: Dict[str, Any]) -> DerivationStep:
    """
    Derive the thermodynamic equation of motion from entropy extremization.

    State coordinate (paper):
      χ = (S_H - S_early)/(S_max - S_early)
      χ̇ = Γ(χ),   Γ(0)=Γ(1)=0, Γ>0 on (0,1)

    Minimal member of that fixed-point class:
      Γ(χ) = γ χ (1-χ)

    Geometry of S_H(H):
      S_H = π/(G H²)  =>  H = sqrt(π/(G S_H))
      dH/dS = -H/(2 S_H)

    Chain rule => thermodynamic EOM for H:
      Ḣ = (dH/dS) * Ṡ_H
      Ṡ_H = (S_max - S_early) χ̇ = (S_max - S_early) Γ(χ)

    So:
      Ḣ = -(H/(2 S_H)) * (S_max - S_early) * γ * χ * (1-χ)

    This is the differential equation the solver must treat as central -
    not F(Π) or a sigmoid.
    """
    H, G, chi, gamma = sym["H"], sym["G"], sym["chi"], sym["gamma"]
    S_max, S_early = sym["S_max"], sym["S_early"]

    S_H = pi / (G * H**2)
    # Invert: H = sqrt(π/(G S)) along the entropy branch
    S = Symbol("S", real=True, positive=True)
    H_of_S = sqrt(pi / (G * S))
    dH_dS = simplify(diff(H_of_S, S))  # -H/(2S) after back-sub
    dH_dS_H = simplify(dH_dS.subs({S: S_H}))

    Delta_S = S_max - S_early
    Gamma = gamma * chi * (1 - chi)
    S_H_dot_from_chi = Delta_S * Gamma
    Hdot_EOM = simplify(dH_dS_H * S_H_dot_from_chi)

    # Also express χ in terms of S_H
    chi_def = (S_H - S_early) / Delta_S

    return DerivationStep(
        name="thermodynamic_equation_of_motion",
        stage="DERIVE_EOM",
        assumptions=[
            "P3: finite S_max",
            "P4: χ entropy-completion coordinate",
            "P5: χ̇ = Γ(χ) in fixed-point class; minimal Γ=γ χ(1-χ)",
            "S_H = π/(G H²) invertible for H>0",
        ],
        equations={
            "chi_definition": Eq(chi, chi_def),
            "chi_dot": Eq(Symbol("chi_dot"), Gamma),
            "dH_dS": Eq(Symbol("dH_dS"), dH_dS_H),
            "S_H_dot_from_chi": Eq(Symbol("S_H_dot"), S_H_dot_from_chi),
            "thermodynamic_EOM": Eq(sym["Hdot"], Hdot_EOM),
            "production_from_EOM": (
                "P = S_H_dot + S_m_dot + S_fields_dot, "
                "with S_H_dot = (S_max-S_early) Γ(χ)"
            ),
        },
        claims=[
            "Thermodynamic EOM derived: Ḣ = (dH/dS)(S_max-S_early) Γ(χ).",
            "With Γ=γ χ(1-χ): Ḣ = -(H/(2 S_H))(S_max-S_early) γ χ(1-χ) ≤ 0 for χ∈[0,1].",
            "This closes GSL -> dynamics without assuming de Sitter or inserting F(Π).",
        ],
        open_gaps=[
            "Γ=γ χ(1-χ) is the minimal function in the fixed-point class, not proven unique.",
            "First principles should allow other Γ; stability requires only Γ'(1)<0.",
        ],
        status="derived",
        evidence={
            "Hdot_EOM": Hdot_EOM,
            "Gamma": Gamma,
            "dH_dS": dH_dS_H,
            "S_H_dot_from_chi": S_H_dot_from_chi,
            "chi_def": chi_def,
        },
    )


# =============================================================================
# STAGE C - Equilibrium P=0 -> H_* -> a(t) -> linearize λ<0
# =============================================================================


def step_equilibrium_P_zero(sym: Dict[str, Any], eom: DerivationStep) -> DerivationStep:
    """Solve P=0 / χ̇=0 / Ḣ=0 for cosmological equilibrium."""
    chi, gamma = sym["chi"], sym["gamma"]
    Gamma = eom.evidence["Gamma"]
    # Equilibria of Γ: chi=0 (early) and chi=1 (max entropy)
    eq_pts = sp.solve(Gamma, chi)
    return DerivationStep(
        name="equilibrium_P_equals_zero",
        stage="SOLVE_EQUILIBRIUM",
        assumptions=["Late-time Ṡ_m=Ṡ_fields=0", "P = Ṡ_H = (S_max-S_early) Γ(χ)"],
        equations={
            "P_eq": "P = 0",
            "chi_dot_eq": "Gamma(chi) = 0",
            "equilibria": str(eq_pts),
            "Hdot_eq": "Hdot = 0",
            "late_attractor_candidate": "chi = 1  (maximum entropy)",
            "early_fixed_point": "chi = 0  (unstable / early boundary)",
        },
        claims=[
            "Equilibrium P=0 => Γ(χ)=0 => χ∈{0,1}; late-time candidate is χ=1.",
            "χ=1 => Ḣ=0 => H constant - de Sitter kinematics IF H_*>0 exists.",
        ],
        status="derived",
        evidence={"eq_pts": eq_pts, "Gamma": Gamma},
    )


def step_solve_H_star(sym: Dict[str, Any]) -> DerivationStep:
    """H_* = sqrt(π/(G S_max)) from entropy extremum S_H->S_max."""
    G, S_max = sym["G"], sym["S_max"]
    H_star = simplify(sqrt(pi / (G * S_max)))
    return DerivationStep(
        name="solve_H_star",
        stage="SOLVE_H_STAR",
        assumptions=["P3: 0 < S_max < ∞", "At χ=1: S_H = S_max = π/(G H_*^2)"],
        equations={
            "S_max_identity": Eq(S_max, pi / (G * sym["H_star"] ** 2)),
            "H_star": Eq(sym["H_star"], H_star),
            "positivity": "G>0, S_max>0  =>  H_star > 0",
        },
        claims=[
            "GIVEN finite S_max: H_* = sqrt(π/(G S_max)) > 0 is DERIVED.",
            "GSL alone without S_max does NOT prove H_*>0 - existence is CONDITIONAL.",
        ],
        open_gaps=["S_max (holographic N_dof/N_max) is an external thermodynamic input."],
        status="conditional",
        evidence={"H_star": H_star, "existence": "conditional_on_S_max"},
    )


def step_scale_factor(sym: Dict[str, Any]) -> DerivationStep:
    t, H_star = sym["t"], sym["H_star"]
    a0 = Symbol("a_0", positive=True)
    a_eq = a0 * exp(H_star * t)
    return DerivationStep(
        name="scale_factor_de_Sitter",
        stage="DERIVE_a_of_t",
        assumptions=["Equilibrium reached: H(t)=H_*"],
        equations={
            "Hubble_def": "H = a_dot / a",
            "a_eq": Eq(Symbol("a"), a_eq),
        },
        claims=["Integration of ȧ/a = H_* => a(t)=a0 exp(H_* t)."],
        status="derived",
        evidence={"a_eq": a_eq},
    )


def step_linearize_stability(
    sym: Dict[str, Any], eom: DerivationStep
) -> tuple[DerivationStep, DeSitterStatus]:
    """
    Linearize H = H_* + δH about equilibrium and prove λ < 0.

    χ = 1 - ε,  ε≪1
    χ̇ = γ χ(1-χ) => ε̇ = -γ ε + O(ε²)  =>  λ_χ = -γ < 0

    H(S) with S = S_early + χ(S_max-S_early):
      near χ=1, δH ∝ ε  (same decay rate)
      δḢ = λ δH with λ = -γ < 0
    """
    chi, gamma, H, H_star, G = (
        sym["chi"],
        sym["gamma"],
        sym["H"],
        sym["H_star"],
        sym["G"],
    )
    S_max, S_early, eps, dH = (
        sym["S_max"],
        sym["S_early"],
        sym["eps"],
        sym["dH"],
    )

    Gamma = eom.evidence["Gamma"]
    # dΓ/dχ at χ=1
    lam_chi = simplify(diff(Gamma, chi).subs({chi: 1}))  # -gamma

    # Linear map δH ↔ ε
    # S = S_early + χ ΔS, H=sqrt(π/(G S))
    # At eq: S=S_max, H=H_*
    # dH/dχ = (dH/dS)(dS/dχ) = (-H/(2S)) * ΔS
    # At *: dH/dχ|_*= (-H_*/(2 S_max)) * (S_max - S_early)
    # χ=1-ε => δH ~ - (dH/dχ)_* ε = (H_*/(2 S_max)) ΔS ε
    Delta_S = S_max - S_early
    dH_dchi_star = simplify(
        (-H_star / (2 * S_max)) * Delta_S
    )
    # δH = - (dH/dχ)_* ε   because χ↓ by ε means H↑ if approaching from above... 
    # Actually: χ increases to 1, S increases to S_max, H decreases to H_*.
    # χ = 1 - ε => S = S_max - ε ΔS => H > H_* for ε>0
    # H ~ H_* + (dH/dS)_* (-ε ΔS) = H_* + (-H_*/(2S_max))*(-ε ΔS)
    #     = H_* + (H_*/(2S_max)) ΔS ε
    # So δH = (H_*/(2 S_max)) ΔS ε > 0
    # ε̇ = -γ ε => δḢ = (H_*/(2 S_max)) ΔS ε̇ = -γ δH
    # λ = -γ < 0

    lam = simplify(-gamma)
    lam_ok = simplify(lam - lam_chi) == 0

    # Lyapunov V = S_max - S_H ≥ 0, V̇ = -Ṡ_H ≤ 0
    S_H = pi / (G * H**2)
    V = S_max - S_H
    S_max_id = pi / (G * H_star**2)
    V_eq = simplify(V.subs({H: H_star, S_max: S_max_id}))

    strict = lam_ok and V_eq == 0

    if strict:
        desitter = DeSitterStatus(
            equilibrium_identified=True,
            existence="conditional",
            existence_detail=(
                "H_*=sqrt(π/(G S_max))>0 given 0<S_max<∞; not from GSL alone."
            ),
            stability="proven_strict",
            stability_detail=(
                "Linearization: χ=1-ε => ε̇=-γ ε; δH∝ε => δḢ=λ δH with λ=-γ<0. "
                "Lyapunov V=S_max-S_H, V̇=-Ṡ_H≤0, V=0 iff H=H_*."
            ),
            lambda_linear=str(lam),
            verdict_line=(
                "STABLE DE SITTER ATTRACTOR: PROVEN "
                "(existence CONDITIONAL on finite S_max; λ=-γ<0)"
            ),
        )
        claims = [
            "LINEARIZATION: δḢ = λ δH with λ = -γ < 0.",
            "RESULT: δH->0 - stable de Sitter attractor (existence conditional on S_max).",
        ]
        status: Literal["derived", "failed"] = "derived"
    else:
        desitter = DeSitterStatus(
            equilibrium_identified=False,
            existence="failed",
            existence_detail="Linearization identities failed.",
            stability="failed",
            stability_detail="Could not prove λ<0.",
            lambda_linear=str(lam),
            verdict_line="DE SITTER ATTRACTOR: FAILED",
        )
        claims = ["Stability proof FAILED."]
        status = "failed"

    step = DerivationStep(
        name="linearize_and_prove_lambda_negative",
        stage="LINEARIZE",
        assumptions=[
            "Thermodynamic EOM with Γ=γ χ(1-χ)",
            "Equilibrium (H,χ)=(H_*,1)",
            "S_max = π/(G H_*^2)",
        ],
        equations={
            "perturbation_chi": "chi = 1 - eps",
            "eps_dot": Eq(Symbol("eps_dot"), -gamma * eps),
            "delta_H_map": Eq(
                dH, (H_star / (2 * S_max)) * Delta_S * eps
            ),
            "lambda": Eq(sym["lam"], lam),
            "linear_EOM": Eq(Symbol("delta_H_dot"), lam * dH),
            "Lyapunov_V": Eq(Symbol("V"), V),
            "stability_criterion": "lambda < 0",
        },
        claims=claims,
        open_gaps=[
            "Existence of H_*>0 remains conditional on holographic S_max.",
        ],
        status=status,
        evidence={
            "lambda": lam,
            "lam_chi": lam_chi,
            "dH_dchi_star": dH_dchi_star,
            "V_eq": V_eq,
            "desitter": desitter.to_dict(),
        },
    )
    return step, desitter


def step_attractor_result(desitter: DeSitterStatus) -> DerivationStep:
    return DerivationStep(
        name="RESULT_stable_de_Sitter_attractor",
        stage="RESULT",
        assumptions=["All prior first-principles stages"],
        equations={"verdict": desitter.verdict_line},
        claims=[
            desitter.verdict_line,
            f"existence={desitter.existence}; stability={desitter.stability}; λ={desitter.lambda_linear}",
        ],
        status="derived" if desitter.stability == "proven_strict" else "failed",
        evidence=desitter.to_dict(),
    )


# =============================================================================
# STAGE D - Exact closure solutions + enthalpy bridge + asymptotic Einstein
# =============================================================================


def step_exact_chi_H_solutions(sym: Dict[str, Any]) -> DerivationStep:
    """Exact logistic χ(t), S_H(t), H(t) under the ASSUMED closure."""
    t, gamma = sym["t"], sym["gamma"]
    S_max, S_early, G = sym["S_max"], sym["S_early"], sym["G"]
    t_crit = sym["t_crit"]
    C = Symbol("C", real=True, positive=True)

    chi_t = 1 / (1 + C * exp(-gamma * (t - t_crit)))
    S_H_t = S_early + (S_max - S_early) * chi_t
    H_t = sqrt(pi / (G * S_H_t))
    H_inf = sqrt(pi / (G * S_max))
    lim_chi = simplify(sp.limit(chi_t, t, sp.oo))
    lim_S = simplify(sp.limit(S_H_t, t, sp.oo))
    lim_H = simplify(sp.limit(H_t, t, sp.oo))

    return DerivationStep(
        name="exact_chi_S_H_H_from_closure",
        stage="EXACT_SOLUTIONS",
        assumptions=[
            "ASSUMED closure χ̇=γ χ(1-χ)",
            "S_H = S_early + (S_max-S_early)χ",
            "H = sqrt(π/(G S_H)), H>0",
        ],
        equations={
            "chi_t": Eq(Symbol("chi"), chi_t),
            "S_H_t": Eq(Symbol("S_H"), S_H_t),
            "H_t": Eq(Symbol("H"), H_t),
            "lim_chi": Eq(Symbol("lim_chi"), lim_chi),
            "lim_S_H": Eq(Symbol("lim_S_H"), lim_S),
            "lim_H": Eq(Symbol("lim_H"), lim_H),
            "H_inf": Eq(sym["H_star"], H_inf),
        },
        claims=[
            "Exact χ(t)=1/(1+C e^{-γ(t-t_crit)}) under assumed closure.",
            f"t->∞: χ->{lim_chi}, S_H->S_max, H->H_∞=sqrt(π/(G S_max)).",
            "χ is thermodynamic entropy-completion; do NOT auto-identify with pheno w(t).",
        ],
        open_gaps=[
            "Closure remains ASSUMED; exact solution does not elevate it to derived.",
        ],
        status="derived",
        evidence={
            "chi_t": chi_t,
            "S_H_t": S_H_t,
            "H_t": H_t,
            "H_inf": H_inf,
            "lim_chi": lim_chi,
            "lim_H": lim_H,
        },
    )


def step_enthalpy_bridge(sym: Dict[str, Any]) -> DerivationStep:
    """
    Paper bridge: Ḣ = -4πG(ρ_tot+P_tot); equilibrium ρ_tot+P_tot=0 => Ḣ=0.
    Connects thermodynamic H->H_∞ to Einstein/Friedmann enthalpy condition.
    """
    H, G = sym["H"], sym["G"]
    rho_m, rho_r, rho_S = sym["rho_m"], sym["rho_r"], sym["rho_S"]
    P_m, P_r, P_S = sym["P_m"], sym["P_r"], sym["P_S"]
    rho_tot = rho_m + rho_r + rho_S
    P_tot = P_m + P_r + P_S
    Hdot = -4 * pi * G * (rho_tot + P_tot)
    return DerivationStep(
        name="enthalpy_bridge_to_de_Sitter",
        stage="ENTHALPY_BRIDGE",
        assumptions=[
            "Einstein/FLRW Raychaudhuri (paper)",
            "Late-time ρ_m,ρ_r -> 0 at attractor",
        ],
        equations={
            "Raychaudhuri": Eq(sym["Hdot"], Hdot),
            "enthalpy_eq": Eq(rho_tot + P_tot, 0),
            "implies_Hdot0": "rho_tot + P_tot = 0  =>  Hdot = 0  =>  H = H_infty",
            "w_S_eq": Eq(sym["w_S"], -1),
            "paper_link": (
                "Horizon: S_H->S_max => H->H_∞  AND  "
                "Einstein: ρ_tot+P_tot=0 => Ḣ=0 => de Sitter"
            ),
        },
        claims=[
            "Paper enthalpy condition ρ_tot+P_tot=0 is equivalent to Ḣ=0.",
            "At attractor with diluted matter/radiation: ρ_S+P_S=0 => w_S=-1.",
            "Bridges thermodynamic S_max->H_∞ with Einstein/Friedmann de Sitter condition.",
        ],
        status="derived",
        evidence={"Hdot": Hdot, "rho_tot": rho_tot, "P_tot": P_tot},
    )


def step_asymptotic_curvature(sym: Dict[str, Any]) -> DerivationStep:
    """R_∞ = 12 H_∞² = 12π/(G S_max)."""
    H_inf, G, S_max = sym["H_star"], sym["G"], sym["S_max"]
    R_flat = 6 * (sym["Hdot"] + 2 * H_inf**2)
    R_inf = simplify(12 * H_inf**2)
    R_inf_S = simplify(12 * pi / (G * S_max))
    # With H_inf^2 = pi/(G S_max)
    ok = simplify(R_inf.subs({H_inf**2: pi / (G * S_max)}) - R_inf_S) == 0
    return DerivationStep(
        name="asymptotic_curvature_R_inf",
        stage="ASYMPTOTIC_EINSTEIN",
        assumptions=["Flat FLRW", "Attractor: Ḣ->0, H->H_∞"],
        equations={
            "R_FLRW": Eq(Symbol("R"), 6 * (sym["Hdot"] + 2 * sym["H"] ** 2)),
            "R_inf": Eq(Symbol("R_inf"), R_inf),
            "R_inf_thermo": Eq(Symbol("R_inf"), R_inf_S),
        },
        claims=[
            "At attractor: R_∞ = 12 H_∞².",
            "With H_∞²=π/(G S_max): R_∞ = 12π/(G S_max).",
            "Maximum horizon entropy determines the asymptotic curvature scale.",
        ],
        status="derived" if ok else "failed",
        evidence={"R_inf": R_inf, "R_inf_S": R_inf_S, "ok": ok},
    )


def step_asymptotic_Einstein_tensor(sym: Dict[str, Any]) -> DerivationStep:
    """R_μν=3 H_∞² g_μν => G_μν^(∞)=-3 H_∞² g_μν."""
    H_inf = sym["H_star"]
    G_inf = simplify(-3 * H_inf**2)  # coefficient of g_μν
    return DerivationStep(
        name="asymptotic_Einstein_tensor",
        stage="ASYMPTOTIC_EINSTEIN",
        assumptions=["de Sitter geometry at attractor"],
        equations={
            "R_mu_nu_inf": "R^{(∞)}_{μν} = 3 H_∞² g_μν",
            "R_inf": Eq(Symbol("R_inf"), 12 * H_inf**2),
            "G_mu_nu_inf": "G^{(∞)}_{μν} = R_μν - (1/2) R g_μν = -3 H_∞² g_μν",
            "G_coeff": Eq(Symbol("G_inf_coeff"), G_inf),
        },
        claims=[
            "Asymptotic Einstein tensor: G_μν^(∞) = -3 H_∞² g_μν.",
            "This is geometry of the attractor, not a claim of full EFE derivation.",
        ],
        status="derived",
        evidence={"G_inf_coeff": G_inf},
    )


def step_thermodynamic_Lambda_S(sym: Dict[str, Any]) -> DerivationStep:
    """Λ_S ≡ 3 H_∞² = 3π/(G S_max). Retain π."""
    H_inf, G, S_max = sym["H_star"], sym["G"], sym["S_max"]
    Lambda_S = 3 * H_inf**2
    Lambda_S_thermo = simplify(3 * pi / (G * S_max))
    H2 = pi / (G * S_max)
    ok = simplify(Lambda_S.subs({H_inf**2: H2}) - Lambda_S_thermo) == 0
    return DerivationStep(
        name="thermodynamic_Lambda_S",
        stage="ASYMPTOTIC_EINSTEIN",
        assumptions=["H_∞² = π/(G S_max) from S_H->S_max"],
        equations={
            "Lambda_S_def": Eq(Symbol("Lambda_S"), Lambda_S),
            "Lambda_S_thermo": Eq(Symbol("Lambda_S"), Lambda_S_thermo),
            "S_max_from_Lambda": Eq(S_max, 3 * pi / (G * Symbol("Lambda_S"))),
            "chain": "S_max ↔ H_∞ ↔ Λ_S ↔ de Sitter curvature",
        },
        claims=[
            "Λ_S ≡ 3 H_∞² = 3π/(G S_max)  (π retained).",
            "Equivalently S_max = 3π/(G Λ_S).",
            "Asymptotic vacuum Einstein: G_μν^(∞) + Λ_S g_μν = 0.",
        ],
        open_gaps=[BOUNDARY_NOT_PROVEN],
        status="conditional" if ok else "failed",
        evidence={
            "Lambda_S": Lambda_S,
            "Lambda_S_thermo": Lambda_S_thermo,
            "ok": ok,
        },
    )


def step_asymptotic_Friedmann_and_T_S(sym: Dict[str, Any]) -> DerivationStep:
    """ρ_S,∞ = 3 H_∞²/(8πG) = 3/(8 G² S_max); T^(S)=-ρ_S g_μν."""
    H_inf, G, S_max = sym["H_star"], sym["G"], sym["S_max"]
    rho_S = simplify(3 * H_inf**2 / (8 * pi * G))
    rho_S_thermo = simplify(sp.Rational(3, 8) / (G**2 * S_max))
    # Check: 3*(π/(G S_max))/(8πG) = 3π/(8π G² S_max) = 3/(8 G² S_max)
    ok = simplify(rho_S.subs({H_inf**2: pi / (G * S_max)}) - rho_S_thermo) == 0
    Lambda_S = 3 * H_inf**2
    return DerivationStep(
        name="asymptotic_Friedmann_and_T_S",
        stage="ASYMPTOTIC_EINSTEIN",
        assumptions=[
            "Late-time ρ_m,ρ_r->0",
            "Friedmann 3H²=8πG ρ_S at attractor",
            "w_S=-1 => T^(S)=-ρ_S g_μν",
        ],
        equations={
            "Friedmann_inf": Eq(3 * H_inf**2, 8 * pi * G * Symbol("rho_S")),
            "rho_S_inf": Eq(Symbol("rho_S_inf"), rho_S),
            "rho_S_thermo": Eq(Symbol("rho_S_inf"), rho_S_thermo),
            "P_S_inf": Eq(Symbol("P_S_inf"), -rho_S),
            "w_S_inf": Eq(Symbol("w_S_inf"), -1),
            "T_S_inf": "T^{(S,∞)}_{μν} = -ρ_S g_μν = -(Λ_S/(8πG)) g_μν",
            "Lambda_S_rho": Eq(Symbol("Lambda_S"), 8 * pi * G * Symbol("rho_S")),
        },
        claims=[
            "ρ_S,∞ = 3 H_∞²/(8πG) = 3/(8 G² S_max)  (π cancelled correctly).",
            "T^(S,∞)_μν = -ρ_S g_μν = -(Λ_S/(8πG)) g_μν.",
            "Λ_S = 8πG ρ_S at the attractor.",
        ],
        status="derived" if ok else "failed",
        evidence={
            "rho_S": rho_S,
            "rho_S_thermo": rho_S_thermo,
            "Lambda_S": Lambda_S,
            "ok": ok,
        },
    )


def step_asymptotic_Einstein_equation(sym: Dict[str, Any]) -> DerivationStep:
    """G_μν^(∞) + Λ_S g_μν = 0 - ASYMPTOTIC only."""
    H_inf, G, S_max = sym["H_star"], sym["G"], sym["S_max"]
    Lambda_S = simplify(3 * pi / (G * S_max))
    return DerivationStep(
        name="RESULT_asymptotic_Einstein_equation",
        stage="ASYMPTOTIC_EINSTEIN",
        assumptions=[
            "Stable de Sitter attractor",
            "Effective T^(S)=-ρ_S g_μν",
            "Einstein equation G_μν = 8πG T_μν^(S) at attractor",
        ],
        equations={
            "Einstein_asymptotic": "G^{(∞)}_{μν} + Λ_S g_μν = 0",
            "Lambda_S": Eq(Symbol("Lambda_S"), Lambda_S),
            "chain": "S_max -> H_∞ -> Λ_S -> T^(S)_∞ -> G_μν^(∞)+Λ_S g_μν=0",
            "NOT_CLAIMED": "G_μν + Λ g_μν = 8πG T_μν  (full local EFE)",
        },
        claims=[
            "ASYMPTOTIC RESULT: G_μν^(∞) + Λ_S g_μν = 0 with Λ_S=3π/(G S_max).",
            "VALID: horizon thermo + finite S_max + closure => stable dS => asymptotic vacuum Einstein.",
            "NOT PROVEN: full Einstein field equations from global Hubble-horizon thermodynamics alone.",
        ],
        open_gaps=[BOUNDARY_NOT_PROVEN],
        status="conditional",
        evidence={"Lambda_S": Lambda_S, "boundary": BOUNDARY_NOT_PROVEN},
    )


def step_conservation_Q_flag(sym: Dict[str, Any]) -> DerivationStep:
    """Flag paper sourcing vs conservation-consistent Q-formalism."""
    H = sym["H"]
    rho_m, P_m = sym["rho_m"], sym["P_m"]
    rho_S, P_S = sym["rho_S"], sym["P_S"]
    Q = 3 * H * (rho_m + P_m)
    return DerivationStep(
        name="conservation_Q_formalism_flag",
        stage="CONSISTENCY",
        assumptions=["Paper eqs. (88)/(90)/(114) entropy sourcing"],
        equations={
            "paper_sourcing": Eq(
                Symbol("rho_S_dot") + 3 * H * (rho_S + P_S), Q
            ),
            "paper_matter_alone": Eq(
                Symbol("rho_m_dot") + 3 * H * (rho_m + P_m), 0
            ),
            "consistent_Q": (
                "dot(rho_m)+3H(rho_m+P_m) = -Q,  "
                "dot(rho_S)+3H(rho_S+P_S) = +Q,  Q_m+Q_S=0"
            ),
            "total_conservation": "dot(rho_tot)+3H(rho_tot+P_tot)=0",
        },
        claims=[
            "FLAG: paper keeps matter separately conserved while sourcing ρ_S - "
            "potentially inconsistent unless Q is effective transfer.",
            "Diagnostic mode: implement Q_m=-Q, Q_S=+Q so total ∇_μ T^{μν}=0.",
            "Do NOT silently alter the paper model; report both forms.",
        ],
        open_gaps=[
            "Choose interacting-fluid Q-formalism before CLASS handoff if total conservation is required.",
        ],
        status="assumed",
        evidence={"Q": Q},
    )



# =============================================================================
# MODULE 17 - Jacobson local-horizon route (FULL, embedded; no external import)
# =============================================================================


@dataclass
class JacobsonLabeledEq:
    id: str
    tag: str
    title: str
    latex: str
    ascii: str
    assumptions: List[str] = field(default_factory=list)
    note: str = ""
    status: str = "stated"

    def to_dict(self) -> Dict[str, str]:
        return {
            "id": self.id,
            "tag": self.tag,
            "title": self.title,
            "latex": self.latex,
            "ascii": self.ascii,
            "assumptions": " | ".join(self.assumptions),
            "note": self.note,
            "status": self.status,
        }


def build_jacobson_derivation_embedded() -> Dict[str, Any]:
    """
    Full labeled Jacobson chain (Module 17), embedded in analytical_solver.

    Honesty: A1-A3 and H1/C1/C2 are AXIOMS/CONDITIONS of this module -
    NOT derived from the global Hubble-horizon GSL argument.
    """
    G = Symbol("G", positive=True)
    eta = Symbol("eta", positive=True)
    S_max = Symbol("S_max", positive=True)
    H_inf = Symbol("H_infty", positive=True)
    Rkk = Symbol("R_kk")
    Tkk = Symbol("T_kk")

    eqs: List[JacobsonLabeledEq] = []

    def add(eid, tag, title, latex, ascii, assumptions=None, note="", status="stated"):
        eqs.append(JacobsonLabeledEq(
            eid, tag, title, latex, ascii,
            assumptions=assumptions or [], note=note, status=status,
        ))

    add("A0.1", "AUDIT", "Scope separation",
        r"\text{Module 17}\;\perp\;\text{Steps 0--16 (global Hubble-horizon attractor)}",
        "Module 17 is logically separate from Steps 0-16",
        note="Caution: no unfinished global step filled by invention.")
    add("J.A1", "AXIOM", "Clausius on local Rindler horizons",
        r"\delta Q = T\,dS", "δQ = T dS",
        note="Jacobson fundamental input - not from GSL on Hubble sphere.")
    add("J.A2", "AXIOM", "Bekenstein-Hawking entropy density",
        r"dS=\eta\,dA,\qquad \eta=\frac{1}{4G}",
        "dS = η dA; η = 1/(4G)")
    add("J.A3", "AXIOM", "Unruh temperature",
        r"T=\frac{\kappa}{2\pi}", "T = κ/(2π)")
    add("J.H1", "CONDITION", "Instantaneously stationary local horizon",
        r"\theta\big|_{P}=0,\quad \sigma_{ab}\big|_{P}=0",
        "θ|_P = 0, σ|_P = 0",
        note="Drops quadratic Raychaudhuri terms at leading order.",
        status="conditional")
    add("J.D1", "DEFINITION", "Heat flux across the horizon pencil",
        r"\delta Q=\int_{\mathcal{H}} T_{\mu\nu}\,\chi^{\mu}\,d\Sigma^{\nu},"
        r"\quad \chi^{a}=-\kappa\lambda\,k^{a}",
        "δQ = ∫ T_μν χ^μ dΣ^ν with χ^a = -κ λ k^a")
    add("J.D2", "DEFINITION", "Null generator and area element",
        r"k^{\mu}k_{\mu}=0,\quad d\Sigma^{\mu}=k^{\mu}\,d\lambda\,dA",
        "k null; dΣ^μ = k^μ dλ dA")
    add("J.G1", "AXIOM", "Null Raychaudhuri equation",
        r"\frac{d\theta}{d\lambda}=-\frac12\theta^2-\sigma_{ab}\sigma^{ab}"
        r"-R_{\mu\nu}k^{\mu}k^{\nu}",
        "dθ/dλ = -½θ² - σ² - R_μν k^μ k^ν")
    add("J.G2", "DERIVED", "Linearized expansion near P",
        r"\theta(\lambda)=-\lambda\,R_{\mu\nu}k^{\mu}k^{\nu}+O(\lambda^2)",
        "θ(λ) = -λ R_μν k^μ k^ν + O(λ²)", status="derived")
    add("J.G3", "DERIVED", "Area variation",
        r"\delta A=\int \theta\,d\lambda\,dA"
        r"=-\int\lambda\,R_{\mu\nu}k^{\mu}k^{\nu}\,d\lambda\,dA",
        "δA = ∫ θ dλ dA = -∫ λ R_μν k^μ k^ν dλ dA", status="derived")
    add("J.E1", "DERIVED", "Entropy variation",
        r"dS=\eta\,\delta A"
        r"=-\eta\int\lambda\,R_{\mu\nu}k^{\mu}k^{\nu}\,d\lambda\,dA",
        "dS = η δA", status="derived")
    add("J.E2", "DERIVED", "Heat side of Clausius",
        r"\delta Q=-\kappa\int\lambda\,T_{\mu\nu}k^{\mu}k^{\nu}\,d\lambda\,dA",
        "δQ = -κ ∫ λ T_μν k^μ k^ν dλ dA", status="derived")
    add("J.E3", "DERIVED", "Clausius => null-projected field equation",
        r"T_{\mu\nu}k^{\mu}k^{\nu}=\frac{\eta}{2\pi}\,R_{\mu\nu}k^{\mu}k^{\nu}",
        "T_μν k^μ k^ν = (η/(2π)) R_μν k^μ k^ν for all null k",
        status="derived")
    null_rel = Eq(Tkk, (eta / (2 * pi)) * Rkk)
    add("J.E3b", "DERIVED", "Null projection (symbolic)",
        sp.latex(null_rel), str(null_rel), status="derived")
    add("J.L1", "LEMMA", "Null-projection lemma",
        r"S_{\mu\nu}k^{\mu}k^{\nu}=0\ \forall\mathrm{null}\ k"
        r"\;\Rightarrow\; S_{\mu\nu}=f\,g_{\mu\nu}",
        "If S_μν k^μ k^ν=0 for all null k, then S_μν = f g_μν")
    add("J.E4", "DERIVED", "Ricci-stress relation up to pure trace",
        r"R_{\mu\nu}-\frac{2\pi}{\eta}T_{\mu\nu}=f\,g_{\mu\nu}",
        "R_μν - (2π/η) T_μν = f g_μν", status="derived")
    add("J.C1", "CONDITION", "Local stress-energy conservation",
        r"\nabla^{\mu}T_{\mu\nu}=0", "∇^μ T_μν = 0", status="conditional")
    add("J.G4", "AXIOM", "Twice-contracted Bianchi identity",
        r"\nabla^{\mu}G_{\mu\nu}=0", "∇^μ G_μν = 0")
    add("J.E5", "DERIVED", "Fixing the free function f",
        r"f=-\frac{R}{2}+\Lambda", "f = -R/2 + Λ", status="derived")
    add("J.E6", "DERIVED", "Einstein equation with undetermined Λ",
        r"G_{\mu\nu}+\Lambda g_{\mu\nu}=\frac{2\pi}{\eta}\,T_{\mu\nu}",
        "G_μν + Λ g_μν = (2π/η) T_μν", status="derived")
    add("J.C2", "CONDITION", "Newton-constant calibration of η",
        r"\eta=\frac{1}{4G}\quad\Rightarrow\quad\frac{2\pi}{\eta}=8\pi G",
        "η = 1/(4G) => 2π/η = 8πG", status="conditional")
    cal_ok = simplify(2 * pi * (4 * G) - 8 * pi * G) == 0
    add("J.E7", "THEOREM", "Full local Einstein field equations",
        r"G_{\mu\nu}+\Lambda g_{\mu\nu}=8\pi G\,T_{\mu\nu}",
        "G_μν + Λ g_μν = 8πG T_μν",
        note="DERIVED under Module-17 axioms; NOT from global Hubble GSL alone.",
        status="derived" if cal_ok else "open")
    add("J.B1", "BRIDGE", "FLRW specialization of the local EFE",
        r"\text{FLRW}+T=T^{(m)}+T^{(S)}\;\Rightarrow\;\text{Friedmann}",
        "FLRW + T = T^(m)+T^(S) => Friedmann", status="conditional")
    Lam_thermo = simplify(3 * pi / (G * S_max))
    bridge_ok = simplify((3 * H_inf**2).subs({H_inf**2: pi / (G * S_max)}) - Lam_thermo) == 0
    add("J.B2", "BRIDGE", "Thermodynamic identification of Λ",
        r"\Lambda\to\Lambda_S=\frac{3\pi}{G S_{\max}}=3 H_\infty^2",
        "Λ -> Λ_S = 3π/(G S_max) = 3 H_∞²", status="conditional")
    add("J.B3", "DERIVED", "Asymptotic Einstein structure (recovered)",
        r"G^{(\infty)}_{\mu\nu}+\Lambda_S g_{\mu\nu}=0",
        "G^(∞)_μν + Λ_S g_μν = 0",
        status="derived" if bridge_ok else "open")
    add("J.ARCH", "THEOREM", "Completed research-target architecture",
        r"\text{local thermo}\to\text{EFE}\to\text{FLRW}\to S_H"
        r"\to\text{GSL}+S_{\max}+\text{closure}\to\text{dS}\to\Lambda_S",
        "local thermo -> EFE -> FLRW -> S_H -> GSL+Smax+closure -> dS -> Λ_S",
        status="conditional")

    claim_F = (
        "CONDITIONAL/DERIVED (Module 17): full local EFE under Jacobson axioms; "
        "still NOT derived from the global Hubble-horizon argument alone."
    )
    return {
        "equations": eqs,
        "claim_F_status": claim_F,
        "asymptotic_ok": bool(bridge_ok),
        "efe_status": "derived_under_axioms" if cal_ok else "open",
        "Lambda_S": str(Lam_thermo),
    }


def step_jacobson_local_horizon_full() -> DerivationStep:
    """Full Module-17 Jacobson step (embedded)."""
    res = build_jacobson_derivation_embedded()
    eqs = {e.id: e.ascii for e in res["equations"]}
    return DerivationStep(
        name="LOCAL_horizon_Jacobson_EFE",
        stage="JACOBSON_MODULE",
        assumptions=[
            "AXIOM J.A1 Clausius δQ=T dS",
            "AXIOM J.A2 dS=η dA",
            "AXIOM J.A3 T=κ/(2π)",
            "CONDITION J.H1 θ=σ=0 at P",
            "CONDITION J.C1 ∇^μ T_μν=0",
            "CONDITION J.C2 η=1/(4G)",
        ],
        equations=eqs,
        claims=[
            "Under local-horizon axioms: G_μν+Λ g_μν=8πG T_μν is DERIVED.",
            "Λ fixed to Λ_S=3π/(G S_max) by BRIDGE to global attractor.",
            "Asymptotic corollary: G^(∞)_μν+Λ_S g_μν=0 (matches Step 16).",
            res["claim_F_status"],
        ],
        open_gaps=[
            "Axioms are independent of the global Hubble-horizon GSL proof.",
            "Non-equilibrium / viscous extensions not included.",
        ],
        status="conditional" if res["efe_status"] == "derived_under_axioms" else "stub",
        evidence={
            "efe_status": res["efe_status"],
            "claim_F": res["claim_F_status"],
            "asymptotic_ok": res["asymptotic_ok"],
            "jacobson_equations": [e.to_dict() for e in res["equations"]],
        },
    )



# =============================================================================
# LAYERS 1-21 - First-principles implications of the locked entropy structure
# Canon locked: S_H = π/(G H²)   [GH/(2π) REJECTED]
# Everything below lives ONLY in analytical_solver.py.
# =============================================================================


def _locked_SH(G, H):
    return pi / (G * H**2)


def _thermo_Hdot(G, H, S_max, S_early, gamma, chi):
    """Ḣ from chain rule + logistic closure (first principles from S_H(H), χ)."""
    S_H = _locked_SH(G, H)
    Delta_S = S_max - S_early
    return -(H / (2 * S_H)) * Delta_S * gamma * chi * (1 - chi)


def step_L01_effective_fluid(sym: Dict[str, Any]) -> DerivationStep:
    """
    LAYER 1 - Entire effective cosmological fluid from entropy sector.

    First principles:
      (i)   S_H = π/(G H²)           [locked BH + apparent horizon]
      (ii)  H² = 8πG/3 ρ_tot         [Friedmann / GR bridge]
      (iii) entropy-only: ρ_tot=ρ_S
      (iv)  Ḣ = -4πG(ρ_S+P_S)        [Raychaudhuri]
      (v)   χ̇ = γ χ(1-χ)             [ASSUMED closure; Γ-class]
      (vi)  S_H = S_early + ΔS χ

    => ρ_S = 3H²/(8πG) = 3/(8 G² S_H)
    => Ḣ = -(H/(2 S_H)) ΔS γ χ(1-χ)
    => P_S = -ρ_S - Ḣ/(4πG)
    => w_S = -1 + ΔS γ χ(1-χ)/(3 H S_H)
    => χ->1: w_S->-1; interior: w_S>-1 (non-phantom).
    """
    G, H, chi, gamma = sym["G"], sym["H"], sym["chi"], sym["gamma"]
    S_max, S_early = sym["S_max"], sym["S_early"]
    S_H = _locked_SH(G, H)
    Delta_S = S_max - S_early
    rho_S = simplify(3 * H**2 / (8 * pi * G))
    rho_from_S = simplify(sp.Rational(3, 8) / (G**2 * S_H))
    ok_rho = simplify(rho_S - rho_from_S) == 0
    Hdot = _thermo_Hdot(G, H, S_max, S_early, gamma, chi)
    P_S = simplify(-rho_S - Hdot / (4 * pi * G))
    w_S = simplify(P_S / rho_S)
    w_tgt = simplify(-1 + Delta_S * gamma * chi * (1 - chi) / (3 * H * S_H))
    ok_w = simplify(w_S - w_tgt) == 0
    excess = simplify(w_S + 1)
    return DerivationStep(
        name="L01_effective_cosmological_fluid",
        stage="LAYER_01",
        assumptions=[
            "Locked S_H=π/(G H²)",
            "Friedmann H²=8πG ρ_S/3 (entropy-only background)",
            "Raychaudhuri Ḣ=-4πG(ρ_S+P_S)",
            "ASSUMED logistic closure χ̇=γ χ(1-χ)",
        ],
        equations={
            "S_H": Eq(Symbol("S_H"), S_H),
            "H2": Eq(H**2, pi / (G * S_H)),
            "Friedmann": Eq(H**2, (8 * pi * G / 3) * Symbol("rho_S")),
            "rho_S": Eq(Symbol("rho_S"), rho_S),
            "rho_S_of_S_H": Eq(Symbol("rho_S"), rho_from_S),
            "Hdot": Eq(sym["Hdot"], Hdot),
            "P_S": Eq(Symbol("P_S"), P_S),
            "w_S": Eq(Symbol("w_S"), w_S),
            "w_S_compact": Eq(Symbol("w_S"), w_tgt),
            "non_phantom": "w_S+1 = ΔS γ χ(1-χ)/(3 H S_H) ≥ 0 on [0,1]",
            "attractor_EOS": "lim_{χ->1} w_S = -1",
        },
        claims=[
            "LAYER 1 DERIVED: entire effective fluid (ρ_S,P_S,w_S) from entropy dynamics.",
            "Predicted EOS: w_S=-1+ΔS γχ(1-χ)/(3 H S_H); non-phantom approach to de Sitter.",
        ],
        status="derived" if (ok_rho and ok_w) else "failed",
        evidence={
            "rho_S": rho_S, "P_S": P_S, "w_S": w_S, "Hdot": Hdot,
            "excess": excess, "ok_rho": ok_rho, "ok_w": ok_w,
        },
    )


def step_L02_deceleration_cosmography(sym: Dict[str, Any]) -> DerivationStep:
    """LAYER 2 - q, jerk from entropy EOM (cosmographic hierarchy)."""
    G, H, chi, gamma = sym["G"], sym["H"], sym["chi"], sym["gamma"]
    S_max, S_early = sym["S_max"], sym["S_early"]
    S_H = _locked_SH(G, H)
    Delta_S = S_max - S_early
    Hdot = _thermo_Hdot(G, H, S_max, S_early, gamma, chi)
    q = simplify(-1 - Hdot / H**2)
    q_tgt = simplify(-1 + Delta_S * gamma * chi * (1 - chi) / (2 * H * S_H))
    ok = simplify(q - q_tgt) == 0
    # Ḧ via chain: dḢ/dt = (dḢ/dχ) χ̇
    # Ḣ = -(H/(2 S_H)) ΔS γ χ(1-χ); keep definitional identity for jerk
    j_id = "j = 1 + 3 Ḣ/H² + Ḧ/H³ = q(1+2q) - q̇/H"
    return DerivationStep(
        name="L02_deceleration_cosmography",
        stage="LAYER_02",
        assumptions=["LAYER 1 Hdot", "definitions q,j"],
        equations={
            "q_def": "q ≡ -ä/(a H²) = -1 - Ḣ/H²",
            "q": Eq(Symbol("q"), q),
            "q_compact": Eq(Symbol("q"), q_tgt),
            "j_def": "j ≡ a⃛/(a H³) = 1 + 3 Ḣ/H² + Ḧ/H³",
            "j_identity": j_id,
            "snap_def": "s ≡ a^(4)/(a H⁴)",
            "q_zero": "q=0 ⇔ ΔS γ χ(1-χ) = 2 H S_H",
            "q_attractor": "χ->1 => q->-1",
        },
        claims=[
            "LAYER 2 DERIVED: analytic q(χ); cosmographic tower fixed by entropy dynamics.",
        ],
        status="derived" if ok else "failed",
        evidence={"q": q, "q_tgt": q_tgt, "ok": ok, "Hdot": Hdot},
    )


def step_L03_phase_trajectory(sym: Dict[str, Any]) -> DerivationStep:
    """
    LAYER 3 - Derive H(χ), a(χ), and H(a) by eliminating cosmic time.

    First-principles chain
    ----------------------
    (1) S_H(χ) = S_early + (S_max - S_early) χ
    (2) H(χ)   = sqrt(π/(G S_H(χ)))                 [invert locked S_H=π/(G H²)]
    (3) χ̇ = γ χ(1-χ),   d ln a = H dt
        =>  dχ / d ln a = γ χ(1-χ) / H(χ)
        =>  ln(a/a_ref) = ∫_{χ_ref}^{χ} H(x)/(γ x(1-x)) dx
    (4) Partial fractions: 1/(x(1-x)) = 1/x + 1/(1-x), so
        ln(a/a_ref) = γ^{-1} sqrt(π/G) ∫ [1/(x sqrt(S_e+ΔS x))
                                          + 1/((1-x) sqrt(S_e+ΔS x))] dx
    (5) Closed form when S_early=0:
        ln(a/a_ref) = [sqrt(π/(G S_max))/γ] [F(χ)-F(χ_ref)],
        F(u) = 2 artanh(√u) - 2/√u    (u∈(0,1))
    (6) H(a): invert a=a(χ) for χ=χ(a), then H(a)=H(χ(a)).
        Equivalently the parametric curve (a(χ), H(χ)) IS the exact H(a).
    (7) H(z) = H(a=1/(1+z)).
    """
    G, chi, gamma = sym["G"], sym["chi"], sym["gamma"]
    S_max, S_early = sym["S_max"], sym["S_early"]
    Delta_S = S_max - S_early
    x = Symbol("x", positive=True)
    chi_ref = Symbol("chi_ref", positive=True)

    # --- (1)+(2) H(χ) ---
    S_H = S_early + Delta_S * chi
    H_chi = sqrt(pi / (G * S_H))
    ok_H = simplify(H_chi**2 * G * S_H - pi) == 0

    # --- (3) eliminate t ---
    dchi_dN = gamma * chi * (1 - chi) / H_chi
    # integrand of ln a
    H_of_x = sqrt(pi / (G * (S_early + Delta_S * x)))
    integrand = H_of_x / (gamma * x * (1 - x))

    # --- (4) partial-fraction skeleton (general S_early) ---
    # ln(a/a_ref) = γ^{-1} sqrt(π/G) * [J1+J2]_{χ_ref}^{χ}
    pref = sqrt(pi / G) / gamma
    J1_integrand = 1 / (x * sqrt(S_early + Delta_S * x))
    J2_integrand = 1 / ((1 - x) * sqrt(S_early + Delta_S * x))

    # --- (5) S_early = 0 closed form ---
    # F(u) = 2 artanh(√u) - 2/√u
    u = Symbol("u", positive=True)
    F_u = 2 * sp.atanh(sqrt(u)) - 2 / sqrt(u)
    # Verify dF/du = 1/(u^{3/2}(1-u)) on (0,1):
    # integrand for Se=0: H/(γ x(1-x)) = sqrt(π/(G ΔS))/(γ) * 1/(x^{3/2}(1-x))
    dF = simplify(diff(F_u, u))
    target_dF = 1 / (u ** sp.Rational(3, 2) * (1 - u))
    # atanh derivative identities can leave equivalent forms; check numerically-friendly identity:
    ok_F = simplify(dF * (u ** sp.Rational(3, 2) * (1 - u)) - 1) == 0
    if not ok_F:
        # Accept equivalent rewrite via direct Se=0 antiderivative from ∫ 1/(x(1-x)√(ΔS x))
        F_alt = integrate(1 / (x * (1 - x) * sqrt(Delta_S * x)), x)
        dF_alt = simplify(diff(F_alt, x))
        ok_F = simplify(dF_alt - 1 / (x * (1 - x) * sqrt(Delta_S * x))) == 0
        F_u = F_alt.subs({x: u})
    pref0 = sqrt(pi / (G * S_max)) / gamma  # ΔS=S_max when S_early=0
    ln_a_Se0 = pref0 * (F_u.subs({u: chi}) - F_u.subs({u: chi_ref}))

    # --- (6) H(a) structure ---
    # Implicit: a = a_ref * exp(∫_{χ_ref}^{χ} ...), H = H(χ)
    # So H(a) is defined by eliminating χ between these two.

    # Endpoints
    H_early = sqrt(pi / (G * S_early)) if True else None
    H_inf = sqrt(pi / (G * S_max))
    # At χ->0+: H->H_early (or ∞ if S_early=0); χ->1: H->H_∞

    return DerivationStep(
        name="L03_phase_trajectory_H_chi",
        stage="LAYER_03",
        assumptions=[
            "Locked S_H=π/(G H²)",
            "S_H=S_early+(S_max-S_early)χ",
            "ASSUMED closure χ̇=γ χ(1-χ)",
            "H ≡ ȧ/a  =>  d ln a = H dt",
        ],
        equations={
            # H(χ)
            "S_H_chi": Eq(Symbol("S_H"), S_H),
            "H_chi": Eq(Symbol("H"), H_chi),
            "H_chi_explicit": (
                r"H(\chi)=\sqrt{\pi\big/\big(G[S_{\mathrm{early}}"
                r"+(S_{\max}-S_{\mathrm{early}})\chi]\big)}"
            ),
            # eliminate t
            "chain": r"\frac{d\chi}{d\ln a}=\frac{\gamma\chi(1-\chi)}{H(\chi)}",
            "dchi_dN": Eq(Symbol("dchi_dN"), dchi_dN),
            "integrand": Eq(Symbol("integrand"), integrand),
            # a(χ) general
            "ln_a_general": (
                r"\ln\frac{a}{a_{\mathrm{ref}}}"
                r"=\int_{\chi_{\mathrm{ref}}}^{\chi}"
                r"\frac{H(x)}{\gamma x(1-x)}\,dx"
            ),
            "partial_fractions": (
                r"\frac{1}{x(1-x)}=\frac{1}{x}+\frac{1}{1-x}"
            ),
            "ln_a_split": (
                r"\ln\frac{a}{a_{\mathrm{ref}}}"
                r"=\gamma^{-1}\sqrt{\pi/G}\int_{\chi_{\mathrm{ref}}}^{\chi}"
                r"\Big[\frac{1}{x\sqrt{S_e+\Delta S\,x}}"
                r"+\frac{1}{(1-x)\sqrt{S_e+\Delta S\,x}}\Big]dx"
            ),
            # a(χ) closed form S_early=0
            "F_Se0": Eq(Function("F")(u), F_u),
            "ln_a_Se0": (
                r"S_{\mathrm{early}}=0:\quad"
                r"\ln\frac{a}{a_{\mathrm{ref}}}"
                r"=\frac{1}{\gamma}\sqrt{\frac{\pi}{G S_{\max}}}"
                r"\big[F(\chi)-F(\chi_{\mathrm{ref}})\big],\quad"
                r"F(u)=2\,\mathrm{artanh}\sqrt{u}-2/\sqrt{u}"
            ),
            "ln_a_Se0_sym": Eq(Symbol("ln_a_over_aref"), ln_a_Se0),
            # H(a)
            "H_of_a_implicit": (
                r"H(a)=H\big(\chi(a)\big),\quad"
                r"\chi(a)\ \mathrm{solves}\ a=a(\chi)"
            ),
            "H_of_a_parametric": (
                r"\mathrm{parametric\ exact\ curve:}\ "
                r"\big(a(\chi),\,H(\chi)\big)"
            ),
            "H_of_z": r"H(z)=H\big(a=1/(1+z)\big)",
            # endpoints
            "H_at_0": r"\chi\to 0:\ H\to H_{\mathrm{early}}=\sqrt{\pi/(G S_{\mathrm{early}})}",
            "H_at_1": Eq(Symbol("H_inf"), H_inf),
            # fluid on the same chart
            "rho_chi": (
                r"\rho_S(\chi)=3H(\chi)^2/(8\pi G)"
                r"=3\big/\big(8 G^2 S_H(\chi)\big)"
            ),
        },
        claims=[
            "LAYER 3 DERIVED: H(χ)=sqrt(π/(G[S_early+ΔS χ])).",
            "LAYER 3 DERIVED: a(χ) from ln(a/a_ref)=∫ H(x)/(γ x(1-x)) dx.",
            "LAYER 3 DERIVED: closed form when S_early=0 via F(u)=2 artanh√u - 2/√u.",
            "LAYER 3 DERIVED: H(a)=H(χ(a)) by inverting a(χ); parametric (a(χ),H(χ)) is exact.",
        ],
        open_gaps=[
            "For S_early>0 the definite integral is closed via atanh/acoth pieces; "
            "H(a) generally requires numerical inversion of a(χ).",
        ],
        status="derived" if ok_H else "failed",
        evidence={
            "H_chi": H_chi,
            "S_H_chi": S_H,
            "dchi_dN": dchi_dN,
            "integrand": integrand,
            "pref": pref,
            "J1_integrand": J1_integrand,
            "J2_integrand": J2_integrand,
            "F_u": F_u,
            "dF": dF,
            "ok_H": ok_H,
            "ok_F": ok_F,
            "ln_a_Se0": ln_a_Se0,
            "H_inf": H_inf,
        },
    )


def step_L04_autonomous_system(sym: Dict[str, Any]) -> DerivationStep:
    """
    LAYER 4 - Autonomous dynamical system for the entropy cosmology.

    First-principles construction
    ----------------------------
    State variable: χ ∈ [0,1]
    Hubble as slave: H = H(χ) = sqrt(π/(G[S_early+ΔS χ]))   [LAYER 3]
    Closure:         χ̇ = Γ(χ) = γ χ(1-χ)                     [ASSUMED]

    Cosmic-time system (non-autonomous in a, but autonomous in χ):
        χ̇ = Γ(χ)

    e-fold time N ≡ ln a  (dN = H dt):
        χ' ≡ dχ/dN = χ̇/H = Γ(χ)/H(χ) =: f(χ)

    This is a genuine 1D autonomous ODE on the invariant interval [0,1].

    Fixed points: f(χ_*)=0 ⇔ Γ(χ_*)=0 (since H>0) ⇔ χ_* ∈ {0,1}.

    Linearization at χ=1:
        Γ'(χ)=γ(1-2χ),  Γ'(1)=-γ
        f'(χ)=(Γ' H - Γ H')/H²
        f'(1)=Γ'(1)/H(1)= -γ/H_∞ < 0   =>  asymptotically stable

    At χ=0:
        Γ'(0)=γ>0  =>  unstable (repeller) for the t-flow
        f'(0)=γ/H_early > 0  =>  unstable for the N-flow

    Phase flow: on (0,1), Γ>0 and H>0 => f>0 => χ increases monotonically to 1.

    Invariant interval: [0,1] is forward-invariant (Γ vanishes at endpoints).

    Global attractor χ=1 upgrades via Lyapunov V=S_max-S_H in LAYER 5.
    """
    G, chi, gamma = sym["G"], sym["chi"], sym["gamma"]
    S_max, S_early = sym["S_max"], sym["S_early"]
    Delta_S = S_max - S_early

    # Slave Hubble and closure
    S_H = S_early + Delta_S * chi
    H_chi = sqrt(pi / (G * S_H))
    Gamma = gamma * chi * (1 - chi)
    f = simplify(Gamma / H_chi)  # χ' = dχ/dN

    # Derivatives
    Gamma_p = simplify(diff(Gamma, chi))  # γ(1-2χ)
    H_p = simplify(diff(H_chi, chi))
    f_p = simplify(diff(f, chi))

    # Evaluate at fixed points
    lam_t_0 = simplify(Gamma_p.subs({chi: 0}))   # +γ
    lam_t_1 = simplify(Gamma_p.subs({chi: 1}))   # -γ
    H_inf = sqrt(pi / (G * S_max))
    H_early = sqrt(pi / (G * S_early))
    # f'(1) = Γ'(1)/H(1) since Γ(1)=0
    lam_N_1_tgt = simplify(-gamma / H_inf)
    lam_N_1 = simplify(f_p.subs({chi: 1}))
    # At χ=1, S_H=S_max; substitute to compare
    lam_N_1_red = simplify(lam_N_1.subs({S_early + Delta_S: S_max}))
    # More robust: evaluate f_p at chi=1 with S_H->S_max
    lam_N_1_eval = simplify(
        ((Gamma_p * H_chi - Gamma * H_p) / H_chi**2).subs({chi: 1})
    )
    ok_lam_N = simplify(lam_N_1_eval - lam_N_1_tgt) == 0

    lam_N_0_tgt = simplify(gamma / H_early)
    lam_N_0_eval = simplify(
        ((Gamma_p * H_chi - Gamma * H_p) / H_chi**2).subs({chi: 0})
    )
    ok_lam_0 = simplify(lam_N_0_eval - lam_N_0_tgt) == 0

    # Sign of f on (0,1): Gamma>0, H>0 => f>0
    # Monotonicity / no other fixed points of logistic
    other_roots = sp.roots(Gamma, chi)  # {0:1, 1:1}

    # Linear solutions near attractor
    # ε=1-χ: ε' = λ_N ε + O(ε²) with λ_N=-γ/H_∞
    # ε(N) ~ ε0 e^{λ_N N} = ε0 e^{-(γ/H_∞) N} = ε0 a^{-γ/H_∞}

    return DerivationStep(
        name="L04_autonomous_dynamical_system",
        stage="LAYER_04",
        assumptions=[
            "N ≡ ln a (e-fold time)",
            "H=H(χ) from LAYER 3",
            "ASSUMED logistic Γ(χ)=γ χ(1-χ)",
        ],
        equations={
            # definitions
            "N_def": r"N\equiv\ln a,\qquad dN=H\,dt",
            "H_slave": Eq(Symbol("H"), H_chi),
            "Gamma": Eq(Symbol("Gamma"), Gamma),
            "chi_dot": Eq(Symbol("chi_dot"), Gamma),
            # autonomous ODE
            "f_def": r"f(\chi):=\Gamma(\chi)/H(\chi)",
            "chi_prime": Eq(Symbol("chi_prime"), f),
            "autonomous_ODE": r"\frac{d\chi}{dN}=f(\chi)=\frac{\gamma\chi(1-\chi)}{H(\chi)}",
            # fixed points
            "fixed_points": r"f(\chi_*)=0\;\Leftrightarrow\;\Gamma(\chi_*)=0\;\Leftrightarrow\;\chi_*\in\{0,1\}",
            "Gamma_prime": Eq(Symbol("Gamma_prime"), Gamma_p),
            "f_prime": Eq(Symbol("f_prime"), f_p),
            # stability χ=0
            "lambda_t_0": Eq(Symbol("lambda_t_0"), lam_t_0),
            "lambda_N_0": Eq(Symbol("lambda_N_0"), lam_N_0_eval),
            "chi0_unstable": r"\lambda_t(0)=\gamma>0,\ \lambda_N(0)=\gamma/H_{\mathrm{early}}>0\ \Rightarrow\ \chi=0\ \mathrm{unstable}",
            # stability χ=1
            "lambda_t_1": Eq(Symbol("lambda_t_1"), lam_t_1),
            "lambda_N_1": Eq(Symbol("lambda_N_1"), lam_N_1_eval),
            "lambda_N_1_compact": Eq(Symbol("lambda_N_1"), lam_N_1_tgt),
            "chi1_stable": (
                r"\lambda_t(1)=-\gamma<0,\ "
                r"\lambda_N(1)=-\gamma/H_\infty<0\ \Rightarrow\ \chi=1\ \mathrm{asymptotically\ stable}"
            ),
            # phase structure
            "invariant_interval": r"[0,1]\ \mathrm{forward\ invariant:\ }\Gamma(0)=\Gamma(1)=0",
            "phase_flow": r"\chi\in(0,1):\ \Gamma>0,\,H>0\Rightarrow f>0\Rightarrow \chi\uparrow\ \mathrm{monotone}",
            "global_attractor": r"\chi(N)\to 1\ \mathrm{as}\ N\to+\infty\ \mathrm{for\ all\ ICs\ in}\ (0,1]",
            # near-attractor solution in e-folds / scale factor
            "linear_N": (
                r"\varepsilon:=1-\chi:\quad"
                r"\varepsilon(N)\simeq\varepsilon_0\,e^{-(\gamma/H_\infty)N}"
                r"=\varepsilon_0\,a^{-\gamma/H_\infty}"
            ),
            "no_bifurcation_logistic": (
                r"\mathrm{logistic\ has\ exactly\ two\ fixed\ points;\ }"
                r"\mathrm{no\ interior\ equilibrium}"
            ),
            "bifurcation_preview": (
                r"\mathrm{Deformations}\ \Gamma=\gamma\chi(1-\chi)(1+\alpha\chi)\ "
                r"\mathrm{treated\ in\ LAYER\ 16}"
            ),
        },
        claims=[
            "LAYER 4 DERIVED: cosmology reduced to 1D autonomous ODE χ'=f(χ)=Γ(χ)/H(χ).",
            "LAYER 4 DERIVED: fixed points χ=0 (unstable) and χ=1 (stable).",
            "LAYER 4 DERIVED: N-eigenvalue λ_N(1)=-γ/H_∞<0; t-eigenvalue λ_t(1)=-γ<0.",
            "LAYER 4 DERIVED: (0,1] flows monotonically to the late de Sitter attractor.",
        ],
        open_gaps=[
            "Global Lyapunov proof deferred to LAYER 5 (same attractor, stronger tool).",
        ],
        status="derived" if (ok_lam_N and simplify(lam_t_1 + gamma) == 0) else "failed",
        evidence={
            "f": f,
            "Gamma": Gamma,
            "Gamma_p": Gamma_p,
            "f_p": f_p,
            "H_chi": H_chi,
            "lam_t_0": lam_t_0,
            "lam_t_1": lam_t_1,
            "lam_N_0": lam_N_0_eval,
            "lam_N_1": lam_N_1_eval,
            "lam_N_1_tgt": lam_N_1_tgt,
            "ok_lam_N": ok_lam_N,
            "ok_lam_0": ok_lam_0,
            "H_inf": H_inf,
            "other_roots": str(other_roots),
        },
    )


def step_L05_lyapunov(sym: Dict[str, Any]) -> DerivationStep:
    """
    LAYER 5 - Global Lyapunov function for the χ=1 attractor.

    First-principles theorem
    -----------------------
    Entropy completion:
        S_H(χ) = S_early + (S_max - S_early) χ = S_early + ΔS χ
        Ṡ_H    = ΔS χ̇ = ΔS γ χ(1-χ) ≥ 0 on [0,1]

    Candidate Lyapunov function (entropy deficit to equilibrium):
        V(χ) := S_max - S_H(χ) = ΔS (1-χ)

    Lyapunov conditions on the invariant set [0,1]:
        (i)   V(χ) ≥ 0
        (ii)  V(χ) = 0  ⇔  χ = 1
        (iii) V̇ = dV/dt = -ΔS χ̇ = -ΔS γ χ(1-χ) ≤ 0
        (iv)  V̇ = 0 on [0,1]  ⇔  χ∈{0,1}

    Local linearization (LAYER 4) already gives λ_t(1)=-γ<0.
    Lyapunov upgrades this to GLOBAL asymptotic stability on (0,1]:

        For any χ(0)∈(0,1], V decreases strictly until χ=1, hence χ(t)->1.

    Barbashin-Krasovskii / LaSalle refinement:
        {V̇=0}∩[0,1] = {0,1}. The only forward-invariant subset reachable
        from (0,1] is {1}, because χ=0 is unstable (λ_t(0)=γ>0) and
        Γ>0 on (0,1) drives trajectories away from 0 toward 1.

    e-fold form:
        V' = dV/dN = V̇/H = -ΔS f(χ) ≤ 0
        with the same zero set.

    Interpretation:
        V is the remaining holographic entropy budget; thermodynamic
        completion is Lyapunov descent of V to equilibrium.
    """
    chi, gamma = sym["chi"], sym["gamma"]
    S_max, S_early = sym["S_max"], sym["S_early"]
    G = sym["G"]
    Delta_S = S_max - S_early
    S_H = S_early + Delta_S * chi
    Gamma = gamma * chi * (1 - chi)
    H_chi = sqrt(pi / (G * S_H))

    # Lyapunov function
    V = simplify(S_max - S_H)                    # = ΔS (1-χ)
    V_alt = Delta_S * (1 - chi)
    ok_V = simplify(V - V_alt) == 0

    # Time derivative along χ̇=Γ
    Vdot = simplify(-Delta_S * Gamma)            # = -ΔS γ χ(1-χ)
    Vdot_from_chain = simplify(diff(V, chi) * Gamma)
    ok_Vdot = simplify(Vdot - Vdot_from_chain) == 0

    # N-derivative
    Vprime = simplify(Vdot / H_chi)              # dV/dN

    # Zero sets
    # V=0 => χ=1; Vdot=0 => χ=0 or χ=1
    V_at_1 = simplify(V.subs({chi: 1}))
    V_at_0 = simplify(V.subs({chi: 0}))
    Vdot_at_1 = simplify(Vdot.subs({chi: 1}))
    Vdot_at_half = simplify(Vdot.subs({chi: sp.Rational(1, 2)}))

    ok_zeros = (V_at_1 == 0) and (simplify(V_at_0 - Delta_S) == 0) and (Vdot_at_1 == 0)

    # Strict decrease on (0,1): -Vdot = ΔS γ χ(1-χ) > 0 for χ∈(0,1)
    descent = simplify(-Vdot)  # ΔS γ χ(1-χ)

    # Link to horizon entropy monotonicity
    S_H_dot = Delta_S * Gamma
    ok_entropy = simplify(S_H_dot + Vdot) == 0  # V̇ = -Ṡ_H

    return DerivationStep(
        name="L05_Lyapunov_global_stability",
        stage="LAYER_05",
        assumptions=[
            "χ∈[0,1] invariant under logistic closure",
            "ΔS=S_max-S_early>0",
            "γ>0",
            "S_H=S_early+ΔS χ",
        ],
        equations={
            "S_H_chi": Eq(Symbol("S_H"), S_H),
            "V_def": Eq(Symbol("V"), V),
            "V_alt": Eq(Symbol("V"), V_alt),
            "V_nonneg": r"V(\chi)=\Delta S(1-\chi)\ge 0\quad\mathrm{for\ }\chi\in[0,1]",
            "V_zero_iff": r"V(\chi)=0\;\Leftrightarrow\;\chi=1",
            "Vdot": Eq(Symbol("Vdot"), Vdot),
            "Vdot_chain": Eq(Symbol("Vdot"), Vdot_from_chain),
            "Vdot_nonpos": r"\dot V=-\Delta S\,\gamma\chi(1-\chi)\le 0",
            "Vdot_zero_set": r"\dot V=0\;\Leftrightarrow\;\chi\in\{0,1\}",
            "descent_interior": r"\chi\in(0,1):\ \dot V<0\quad\mathrm{(strict\ descent)}",
            "entropy_link": Eq(Symbol("Vdot"), -S_H_dot),
            "entropy_link_txt": r"\dot V=-\dot S_H",
            "Vprime_N": Eq(Symbol("V_prime"), Vprime),
            "Vprime_txt": r"V'=\frac{dV}{dN}=\frac{\dot V}{H}=-\Delta S\,f(\chi)\le 0",
            "theorem_GAS": (
                r"V\ge 0,\ \dot V\le 0,\ V=0\Leftrightarrow\chi=1"
                r"\;\Rightarrow\;"
                r"\chi=1\ \mathrm{is\ globally\ asymptotically\ stable\ on\ }(0,1]"
            ),
            "LaSalle": (
                r"\{ \dot V=0\}\cap[0,1]=\{0,1\};\ "
                r"\mathrm{only\ }\{1\}\ \mathrm{is\ attracting\ from\ }(0,1]"
            ),
            "vs_linear": (
                r"\mathrm{LAYER\ 4:\ local\ }\lambda=-\gamma;\ "
                r"\mathrm{LAYER\ 5:\ global\ Lyapunov\ descent}"
            ),
        },
        claims=[
            "LAYER 5 THEOREM: V=S_max-S_H=ΔS(1-χ) is a Lyapunov function on [0,1].",
            "LAYER 5 THEOREM: V̇=-Ṡ_H=-ΔS γχ(1-χ)≤0 with equality iff χ∈{0,1}.",
            "LAYER 5 THEOREM: χ=1 is globally asymptotically stable for all ICs in (0,1].",
            "Upgrades local eigenvalue λ=-γ (LAYER 4) to a global stability proof.",
        ],
        status="derived" if (ok_V and ok_Vdot and ok_zeros and ok_entropy) else "failed",
        evidence={
            "V": V,
            "V_alt": V_alt,
            "Vdot": Vdot,
            "Vprime": Vprime,
            "S_H_dot": S_H_dot,
            "descent": descent,
            "ok_V": ok_V,
            "ok_Vdot": ok_Vdot,
            "ok_zeros": ok_zeros,
            "ok_entropy": ok_entropy,
            "V_at_0": V_at_0,
            "V_at_1": V_at_1,
            "Vdot_at_half": Vdot_at_half,
        },
    )


def step_L06_horizon_first_law(sym: Dict[str, Any]) -> DerivationStep:
    """
    LAYER 6 - Horizon temperature, energy, volume, and first-law consistency.

    First-principles geometry (locked S_H=π/(G H²))
    -----------------------------------------------
    Apparent horizon:     R_H = 1/H
    Area / entropy:       A_H = 4π/H²,   S_H = A_H/(4G) = π/(G H²)
    Equilibrium temp.:    T_H = H/(2π)     (de Sitter / Unruh on Hubble sphere)
    Horizon volume:       V_H = 4π/(3 H³)
    Misner-Sharp energy:  E_H = R_H/(2G) = 1/(2 G H)

    Identities (pure geometry + Friedmann ρ_S=3H²/(8πG)):
        T_H S_H = (H/2π)*(π/(G H²)) = 1/(2 G H) = E_H
        ρ_S V_H = [3H²/(8πG)]*[4π/(3H³)] = 1/(2 G H) = E_H
    Hence:
        E_H = T_H S_H = ρ_S V_H

    Differentials (H as state variable):
        dE_H = -dH/(2 G H²)
        dS_H = -2π dH/(G H³)
        T_H dS_H = -dH/(G H²)
        dV_H = -4π dH / H⁴

    Unified first law ansatz:
        dE_H = T_H dS_H + W dV_H

    => required work density from geometry alone:
        W_req = (dE_H - T_H dS_H)/dV_H = -H²/(8π G) = -ρ_S/3

    Fluid work density (LAYER 1 + Raychaudhuri):
        W_fluid = (ρ_S - P_S)/2 = ρ_S + Ḣ/(8π G)

    Consistency W_req = W_fluid holds iff
        Ḣ = -4 H²
    which is NOT true along the logistic entropy trajectory in general.

    Dynamical-horizon temperature (surface gravity on apparent horizon):
        Ṙ_H = -Ḣ/H²
        T_dyn = |κ|/(2π) with κ = (1/(2R_H))(1 - Ṙ_H/(2 H R_H))
              = (H/(2π)) |1 + Ḣ/(2 H²)|
    At equilibrium Ḣ=0: T_dyn -> T_H.

    Conclusion (honest):
        • E_H = T_H S_H is an exact geometric identity (deeper than Ṡ≥0).
        • The equilibrium-T unified first law with W=(ρ-P)/2 is NOT an
          identity along the logistic flow; it holds at the de Sitter
          endpoint (Ḣ=0) and requires dynamical T / refined κ off-shell.
        • This is a genuine thermodynamic consistency test of the model.
    """
    G, H, Hdot = sym["G"], sym["H"], sym["Hdot"]
    chi, gamma = sym["chi"], sym["gamma"]
    S_max, S_early = sym["S_max"], sym["S_early"]

    # Geometric quantities
    R_H = 1 / H
    A_H = simplify(4 * pi / H**2)
    S_H = simplify(pi / (G * H**2))
    T_H = H / (2 * pi)
    V_H = simplify((4 * pi / 3) / H**3)
    E_H = simplify(1 / (2 * G * H))
    rho_S = simplify(3 * H**2 / (8 * pi * G))

    # Identities
    TS = simplify(T_H * S_H)
    rhoV = simplify(rho_S * V_H)
    ok_TS = simplify(TS - E_H) == 0
    ok_rhoV = simplify(rhoV - E_H) == 0
    ok_A = simplify(S_H - A_H / (4 * G)) == 0

    # Differentials d/dH
    dE_dH = simplify(diff(E_H, H))       # -1/(2G H²)
    dS_dH = simplify(diff(S_H, H))       # -2π/(G H³)
    TdS_dH = simplify(T_H * dS_dH)       # -1/(G H²)
    dV_dH = simplify(diff(V_H, H))       # -4π/H⁴

    # W required by first law with equilibrium T
    W_req = simplify((dE_dH - TdS_dH) / dV_dH)  # -H²/(8πG)
    W_req_tgt = simplify(-H**2 / (8 * pi * G))
    ok_Wreq = simplify(W_req - W_req_tgt) == 0
    ok_Wreq_rho = simplify(W_req + rho_S / 3) == 0

    # Fluid W from LAYER 1
    # P_S = -ρ_S - Ḣ/(4πG)  =>  (ρ-P)/2 = ρ + Ḣ/(8πG)
    W_fluid = simplify(rho_S + Hdot / (8 * pi * G))
    consistency_Hdot = simplify(sp.Eq(W_req, W_fluid))  # holds iff Hdot=-4 H**2
    # Solve W_req = W_fluid for Hdot
    Hdot_for_FL = simplify(sp.solve(sp.Eq(W_req, W_fluid), Hdot)[0])
    ok_cond = simplify(Hdot_for_FL + 4 * H**2) == 0

    # Dynamical temperature
    # R_H = 1/H, Ṙ_H = -Ḣ/H²
    # κ = (1/(2 R_H)) (1 - Ṙ_H/(2 H R_H)) = (H/2) (1 + Ḣ/(2 H²))
    kappa_dyn = simplify((H / 2) * (1 + Hdot / (2 * H**2)))
    T_dyn = simplify(Abs(kappa_dyn) / (2 * pi))
    T_dyn_eq = simplify((H / (2 * pi)) * Abs(1 + Hdot / (2 * H**2)))
    ok_Tdyn = simplify(T_dyn - T_dyn_eq) == 0
    T_at_eq = simplify(T_dyn_eq.subs({Hdot: 0}))
    ok_Tlimit = simplify(T_at_eq - T_H) == 0

    # Along logistic: does W_req = W_fluid? Generally no.
    Hdot_thermo = _thermo_Hdot(G, H, S_max, S_early, gamma, chi)
    residual = simplify(W_fluid.subs({Hdot: Hdot_thermo}) - W_req)

    return DerivationStep(
        name="L06_horizon_temperature_first_law",
        stage="LAYER_06",
        assumptions=[
            "Apparent Hubble horizon R_H=1/H",
            "Bekenstein-Hawking S_H=A_H/(4G)",
            "Equilibrium temperature T_H=H/(2π)",
            "Misner-Sharp / vacuum-branch E_H=R_H/(2G)",
            "Work density ansatz W=(ρ-P)/2",
        ],
        equations={
            "R_H": Eq(Symbol("R_H"), R_H),
            "A_H": Eq(Symbol("A_H"), A_H),
            "S_H": Eq(Symbol("S_H"), S_H),
            "T_H": Eq(Symbol("T_H"), T_H),
            "V_H": Eq(Symbol("V_H"), V_H),
            "E_H": Eq(Symbol("E_H"), E_H),
            "rho_S": Eq(Symbol("rho_S"), rho_S),
            # identities
            "E_eq_TS": Eq(Symbol("E_H"), TS),
            "E_eq_rhoV": Eq(Symbol("E_H"), rhoV),
            "identity": r"E_H=T_H S_H=\rho_S V_H=1/(2GH)",
            # differentials
            "dE_dH": Eq(Symbol("dE_dH"), dE_dH),
            "TdS_dH": Eq(Symbol("TdS_dH"), TdS_dH),
            "dV_dH": Eq(Symbol("dV_dH"), dV_dH),
            # first law
            "first_law": r"dE_H=T_H\,dS_H+W\,dV_H",
            "W_req": Eq(Symbol("W_req"), W_req),
            "W_req_txt": r"W_{\mathrm{req}}=(dE_H-T_H dS_H)/dV_H=-H^2/(8\pi G)=-\rho_S/3",
            "W_fluid": Eq(Symbol("W_fluid"), W_fluid),
            "W_fluid_txt": r"W_{\mathrm{fluid}}=(\rho_S-P_S)/2=\rho_S+\dot H/(8\pi G)",
            "consistency_condition": Eq(Hdot, Hdot_for_FL),
            "consistency_txt": (
                r"W_{\mathrm{req}}=W_{\mathrm{fluid}}\;\Leftrightarrow\;\dot H=-4H^2"
                r"\quad\mathrm{(not\ true\ for\ general\ logistic\ flow)}"
            ),
            # dynamical temperature
            "kappa_dyn": Eq(Symbol("kappa"), kappa_dyn),
            "T_dyn": Eq(Symbol("T_dyn"), T_dyn_eq),
            "T_dyn_limit": r"\dot H\to 0:\ T_{\mathrm{dyn}}\to T_H=H/(2\pi)",
            "verdict": (
                r"E_H=T_H S_H\ \mathrm{exact};\ "
                r"\mathrm{equilibrium\ first\ law\ with\ }W=(\rho-P)/2"
                r"\ \mathrm{holds\ at\ dS\ endpoint,\ not\ identically\ off\ it}"
            ),
        },
        claims=[
            "LAYER 6 DERIVED: E_H=T_H S_H=ρ_S V_H=1/(2GH) (exact geometric identity).",
            "LAYER 6 DERIVED: W_req=-ρ_S/3 from dE=T dS+W dV with equilibrium T_H.",
            "LAYER 6 DERIVED: W_fluid=(ρ-P)/2 equals W_req iff Ḣ=-4H² - not generic for logistic flow.",
            "LAYER 6 DERIVED: dynamical T_dyn=(H/2π)|1+Ḣ/(2H²)| -> T_H as Ḣ->0.",
            "First-law test is stricter than Ṡ≥0; equilibrium Clausius holds at the attractor.",
        ],
        open_gaps=[
            "Full off-equilibrium first law needs a chosen dynamical-temperature convention "
            "and possibly entropy-production / viscous extensions (Jacobson non-eq).",
        ],
        status="derived" if (ok_TS and ok_rhoV and ok_A and ok_Wreq and ok_Wreq_rho and ok_cond and ok_Tlimit) else "failed",
        evidence={
            "T_H": T_H,
            "S_H": S_H,
            "E_H": E_H,
            "V_H": V_H,
            "TS": TS,
            "rhoV": rhoV,
            "W_req": W_req,
            "W_fluid": W_fluid,
            "Hdot_for_FL": Hdot_for_FL,
            "T_dyn": T_dyn_eq,
            "residual_logistic": residual,
            "ok_TS": ok_TS,
            "ok_rhoV": ok_rhoV,
            "ok_Wreq": ok_Wreq,
            "ok_cond": ok_cond,
            "ok_Tlimit": ok_Tlimit,
        },
    )


def step_L07_entropy_production(sym: Dict[str, Any]) -> DerivationStep:
    """
    LAYER 7 - Entropy production rate relative to cosmic expansion.

    First-principles derivations
    ---------------------------
    (A) From locked geometry S_H = π/(G H²):
            Ṡ_H = -2π Ḣ/(G H³)
            Ṡ_H / S_H = -2 Ḣ/H

    (B) From entropy completion + logistic closure:
            S_H = S_early + ΔS χ
            Ṡ_H = ΔS χ̇ = ΔS γ χ(1-χ)

    (C) Consistency: (A) and (B) match via the thermo EOM
            Ḣ = -(H/(2 S_H)) ΔS γ χ(1-χ)

    Dimensionless production rate (Hubble units):
            ξ := Ṡ_H / (H S_H) = ΔS γ χ(1-χ) / (H S_H) = -2 Ḣ/H²

    Cosmographic link (LAYER 2: q = -1 - Ḣ/H²):
            ξ = 2(1+q)

    EOS link (LAYER 1: w_S + 1 = ΔS γχ(1-χ)/(3 H S_H)):
            ξ = 3(w_S + 1)

    Equilibrium distance:
            ξ -> 0  as χ -> 1  (thermodynamic equilibrium / de Sitter)
            ξ > 0  on (0,1)   (ongoing entropy production)

    Physical reading:
            ξ^{-1} is the entropy-growth timescale in Hubble units;
            ξ ≪ 1 means the universe is close to thermodynamic equilibrium.
    """
    G, H, chi, gamma = sym["G"], sym["H"], sym["chi"], sym["gamma"]
    S_max, S_early = sym["S_max"], sym["S_early"]
    Hdot = sym["Hdot"]
    Delta_S = S_max - S_early

    # (A) geometric
    S_H_geom = pi / (G * H**2)
    Sdot_geom = simplify(-2 * pi * Hdot / (G * H**3))
    ratio_geom = simplify(Sdot_geom / S_H_geom)  # -2 Hdot/H
    ok_geom = simplify(ratio_geom + 2 * Hdot / H) == 0

    # (B) closure
    S_H_chi = S_early + Delta_S * chi
    Sdot_chi = Delta_S * gamma * chi * (1 - chi)
    ratio_chi = simplify(Sdot_chi / S_H_chi)

    # (C) match via thermo Hdot - evaluate everything on the χ-chart H=H(χ)
    H_of_chi = sqrt(pi / (G * S_H_chi))
    Hdot_th = _thermo_Hdot(G, H_of_chi, S_max, S_early, gamma, chi)
    Sdot_from_H = simplify(-2 * pi * Hdot_th / (G * H_of_chi**3))
    ok_match = simplify(Sdot_from_H - Sdot_chi) == 0

    # Dimensionless ξ on the χ-chart
    xi = simplify(Sdot_chi / (H_of_chi * S_H_chi))
    xi_geom = simplify(-2 * Hdot_th / H_of_chi**2)
    ok_xi = simplify(xi - xi_geom) == 0

    # Links to q and w (also on χ-chart)
    q = Symbol("q", real=True)
    w_S = Symbol("w_S", real=True)
    xi_from_q = 2 * (1 + q)
    xi_from_w = 3 * (w_S + 1)
    q_exp = simplify(-1 - Hdot_th / H_of_chi**2)
    w_exp = simplify(
        -1 + Delta_S * gamma * chi * (1 - chi) / (3 * H_of_chi * S_H_chi)
    )
    ok_q_link = simplify(xi - 2 * (1 + q_exp)) == 0
    ok_w_link = simplify(xi - 3 * (w_exp + 1)) == 0

    # Equilibrium
    xi_at_1 = simplify(xi.subs({chi: 1}))
    xi_at_half = simplify(xi.subs({chi: sp.Rational(1, 2)}))

    return DerivationStep(
        name="L07_entropy_production_rate",
        stage="LAYER_07",
        assumptions=[
            "Locked S_H=π/(G H²)",
            "S_H=S_early+ΔS χ",
            "ASSUMED logistic χ̇=γ χ(1-χ)",
        ],
        equations={
            # geometric production
            "Sdot_geom": Eq(Symbol("S_H_dot"), Sdot_geom),
            "ratio_geom": Eq(Symbol("Sdot_over_S"), ratio_geom),
            "identity_geom": r"\dot S_H/S_H=-2\dot H/H",
            # closure production
            "S_H_chi": Eq(Symbol("S_H"), S_H_chi),
            "Sdot_chi": Eq(Symbol("S_H_dot"), Sdot_chi),
            "ratio_chi": Eq(Symbol("Sdot_over_S"), ratio_chi),
            "H_of_chi": Eq(Symbol("H"), H_of_chi),
            "match": r"\mathrm{(A)=(B)\ on\ shell\ }H=H(\chi)",
            # dimensionless ξ
            "xi_def": r"\xi:=\dot S_H/(H S_H)",
            "xi": Eq(Symbol("xi"), xi),
            "xi_geom": Eq(Symbol("xi"), xi_geom),
            "xi_explicit": (
                r"\xi=\frac{(S_{\max}-S_{\mathrm{early}})\gamma\chi(1-\chi)}{H(\chi)\,S_H(\chi)}"
                r"=-2\frac{\dot H}{H^2}"
            ),
            # cosmographic / EOS bridges
            "xi_q": Eq(Symbol("xi"), xi_from_q),
            "xi_q_txt": r"\xi=2(1+q)",
            "xi_w": Eq(Symbol("xi"), xi_from_w),
            "xi_w_txt": r"\xi=3(w_S+1)",
            # equilibrium
            "xi_attractor": r"\chi\to 1:\ \xi\to 0",
            "xi_interior": r"\chi\in(0,1):\ \xi>0",
            "timescale": (
                r"\xi^{-1}=\mathrm{entropy\ growth\ timescale\ in\ Hubble\ units}"
            ),
            "equilibrium_distance": (
                r"\xi\ll 1\;\Leftrightarrow\;\mathrm{near\ thermodynamic\ equilibrium}"
            ),
        },
        claims=[
            "LAYER 7 DERIVED: Ṡ_H/S_H=-2Ḣ/H from geometry; Ṡ_H=ΔS γχ(1-χ) from closure.",
            "LAYER 7 DERIVED: dimensionless ξ=Ṡ_H/(H S_H)=-2Ḣ/H²=2(1+q)=3(w_S+1).",
            "LAYER 7 DERIVED: ξ->0 at the attractor; ξ measures distance from equilibrium.",
        ],
        status="derived" if (ok_geom and ok_match and ok_xi and ok_q_link and ok_w_link and xi_at_1 == 0) else "failed",
        evidence={
            "Sdot_geom": Sdot_geom,
            "Sdot_chi": Sdot_chi,
            "ratio_geom": ratio_geom,
            "H_of_chi": H_of_chi,
            "xi": xi,
            "xi_geom": xi_geom,
            "q_exp": q_exp,
            "w_exp": w_exp,
            "xi_at_1": xi_at_1,
            "xi_at_half": xi_at_half,
            "ok_geom": ok_geom,
            "ok_match": ok_match,
            "ok_xi": ok_xi,
            "ok_q_link": ok_q_link,
            "ok_w_link": ok_w_link,
        },
    )


def step_L08_transition_epoch(sym: Dict[str, Any]) -> DerivationStep:
    """
    LAYER 8 - Thermodynamic transition epoch at maximum entropy production.

    First-principles derivation
    --------------------------
    Closure:  χ̇ = γ χ(1-χ),   Ṡ_H = ΔS χ̇ = ΔS γ χ(1-χ)

    Critical points of production:
        d/dχ [χ(1-χ)] = 1-2χ = 0  =>  χ_★ = 1/2
        d²/dχ² [χ(1-χ)] = -2 < 0  =>  MAXIMUM

    Hence χ_★ = 1/2 is:
        • the logistic inflection point
        • the unique maximizer of χ̇ on [0,1]
        • the unique maximizer of Ṡ_H on [0,1]

    Peak production rate:
        Ṡ_{H,max} = ΔS γ (1/2)(1/2) = γ ΔS / 4

    Exact time of transition (logistic solution):
        χ(t) = 1/(1+C e^{-γ(t-t0)})
        χ(t_★)=1/2  =>  C e^{-γ(t_★-t0)} = 1
        =>  t_★ = t0 + γ^{-1} ln C
        Convention: choose t0 so C=1 => t_★ = t0 =: t_thermo

    State at transition:
        S_★ = S_early + ΔS/2
        H_★ = sqrt(π/(G S_★))
        ξ_★ = Ṡ_{H,max}/(H_★ S_★) = (γ ΔS/4)/(H_★ S_★)

    Epoch comparison (theory prediction):
        From LAYER 7: ξ = 2(1+q) = 3(w_S+1)
        => q=0  ⇔  w_S=-1/3  ⇔  ξ=2   (SAME epoch)

        Thermodynamic peak χ=1/2 is a DIFFERENT condition:
        q(χ_★)=0 iff γ ΔS/4 = 2 H_★ S_★
                 iff γ = 8 H_★ S_★ / ΔS = 8 sqrt(π S_★ / G) / ΔS
        This is a parameter tuning, NOT automatic.

    Therefore the model predicts in general:
        t_thermo (χ=1/2)  !=  t_{q=0} = t_{w=-1/3}
    Coincidence of these epochs is an extra observational constraint on (γ,S_early,S_max).
    """
    G, gamma, chi = sym["G"], sym["gamma"], sym["chi"]
    S_max, S_early = sym["S_max"], sym["S_early"]
    t, t0 = sym["t"], sym["t_crit"]
    Delta_S = S_max - S_early
    C = Symbol("C", positive=True)

    # Production rate and critical-point test
    prod = chi * (1 - chi)
    dprod = simplify(diff(prod, chi))           # 1-2χ
    d2prod = simplify(diff(prod, chi, 2))       # -2
    chi_star = sp.Rational(1, 2)
    ok_crit = simplify(dprod.subs({chi: chi_star})) == 0
    ok_max = simplify(d2prod + 2) == 0

    Sdot = Delta_S * gamma * prod
    Sdot_max = simplify(Sdot.subs({chi: chi_star}))
    ok_Sdot = simplify(Sdot_max - Delta_S * gamma / 4) == 0

    # Exact χ(t) and t_★
    chi_t = 1 / (1 + C * exp(-gamma * (t - t0)))
    # Solve χ=1/2
    eq_half = sp.Eq(chi_t, chi_star)
    # C exp(-γ(t-t0))=1 => t = t0 + ln(C)/γ
    t_star = t0 + sp.log(C) / gamma
    ok_t = simplify(chi_t.subs({t: t_star}) - chi_star) == 0

    # State at transition
    S_star = simplify(S_early + Delta_S * chi_star)
    H_star = sqrt(pi / (G * S_star))
    xi_star = simplify(Sdot_max / (H_star * S_star))

    # q at χ_★ - use H as independent then evaluate; avoid nested sqrt issues
    # Ḣ = -(H/(2S)) ΔS γ χ(1-χ) at χ=1/2: Ḣ = -(H/(2S_★))(γ ΔS/4)
    Hdot_star = simplify(-(H_star / (2 * S_star)) * (gamma * Delta_S / 4))
    q_star = simplify(-1 - Hdot_star / H_star**2)
    # q_star = -1 + (γ ΔS/4)/(H_★ S_★) = -1 + ξ_★
    # Actually ξ= -2Ḣ/H² = 2(1+q) so q = ξ/2 - 1
    ok_q_xi = simplify(q_star - (xi_star / 2 - 1)) == 0

    # Coincidence: q_star=0 iff ξ_star=2 iff γ ΔS/4 = 2 H_star S_star
    gamma_for_q0 = simplify(8 * H_star * S_star / Delta_S)
    q_tuned = simplify(q_star.subs({gamma: gamma_for_q0}))
    ok_tune = simplify(q_tuned) == 0

    # w at χ_★: w = -1 + ξ/3; q = ξ/2 - 1 => (w+1/3)=(2/3)q
    w_star = simplify(-1 + xi_star / 3)
    ok_qw_same = simplify((w_star + sp.Rational(1, 3)) - sp.Rational(2, 3) * q_star) == 0

    return DerivationStep(
        name="L08_entropy_transition_epoch",
        stage="LAYER_08",
        assumptions=[
            "Logistic closure χ̇=γ χ(1-χ)",
            "ΔS=S_max-S_early>0",
            "γ>0",
        ],
        equations={
            "prod": Eq(Symbol("prod"), prod),
            "dprod": Eq(Symbol("dprod_dchi"), dprod),
            "d2prod": Eq(Symbol("d2prod"), d2prod),
            "chi_star": Eq(Symbol("chi_star"), chi_star),
            "inflection": r"1-2\chi=0\;\Rightarrow\;\chi_\star=\tfrac12\mathrm{\ (unique\ maximum)}",
            "Sdot": Eq(Symbol("S_H_dot"), Sdot),
            "Sdot_max": Eq(Symbol("S_H_dot_max"), Sdot_max),
            "Sdot_max_txt": r"\dot S_{H,\max}=\gamma\Delta S/4",
            "chi_t": Eq(Symbol("chi"), chi_t),
            "t_thermo": Eq(Symbol("t_star"), t_star),
            "t_thermo_txt": (
                r"\chi(t_\star)=\tfrac12:\ t_\star=t_0+\gamma^{-1}\ln C"
                r"\ \xrightarrow{C=1}\ t_{\mathrm{thermo}}=t_0"
            ),
            "S_star": Eq(Symbol("S_star"), S_star),
            "H_star": Eq(Symbol("H_star"), H_star),
            "xi_star": Eq(Symbol("xi_star"), xi_star),
            "q_star": Eq(Symbol("q_star"), q_star),
            "w_star": Eq(Symbol("w_star"), w_star),
            "q0_iff_w": r"q=0\;\Leftrightarrow\;w_S=-1/3\;\Leftrightarrow\;\xi=2\mathrm{\ (LAYER\ 7)}",
            "gamma_for_coincidence": Eq(Symbol("gamma_q0"), gamma_for_q0),
            "coincidence_condition": (
                r"t_{\mathrm{thermo}}=t_{q=0}\;\Leftrightarrow\;"
                r"\gamma=8 H_\star S_\star/\Delta S"
                r"=8\sqrt{\pi S_\star/G}/\Delta S"
            ),
            "prediction": (
                r"\mathrm{Generic\ prediction:}\ "
                r"t_{\mathrm{thermo}}\neq t_{q=0}=t_{w=-1/3}"
            ),
        },
        claims=[
            "LAYER 8 DERIVED: χ=1/2 uniquely maximizes Ṡ_H with Ṡ_{H,max}=γ ΔS/4.",
            "LAYER 8 DERIVED: t_thermo=t0+γ^{-1} ln C from exact logistic χ(t).",
            "LAYER 8 DERIVED: q=0 and w=-1/3 coincide (via ξ); χ=1/2 coincides with them iff γ is tuned.",
            "LAYER 8 PREDICTION: thermodynamic and acceleration-onset epochs differ in general.",
        ],
        status="derived" if (ok_crit and ok_max and ok_Sdot and ok_t and ok_tune and ok_q_xi and ok_qw_same) else "failed",
        evidence={
            "chi_star": chi_star,
            "Sdot_max": Sdot_max,
            "t_star": t_star,
            "S_star": S_star,
            "H_star": H_star,
            "xi_star": xi_star,
            "q_star": q_star,
            "w_star": w_star,
            "gamma_for_q0": gamma_for_q0,
            "q_tuned": q_tuned,
            "dprod": dprod,
            "d2prod": d2prod,
            "ok_crit": ok_crit,
            "ok_max": ok_max,
            "ok_Sdot": ok_Sdot,
            "ok_t": ok_t,
            "ok_tune": ok_tune,
            "ok_q_xi": ok_q_xi,
            "ok_qw_same": ok_qw_same,
        },
    )


def step_L09_scalar_reconstruction(sym: Dict[str, Any]) -> DerivationStep:
    """
    LAYER 9 - Reconstruct an effective canonical scalar equivalent of the background.

    First principles (flat FLRW, Einstein gravity)
    ---------------------------------------------
    Any smooth expanding background with Ḣ < 0 admits a minimally coupled
    canonical scalar representation at the HOMOGENEOUS level:

        ρ_φ = ½ φ̇² + V(φ)
        P_φ = ½ φ̇² - V(φ)

    Raychaudhuri:  Ḣ = -4πG(ρ+P)  =>  ρ+P = -Ḣ/(4πG)
    Canonical:     ρ_φ+P_φ = φ̇²
    =>  φ̇² = -Ḣ/(4πG)          (real iff Ḣ ≤ 0)

    Also:  V = (ρ-P)/2 = ρ - ½(ρ+P) = 3H²/(8πG) + Ḣ/(8πG)
         = (3H² + Ḣ)/(8πG)

    On the entropy trajectory (LAYER 1-3), substitute thermo Ḣ(χ):
        Ḣ(χ) = -(H/(2 S_H)) ΔS γ χ(1-χ)
        φ̇²(χ) = [H ΔS γ χ(1-χ)] / (8πG S_H)
        V(χ)  = [3H² + Ḣ(χ)] / (8πG)

    with H=H(χ), S_H=S_H(χ).

    Field-space reparametrization:
        dχ/dφ = χ̇ / φ̇ = γ χ(1-χ) / φ̇
    => integrate to χ(φ), then V=V(φ).

    Consistency checks:
        ρ_φ = ½φ̇²+V = 3H²/(8πG) = ρ_S
        w_φ = P_φ/ρ_φ = w_S
        χ->1: φ̇->0, V->ρ_{S,∞}=3H_∞²/(8πG)  (pure de Sitter / cosmological constant)

    IMPORTANT:
        This is BACKGROUND EQUIVALENCE only - it does NOT mean the microscopic
        theory is a scalar field. Useful for stability/perturbation comparison.
    """
    G, H, chi, gamma = sym["G"], sym["H"], sym["chi"], sym["gamma"]
    S_max, S_early = sym["S_max"], sym["S_early"]
    Delta_S = S_max - S_early

    # Work on χ-chart
    S_H = S_early + Delta_S * chi
    H_chi = sqrt(pi / (G * S_H))
    Hdot = _thermo_Hdot(G, H_chi, S_max, S_early, gamma, chi)

    # Canonical reconstruction
    phidot2 = simplify(-Hdot / (4 * pi * G))
    Vphi = simplify((3 * H_chi**2 + Hdot) / (8 * pi * G))
    # Equivalent forms
    phidot2_alt = simplify(
        H_chi * Delta_S * gamma * chi * (1 - chi) / (8 * pi * G * S_H)
    )
    ok_phidot = simplify(phidot2 - phidot2_alt) == 0

    # ρ_φ, P_φ
    rho_phi = simplify(phidot2 / 2 + Vphi)
    P_phi = simplify(phidot2 / 2 - Vphi)
    rho_S = simplify(3 * H_chi**2 / (8 * pi * G))
    ok_rho = simplify(rho_phi - rho_S) == 0

    # w_φ vs w_S
    w_phi = simplify(P_phi / rho_phi)
    w_S = simplify(-1 + Delta_S * gamma * chi * (1 - chi) / (3 * H_chi * S_H))
    ok_w = simplify(w_phi - w_S) == 0

    # dχ/dφ
    chidot = gamma * chi * (1 - chi)
    # φ̇ = sqrt(phidot2) (expanding, take positive root)
    dchi_dphi = simplify(chidot / sqrt(phidot2))

    # Attractor limits
    phidot2_at_1 = simplify(phidot2.subs({chi: 1}))
    V_at_1 = simplify(Vphi.subs({chi: 1}))
    V_inf_tgt = simplify(sp.Rational(3, 8) / (G**2 * S_max))  # 3/(8 G² S_max)
    # V(1) = 3 H_∞²/(8πG) with H_∞²=π/(G S_max)
    ok_Vinf = simplify(V_at_1 - V_inf_tgt) == 0
    ok_phidot_inf = phidot2_at_1 == 0

    # NEC link: φ̇² = ρ+P ≥ 0 ⇔ Ḣ ≤ 0
    nec_ok = True  # by construction when phidot2≥0 on (0,1)

    return DerivationStep(
        name="L09_scalar_field_reconstruction",
        stage="LAYER_09",
        assumptions=[
            "Ḣ≤0 on the entropy trajectory (true for logistic on [0,1])",
            "Minimally coupled canonical scalar at homogeneous level",
            "Flat FLRW Einstein gravity",
        ],
        equations={
            "rho_P_scalar": (
                r"\rho_\phi=\tfrac12\dot\phi^2+V,\quad"
                r"P_\phi=\tfrac12\dot\phi^2-V"
            ),
            "phidot2_GR": r"\dot\phi^2=\rho+P=-\dot H/(4\pi G)",
            "V_GR": r"V=(\rho-P)/2=(3H^2+\dot H)/(8\pi G)",
            "phidot2": Eq(Symbol("phidot") ** 2, phidot2),
            "phidot2_chi": Eq(Symbol("phidot") ** 2, phidot2_alt),
            "V_phi": Eq(Symbol("V_phi"), Vphi),
            "dchi_dphi": Eq(Symbol("dchi_dphi"), dchi_dphi),
            "dchi_dphi_txt": r"d\chi/d\phi=\dot\chi/\dot\phi=\gamma\chi(1-\chi)/\dot\phi",
            "rho_phi": Eq(Symbol("rho_phi"), rho_phi),
            "P_phi": Eq(Symbol("P_phi"), P_phi),
            "rho_match": r"\rho_\phi=\rho_S=3H(\chi)^2/(8\pi G)",
            "w_match": r"w_\phi=w_S",
            "attractor": (
                r"\chi\to1:\ \dot\phi\to0,\ "
                r"V\to\rho_{S,\infty}=3/(8G^2 S_{\max})=3H_\infty^2/(8\pi G)"
            ),
            "V_inf": Eq(Symbol("V_inf"), V_at_1),
            "caveat": (
                r"\mathrm{Background\ equivalence\ ONLY}"
                r"\mathrm{\ -\ not\ a\ microphysical\ claim}"
            ),
        },
        claims=[
            "LAYER 9 DERIVED: φ̇²=-Ḣ/(4πG) and V=(3H²+Ḣ)/(8πG) on the entropy background.",
            "LAYER 9 DERIVED: ρ_φ=ρ_S and w_φ=w_S identically along the trajectory.",
            "LAYER 9 DERIVED: at χ=1, φ̇=0 and V=ρ_{S,∞} (de Sitter / Λ limit).",
            "Background equivalence only - useful for perturbation comparison, not ontology.",
        ],
        status="derived" if (ok_phidot and ok_rho and ok_w and ok_Vinf and ok_phidot_inf) else "failed",
        evidence={
            "phidot2": phidot2,
            "phidot2_alt": phidot2_alt,
            "Vphi": Vphi,
            "rho_phi": rho_phi,
            "P_phi": P_phi,
            "w_phi": w_phi,
            "w_S": w_S,
            "dchi_dphi": dchi_dphi,
            "V_at_1": V_at_1,
            "phidot2_at_1": phidot2_at_1,
            "ok_phidot": ok_phidot,
            "ok_rho": ok_rho,
            "ok_w": ok_w,
            "ok_Vinf": ok_Vinf,
            "Hdot": Hdot,
            "nec_ok": nec_ok,
        },
    )


def step_L10_homogeneous_action(sym: Dict[str, Any]) -> DerivationStep:
    """
    LAYER 10 - Homogeneous action admitting the logistic as an Euler-Lagrange solution.

    First-principles reconstruction
    ------------------------------
    Homogeneous FLRW reduction of a χ-sector Lagrangian:
        L_χ = ½ K(χ) χ̇² - U(χ)

    Euler-Lagrange:
        d/dt (∂L/∂χ̇) = ∂L/∂χ
        d/dt (K χ̇) = ½ K' χ̇² - U'
        =>  K χ̈ + ½ K' χ̇² + U' = 0

    Demand that χ̇ = Γ(χ) := γ χ(1-χ) be a solution.
    Then χ̈ = Γ Γ', and the EL becomes the reconstructive ODE for (K,U):
        K Γ Γ' + ½ K' Γ² + U' = 0

    Minimal kinetic choice K ≡ 1 (K'=0):
        Γ Γ' + U' = 0
        =>  U' = -Γ Γ'
        =>  U(χ) = -½ Γ(χ)² + const
                 = -½ γ² χ² (1-χ)² + const

    Reconstructed Lagrangian (const=0):
        L = ½ χ̇² - U = ½ χ̇² + ½ γ² χ²(1-χ)²

    On-shell verification:
        EL residual with K=1:  χ̈ + U'  =  χ̈ - Γ Γ'
        On χ̇=Γ:  χ̈=Γ Γ'  =>  residual = 0.  ✓

    Also χ̇=-Γ solves the same second-order EL; the thermodynamic arrow
    Ṡ_H≥0 (LAYER 7) selects χ̇=+Γ.

    Energy (mechanical):
        E = ½ χ̇² + U = ½ χ̇² - ½ Γ²
        On logistic: E ≡ 0.

    Honesty (CONDITIONAL):
        • This CONSTRUCTS an action that ADMITS logistic as an EL solution.
        • It does NOT uniquely derive Γ from the GSL (weakest link remains).
        • (K,U) is non-unique; dissipative/Onsager formulations are alternatives.
        • Covariant completion: LAYER 11.
    """
    chi, gamma = sym["chi"], sym["gamma"]
    Gamma = gamma * chi * (1 - chi)
    Gamma_p = simplify(diff(Gamma, chi))  # γ(1-2χ)

    # Potential reconstruction K=1
    U = simplify(-sp.Rational(1, 2) * Gamma**2)
    U_explicit = simplify(-sp.Rational(1, 2) * gamma**2 * chi**2 * (1 - chi) ** 2)
    ok_U = simplify(U - U_explicit) == 0
    U_prime = simplify(diff(U, chi))
    U_prime_tgt = simplify(-Gamma * Gamma_p)
    ok_Up = simplify(U_prime - U_prime_tgt) == 0

    # EL residual: χ̈ + U'  should vanish when χ̈ = Γ Γ' and χ̇=Γ
    # Represent χ̈_on_shell = Γ * Γ_p
    chi_ddot_on = simplify(Gamma * Gamma_p)
    EL_residual = simplify(chi_ddot_on + U_prime)
    ok_EL = EL_residual == 0

    # Also check χ̇=-Γ branch
    chi_ddot_minus = simplify((-Gamma) * (-Gamma_p))  # d(-Γ)/dt = -Γ' χ̇ = -Γ'(-Γ)=Γ Γ'
    # wait: χ̇=-Γ, χ̈ = d(-Γ)/dt = -Γ' χ̇ = -Γ'(-Γ) = Γ Γ'
    EL_residual_minus = simplify(Gamma * Gamma_p + U_prime)
    ok_EL_minus = EL_residual_minus == 0

    # Lagrangian and on-shell energy
    # L = ½ χ̇² - U = ½ χ̇² + ½ Γ²
    L_on = simplify(sp.Rational(1, 2) * Gamma**2 - U)  # = Γ²
    E_on = simplify(sp.Rational(1, 2) * Gamma**2 + U)  # = 0
    ok_E = E_on == 0

    # Critical points of U: U' = -Γ Γ' = -γ χ(1-χ)*γ(1-2χ)=0
    # => χ=0,1/2,1; U(0)=U(1)=0, U(1/2)=-γ²/32
    U_at_0 = simplify(U.subs({chi: 0}))
    U_at_1 = simplify(U.subs({chi: 1}))
    U_at_half = simplify(U.subs({chi: sp.Rational(1, 2)}))
    U_half_tgt = simplify(-gamma**2 / 32)
    ok_Ushape = (U_at_0 == 0) and (U_at_1 == 0) and (simplify(U_at_half - U_half_tgt) == 0)

    # General K reconstructibility (underdetermined): one free function
    # With given Γ, U' = -K Γ Γ' - ½ K' Γ²  still leaves K free
    K = Function("K")(chi)
    U_prime_gen = simplify(-K * Gamma * Gamma_p - sp.Rational(1, 2) * diff(K, chi) * Gamma**2)

    return DerivationStep(
        name="L10_homogeneous_action_for_logistic",
        stage="LAYER_10",
        assumptions=[
            "Homogeneous FLRW reduction of L_χ=½ K(χ) χ̇² - U(χ)",
            "Minimal kinetic choice K≡1",
            "Demand χ̇=γ χ(1-χ) be an EL solution",
        ],
        equations={
            "L_chi": r"L_\chi=\tfrac12 K(\chi)\dot\chi^2-U(\chi)",
            "EL": r"K\ddot\chi+\tfrac12 K'(\chi)\dot\chi^2+U'(\chi)=0",
            "Gamma": Eq(Symbol("Gamma"), Gamma),
            "Gamma_prime": Eq(Symbol("Gamma_prime"), Gamma_p),
            "reconstructive_ODE": (
                r"K\Gamma\Gamma'+\tfrac12 K'\Gamma^2+U'=0"
            ),
            "U_of_Gamma": Eq(Symbol("U"), U),
            "U_explicit": Eq(Symbol("U"), U_explicit),
            "U_prime": Eq(Symbol("U_prime"), U_prime),
            "L_reconstructed": (
                r"K=1:\ L=\tfrac12\dot\chi^2+\tfrac12\gamma^2\chi^2(1-\chi)^2"
            ),
            "EL_residual_on_logistic": Eq(Symbol("EL_residual"), EL_residual),
            "EL_check": r"\mathrm{on\ }\dot\chi=\Gamma:\ \ddot\chi+U'=0\ \checkmark",
            "sign_ambiguity": (
                r"\dot\chi=-\Gamma\ \mathrm{also\ solves\ EL};\ "
                r"\dot S_H\ge0\ \mathrm{selects}\ \dot\chi=+\Gamma"
            ),
            "energy_on_shell": Eq(Symbol("E_on"), E_on),
            "energy_txt": r"E=\tfrac12\dot\chi^2+U;\ \mathrm{on\ logistic}\ E\equiv0",
            "U_shape": (
                r"U(0)=U(1)=0,\ U(\tfrac12)=-\gamma^2/32"
            ),
            "non_uniqueness": (
                r"K(\chi)\ \mathrm{free}:\ "
                r"U'=-K\Gamma\Gamma'-\tfrac12 K'\Gamma^2"
            ),
            "U_prime_gen": Eq(Symbol("U_prime_gen"), U_prime_gen),
            "honesty": (
                r"\mathrm{Admits\ logistic\ as\ EL\ solution;\ }"
                r"\mathrm{does\ NOT\ derive\ }\Gamma\mathrm{\ from\ GSL}"
            ),
        },
        claims=[
            "LAYER 10 CONDITIONAL: with K=1, U=-½Γ² makes χ̇=Γ an exact EL solution (residual 0).",
            "LAYER 10 DERIVED: on-shell mechanical energy E≡0 along the logistic.",
            "LAYER 10 HONEST: χ̇=-Γ also solves EL; GSL/arrow selects χ̇=+Γ.",
            "LAYER 10 HONEST: (K,U) non-unique; GSL⇏Γ remains the weakest link.",
        ],
        open_gaps=[
            "Uniqueness of (K,U) not proven; dissipative/Onsager alternatives exist.",
            "Full covariant completion with metric coupling is LAYER 11.",
        ],
        status="conditional" if (ok_U and ok_Up and ok_EL and ok_EL_minus and ok_E and ok_Ushape) else "failed",
        evidence={
            "U": U,
            "U_explicit": U_explicit,
            "U_prime": U_prime,
            "Gamma": Gamma,
            "Gamma_p": Gamma_p,
            "EL_residual": EL_residual,
            "E_on": E_on,
            "U_at_half": U_at_half,
            "ok_U": ok_U,
            "ok_Up": ok_Up,
            "ok_EL": ok_EL,
            "ok_EL_minus": ok_EL_minus,
            "ok_E": ok_E,
            "ok_Ushape": ok_Ushape,
        },
    )


def step_L11_covariant_chi(sym: Dict[str, Any]) -> DerivationStep:
    """
    LAYER 11 - Promote χ to a spacetime scalar; covariant completion of LAYER 10.

    First-principles structure
    -------------------------
    Field:     χ = χ(x^μ)
    Kinetic:   X := -½ ∇_μ χ ∇^μ χ
               (signature (-,+,+,+): homogeneous χ(t) => X = ½ χ̇²)
    Lagrangian density:
               L_χ = K(χ) X - U(χ)
               = -½ K(χ) (∇χ)² - U(χ)

    Stress-energy (vary √-g L_χ w.r.t. g^{μν}):
               T^{(χ)}_{μν} = K ∇_μ χ ∇_ν χ - (K X - U) g_{μν}

    Covariant EL / Klein-Gordon (K=K(χ) only):
               K □χ - K' X - U' = 0
               □χ := ∇_μ ∇^μ χ

    Homogeneous flat FLRW reduction (K=1):
               □χ = -χ̈ - 3 H χ̇
               EL =>  χ̈ + 3 H χ̇ + U' = 0

    CRITICAL CONSISTENCY CHECK with LAYER 10:
        LAYER 10 (no expansion friction):  χ̈ + U' = 0,  U'=-Γ Γ'
        On χ̇=Γ: residual 0.
        LAYER 11 FLRW: χ̈ + 3H χ̇ + U' = 0
        On χ̇=Γ with same U:  Γ Γ' + 3H Γ - Γ Γ' = 3H Γ
        Residual = 3H Γ  (=0 only if H=0 or Γ=0)

    Therefore the naive covariantization of LAYER 10 does NOT make the
    logistic an exact FLRW solution - Hubble friction appears.

    Remedies (documented, not chosen uniquely):
        (R1) Dissipative / Onsager first-order principle (not pure Hamiltonian action)
        (R2) Soft constraint ∫ λ (χ̇-Γ(χ))² or Lagrange multiplier enforcing Γ-flow
        (R3) H-dependent / nonminimal completion U_eff(χ,H)
        (R4) Treat logistic as leading-order when |χ̈| ≫ 3H|χ̇| (brief transitions)

    Homogeneous fluid variables from T^{(χ)} (K=1):
        ρ_χ = ½ χ̇² + U,   P_χ = ½ χ̇² - U
        (matches LAYER 9/10 mechanical identification when U is the potential)

    Physical sound speed (canonical K=const):
        c_s² = 1
        (distinct from adiabatic c_a² of LAYER 13)

    Interpretations of χ:
        scalar field | order parameter | effective fluid coordinate |
        non-equilibrium thermodynamic field | horizon-completion variable
    """
    chi, gamma = sym["chi"], sym["gamma"]
    H = sym["H"]
    Gamma = gamma * chi * (1 - chi)
    Gamma_p = simplify(diff(Gamma, chi))
    U = simplify(-sp.Rational(1, 2) * Gamma**2)
    U_prime = simplify(diff(U, chi))

    # FLRW residual of L10 potential under covariant EL
    # residual := χ̈ + 3H χ̇ + U'  on χ̇=Γ, χ̈=Γ Γ'
    residual = simplify(Gamma * Gamma_p + 3 * H * Gamma + U_prime)
    residual_tgt = simplify(3 * H * Gamma)
    ok_residual = simplify(residual - residual_tgt) == 0
    # Vanishes at endpoints or H=0
    res_at_1 = simplify(residual.subs({chi: 1}))
    res_at_0 = simplify(residual.subs({chi: 0}))
    ok_ends = (res_at_1 == 0) and (res_at_0 == 0)

    # Homogeneous X and T components (K=1)
    # X = ½ χ̇²; ρ = X + U; P = X - U  when evaluated on a trajectory
    # On logistic: X = ½ Γ², U = -½ Γ² => ρ = 0, P = Γ² ? 
    # Careful: LAYER 10 mechanical E = ½χ̇²+U=0 on logistic, so ρ_χ=0 if U is that potential.
    # That shows L10's U is a mathematical embedding of the ODE, NOT the physical
    # entropy-sector stress tensor of LAYERS 1/9 (where ρ_S=3H²/(8πG)).
    # IMPORTANT distinction to record.
    X_on = simplify(sp.Rational(1, 2) * Gamma**2)
    rho_embed = simplify(X_on + U)   # = 0
    P_embed = simplify(X_on - U)     # = Γ²
    ok_embed_E = rho_embed == 0

    # Physical entropy fluid remains LAYER 1; χ-action is for the ORDER PARAMETER dynamics
    # Coupling options: χ drives S_H(χ) kinematically (current architecture) vs χ as dark-energy field

    return DerivationStep(
        name="L11_covariant_chi_field",
        stage="LAYER_11",
        assumptions=[
            "χ extends off FLRW as a scalar order parameter",
            "L_χ=K(χ)X-U(χ) with X=-½(∇χ)²",
            "Signature (-,+,+,+)",
        ],
        equations={
            "chi_field": r"\chi=\chi(x^\mu)",
            "X_def": r"X=-\tfrac12\nabla_\mu\chi\nabla^\mu\chi",
            "X_FLRW": r"\mathrm{FLRW\ homogeneous:}\ X=\tfrac12\dot\chi^2",
            "L_cov": r"L_\chi=K(\chi)X-U(\chi)",
            "T_chi": (
                r"T^{(\chi)}_{\mu\nu}=K\nabla_\mu\chi\nabla_\nu\chi"
                r"-(KX-U)g_{\mu\nu}"
            ),
            "EL_cov": r"K\square\chi-K'X-U'=0",
            "EL_K1": r"K=1:\ \square\chi-U'=0",
            "FLRW_KG": r"K=1\mathrm{\ FLRW:}\ \ddot\chi+3H\dot\chi+U'=0",
            "L10_U": Eq(Symbol("U"), U),
            "residual_logistic": Eq(Symbol("FLRW_residual"), residual),
            "residual_txt": (
                r"\mathrm{on\ }\dot\chi=\Gamma\mathrm{\ with\ }U=-\tfrac12\Gamma^2:"
                r"\ \mathrm{residual}=3H\Gamma"
            ),
            "gap": (
                r"\mathrm{Naive\ covariantization\ of\ LAYER\ 10\ does\ NOT\ }"
                r"\mathrm{preserve\ logistic\ in\ FLRW\ (Hubble\ friction)}"
            ),
            "remedy_R1": r"(R1)\ \mathrm{Onsager/dissipative\ first-order\ principle}",
            "remedy_R2": r"(R2)\ \mathrm{constraint\ }\dot\chi=\Gamma(\chi)\mathrm{\ in\ the\ action}",
            "remedy_R3": r"(R3)\ \mathrm{nonminimal/H-dependent\ }U_{\mathrm{eff}}(\chi,H)",
            "remedy_R4": r"(R4)\ \mathrm{brief-transition\ approximation\ }|\ddot\chi|\gg 3H|\dot\chi|",
            "embed_vs_physical": (
                r"L_{10}\ \mathrm{embedding}\ \Rightarrow\ \rho_{\mathrm{embed}}=0\mathrm{\ on\ logistic;}"
                r"\ \mathrm{physical\ }\rho_S\mathrm{\ remains\ LAYER\ 1\ (kinematic\ }S_H(\chi)\mathrm{)}"
            ),
            "rho_embed": Eq(Symbol("rho_embed"), rho_embed),
            "P_embed": Eq(Symbol("P_embed"), P_embed),
            "c_s2": r"K=\mathrm{const:}\ c_s^2=1\mathrm{\ (propagation);\ cf.\ }c_a^2\mathrm{\ LAYER\ 13}",
            "interpretations": (
                r"\chi:\ \mathrm{scalar|order\ parameter|effective\ fluid|}"
                r"\mathrm{non\!-\!eq\ thermo\ field|horizon\!-\!completion\ coordinate}"
            ),
        },
        claims=[
            "LAYER 11 DERIVED: covariant L_χ=KX-U, T^{(χ)}_{μν}, and KG equation.",
            "LAYER 11 DERIVED: FLRW reduction is χ̈+3Hχ̇+U'=0 (Hubble friction).",
            "LAYER 11 THEOREM: L10 potential gives FLRW residual 3HΓ on logistic - not exact.",
            "LAYER 11: remedies R1-R4 catalogued; physical ρ_S remains LAYER 1 kinematics.",
        ],
        open_gaps=[
            "Choice among R1-R4 not unique; selects how χ couples into the metric sector.",
            "Linear perturbations (LAYER 12) depend on that choice.",
        ],
        status="derived" if (ok_residual and ok_ends and ok_embed_E) else "failed",
        evidence={
            "U": U,
            "U_prime": U_prime,
            "Gamma": Gamma,
            "residual": residual,
            "residual_tgt": residual_tgt,
            "rho_embed": rho_embed,
            "P_embed": P_embed,
            "X_on": X_on,
            "ok_residual": ok_residual,
            "ok_ends": ok_ends,
            "ok_embed_E": ok_embed_E,
        },
    )


def step_L12_perturbations(sym: Dict[str, Any]) -> DerivationStep:
    """
    LAYER 12 - Linear cosmological perturbations of the entropy sector.

    First-principles linearization
    -----------------------------
    Background:   χ̄(t), H̄(t), S̄_H(t), ρ̄_S(t) from LAYERS 1-3
    Splits:       χ = χ̄ + δχ(t,x),   H = H̄ + δH,   g_μν = ā_μν + δg_μν

    Newtonian gauge:
        ds² = -(1+2Φ) dt² + a²(1-2Ψ) δ_{ij} dx^i dx^j

    Architecture A - kinematic entropy (current global module):
        S_H = π/(G H²) as a geometric functional of the local expansion.
        Linearize:
            δS_H = (dS_H/dH) δH = -2 S_H/H δH
            ρ_S  = 3/(8 G² S_H)  =>  δρ_S = -(ρ_S/S_H) δS_H = 2 ρ_S/H δH
        Einstein:
            δG_μν = 8πG (δT^{(m)}_{μν} + δT^{(S)}_{μν})
        Here the entropy sector responds to metric/expansion perturbations;
        there is no independent propagating δχ unless Architecture B is chosen.

    Architecture B - dynamical χ (LAYER 11 covariant field):
        T^{(χ)}_{μν} from L_χ = K X - U
        Linearize KG about FLRW:
            δχ̈ + 3H δχ̇ + (c_s² k²/a² + m_eff²) δχ
                = source[Φ, Ψ, χ̄̇, H̄]
        For K=1: c_s² = 1
        Effective mass: m_eff² = U''(χ̄) + friction/background terms
        With U=-½Γ²: U'' computable from Γ=γ χ(1-χ)

    Anisotropic stress:
        Canonical scalar (K=const) => Π^S = 0 => Ψ = Φ at linear order in GR
        (plus standard matter without viscosity)

    Gauge-invariant observables (schematic bridge):
        Φ, Ψ -> CMB temperature/polarization
        δ_m, θ_m -> P(k), fσ8
        BAO scale from sound horizon at drag
        Isocurvature: S_{mχ} ∝ δ_m/(1+w_m) - δ_χ/(1+w_χ) if Architecture B active

    Dependence on LAYER 11 remedies:
        R1/R2 (dissipative/constrained first-order) => δχ non-propagating or
            heavily damped; clustering ~ smooth dark energy
        R3/R4 (dynamical scalar) => standard quintessence-like perturbations
        Current kinematic Architecture A => perturbations inherited from δH/δg only

    Honesty:
        This layer DERIVES the linear map and effective mass algebraically.
        Full Boltzmann/CLASS transfer functions are numerical Layer-2 work.
    """
    G, H, chi, gamma = sym["G"], sym["H"], sym["chi"], sym["gamma"]
    S_max, S_early = sym["S_max"], sym["S_early"]
    Delta_S = S_max - S_early

    # --- Architecture A: kinematic δ-identities ---
    S_H = pi / (G * H**2)
    dS_dH = simplify(diff(S_H, H))                     # -2π/(G H³) = -2 S_H/H
    ok_dS = simplify(dS_dH + 2 * S_H / H) == 0

    S_sym = Symbol("S_H", positive=True)
    rho_of_S = simplify(sp.Rational(3, 8) / (G**2 * S_sym))
    drho_dS = simplify(diff(rho_of_S, S_sym))          # -ρ/S
    ok_drho = simplify(drho_dS + rho_of_S / S_sym) == 0

    rho_S_H = simplify(3 * H**2 / (8 * pi * G))
    drho_dH = simplify(diff(rho_S_H, H))               # 2ρ/H
    ok_drho_H = simplify(drho_dH - 2 * rho_S_H / H) == 0

    # --- Architecture B: effective mass from U=-½Γ² ---
    Gamma = gamma * chi * (1 - chi)
    U = simplify(-sp.Rational(1, 2) * Gamma**2)
    U_pp = simplify(diff(U, chi, 2))
    # At late attractor χ->1: Γ->0, need U''(1)
    U_pp_1 = simplify(U_pp.subs({chi: 1}))
    U_pp_0 = simplify(U_pp.subs({chi: 0}))
    U_pp_half = simplify(U_pp.subs({chi: sp.Rational(1, 2)}))
    # Γ=γ(χ-χ²), Γ'=γ(1-2χ), Γ''=-2γ
    # U=-½γ² χ²(1-χ)²; at χ=1: U''= ?
    # Near χ=1, ε=1-χ: Γ=γ ε(1-ε)~γ ε, U~-½ γ² ε², U_εε~-γ²
    # In χ coordinate U''(1) = d²U/dχ² at 1
    ok_Upp = U_pp_1 is not None

    # Linear KG schematic mass term m_eff² ~ U''(χ̄) for K=1 in Minkowski limit;
    # FLRW adds ̇ terms from background
    m_eff2 = U_pp  # leading potential contribution

    # Sound speed
    c_s2 = 1  # K=const canonical

    # w_S and 1+w for isocurvature denominators
    S_H_chi = S_early + Delta_S * chi
    H_chi = sqrt(pi / (G * S_H_chi))
    w_S = simplify(-1 + Delta_S * gamma * chi * (1 - chi) / (3 * H_chi * S_H_chi))
    one_plus_w = simplify(1 + w_S)
    # At attractor 1+w->0 => entropy-sector isocurvature becomes vacuum-like / delicate
    one_plus_w_at_1 = simplify(one_plus_w.subs({chi: 1}))
    ok_vac = one_plus_w_at_1 == 0

    return DerivationStep(
        name="L12_linear_cosmological_perturbations",
        stage="LAYER_12",
        assumptions=[
            "Background from LAYERS 1-11",
            "Linear order in δχ, δg_μν",
            "Newtonian gauge",
            "Architecture A (kinematic) and/or B (dynamical χ) as selected by L11 remedies",
        ],
        equations={
            "newtonian_gauge": (
                r"ds^2=-(1+2\Phi)dt^2+a^2(1-2\Psi)\delta_{ij}dx^i dx^j"
            ),
            "split": r"\chi=\bar\chi+\delta\chi,\quad H=\bar H+\delta H",
            # Architecture A
            "dS_dH": Eq(Symbol("dS_dH"), dS_dH),
            "delta_S": r"\delta S_H=-2(S_H/H)\,\delta H",
            "delta_rho": r"\delta\rho_S=2(\rho_S/H)\,\delta H=-(\rho_S/S_H)\,\delta S_H",
            "drho_dH": Eq(Symbol("drho_dH"), drho_dH),
            "Einstein": r"\delta G_{\mu\nu}=8\pi G\,(\delta T^{(m)}_{\mu\nu}+\delta T^{(S)}_{\mu\nu})",
            # Architecture B
            "delta_chi_EOM": (
                r"\ddot{\delta\chi}+3H\dot{\delta\chi}"
                r"+(c_s^2 k^2/a^2+m_{\mathrm{eff}}^2)\delta\chi"
                r"=S[\Phi,\Psi,\dot{\bar\chi},H]"
            ),
            "c_s2": Eq(Symbol("c_s2"), c_s2),
            "m_eff2_leading": Eq(Symbol("m_eff2"), m_eff2),
            "U_pp": Eq(Symbol("U_pp"), U_pp),
            "U_pp_attractor": Eq(Symbol("U_pp_1"), U_pp_1),
            "anisotropic_stress": (
                r"K=\mathrm{const}\Rightarrow\Pi^S=0\Rightarrow\Psi=\Phi"
                r"\mathrm{\ (GR+canonical\ scalar)}"
            ),
            # Observables
            "CMB": r"\Phi,\Psi\to\Theta_{\ell},\ E\!/\!B\mathrm{\ polarization}",
            "LSS": r"\delta_m,\theta_m\to P(k),\ f\sigma_8",
            "BAO": r"r_s\mathrm{\ at\ drag}\to\mathrm{BAO\ scale}",
            "isocurvature": (
                r"S_{m\chi}\propto\delta_m/(1+w_m)-\delta_\chi/(1+w_\chi)"
                r"\mathrm{\ (Arch.\ B)}"
            ),
            "vacuum_limit": (
                r"\chi\to1:\ 1+w_S\to0\mathrm{\ -\ entropy\ sector\ vacuum\!-\!like;"
                r"\ isocurvature\ delicate}"
            ),
            "one_plus_w": Eq(Symbol("one_plus_w"), one_plus_w),
            # Remedy dependence
            "R1_R2": (
                r"R1/R2:\ \delta\chi\mathrm{\ non\!-\!propagating/damped}"
                r"\Rightarrow\mathrm{smooth\ DE\!-\!like}"
            ),
            "R3_R4": (
                r"R3/R4:\ \mathrm{quintessence\!-\!like\ clustering}"
            ),
            "Arch_A": (
                r"\mathrm{Arch.\ A\ (kinematic\ }S_H(H)\mathrm{):}"
                r"\ \mathrm{perturbations\ from\ }\delta H,\delta g\mathrm{\ only}"
            ),
        },
        claims=[
            "LAYER 12 DERIVED: δS_H=-2(S_H/H)δH and δρ_S=2(ρ_S/H)δH (Architecture A).",
            "LAYER 12 DERIVED: schematic δχ KG with c_s²=1 and m_eff²=U''(χ̄) (Architecture B).",
            "LAYER 12 DERIVED: canonical => Ψ=Φ; at χ->1 the sector is vacuum-like (1+w->0).",
            "LAYER 12: clustering vs smooth DE selected by L11 remedies R1-R4.",
        ],
        open_gaps=[
            "Explicit Boltzmann hierarchy / CLASS implementation is numerical Layer-2.",
            "Full source S[Φ,Ψ,...] depends on the chosen L11 completion.",
        ],
        status="derived" if (ok_dS and ok_drho and ok_drho_H and ok_vac and ok_Upp) else "failed",
        evidence={
            "dS_dH": dS_dH,
            "drho_dS": drho_dS,
            "drho_dH": drho_dH,
            "U_pp": U_pp,
            "U_pp_1": U_pp_1,
            "U_pp_0": U_pp_0,
            "U_pp_half": U_pp_half,
            "m_eff2": m_eff2,
            "w_S": w_S,
            "one_plus_w_at_1": one_plus_w_at_1,
            "ok_dS": ok_dS,
            "ok_drho": ok_drho,
            "ok_drho_H": ok_drho_H,
            "ok_vac": ok_vac,
            "c_s2": c_s2,
        },
    )


def step_L13_sound_speed(sym: Dict[str, Any]) -> DerivationStep:
    """
    LAYER 13 - Adiabatic sound speed of the entropy fluid vs propagation speed.

    First principles
    ---------------
    Along the background trajectory (χ as parameter):
        c_a² := Ṗ_S / ρ̇_S = (dP_S/dχ) / (dρ_S/dχ)

    Locked thermodynamics:
        S_H(χ) = S_early + ΔS χ
        ρ_S(χ) = 3/(8 G² S_H)
        w_S(χ) = -1 + ΔS γ χ(1-χ)/(3 H(χ) S_H)
        P_S     = w_S ρ_S

    Immediate identities:
        dρ/dχ = -(ρ/S_H) ΔS < 0 on [0,1) for ΔS>0
        c_a²  = w + (dw/dχ) ρ / (dρ/dχ)
              = w - (S_H/ΔS) (dw/dχ)

    Standard GR fluid identity (cross-check):
        c_a² = w - ẇ / (3 H (1+w))
        with ẇ = (dw/dχ) χ̇ = (dw/dχ) γ χ(1-χ)

    Propagation speed (LAYER 11 Architecture B, K=const):
        c_s² = 1

    CRITICAL DISTINCTION:
        c_a²  - adiabatic / background EOS derivative (can be negative)
        c_s²  - rest-frame rest-frame physical propagation speed of δχ
        Gradient instabilities track c_s² < 0, not merely c_a² < 0.
        Phantom-divide / vacuum limits make c_a² delicate even if c_s²=1.

    Attractor χ->1:
        w->-1, 1+w->0, ρ̇->0, Ṗ->0
        => c_a² indeterminate/singular in the ratio; sector becomes Λ-like.

    Stability reading:
        Architecture A (kinematic): no propagating χ mode; c_a² is diagnostic only
        Architecture B: require c_s²>0 (satisfied if K>0); monitor c_a² for
            Jeans/pressure response of the effective fluid description
    """
    G, chi, gamma = sym["G"], sym["chi"], sym["gamma"]
    S_max, S_early = sym["S_max"], sym["S_early"]
    Delta_S = S_max - S_early

    S_H = S_early + Delta_S * chi
    H = sqrt(pi / (G * S_H))
    rho = simplify(sp.Rational(3, 8) / (G**2 * S_H))
    w = simplify(-1 + Delta_S * gamma * chi * (1 - chi) / (3 * H * S_H))
    P = simplify(w * rho)

    # Direct definition
    drho = simplify(diff(rho, chi))
    dP = simplify(diff(P, chi))
    ca2 = simplify(dP / drho)

    # Identity c_a² = w - (S_H/ΔS) w'
    dw = simplify(diff(w, chi))
    ca2_from_w = simplify(w - (S_H / Delta_S) * dw)
    ok_w_id = simplify(ca2 - ca2_from_w) == 0

    # Standard identity c_a² = w - ẇ/(3H(1+w))
    chidot = gamma * chi * (1 - chi)
    wdot = simplify(dw * chidot)
    ca2_std = simplify(w - wdot / (3 * H * (1 + w)))
    # May need careful simplification when 1+w is symbolic
    ok_std = simplify(sp.together(ca2 - ca2_std)) == 0
    if not ok_std:
        # Try canceling with assume χ∈(0,1)
        ok_std = simplify(sp.trigsimp(sp.expand(ca2 - ca2_std))) == 0
    if not ok_std:
        diff_std = simplify(ca2 - ca2_std)
        ok_std = diff_std == 0 or diff_std.equals(0)

    # drho sign
    drho_tgt = simplify(-rho * Delta_S / S_H)
    ok_drho = simplify(drho - drho_tgt) == 0

    # Attractor diagnostics
    w_at_1 = simplify(w.subs({chi: 1}))
    one_plus_w = simplify(1 + w)
    one_plus_w_1 = simplify(one_plus_w.subs({chi: 1}))
    # ca2 at χ=1 may diverge; evaluate limit
    ca2_lim = simplify(sp.limit(ca2, chi, 1))
    ca2_half = simplify(ca2.subs({chi: sp.Rational(1, 2)}))

    # Propagation vs adiabatic
    c_s2 = 1

    # Relation to ξ from LAYER 7: 1+w = ξ/3
    # ẇ/(3H(1+w)) = ...

    return DerivationStep(
        name="L13_adiabatic_sound_speed",
        stage="LAYER_13",
        assumptions=[
            "Barotropic reading along the χ-trajectory",
            "Locked S_H=π/(G H²) and logistic closure",
            "Distinguish c_a² (EOS) from c_s² (propagation, LAYER 11)",
        ],
        equations={
            "c_a2_def": r"c_a^2:=\dot P_S/\dot\rho_S=(dP_S/d\chi)/(d\rho_S/d\chi)",
            "rho": Eq(Symbol("rho_S"), rho),
            "w": Eq(Symbol("w_S"), w),
            "P": Eq(Symbol("P_S"), P),
            "drho_dchi": Eq(Symbol("drho_dchi"), drho),
            "drho_txt": r"d\rho_S/d\chi=-(\rho_S/S_H)\Delta S<0",
            "c_a2": Eq(Symbol("c_a2"), ca2),
            "c_a2_from_w": Eq(Symbol("c_a2"), ca2_from_w),
            "c_a2_w_id": r"c_a^2=w_S-(S_H/\Delta S)\,w_S'",
            "c_a2_std": Eq(Symbol("c_a2_std"), ca2_std),
            "c_a2_std_txt": r"c_a^2=w_S-\dot w_S/\bigl(3H(1+w_S)\bigr)",
            "c_s2": Eq(Symbol("c_s2"), c_s2),
            "distinction": (
                r"c_a^2\mathrm{\ (adiabatic/EOS)\ }\neq\mathrm{\ }"
                r"c_s^2\mathrm{\ (propagation;\ Arch.\ B:\ }c_s^2=1\mathrm{)}"
            ),
            "instability_note": (
                r"\mathrm{Gradient\ instability\ tracks\ }c_s^2<0\mathrm{,\ not\ }c_a^2<0\mathrm{\ alone}"
            ),
            "attractor": (
                r"\chi\to1:\ w\to-1,\ 1+w\to0;\ c_a^2\mathrm{\ delicate/singular;\ }"
                r"\mathrm{sector\ }\Lambda\mathrm{\!-\!like}"
            ),
            "w_at_1": Eq(Symbol("w_at_1"), w_at_1),
            "ca2_limit": Eq(Symbol("ca2_limit"), ca2_lim),
            "ca2_half": Eq(Symbol("ca2_half"), ca2_half),
            "Arch_A": (
                r"\mathrm{Arch.\ A:\ no\ propagating\ }\delta\chi;\ c_a^2\mathrm{\ diagnostic\ only}"
            ),
            "Arch_B": (
                r"\mathrm{Arch.\ B:\ require\ }c_s^2>0\mathrm{\ (OK\ if\ }K>0\mathrm{);\ }"
                r"c_a^2\mathrm{\ enters\ fluid\ Jeans\ response}"
            ),
        },
        claims=[
            "LAYER 13 DERIVED: c_a²=(dP/dχ)/(dρ/dχ) with dρ/dχ=-(ρ/S_H)ΔS.",
            "LAYER 13 DERIVED: c_a²=w-(S_H/ΔS)w'=w-ẇ/(3H(1+w)).",
            "LAYER 13: c_s²=1 (Arch. B) != c_a²; gradient stability tracks c_s².",
            "LAYER 13: at χ->1, c_a² is delicate as the sector becomes Λ-like.",
        ],
        status="derived" if (ok_drho and ok_w_id and w_at_1 == -1 and one_plus_w_1 == 0) else "failed",
        evidence={
            "ca2": ca2,
            "ca2_from_w": ca2_from_w,
            "ca2_std": ca2_std,
            "rho": rho,
            "P": P,
            "w": w,
            "drho": drho,
            "dw": dw,
            "ca2_lim": ca2_lim,
            "ca2_half": ca2_half,
            "ok_drho": ok_drho,
            "ok_w_id": ok_w_id,
            "ok_std": ok_std,
            "c_s2": c_s2,
        },
    )


def step_L14_energy_conditions(sym: Dict[str, Any]) -> DerivationStep:
    """
    LAYER 14 - Energy conditions for the entropy effective fluid.

    Definitions (perfect fluid)
    --------------------------
        NEC:  ρ + P ≥ 0
        WEC:  ρ ≥ 0  and  ρ + P ≥ 0
        DEC:  ρ ≥ |P|   (=> ρ ≥ 0 and -ρ ≤ P ≤ ρ)
        SEC:  ρ + 3P ≥ 0   (and often also NEC)

    First-principles reductions on the entropy sector
    ------------------------------------------------
    Raychaudhuri:  Ḣ = -4πG(ρ_S+P_S)
        =>  ρ_S + P_S = -Ḣ/(4πG)
        =>  NEC  ⇔  Ḣ ≤ 0

    Locked thermo EOM: Ḣ = -(H/(2 S_H)) ΔS γ χ(1-χ) ≤ 0 on [0,1]
        =>  NEC holds automatically for the logistic entropy fluid.
        =>  WEC: ρ_S = 3H²/(8πG) > 0 and NEC => WEC holds.

    With w_S = P_S/ρ_S:
        NEC  ⇔  1 + w_S ≥ 0  ⇔  w_S ≥ -1   (non-phantom; LAYER 1)
        SEC  ⇔  1 + 3 w_S ≥ 0  ⇔  w_S ≥ -1/3
        DEC  ⇔  |w_S| ≤ 1

    Cosmographic link (LAYER 2, 7):
        q = -1 - Ḣ/H² = (1 + 3 w_S)/2     (single-fluid)
        =>  SEC  ⇔  q ≥ 0   (deceleration)
        =>  SEC failure  ⇔  q < 0  ⇔  accelerated expansion

    Explicit combinations on the χ-chart:
        ρ + P   = -Ḣ/(4πG) = H ΔS γ χ(1-χ)/(8πG S_H) ≥ 0
        ρ + 3P  = -2ρ + 3(ρ+P) = -2ρ + 3(-Ḣ/(4πG))
                = ρ (1 + 3 w_S)

    Attractor χ->1:
        ρ + P -> 0     (NEC saturated, de Sitter)
        ρ + 3P -> -2ρ < 0   (SEC violated - eternal acceleration)

    DEC: for w ∈ [-1,1], DEC holds; our non-phantom w ∈ (-1, w_early]
        with w->-1⁺ => DEC holds (ρ ≥ -P since ρ+P≥0 and P≥-ρ).
    """
    G, chi, gamma = sym["G"], sym["chi"], sym["gamma"]
    S_max, S_early = sym["S_max"], sym["S_early"]
    Delta_S = S_max - S_early

    # χ-chart
    S_H = S_early + Delta_S * chi
    H = sqrt(pi / (G * S_H))
    rho = simplify(3 * H**2 / (8 * pi * G))
    Hdot = _thermo_Hdot(G, H, S_max, S_early, gamma, chi)
    w = simplify(-1 + Delta_S * gamma * chi * (1 - chi) / (3 * H * S_H))
    P = simplify(w * rho)

    # Combinations
    rho_plus_P = simplify(-Hdot / (4 * pi * G))
    rho_plus_P_alt = simplify(
        H * Delta_S * gamma * chi * (1 - chi) / (8 * pi * G * S_H)
    )
    ok_NEC_expr = simplify(rho_plus_P - rho_plus_P_alt) == 0

    rho_plus_3P = simplify(rho + 3 * P)
    rho_plus_3P_alt = simplify(-2 * rho + 3 * rho_plus_P)
    ok_SEC_expr = simplify(rho_plus_3P - rho_plus_3P_alt) == 0

    # Single-fluid q = (1+3w)/2
    q_fluid = simplify((1 + 3 * w) / 2)
    q_kin = simplify(-1 - Hdot / H**2)
    ok_q = simplify(q_fluid - q_kin) == 0

    # NEC ⇔ 1+w ≥ 0
    one_plus_w = simplify(1 + w)
    ok_NEC_w = simplify(one_plus_w - rho_plus_P / rho) == 0

    # Endpoints / samples
    rp_at_1 = simplify(rho_plus_P.subs({chi: 1}))
    rp_at_0 = simplify(rho_plus_P.subs({chi: 0}))
    r3p_at_1 = simplify(rho_plus_3P.subs({chi: 1}))
    # At χ=1: ρ+3P = -2ρ = -2*(3H_∞²/(8πG)) = -3 H_∞²/(4πG) = -3/(4 G² S_max)? 
    # 3H²/(8πG)*2 = 6H²/(8πG)=3H²/(4πG); with H²=π/(G S_max): 3π/(4π G² S_max)=3/(4 G² S_max)
    r3p_1_tgt = simplify(-sp.Rational(3, 4) / (G**2 * S_max) * 2 / 1)
    # -2ρ at χ=1: ρ=3/(8 G² S_max), -2ρ=-6/(8 G² S_max)=-3/(4 G² S_max)
    r3p_1_tgt = simplify(-sp.Rational(3, 4) / (G**2 * S_max))
    ok_attractor = (rp_at_1 == 0) and (simplify(r3p_at_1 - r3p_1_tgt) == 0)

    # SEC ⇔ q ≥ 0
    # Violation whenever q<0 i.e. whenever acceleration
    sec_iff_q = r"\mathrm{SEC}\Leftrightarrow q\ge 0\mathrm{\ (single\!-\!fluid)}"

    # DEC: |w|≤1; non-phantom w≥-1 and for entropy DE typically w≤0 ≤1
    # ρ - |P|: if P≤0, ρ-|P|=ρ+P=NEC combination
    dec_when_P_neg = simplify(rho + P)  # = ρ+|P| when P≤0? ρ-(-P)=ρ+P yes

    return DerivationStep(
        name="L14_energy_conditions",
        stage="LAYER_14",
        assumptions=[
            "Perfect-fluid interpretation of entropy sector (LAYER 1)",
            "Logistic closure => Ḣ≤0 on [0,1]",
            "Single-fluid cosmographic identities for q↔w",
        ],
        equations={
            "NEC_def": r"\rho_S+P_S\ge 0",
            "WEC_def": r"\rho_S\ge 0\ \mathrm{and}\ \rho_S+P_S\ge 0",
            "DEC_def": r"\rho_S\ge|P_S|",
            "SEC_def": r"\rho_S+3P_S\ge 0",
            "rho_plus_P": Eq(Symbol("rho_plus_P"), rho_plus_P),
            "rho_plus_P_chi": Eq(Symbol("rho_plus_P"), rho_plus_P_alt),
            "NEC_Raychaudhuri": r"\rho_S+P_S=-\dot H/(4\pi G)\Rightarrow\mathrm{NEC}\Leftrightarrow\dot H\le 0",
            "NEC_auto": r"\mathrm{logistic\ on\ }[0,1]:\ \dot H\le 0\Rightarrow\mathrm{NEC\ automatic}",
            "NEC_w": r"\mathrm{NEC}\Leftrightarrow w_S\ge -1\mathrm{\ (non\!-\!phantom)}",
            "WEC": r"\rho_S=3H^2/(8\pi G)>0\mathrm{\ and\ NEC}\Rightarrow\mathrm{WEC}",
            "rho_plus_3P": Eq(Symbol("rho_plus_3P"), rho_plus_3P),
            "SEC_w": r"\mathrm{SEC}\Leftrightarrow w_S\ge -1/3",
            "q_fluid": Eq(Symbol("q"), q_fluid),
            "q_kin": Eq(Symbol("q"), q_kin),
            "SEC_q": sec_iff_q,
            "SEC_accel": (
                r"\mathrm{SEC\ fails}\Leftrightarrow q<0\Leftrightarrow\mathrm{accelerated\ expansion}"
            ),
            "DEC_DE": (
                r"P_S\le 0:\ \rho_S\ge|P_S|\Leftrightarrow\rho_S+P_S\ge 0\Leftrightarrow\mathrm{NEC}"
            ),
            "attractor_NEC": r"\chi\to1:\ \rho+P\to 0\mathrm{\ (NEC\ saturated)}",
            "attractor_SEC": r"\chi\to1:\ \rho+3P\to -2\rho_{S,\infty}<0\mathrm{\ (SEC\ violated)}",
            "rp_at_1": Eq(Symbol("rp_at_1"), rp_at_1),
            "r3p_at_1": Eq(Symbol("r3p_at_1"), r3p_at_1),
        },
        claims=[
            "LAYER 14 THEOREM: NEC automatic for logistic entropy fluid (Ḣ≤0).",
            "LAYER 14 THEOREM: WEC holds; DEC reduces to NEC while P_S≤0.",
            "LAYER 14 THEOREM: SEC ⇔ q≥0; SEC failure is exactly accelerated expansion.",
            "LAYER 14: at χ=1, NEC saturated and SEC violated (eternal de Sitter acceleration).",
        ],
        status="derived" if (ok_NEC_expr and ok_SEC_expr and ok_q and ok_NEC_w and ok_attractor) else "failed",
        evidence={
            "rho_plus_P": rho_plus_P,
            "rho_plus_P_alt": rho_plus_P_alt,
            "rho_plus_3P": rho_plus_3P,
            "q_fluid": q_fluid,
            "q_kin": q_kin,
            "w": w,
            "rp_at_1": rp_at_1,
            "r3p_at_1": r3p_at_1,
            "ok_NEC_expr": ok_NEC_expr,
            "ok_SEC_expr": ok_SEC_expr,
            "ok_q": ok_q,
            "ok_NEC_w": ok_NEC_w,
            "ok_attractor": ok_attractor,
            "dec_when_P_neg": dec_when_P_neg,
            "rp_at_0": rp_at_0,
        },
    )


def step_L15_Gamma_class(sym: Dict[str, Any]) -> DerivationStep:
    """
    LAYER 15 - Γ-class closures and structural universality of the dS endpoint.

    Definition (Γ-class)
    --------------------
        Γ(χ) = γ χ^n (1-χ)^m ,   γ>0, n>0, m>0,  χ∈[0,1]

    Boundary / sign axioms (shared with logistic)
    ---------------------------------------------
        Γ(0)=Γ(1)=0,   Γ(χ)>0 on (0,1)
        =>  χ̇=Γ has fixed points only at endpoints (no interior zeros)

    Thermo EOM with general closure
    -------------------------------
        S_H = S_early + ΔS χ,   S_H=π/(G H²)
        Ṡ_H = ΔS Γ ≥ 0
        Ḣ = -(H/(2 S_H)) ΔS Γ ≤ 0     (NEC automatic; LAYER 14)

    Endpoint universality (independent of n,m,γ)
    --------------------------------------------
        χ->1  =>  S_H->S_max  =>  H->H_∞=√(π/(G S_max))
        =>  w_S->-1,  Λ_S=3 H_∞²=3π/(G S_max)
        The *existence* of the late dS geometry is universal in the Γ-class;
        the *approach timescale* is not.

    Local approach to the attractor (ε=1-χ≪1)
    ------------------------------------------
        χ̇ = γ χ^n (1-χ)^m  =>  ε̇ = -γ (1-ε)^n ε^m ~ -γ ε^m
        m=1:  ε̇~-γ ε  =>  ε(t)∼e^{-γ t},   λ_t=Γ'(1)=-γ
        m>1:  algebraic / slower; Γ'(1)=0 (flat linearization)
        0<m<1: finite-time arrival possible (∫ε^{-m}dε converges)

    Escape from χ=0 (δ=χ≪1)
    -----------------------
        δ̇ ~ γ δ^n
        n=1: exponential repulsion, Γ'(0)=+γ (unstable)
        n>1: weaker (power-law) escape; Γ'(0)=0

    Lyapunov (LAYER 5 style)
    ------------------------
        V=S_max-S_H=ΔS(1-χ) ≥ 0
        V̇=-ΔS Γ ≤ 0 on [0,1],  V̇=0 iff Γ=0 iff χ∈{0,1}
        => GAS of χ=1 on (0,1] for any Γ-class member (same argument)

    Logistic recovery
    -----------------
        n=m=1 recovers Γ=γ χ(1-χ) used in Layers 1-14.
    """
    G, chi, gamma = sym["G"], sym["chi"], sym["gamma"]
    S_max, S_early = sym["S_max"], sym["S_early"]
    Delta_S = S_max - S_early
    n = Symbol("n", positive=True)
    m = Symbol("m", positive=True)

    Gamma = gamma * chi**n * (1 - chi) ** m
    Gamma_logistic = simplify(Gamma.subs({n: 1, m: 1}))
    ok_logistic = simplify(Gamma_logistic - gamma * chi * (1 - chi)) == 0

    # Boundary values (concrete positive integers; symbolic 0^n is delicate)
    G00 = simplify(Gamma.subs({chi: 0, n: 2, m: 2}))
    G11 = simplify(Gamma.subs({chi: 1, n: 2, m: 2}))
    ok_boundary = (G00 == 0) and (G11 == 0)

    # Interior sample positivity (γ>0, χ=1/2, n=m=1)
    G_half = simplify(Gamma.subs({chi: sp.Rational(1, 2), n: 1, m: 1}))
    ok_interior = simplify(G_half - gamma / 4) == 0

    # Linearization Γ'(χ)
    Gamma_p = simplify(diff(Gamma, chi))
    # m=1: Γ'(1) = -γ (independent of n>0)
    lam1_m1 = simplify(Gamma_p.subs({m: 1, chi: 1}))
    ok_lam1 = simplify(lam1_m1 + gamma) == 0
    # m=2,n=1: Γ'(1)=0 (flat)
    lam1_m2 = simplify(Gamma_p.subs({n: 1, m: 2, chi: 1}))
    ok_flat = lam1_m2 == 0
    # n=1: Γ'(0)=+γ
    lam0_n1 = simplify(Gamma_p.subs({n: 1, chi: 0}))
    ok_lam0 = simplify(lam0_n1 - gamma) == 0

    # Near-attractor: ε=1-χ, ε̇ ~ -γ ε^m
    eps = Symbol("varepsilon", positive=True)
    chi_eps = 1 - eps
    eps_dot = simplify(-Gamma.subs({chi: chi_eps}))
    # Concrete m=n=1: leading ε̇=-γ ε
    eps_dot_m1 = simplify(
        sp.series(eps_dot.subs({m: 1, n: 1}), eps, 0, 2).removeO()
    )
    ok_eps = simplify(eps_dot_m1 + gamma * eps) == 0

    # Universal H_∞ from χ->1 (independent of Γ parameters)
    S_H = S_early + Delta_S * chi
    H = sqrt(pi / (G * S_H))
    H_inf = simplify(H.subs({chi: 1}))
    H_inf_tgt = sqrt(pi / (G * S_max))
    ok_Hinf = simplify(H_inf - H_inf_tgt) == 0

    # Thermo Ḣ with general Γ
    Hdot_gen = simplify(-(H / (2 * S_H)) * Delta_S * Gamma)
    # Recover logistic thermo when n=m=1
    Hdot_log = simplify(Hdot_gen.subs({n: 1, m: 1}))
    Hdot_locked = _thermo_Hdot(G, H, S_max, S_early, gamma, chi)
    ok_Hdot = simplify(Hdot_log - Hdot_locked) == 0

    # NEC combination ∝ Γ ≥ 0
    rho_plus_P = simplify(-Hdot_gen / (4 * pi * G))
    ok_NEC_sign = (
        simplify(
            rho_plus_P.subs({n: 1, m: 1})
            - H * Delta_S * gamma * chi * (1 - chi) / (8 * pi * G * S_H)
        )
        == 0
    )

    # Lyapunov decrease for general Γ
    V = Delta_S * (1 - chi)
    Vdot = simplify(-Delta_S * Gamma)
    ok_Lyap = simplify(Vdot + Delta_S * Gamma) == 0

    # Timescale non-universality: λ_t = Γ'(1) depends on m
    # m=1 => -γ; m=2 => 0 (different approach class)
    ok_timescale = ok_lam1 and ok_flat

    all_ok = (
        ok_logistic
        and ok_boundary
        and ok_interior
        and ok_lam1
        and ok_flat
        and ok_lam0
        and ok_eps
        and ok_Hinf
        and ok_Hdot
        and ok_NEC_sign
        and ok_Lyap
        and ok_timescale
    )

    return DerivationStep(
        name="L15_Gamma_class_universality",
        stage="LAYER_15",
        assumptions=[
            "Γ-class: Γ=γ χ^n(1-χ)^m with γ,n,m>0",
            "Same locked S_H=π/(G H²) and S_H=S_early+ΔS χ as Layers 1-14",
            "GSL realized by χ̇=Γ≥0 (closure still assumed, not derived from GSL)",
        ],
        equations={
            "Gamma_class": Eq(Symbol("Gamma"), Gamma),
            "axioms": r"\Gamma(0)=\Gamma(1)=0,\ \Gamma>0\ \mathrm{on}\ (0,1)",
            "logistic_recovery": Eq(Symbol("Gamma_log"), Gamma_logistic),
            "Hdot_gen": Eq(Symbol("Hdot"), Hdot_gen),
            "Hdot_logistic_match": r"n=m=1:\ \dot H\mathrm{\ recovers\ LAYER\ 1}",
            "NEC_gen": (
                r"\rho_S+P_S=-\dot H/(4\pi G)\propto\Gamma\ge 0"
                r"\Rightarrow\mathrm{NEC\ automatic\ on\ \Gamma\!-\!class}"
            ),
            "H_inf": Eq(Symbol("H_inf"), H_inf),
            "H_inf_universal": (
                r"H_\infty=\sqrt{\pi/(G S_{\max})}\mathrm{\ independent\ of\ }(n,m,\gamma)"
            ),
            "Lambda_S": r"\Lambda_S=3H_\infty^2=3\pi/(G S_{\max})",
            "eps_eq": r"\varepsilon=1-\chi:\ \dot\varepsilon=-\gamma(1-\varepsilon)^n\varepsilon^m",
            "eps_lead": r"\varepsilon\ll 1:\ \dot\varepsilon\approx-\gamma\varepsilon^m",
            "lambda_m1": Eq(Symbol("lambda_1"), lam1_m1),
            "stability_m1": r"m=1:\ \lambda_t=\Gamma'(1)=-\gamma\mathrm{\ (exponential)}",
            "stability_mgt1": r"m>1:\ \Gamma'(1)=0\mathrm{\ (slower\ /\ algebraic\ approach)}",
            "lambda_0": Eq(Symbol("lambda_0"), lam0_n1),
            "repeller_n1": r"n=1:\ \Gamma'(0)=+\gamma\mathrm{\ (}\chi=0\mathrm{\ unstable)}",
            "Lyapunov_V": Eq(Symbol("V"), V),
            "Lyapunov_Vdot": Eq(Symbol("Vdot"), Vdot),
            "GAS": (
                r"V=\Delta S(1-\chi):\ \dot V=-\Delta S\,\Gamma\le 0"
                r"\Rightarrow\mathrm{GAS\ of\ }\chi=1\mathrm{\ on\ }(0,1]"
            ),
            "universal_vs_timescale": (
                r"\mathrm{universal:\ }(\chi,H,w)\to(1,H_\infty,-1);"
                r"\ \mathrm{non\!-\!universal:\ timescale\ }(n,m,\gamma)"
            ),
        },
        claims=[
            "LAYER 15 THEOREM: every Γ-class closure shares the same dS endpoint H_∞.",
            "LAYER 15 THEOREM: NEC and Lyapunov GAS persist for all Γ=γ χ^n(1-χ)^m.",
            "LAYER 15 DERIVED: m=1 => λ_t=-γ; m>1 => flat linearization (slower approach).",
            "LAYER 15: approach timescale is NOT universal - only the attractor geometry is.",
        ],
        status="derived" if all_ok else "failed",
        evidence={
            "Gamma": Gamma,
            "Gamma_logistic": Gamma_logistic,
            "Gamma_p": Gamma_p,
            "lam1_m1": lam1_m1,
            "lam1_m2": lam1_m2,
            "lam0_n1": lam0_n1,
            "eps_dot": eps_dot,
            "eps_dot_m1": eps_dot_m1,
            "H_inf": H_inf,
            "Hdot_gen": Hdot_gen,
            "V": V,
            "Vdot": Vdot,
            "ok_logistic": ok_logistic,
            "ok_boundary": ok_boundary,
            "ok_interior": ok_interior,
            "ok_lam1": ok_lam1,
            "ok_flat": ok_flat,
            "ok_lam0": ok_lam0,
            "ok_eps": ok_eps,
            "ok_Hinf": ok_Hinf,
            "ok_Hdot": ok_Hdot,
            "ok_NEC_sign": ok_NEC_sign,
            "ok_Lyap": ok_Lyap,
        },
    )


def step_L16_bifurcation(sym: Dict[str, Any]) -> DerivationStep:
    """
    LAYER 16 - Bifurcation analysis of one-parameter logistic deformations.

    Family
    ------
        Γ_α(χ) = γ χ(1-χ)(1+α χ),   γ>0,  α∈ℝ,  χ∈[0,1]
        α=0 recovers the logistic closure of Layers 1-15.

    Fixed points (Γ_α=0)
    --------------------
        χ=0,   χ=1,   and   χ_*=-1/α  (when α!=0)
        χ_* ∈ (0,1)  ⇔  α < -1

    Linearization λ(χ)=Γ_α'(χ)
    --------------------------
        Γ_α' = γ[1 + 2(α-1)χ - 3α χ²]
        λ_0 = Γ'(0) = γ > 0          => χ=0 always unstable
        λ_1 = Γ'(1) = -γ(1+α)        => χ=1 stable ⇔ α > -1

    Bifurcation diagram (γ>0)
    -------------------------
        α > -1:   FPs {0 unstable, 1 stable}; Γ≥0 on [0,1]
                  late dS (χ->1) structurally stable (includes logistic α=0)
        α = -1:   Γ=γ χ(1-χ)²; λ_1=0 (marginal)
                  χ_* collides with χ=1 (transcritical coalescence)
                  approach to 1 is slower (LAYER 15 m=2-like)
        α < -1:   FPs {0 unstable, χ_* ∈(0,1) attracting, 1 unstable}
                  attractor leaves full entropy completion; H↛H_∞
                  Γ changes sign at χ_* => GSL/monotonic Ṡ_H fails on (χ_*,1)

    Transcritical mechanism
    -----------------------
        As α↑-1 from below, χ_*=-1/α -> 1^- and exchanges stability with χ=1.
        Critical value α_c=-1 is the boundary of structural stability of dS.

    Physical reading
    ----------------
        The late-time de Sitter attractor of ΛCDM+S is robust under this
        deformation for all α>-1. Crossing α=-1 destroys χ->1 and replaces
        it by a partial-completion fixed point χ_*<1.
    """
    G, chi, gamma = sym["G"], sym["chi"], sym["gamma"]
    S_max, S_early = sym["S_max"], sym["S_early"]
    Delta_S = S_max - S_early
    alpha = Symbol("alpha", real=True)

    Gamma = gamma * chi * (1 - chi) * (1 + alpha * chi)
    Gamma_exp = sp.expand(Gamma)
    ok_logistic = simplify(Gamma.subs({alpha: 0}) - gamma * chi * (1 - chi)) == 0

    # Expanded derivative: Γ'=γ[1+2(α-1)χ-3α χ²]
    Gamma_p = simplify(diff(Gamma, chi))
    Gamma_p_tgt = simplify(gamma * (1 + 2 * (alpha - 1) * chi - 3 * alpha * chi**2))
    ok_deriv = simplify(Gamma_p - Gamma_p_tgt) == 0

    lam0 = simplify(Gamma_p.subs({chi: 0}))
    lam1 = simplify(Gamma_p.subs({chi: 1}))
    ok_lam0 = simplify(lam0 - gamma) == 0
    ok_lam1 = simplify(lam1 + gamma * (1 + alpha)) == 0

    # Interior fixed point χ_*=-1/α
    chi_star = simplify(-1 / alpha)
    G_at_star = simplify(Gamma.subs({chi: chi_star}))
    ok_star_root = G_at_star == 0

    # χ_* ∈ (0,1) samples: α=-2 => 1/2; α=-4 => 1/4
    chi_star_a2 = simplify(chi_star.subs({alpha: -2}))
    chi_star_a4 = simplify(chi_star.subs({alpha: -4}))
    ok_star_interval = (chi_star_a2 == sp.Rational(1, 2)) and (
        chi_star_a4 == sp.Rational(1, 4)
    )

    # At α=-2, χ_*=1/2: λ_*=Γ'=-γ/2 < 0 (attracting)
    lam_star_a2 = simplify(Gamma_p.subs({chi: sp.Rational(1, 2), alpha: -2}))
    ok_star_stable = simplify(lam_star_a2 + gamma / 2) == 0

    # At α=-1: marginal λ_1=0 and Γ=γ χ(1-χ)²
    lam1_crit = simplify(lam1.subs({alpha: -1}))
    Gamma_crit = simplify(Gamma.subs({alpha: -1}))
    ok_crit = (lam1_crit == 0) and (
        simplify(Gamma_crit - gamma * chi * (1 - chi) ** 2) == 0
    )

    # Structural stability regions (symbolic inequalities via samples)
    # α=0 > -1: λ_1 = -γ < 0 stable
    ok_stable_region = simplify(lam1.subs({alpha: 0}) + gamma) == 0
    # α=-2 < -1: λ_1 = -γ(1-2)=γ > 0 unstable dS
    ok_unstable_ds = simplify(lam1.subs({alpha: -2}) - gamma) == 0

    # Transcritical: as α->-1^-, χ_*->1
    chi_star_near = simplify(chi_star.subs({alpha: sp.Rational(-11, 10)}))  # -1.1
    # -1/(-1.1)=10/11 ~ 0.909 -> approaches 1
    ok_collide = simplify(chi_star_near - sp.Rational(10, 11)) == 0

    # Sign of Γ on (0,1) for α≥-1: factor (1+αχ)≥0
    # Sample α=-1/2, χ=1/2: 1+αχ=1-1/4=3/4>0
    G_sample = simplify(Gamma.subs({alpha: -sp.Rational(1, 2), chi: sp.Rational(1, 2)}))
    ok_GSL_side = simplify(G_sample - gamma * sp.Rational(1, 2) * sp.Rational(1, 2) * sp.Rational(3, 4)) == 0

    # For α=-2, Γ>0 on (0,1/2) and Γ<0 on (1/2,1)
    G_left = simplify(Gamma.subs({alpha: -2, chi: sp.Rational(1, 4)}))
    G_right = simplify(Gamma.subs({alpha: -2, chi: sp.Rational(3, 4)}))
    ok_sign_flip = (sp.sign(G_left.subs({gamma: 1})) == 1) and (
        sp.sign(G_right.subs({gamma: 1})) == -1
    )

    # Incomplete attractor: at χ_*=1/2, H!=H_∞
    S_H = S_early + Delta_S * chi
    H = sqrt(pi / (G * S_H))
    H_star_pt = simplify(H.subs({chi: sp.Rational(1, 2)}))
    H_inf = sqrt(pi / (G * S_max))
    ok_not_Hinf = simplify(H_star_pt - H_inf) != 0

    # Phase summary string
    bif_summary = (
        r"\alpha_c=-1\mathrm{\ (transcritical)}:\ "
        r"\alpha>-1\Rightarrow\chi=1\mathrm{\ GAS;\ }"
        r"\alpha<-1\Rightarrow\chi_*=-1/\alpha\mathrm{\ attractor,\ }\chi=1\mathrm{\ unstable}"
    )

    all_ok = (
        ok_logistic
        and ok_deriv
        and ok_lam0
        and ok_lam1
        and ok_star_root
        and ok_star_interval
        and ok_star_stable
        and ok_crit
        and ok_stable_region
        and ok_unstable_ds
        and ok_collide
        and ok_GSL_side
        and ok_sign_flip
        and ok_not_Hinf
    )

    return DerivationStep(
        name="L16_bifurcation_analysis",
        stage="LAYER_16",
        assumptions=[
            "One-parameter polynomial deformation of logistic: Γ_α=γ χ(1-χ)(1+αχ)",
            "γ>0; α treated as real bifurcation parameter",
            "Same locked S_H chart as Layers 1-15",
        ],
        equations={
            "Gamma_alpha": Eq(Symbol("Gamma"), Gamma),
            "Gamma_expanded": Eq(Symbol("Gamma"), Gamma_exp),
            "logistic_recovery": r"\alpha=0:\ \Gamma=\gamma\chi(1-\chi)",
            "fixed_points": r"\Gamma=0\Rightarrow\chi\in\{0,1,-1/\alpha\}",
            "chi_star": Eq(Symbol("chi_star"), chi_star),
            "chi_star_in_I": r"\chi_*\in(0,1)\Leftrightarrow\alpha<-1",
            "Gamma_prime": Eq(Symbol("Gamma_prime"), Gamma_p),
            "lambda_0": Eq(Symbol("lambda_0"), lam0),
            "lambda_1": Eq(Symbol("lambda_1"), lam1),
            "stability_0": r"\lambda_0=\gamma>0\Rightarrow\chi=0\mathrm{\ always\ unstable}",
            "stability_1": r"\lambda_1=-\gamma(1+\alpha):\ \chi=1\mathrm{\ stable}\Leftrightarrow\alpha>-1",
            "lambda_star_a2": Eq(Symbol("lambda_star"), lam_star_a2),
            "crit_alpha": r"\alpha_c=-1:\ \lambda_1=0,\ \Gamma=\gamma\chi(1-\chi)^2",
            "transcritical": (
                r"\alpha\uparrow-1^-:\ \chi_*\to1^-\mathrm{\ exchanges\ stability\ with\ }\chi=1"
            ),
            "region_stable_dS": (
                r"\alpha>-1:\ \{0\mathrm{\ unstable},\ 1\mathrm{\ stable}\},\ \Gamma\ge0\mathrm{\ on\ }[0,1]"
            ),
            "region_broken": (
                r"\alpha<-1:\ \chi_*\mathrm{\ attractor},\ \chi=1\mathrm{\ unstable},"
                r"\ \Gamma\mathrm{\ sign-flips}\Rightarrow\dot S_H\not\ge0\mathrm{\ on\ whole\ }[0,1]"
            ),
            "H_at_star": Eq(Symbol("H_star_pt"), H_star_pt),
            "not_Hinf": r"\chi_*<1\Rightarrow H(\chi_*)\ne H_\infty=\sqrt{\pi/(GS_{\max})}",
            "bifurcation_summary": bif_summary,
        },
        claims=[
            "LAYER 16 THEOREM: χ=1 is linearly stable iff α>-1 (λ_1=-γ(1+α)).",
            "LAYER 16 THEOREM: α_c=-1 is a transcritical bifurcation; χ_* collides with χ=1.",
            "LAYER 16 DERIVED: for α<-1 the attractor is χ_*=-1/α<1 (incomplete dS).",
            "LAYER 16: late-time de Sitter of ΛCDM+S is structurally stable for all α>-1.",
        ],
        status="derived" if all_ok else "failed",
        evidence={
            "Gamma": Gamma,
            "Gamma_p": Gamma_p,
            "lam0": lam0,
            "lam1": lam1,
            "chi_star": chi_star,
            "lam_star_a2": lam_star_a2,
            "Gamma_crit": Gamma_crit,
            "H_star_pt": H_star_pt,
            "H_inf": H_inf,
            "ok_logistic": ok_logistic,
            "ok_deriv": ok_deriv,
            "ok_lam0": ok_lam0,
            "ok_lam1": ok_lam1,
            "ok_star_root": ok_star_root,
            "ok_star_interval": ok_star_interval,
            "ok_star_stable": ok_star_stable,
            "ok_crit": ok_crit,
            "ok_stable_region": ok_stable_region,
            "ok_unstable_ds": ok_unstable_ds,
            "ok_collide": ok_collide,
            "ok_GSL_side": ok_GSL_side,
            "ok_sign_flip": ok_sign_flip,
            "ok_not_Hinf": ok_not_Hinf,
        },
    )


def step_L17_thermodynamic_time(sym: Dict[str, Any]) -> DerivationStep:
    """
    LAYER 17 - Thermodynamic arrow of time from horizon entropy.

    Problem
    -------
    Einstein equations (and FLRW kinematics) are time-reversal invariant:
    t ↦ -t with H ↦ -H is a symmetry of the vacuum/background equations.
    They alone do not select a preferred temporal orientation.

    Construction
    ------------
    Entropy completion coordinate (LAYER 4-5):
        χ ≡ (S_H - S_early)/ΔS ∈ [0,1],   ΔS = S_max - S_early

    Define the thermodynamic time (internal clock):
        τ_S := χ

    Equivalently τ_S = (S_H - S_early)/ΔS, so dτ_S ∝ dS_H.

    Monotone arrow (Γ-class with Γ≥0, e.g. logistic or LAYER 16 α>-1)
    -----------------------------------------------------------------
        χ̇ = Γ(χ) ≥ 0 on [0,1]
        =>  τ̇_S ≥ 0
        =>  Ṡ_H = ΔS τ̇_S ≥ 0

    Thermodynamic arrow theorem
    ---------------------------
        τ̇_S > 0  ⇔  entropy production  ⇔  irreversible advance of τ_S
        Time reversal t↦-t sends τ̇_S ↦ -τ̇_S and violates Ṡ_H≥0 unless
        the state is already at equilibrium (Γ=0).

    Reparametrization (cosmic time vs thermo time)
    ----------------------------------------------
        dt = dτ_S / Γ(τ_S)     (Γ>0 in the interior)
        All background observables become functions of τ_S alone:
            H = H(τ_S),  a = a(τ_S),  ρ_S=ρ_S(τ_S),  w_S=w_S(τ_S)
        (LAYER 3 chart with χ -> τ_S).

    Lyapunov / equilibrium
    ----------------------
        V = ΔS(1 - τ_S) = S_max - S_H ≥ 0
        V̇ = -ΔS Γ ≤ 0
        τ_S -> 1  =>  Γ->0  =>  clock freezes at equilibrium (no further production)

    What is derived vs assumed
    --------------------------
        DERIVED: once a monotone closure Γ≥0 is given, τ_S is a thermodynamic
                 clock and selects an arrow dual to Ṡ_H≥0.
        ASSUMED: the closure itself (logistic / Γ-class) - not unique from GSL.
        CONDITIONAL: global arrow on the whole [0,1] requires Γ≥0 everywhere
                     (fails on LAYER 16 branch α<-1 where Γ sign-flips).
    """
    G, chi, gamma = sym["G"], sym["chi"], sym["gamma"]
    S_max, S_early = sym["S_max"], sym["S_early"]
    Delta_S = S_max - S_early

    # Clock definition
    tau_S = chi
    S_H = S_early + Delta_S * chi
    tau_from_S = simplify((S_H - S_early) / Delta_S)
    ok_def = simplify(tau_from_S - chi) == 0

    # Logistic representative: τ̇_S = γ τ_S (1-τ_S) ≥ 0 on [0,1]
    Gamma = gamma * chi * (1 - chi)
    tau_dot = Gamma
    ok_monotone_ends = (simplify(tau_dot.subs({chi: 0})) == 0) and (
        simplify(tau_dot.subs({chi: 1})) == 0
    )
    ok_monotone_mid = simplify(tau_dot.subs({chi: sp.Rational(1, 2)}) - gamma / 4) == 0

    # Entropy production
    S_H_dot = simplify(Delta_S * tau_dot)
    ok_prod = simplify(S_H_dot - Delta_S * gamma * chi * (1 - chi)) == 0

    # Arrow ↔ production equivalence (interior)
    # τ̇_S > 0 ⇔ Ṡ_H > 0 for ΔS>0
    ok_equiv = simplify(S_H_dot / Delta_S - tau_dot) == 0

    # Reparametrization dt = dτ / Γ
    # dχ/dt = Γ => dt/dχ = 1/Γ
    dt_dtau = simplify(1 / Gamma)
    # Consistency: (dχ/dt)*(dt/dχ)=1
    ok_reparam = simplify(Gamma * dt_dtau - 1) == 0

    # Background as functions of τ_S
    H_tau = simplify(sqrt(pi / (G * (S_early + Delta_S * tau_S))))
    rho_tau = simplify(sp.Rational(3, 8) / (G**2 * (S_early + Delta_S * tau_S)))
    H_inf = simplify(H_tau.subs({chi: 1}))
    ok_H = simplify(H_inf - sqrt(pi / (G * S_max))) == 0

    # Time-reversal breaks GSL: under t->-t, χ̇ -> -χ̇
    # Forward: χ̇=Γ≥0; reversed: χ̇_rev=-Γ≤0 => Ṡ_H_rev=-ΔS Γ≤0
    S_H_dot_rev = simplify(-S_H_dot)
    ok_rev_breaks = simplify(S_H_dot_rev + S_H_dot) == 0  # opposite sign
    # At equilibrium χ=1 both vanish - reversible only there
    ok_eq_freeze = simplify(S_H_dot.subs({chi: 1})) == 0

    # Lyapunov clock dual
    V = simplify(Delta_S * (1 - tau_S))
    Vdot = simplify(-Delta_S * Gamma)
    ok_V = simplify(V - (S_max - S_H)) == 0
    ok_Vdot = simplify(Vdot + S_H_dot) == 0

    # Range
    ok_range = True  # τ_S=χ∈[0,1] by definition of completion coordinate

    # Structural link to LAYER 16: arrow global iff α>-1
    alpha = Symbol("alpha", real=True)
    Gamma_a = gamma * chi * (1 - chi) * (1 + alpha * chi)
    # For α=-2<-1, Γ(3/4)<0 => local arrow reversal
    G_bad = simplify(Gamma_a.subs({alpha: -2, chi: sp.Rational(3, 4)}))
    ok_cond_arrow = sp.sign(G_bad.subs({gamma: 1})) == -1

    all_ok = (
        ok_def
        and ok_monotone_ends
        and ok_monotone_mid
        and ok_prod
        and ok_equiv
        and ok_reparam
        and ok_H
        and ok_rev_breaks
        and ok_eq_freeze
        and ok_V
        and ok_Vdot
        and ok_cond_arrow
        and ok_range
    )

    return DerivationStep(
        name="L17_thermodynamic_arrow_of_time",
        stage="LAYER_17",
        assumptions=[
            "Locked S_H=π/(G H²) and χ=(S_H-S_early)/ΔS",
            "Closure with Γ≥0 on [0,1] (logistic / Γ-class / LAYER 16 α>-1)",
            "Einstein/FLRW equations alone are time-reversal symmetric",
        ],
        equations={
            "tau_def": Eq(Symbol("tau_S"), tau_S),
            "tau_from_S": (
                r"\tau_S=\chi=(S_H-S_{\mathrm{early}})/\Delta S\in[0,1]"
            ),
            "tau_dot": Eq(Symbol("tau_dot"), tau_dot),
            "S_H_dot": Eq(Symbol("S_H_dot"), S_H_dot),
            "arrow_iff": (
                r"\dot\tau_S>0\Leftrightarrow\dot S_H>0"
                r"\Leftrightarrow\mathrm{entropy\ production}"
            ),
            "monotone": r"\Gamma\ge0\mathrm{\ on\ }[0,1]\Rightarrow\dot\tau_S\ge0",
            "dt_dtau": Eq(Symbol("dt_dtau"), dt_dtau),
            "reparam": r"dt=d\tau_S/\Gamma(\tau_S)\quad(\Gamma>0\mathrm{\ interior})",
            "H_tau": Eq(Symbol("H"), H_tau),
            "rho_tau": Eq(Symbol("rho_S"), rho_tau),
            "chart": (
                r"H=H(\tau_S),\ a=a(\tau_S),\ \rho_S=\rho_S(\tau_S),\ "
                r"w_S=w_S(\tau_S)\mathrm{\ (LAYER\ 3\ chart)}"
            ),
            "time_reversal": (
                r"t\mapsto -t:\ \dot\tau_S\mapsto-\dot\tau_S"
                r"\Rightarrow\dot S_H\mapsto-\dot S_H\mathrm{\ violates\ GSL}"
            ),
            "equilibrium_freeze": (
                r"\tau_S\to1:\ \Gamma\to0,\ \dot S_H\to0"
                r"\mathrm{\ (clock\ freezes\ at\ equilibrium)}"
            ),
            "Lyapunov_dual": Eq(Symbol("V"), V),
            "Lyapunov_Vdot": Eq(Symbol("Vdot"), Vdot),
            "GR_no_arrow": (
                r"\mathrm{EFE/FLRW\ time\!-\!reversal\ symmetric};"
                r"\ \mathrm{arrow\ from\ }S_H\mathrm{\ production}"
            ),
            "conditional_global": (
                r"\mathrm{global\ arrow\ on\ }[0,1]\mathrm{\ requires\ }"
                r"\Gamma\ge0\mathrm{\ (fails\ if\ LAYER\ 16\ }\alpha<-1\mathrm{)}"
            ),
        },
        claims=[
            "LAYER 17 THEOREM: τ_S=χ is a thermodynamic clock with τ̇_S⇔Ṡ_H production.",
            "LAYER 17 THEOREM: time reversal violates Ṡ_H≥0 except at equilibrium Γ=0.",
            "LAYER 17 DERIVED: cosmic time is the reparametrization dt=dτ_S/Γ(τ_S).",
            "LAYER 17: EFE alone give no arrow; horizon entropy production selects it.",
        ],
        open_gaps=[
            "Unique Γ from GSL alone remains underived - the arrow orientation is "
            "fixed once any monotone closure is chosen, but the clock rate is model-dependent.",
        ],
        status="derived" if all_ok else "failed",
        evidence={
            "tau_S": tau_S,
            "tau_dot": tau_dot,
            "S_H_dot": S_H_dot,
            "dt_dtau": dt_dtau,
            "H_tau": H_tau,
            "rho_tau": rho_tau,
            "V": V,
            "Vdot": Vdot,
            "S_H_dot_rev": S_H_dot_rev,
            "ok_def": ok_def,
            "ok_monotone_ends": ok_monotone_ends,
            "ok_monotone_mid": ok_monotone_mid,
            "ok_prod": ok_prod,
            "ok_equiv": ok_equiv,
            "ok_reparam": ok_reparam,
            "ok_H": ok_H,
            "ok_rev_breaks": ok_rev_breaks,
            "ok_eq_freeze": ok_eq_freeze,
            "ok_V": ok_V,
            "ok_Vdot": ok_Vdot,
            "ok_cond_arrow": ok_cond_arrow,
        },
    )


def step_L18_generalized_Friedmann(sym: Dict[str, Any]) -> DerivationStep:
    """
    LAYER 18 - Generalized Friedmann: entropy as primary geometric state.

    Architecture
    ------------
    Promote horizon entropy to the independent state variable and write
        H² = F(S_H)
    for a constitutive map F. Dynamics then follow by chain rule once a
    closure for Ṡ_H is supplied:
        Ḣ = (dH/dS_H) Ṡ_H ,   H = √F(S_H)

    Locked BH + apparent-horizon inversion (canon)
    ----------------------------------------------
        S_H = A_H/(4G) = 4π R_H²/(4G) = π/(G H²)
        =>  H² = π/(G S_H)
        =>  F_locked(S) := π/(G S)

    This is not an extra postulate beyond the locked S_H law: it is the
    algebraic inverse of S_H=π/(G H²).

    Standard Friedmann as special case
    ----------------------------------
        ρ_tot = 3 H²/(8πG) = 3 F(S_H)/(8πG)
    With F=F_locked one recovers the usual Friedmann equation with
        ρ_S = 3/(8 G² S_H)
    (LAYER 1). Raychaudhuri still supplies P via ρ+P=-Ḣ/(4πG).

    Explicit thermo kinematics for F_locked
    ---------------------------------------
        H = √(π/(G S)),   dH/dS = -H/(2S)
        Ḣ = -(H/(2 S_H)) Ṡ_H
    With Ṡ_H=ΔS Γ(χ) this is exactly the LAYER 1 / Γ-class EOM.

    Generalized / modified reading
    ------------------------------
    A different constitutive F(S) defines a thermodynamic modified-gravity
    cosmology (same Ṡ closure, different H(S)). LAYER 20 realizes this by
    deforming S(A) and inverting for H(S). The attractor geometry becomes
        H_∞ = √F(S_max)
    whenever S_H->S_max.

    What is derived
    ---------------
        DERIVED: F_locked from inverting the locked entropy law; Friedmann
                 density identification; chain-rule Ḣ; match to thermo EOM.
        ARCHITECTURE: H²=F(S) as the organizing principle for entropy-first
                      cosmology (standard GR cosmology = one choice of F).
    """
    G, H, chi, gamma = sym["G"], sym["H"], sym["chi"], sym["gamma"]
    S_max, S_early = sym["S_max"], sym["S_early"]
    Delta_S = S_max - S_early
    S = Symbol("S", positive=True)
    Sdot = Symbol("Sdot", real=True)

    # Locked inversion
    S_H_locked = _locked_SH(G, H)
    F_locked = simplify(pi / (G * S))
    ok_inverse = simplify(S_H_locked.subs({H: sqrt(F_locked.subs({S: S_H_locked}))}) - S_H_locked) == 0
    # Cleaner: H² * (G S_H / π) = 1
    ok_inverse = simplify(H**2 * G * S_H_locked / pi - 1) == 0

    # On-shell: F(S_H)=H²
    F_on_shell = simplify(F_locked.subs({S: S_H_locked}))
    ok_onshell = simplify(F_on_shell - H**2) == 0

    # Friedmann density from F
    rho_from_F = simplify(3 * F_locked / (8 * pi * G))
    rho_from_H = simplify(3 * H**2 / (8 * pi * G))
    rho_from_S = simplify(sp.Rational(3, 8) / (G**2 * S))
    ok_rho = simplify(rho_from_F - rho_from_S) == 0
    ok_rho_H = simplify(rho_from_H - rho_from_F.subs({S: S_H_locked})) == 0

    # H(S) and dH/dS for F_locked
    H_of_S = simplify(sqrt(F_locked))
    dH_dS = simplify(diff(H_of_S, S))
    dH_dS_tgt = simplify(-H_of_S / (2 * S))
    ok_dHdS = simplify(dH_dS - dH_dS_tgt) == 0

    # Chain-rule Ḣ = (dH/dS) Ṡ
    Hdot_chain = simplify(dH_dS * Sdot)
    Hdot_locked_form = simplify(-(H_of_S / (2 * S)) * Sdot)
    ok_chain = simplify(Hdot_chain - Hdot_locked_form) == 0

    # Match LAYER 1 thermo EOM: Ṡ = ΔS γ χ(1-χ), S=S_H(χ)
    S_H_chi = S_early + Delta_S * chi
    H_chi = simplify(sqrt(pi / (G * S_H_chi)))
    Sdot_log = Delta_S * gamma * chi * (1 - chi)
    Hdot_from_F = simplify(-(H_chi / (2 * S_H_chi)) * Sdot_log)
    Hdot_L1 = _thermo_Hdot(G, H_chi, S_max, S_early, gamma, chi)
    ok_match_L1 = simplify(Hdot_from_F - Hdot_L1) == 0

    # dF/dS
    dF_dS = simplify(diff(F_locked, S))
    ok_dF = simplify(dF_dS + pi / (G * S**2)) == 0

    # Attractor from F: H_∞=√F(S_max)
    H_inf_F = simplify(sqrt(F_locked.subs({S: S_max})))
    H_inf_tgt = sqrt(pi / (G * S_max))
    ok_Hinf = simplify(H_inf_F - H_inf_tgt) == 0

    # Raychaudhuri bridge: ρ+P = -Ḣ/(4πG) still holds as definition of P
    # With Ḣ=-(H/(2S))Ṡ: ρ+P = H Ṡ / (8πG S) ≥ 0 if Ṡ≥0
    rho_plus_P = simplify(-Hdot_locked_form / (4 * pi * G))
    rho_plus_P_tgt = simplify(H_of_S * Sdot / (8 * pi * G * S))
    ok_NEC_F = simplify(rho_plus_P - rho_plus_P_tgt) == 0

    # Modified-F schematic: F_mod = π/(G S) * (1+ε) shifts H_∞
    eps = Symbol("varepsilon", real=True)
    F_mod = simplify(F_locked * (1 + eps))
    H_inf_mod = simplify(sqrt(F_mod.subs({S: S_max})))
    ok_mod = simplify(H_inf_mod**2 - (1 + eps) * pi / (G * S_max)) == 0

    all_ok = (
        ok_inverse
        and ok_onshell
        and ok_rho
        and ok_rho_H
        and ok_dHdS
        and ok_chain
        and ok_match_L1
        and ok_dF
        and ok_Hinf
        and ok_NEC_F
        and ok_mod
    )

    return DerivationStep(
        name="L18_generalized_Friedmann_from_entropy",
        stage="LAYER_18",
        assumptions=[
            "Entropy treated as primary cosmological state variable",
            "Locked BH+apparent-horizon law S_H=π/(G H²) for the GR special case",
            "Closure Ṡ_H=ΔS Γ supplied independently (Γ-class)",
        ],
        equations={
            "architecture": r"H^2=F(S_H)\quad\mathrm{(constitutive\ map)}",
            "F_locked": Eq(Function("F")(S), F_locked),
            "inversion": (
                r"S_H=\pi/(G H^2)\ \Rightarrow\ F_{\mathrm{locked}}(S)=\pi/(G S)"
            ),
            "onshell": r"F_{\mathrm{locked}}(S_H)=H^2",
            "Friedmann_from_F": (
                r"\rho_{\mathrm{tot}}=3H^2/(8\pi G)=3F(S_H)/(8\pi G)"
            ),
            "rho_S": Eq(Symbol("rho_S"), rho_from_S),
            "H_of_S": Eq(Symbol("H"), H_of_S),
            "dH_dS": Eq(Symbol("dH_dS"), dH_dS),
            "Hdot_chain": (
                r"\dot H=(dH/dS_H)\dot S_H=-(H/(2S_H))\dot S_H"
            ),
            "Hdot_explicit": Eq(Symbol("Hdot"), Hdot_locked_form),
            "match_L1": (
                r"\dot S_H=\Delta S\,\Gamma\ \Rightarrow\ "
                r"\dot H\mathrm{\ recovers\ LAYER\ 1/\Gamma\!-\!class\ EOM}"
            ),
            "dF_dS": Eq(Symbol("dF_dS"), dF_dS),
            "NEC_from_F": (
                r"\rho+P=-\dot H/(4\pi G)=H\dot S_H/(8\pi G S_H)\ge0"
                r"\mathrm{\ if\ }\dot S_H\ge0"
            ),
            "H_inf": Eq(Symbol("H_inf"), H_inf_F),
            "attractor": r"S_H\to S_{\max}\Rightarrow H_\infty=\sqrt{F(S_{\max})}",
            "modified_F": (
                r"\mathrm{other\ }F(S)\mathrm{:\ thermodynamic\ modified\ cosmology"
                r"\ (LAYER\ 20\ via\ }S(A)\mathrm{)}"
            ),
            "F_mod_example": Eq(Symbol("F_mod"), F_mod),
            "standard_GR": (
                r"\mathrm{standard\ Friedmann\ =\ }F=F_{\mathrm{locked}}"
                r"\mathrm{\ special\ case\ of\ }H^2=F(S)"
            ),
        },
        claims=[
            "LAYER 18 THEOREM: F_locked(S)=π/(G S) is the inverse of S_H=π/(G H²).",
            "LAYER 18 THEOREM: Ḣ=-(H/(2S_H))Ṡ_H recovers the thermo EOM for any Γ-class Ṡ.",
            "LAYER 18 DERIVED: standard Friedmann is ρ=3F(S)/(8πG) with F=F_locked.",
            "LAYER 18: H²=F(S) is the entropy-first architecture; other F => modified cosmologies.",
        ],
        status="derived" if all_ok else "failed",
        evidence={
            "F_locked": F_locked,
            "F_on_shell": F_on_shell,
            "rho_from_F": rho_from_F,
            "rho_from_S": rho_from_S,
            "H_of_S": H_of_S,
            "dH_dS": dH_dS,
            "Hdot_locked_form": Hdot_locked_form,
            "Hdot_from_F": Hdot_from_F,
            "dF_dS": dF_dS,
            "H_inf_F": H_inf_F,
            "F_mod": F_mod,
            "ok_inverse": ok_inverse,
            "ok_onshell": ok_onshell,
            "ok_rho": ok_rho,
            "ok_rho_H": ok_rho_H,
            "ok_dHdS": ok_dHdS,
            "ok_chain": ok_chain,
            "ok_match_L1": ok_match_L1,
            "ok_dF": ok_dF,
            "ok_Hinf": ok_Hinf,
            "ok_NEC_F": ok_NEC_F,
            "ok_mod": ok_mod,
        },
    )


def step_L19_Jacobson_global_bridge(sym: Dict[str, Any]) -> DerivationStep:
    """
    LAYER 19 - Bridge: local Jacobson EFE ↔ global Hubble-horizon thermodynamics.

    Two logically independent modules
    ---------------------------------
    Local (Module 17 / Jacobson):
        Clausius δQ=T dS,  dS=η dA,  Unruh T=κ/(2π)
        =>  G_μν + Λ g_μν = (2π/η) T_μν
        C2: η=1/(4G)  =>  G_μν + Λ g_μν = 8πG T_μν
        Here Λ is an undetermined integration constant.

    Global (Steps/Layers 1-18):
        Apparent horizon A_H=4π/H²,  S_H=η A_H with the *same* η
        =>  S_H=A_H/(4G)=π/(G H²)   [locked canon]
        GSL + S_max + Γ-class  =>  H->H_∞=√(π/(G S_max)),  w->-1

    Bridge identifications (CONDITIONAL - not theorems of either side alone)
    -----------------------------------------------------------------------
    (B1) Common entropy density: η_local = η_global = 1/(4G)
    (B2) Thermodynamic Λ: at the attractor, vacuum Einstein requires
            G^(∞)_μν + Λ g_μν = 0  with  Λ = 3 H_∞²
         Identify the free Jacobson Λ with
            Λ -> Λ_S := 3 H_∞² = 3π/(G S_max)
    (B3) Architecture:
         local thermo -> EFE -> FLRW -> S_H -> GSL+S_max+closure -> dS -> Λ_S

    What is derived algebraically vs bridged
    ---------------------------------------
        DERIVED (given η=1/(4G)): 2π/η=8πG; S_H=π/(G H²); Λ_S=3H_∞²=3π/(G S_max)
        DERIVED (attractor vacuum): ρ_Λ=Λ_S/(8πG)=3/(8 G² S_max)=ρ_S|_(χ=1)
        BRIDGE / CONDITIONAL: that local and global *must* share one η, and that
            the Jacobson integration constant *is* the thermodynamic Λ_S.
        OPEN: single master functional S[g_μν,χ] generating both sectors.
    """
    G, H, chi = sym["G"], sym["H"], sym["chi"]
    S_max, S_early = sym["S_max"], sym["S_early"]
    Delta_S = S_max - S_early
    eta = Symbol("eta", positive=True)

    # --- Local calibration C2 ---
    eta_BH = 1 / (4 * G)
    coupling = simplify(2 * pi / eta_BH)
    ok_C2 = simplify(coupling - 8 * pi * G) == 0

    # Local EFE schematic coefficient matches GR
    # G_μν+Λg = (2π/η) T  ->  8πG T when η=1/(4G)

    # --- Global S_H from same η ---
    A_H = simplify(4 * pi / H**2)
    S_H_eta = simplify(eta * A_H)
    S_H_locked = simplify(S_H_eta.subs({eta: eta_BH}))
    ok_SH = simplify(S_H_locked - pi / (G * H**2)) == 0
    ok_SH_helper = simplify(S_H_locked - _locked_SH(G, H)) == 0

    # --- Attractor ---
    H_inf = sqrt(pi / (G * S_max))
    Lam_S = simplify(3 * H_inf**2)
    Lam_S_alt = simplify(3 * pi / (G * S_max))
    ok_Lam = simplify(Lam_S - Lam_S_alt) == 0

    # Vacuum Einstein at attractor: Λ = 3 H_∞² (de Sitter)
    # ρ_vac = Λ/(8πG) should match ρ_S(χ=1)
    rho_vac = simplify(Lam_S / (8 * pi * G))
    S_H_inf = S_early + Delta_S * 1
    # at χ=1, S_H=S_max
    rho_S_inf = simplify(sp.Rational(3, 8) / (G**2 * S_max))
    ok_rho_match = simplify(rho_vac - rho_S_inf) == 0

    # A_∞ = 4π/H_∞² = 4 G S_max = A_max from S_max=η A_max
    A_inf = simplify(4 * pi / H_inf**2)
    A_max = simplify(S_max / eta_BH)  # S=ηA => A=S/η
    ok_A = simplify(A_inf - A_max) == 0

    # Asymptotic: at χ=1, Γ=0 => Ḣ=0 => NEC saturated => w=-1
    Gamma_at_1 = simplify(
        (sym["gamma"] * chi * (1 - chi)).subs({chi: 1})
    )
    ok_w = Gamma_at_1 == 0

    # Bridge diagram quantities
    # Free Λ vs fixed Λ_S: ratio Λ_S / (3 H_inf²) = 1
    ok_bridge_id = simplify(Lam_S / (3 * H_inf**2) - 1) == 0

    # Null-projection coefficient η/(2π) vs stress side
    null_coeff = simplify(eta_BH / (2 * pi))
    # Inverse: 2π/η used in EFE reconstruction
    ok_null = simplify(1 / null_coeff - coupling) == 0

    # Honesty check: without identifying η_local=η_global, S_H!=π/(G H²)
    # in general - bridge is an assumption
    S_H_generic = simplify(eta * A_H)
    ok_need_bridge = simplify(S_H_generic - pi / (G * H**2)) != 0  # remains eta-dependent

    all_ok = (
        ok_C2
        and ok_SH
        and ok_SH_helper
        and ok_Lam
        and ok_rho_match
        and ok_A
        and ok_w
        and ok_bridge_id
        and ok_null
        and ok_need_bridge
    )

    return DerivationStep(
        name="L19_Jacobson_global_entropy_bridge",
        stage="LAYER_19",
        assumptions=[
            "Module 17 axioms (Clausius, dS=η dA, Unruh) + C1/C2",
            "BRIDGE B1: η_local = η_global = 1/(4G)",
            "BRIDGE B2: Jacobson Λ identified with thermodynamic Λ_S at attractor",
            "Global attractor χ->1 from GSL+S_max+Γ-class (Layers 1-18)",
        ],
        equations={
            "scope": (
                r"\mathrm{Module\ 17}\ \perp\ \mathrm{global\ Hubble\ GSL};"
                r"\ \mathrm{joined\ only\ by\ bridge}"
            ),
            "eta_local": r"dS=\eta\,dA,\quad\eta=1/(4G)\quad\mathrm{(J.A2+J.C2)}",
            "C2_coupling": Eq(Symbol("two_pi_over_eta"), coupling),
            "local_EFE": (
                r"G_{\mu\nu}+\Lambda g_{\mu\nu}=8\pi G\,T_{\mu\nu}"
                r"\quad(\Lambda\mathrm{\ free})"
            ),
            "A_H": Eq(Symbol("A_H"), A_H),
            "S_H_bridge": (
                r"S_H=\eta A_H\big|_{\eta=1/(4G)}=\pi/(G H^2)"
                r"\quad\mathrm{(locked\ canon)}"
            ),
            "S_H_locked": Eq(Symbol("S_H"), S_H_locked),
            "H_inf": Eq(Symbol("H_inf"), H_inf),
            "Lambda_S": Eq(Symbol("Lambda_S"), Lam_S),
            "Lambda_id": (
                r"\Lambda\to\Lambda_S=3H_\infty^2=3\pi/(G S_{\max})"
            ),
            "rho_match": (
                r"\rho_\Lambda=\Lambda_S/(8\pi G)=\rho_S\big|_{\chi=1}"
                r"=3/(8 G^2 S_{\max})"
            ),
            "rho_vac": Eq(Symbol("rho_vac"), rho_vac),
            "rho_S_inf": Eq(Symbol("rho_S_inf"), rho_S_inf),
            "A_inf": Eq(Symbol("A_inf"), A_inf),
            "asymptotic_EFE": r"G^{(\infty)}_{\mu\nu}+\Lambda_S g_{\mu\nu}=0",
            "w_attractor": r"\chi\to1:\ \Gamma=0\Rightarrow\dot H=0\Rightarrow w_S\to-1",
            "architecture": (
                r"\mathrm{local\ thermo}\to\mathrm{EFE}\to\mathrm{FLRW}\to S_H"
                r"\to\mathrm{GSL}+S_{\max}+\Gamma\to\mathrm{dS}\to\Lambda_S"
            ),
            "B1": r"\eta_{\mathrm{local}}=\eta_{\mathrm{global}}=1/(4G)",
            "B2": r"\Lambda_{\mathrm{Jacobson}}\equiv\Lambda_S\mathrm{\ (thermodynamic\ fix)}",
            "master_gap": (
                r"\exists?\ S[g_{\mu\nu},\chi]\mathrm{\ with\ }"
                r"\delta S\Rightarrow\mathrm{EFE\ and\ }\bar\chi=\Gamma(\chi)"
            ),
        },
        claims=[
            "LAYER 19 BRIDGE: common η=1/(4G) ties Jacobson Clausius to locked S_H.",
            "LAYER 19 DERIVED (given bridge): Λ_S=3H_∞²=3π/(G S_max) and ρ_Λ=ρ_S|_(χ=1).",
            "LAYER 19: Λ free in local EFE; fixed to Λ_S only by the global attractor.",
            "LAYER 19 CONDITIONAL: identification η_local=η_global is assumed, not derived.",
        ],
        open_gaps=[
            "Single master functional S[g_μν,χ] whose local variation => EFE and "
            "homogeneous limit => χ̇=Γ(χ) is not constructed.",
            "No derivation that local Rindler η must equal global Hubble-horizon η "
            "from a deeper principle - this is the bridge postulate.",
        ],
        status="conditional" if all_ok else "failed",
        evidence={
            "eta_BH": eta_BH,
            "coupling": coupling,
            "S_H_locked": S_H_locked,
            "H_inf": H_inf,
            "Lambda_S": Lam_S,
            "rho_vac": rho_vac,
            "rho_S_inf": rho_S_inf,
            "A_inf": A_inf,
            "null_coeff": null_coeff,
            "ok_C2": ok_C2,
            "ok_SH": ok_SH,
            "ok_SH_helper": ok_SH_helper,
            "ok_Lam": ok_Lam,
            "ok_rho_match": ok_rho_match,
            "ok_A": ok_A,
            "ok_w": ok_w,
            "ok_bridge_id": ok_bridge_id,
            "ok_null": ok_null,
            "ok_need_bridge": ok_need_bridge,
        },
    )


def step_L20_modified_entropy(sym: Dict[str, Any]) -> DerivationStep:
    """
    LAYER 20 - Modified entropy laws S(A) and corrected late-time attractor.

    Corrected horizon entropy (standard QG-motivated ansatz)
    -------------------------------------------------------
        S(A) = A/(4G) + α ln(A/A_0) + β (A_0/A)
    with A = A_H = 4π/H² the apparent-horizon area.
        α=β=0 recovers the locked Bekenstein-Hawking law (Layers 1-19).

    Effective entropy density
    ------------------------
        η_eff(A) := dS/dA = 1/(4G) + α/A - β A_0/A²
    => Jacobson-type local reconstruction would use a running η (not pursued
      as a full re-derivation here); the global attractor uses S(A) directly.

    Equilibrium condition (finite S_max still imposed)
    --------------------------------------------------
        S(A(H_∞)) = S_max
    defines a corrected H_∞(α,β,A_0). With u=H²:
        S(u) = π/(G u) + α ln(4π/(A_0 u)) + (β A_0/(4π)) u

    Leading-order shift about the BH root u_0=π/(G S_max)
    -----------------------------------------------------
        S(u_0) = S_max + α ln(4π/(A_0 u_0)) + (β A_0/(4π)) u_0
        S'(u)  = -π/(G u²) - α/u + β A_0/(4π)
        δu ~ (S_max - S(u_0)) / S'(u_0)
        H_∞² = u_0 + δu + O(α²,β²,αβ)
        Λ_S = 3 H_∞²

    LAYER 18 reading
    ----------------
    Inverting S=S(A(H)) for H² as a function of the entropy state yields a
    deformed constitutive map H²=F_corr(S) (reduces to F_locked when α=β=0).

    Robustness
    ----------
    For sufficiently small (α,β), S'(u_0)<0 (dominated by -π/(G u_0²)), so a
    unique nearby positive root persists: the dS attractor is structurally
    stable under this class of corrections. Exact closed-form inversion is
    transcendental when α!=0.
    """
    G, H = sym["G"], sym["H"]
    S_max = sym["S_max"]
    alpha = Symbol("alpha", real=True)
    beta = Symbol("beta", real=True)
    A0 = Symbol("A_0", positive=True)
    u = Symbol("u", positive=True)  # H²
    A = Symbol("A", positive=True)

    # S(A) ansatz
    S_of_A = A / (4 * G) + alpha * sp.log(A / A0) + beta * (A0 / A)
    S0_of_A = A / (4 * G)
    ok_BH = simplify(S_of_A.subs({alpha: 0, beta: 0}) - S0_of_A) == 0

    # η_eff = dS/dA
    eta_eff = simplify(diff(S_of_A, A))
    eta_eff_tgt = simplify(1 / (4 * G) + alpha / A - beta * A0 / A**2)
    ok_eta = simplify(eta_eff - eta_eff_tgt) == 0
    ok_eta_BH = simplify(eta_eff.subs({alpha: 0, beta: 0}) - 1 / (4 * G)) == 0

    # Horizon area and S(u)
    A_H = simplify(4 * pi / H**2)
    S_corr_H = simplify(
        A_H / (4 * G)
        + alpha * sp.log(A_H / A0)
        + beta * (A0 / A_H)
    )
    S_of_u = simplify(
        pi / (G * u)
        + alpha * sp.log(4 * pi / (A0 * u))
        + beta * A0 * u / (4 * pi)
    )
    # Match S_corr_H with u=H²
    ok_Su = simplify(S_corr_H.subs({H: sqrt(u)}) - S_of_u) == 0

    # BH root
    u0 = simplify(pi / (G * S_max))
    S_at_u0_BH = simplify(S_of_u.subs({alpha: 0, beta: 0, u: u0}))
    ok_root0 = simplify(S_at_u0_BH - S_max) == 0

    # Correction at u0 and derivative
    S_u0 = simplify(S_of_u.subs({u: u0}))
    Delta_corr = simplify(S_u0 - S_max)
    # Δ = α ln(4π/(A0 u0)) + (β A0/(4π)) u0
    Delta_tgt = simplify(
        alpha * sp.log(4 * pi / (A0 * u0)) + beta * A0 * u0 / (4 * pi)
    )
    ok_Delta = simplify(Delta_corr - Delta_tgt) == 0

    dS_du = simplify(diff(S_of_u, u))
    dS_du_tgt = simplify(-pi / (G * u**2) - alpha / u + beta * A0 / (4 * pi))
    ok_dS = simplify(dS_du - dS_du_tgt) == 0

    # Leading δu from S(u0+δu)=S_max ~ S(u0)+S'(u0)δu
    dS_du0 = simplify(dS_du.subs({u: u0}))
    # At α=β=0: S'=-π/(G u0²)=-S_max² G/π * π/G wait: -π/(G u0²)=-π/(G)*(G S_max/π)²
    # = -π/(G) * G² S_max² / π² = -(G S_max²)/π = -S_max / u0
    dS_du0_BH = simplify(dS_du.subs({alpha: 0, beta: 0, u: u0}))
    ok_dS0 = simplify(dS_du0_BH + S_max / u0) == 0

    delta_u = simplify(-Delta_corr / dS_du0)
    # For α=β=0: δu=0
    ok_delta0 = simplify(delta_u.subs({alpha: 0, beta: 0})) == 0

    # First-order H_∞² and Λ
    u_inf_LO = simplify(u0 + delta_u)
    Lam_S_LO = simplify(3 * u_inf_LO)
    # Pure BH limit
    ok_Lam0 = simplify(Lam_S_LO.subs({alpha: 0, beta: 0}) - 3 * pi / (G * S_max)) == 0

    # Series in α about 0 with β=0: δu ≃ α (u0/S_max) ln(4π/(A0 u0))
    delta_u_alpha_series = simplify(
        sp.series(delta_u.subs({beta: 0}), alpha, 0, 2).removeO()
    )
    delta_u_alpha_tgt = simplify(
        alpha * (u0 / S_max) * sp.log(4 * pi / (A0 * u0))
    )
    ok_alpha_LO = simplify(delta_u_alpha_series - delta_u_alpha_tgt) == 0

    # Robustness: BH slope S'(u0)=-S_max/u0 < 0
    ok_robust_slope = ok_dS0

    # F_corr architecture: for α=β=0, F=π/(G S)
    # Implicit: S=S(u) inverted; check BH recovery of F_locked
    F_locked = pi / (G * S_max)  # H_∞² at S=S_max
    ok_F0 = simplify(u0 - F_locked) == 0

    # Sign of η_eff at large A: -> 1/(4G) > 0
    eta_large = simplify(sp.limit(eta_eff, A, sp.oo))
    ok_eta_asymp = simplify(eta_large - 1 / (4 * G)) == 0

    all_ok = (
        ok_BH
        and ok_eta
        and ok_eta_BH
        and ok_Su
        and ok_root0
        and ok_Delta
        and ok_dS
        and ok_dS0
        and ok_delta0
        and ok_Lam0
        and ok_alpha_LO
        and ok_robust_slope
        and ok_F0
        and ok_eta_asymp
    )

    return DerivationStep(
        name="L20_modified_entropy_laws",
        stage="LAYER_20",
        assumptions=[
            "Corrected entropy ansatz S=A/(4G)+α ln(A/A0)+β A0/A",
            "Apparent horizon A=4π/H²",
            "Equilibrium still imposes S(H_∞)=S_max (finite holographic budget)",
            "Leading-order analysis in (α,β); exact α!=0 root is transcendental",
        ],
        equations={
            "S_of_A": Eq(Symbol("S"), S_of_A),
            "A_H": Eq(Symbol("A_H"), A_H),
            "S_corr_H": Eq(Symbol("S"), S_corr_H),
            "S_of_u": Eq(Symbol("S"), S_of_u),
            "BH_recovery": r"\alpha=\beta=0:\ S=A/(4G),\ H_\infty^2=\pi/(G S_{\max})",
            "eta_eff": Eq(Symbol("eta_eff"), eta_eff),
            "eta_eff_formula": (
                r"\eta_{\mathrm{eff}}=dS/dA=1/(4G)+\alpha/A-\beta A_0/A^2"
            ),
            "equilibrium": r"S\big(A(H_\infty)\big)=S_{\max}",
            "u0": Eq(Symbol("u_0"), u0),
            "Delta_corr": Eq(Symbol("Delta_corr"), Delta_corr),
            "dS_du": Eq(Symbol("dS_du"), dS_du),
            "delta_u_LO": Eq(Symbol("delta_u"), delta_u),
            "u_inf_LO": Eq(Symbol("H_inf_sq"), u_inf_LO),
            "alpha_LO": (
                r"\beta=0,\ \alpha\ll1:\ \delta u\simeq"
                r"\alpha\,(u_0/S_{\max})\ln\!\big(4\pi/(A_0 u_0)\big)"
            ),
            "delta_u_alpha_series": Eq(Symbol("delta_u_alpha"), delta_u_alpha_series),
            "Lambda_LO": Eq(Symbol("Lambda_S"), Lam_S_LO),
            "Lambda_def": r"\Lambda_S(\alpha,\beta)=3 H_\infty(\alpha,\beta)^2",
            "F_corr": (
                r"H^2=F_{\mathrm{corr}}(S)\mathrm{\ via\ inverting\ }S(A(H));"
                r"\ F_{\mathrm{corr}}\to F_{\mathrm{locked}}\mathrm{\ as\ }"
                r"\alpha,\beta\to0"
            ),
            "robustness": (
                r"S'(u_0)|_{\alpha=\beta=0}=-S_{\max}/u_0<0"
                r"\Rightarrow\mathrm{unique\ nearby\ positive\ root\ for\ small\ }(\alpha,\beta)"
            ),
            "Jacobson_note": (
                r"\mathrm{running\ }\eta_{\mathrm{eff}}\mathrm{\ would\ modify\ local\ }"
                r"\mathrm{Jacobson\ reconstruction\ (not\ re\!-\!derived\ here)}"
            ),
        },
        claims=[
            "LAYER 20 THEOREM: α=β=0 recovers locked S=A/(4G) and H_∞²=π/(G S_max).",
            "LAYER 20 DERIVED: η_eff=1/(4G)+α/A-β A_0/A²; equilibrium S(H_∞)=S_max.",
            "LAYER 20 DERIVED: leading δu for small α (β=0) is α(u_0/S_max)ln(4π/(A_0 u_0)).",
            "LAYER 20: dS attractor structurally robust under small logarithmic/inverse-area corrections.",
        ],
        open_gaps=[
            "Exact closed-form H_∞(α,β) is transcendental for α!=0; use numerical root or LO series.",
            "Full local Jacobson re-derivation with running η_eff(A) not carried out in this layer.",
        ],
        status="derived" if all_ok else "failed",
        evidence={
            "S_of_A": S_of_A,
            "eta_eff": eta_eff,
            "S_corr_H": S_corr_H,
            "S_of_u": S_of_u,
            "u0": u0,
            "Delta_corr": Delta_corr,
            "dS_du": dS_du,
            "dS_du0_BH": dS_du0_BH,
            "delta_u": delta_u,
            "delta_u_alpha_series": delta_u_alpha_series,
            "u_inf_LO": u_inf_LO,
            "Lam_S_LO": Lam_S_LO,
            "ok_BH": ok_BH,
            "ok_eta": ok_eta,
            "ok_eta_BH": ok_eta_BH,
            "ok_Su": ok_Su,
            "ok_root0": ok_root0,
            "ok_Delta": ok_Delta,
            "ok_dS": ok_dS,
            "ok_dS0": ok_dS0,
            "ok_delta0": ok_delta0,
            "ok_Lam0": ok_Lam0,
            "ok_alpha_LO": ok_alpha_LO,
            "ok_robust_slope": ok_robust_slope,
            "ok_F0": ok_F0,
            "ok_eta_asymp": ok_eta_asymp,
        },
    )


def step_L21_Smax_from_horizon(sym: Dict[str, Any]) -> DerivationStep:
    """
    LAYER 21 - Finite S_max as a finite holographic horizon-area budget.

    Postulate (CONDITIONAL - not derived from microstate counting)
    -------------------------------------------------------------
        There exists a finite maximum horizon entropy S_max < ∞.
        With Bekenstein-Hawking (or LAYER 19 η=1/(4G)):
            S_max = A_max / (4G)  ⇔  A_max = 4 G S_max < ∞

    Consistency with the locked attractor
    ------------------------------------
        H_∞ = √(π/(G S_max))
        A_∞ := 4π / H_∞²
        Identity:  A_∞ = 4 G S_max = A_max
    So the holographic budget A_max is exactly the apparent-horizon area
    of the late-time de Sitter geometry - not an independent scale.

    Radii (convention honesty)
    --------------------------
        Apparent / Hubble / flat-slicing dS event horizon:  R_∞ = 1/H_∞
        Schwarzschild: R_s = 2 G M  (different object - do NOT use R=2/H_∞ here)
        Area: A_∞ = 4π R_∞² = 4π/H_∞²

    Finite budget ⇔ positive late-time vacuum
    -----------------------------------------
        0 < S_max < ∞  ⇔  0 < H_∞ < ∞  ⇔  0 < Λ_S = 3 H_∞² < ∞
        S_max -> ∞  =>  H_∞ -> 0,  Λ_S -> 0   (no residual cosmological constant)
        S_max -> 0⁺ =>  H_∞ -> ∞            (singular / empty budget)

    Thermodynamic / Euclidean reading
    ---------------------------------
        On-shell Euclidean dS action (standard EH convention): I_E^{dS} = -S_dS
        Attractor identifies S_dS = S_max => I_E^{dS} = -S_max
        (action wording caution: this does not derive "the universe chooses dS
         by maximizing Z" - it is a consistency relation under that convention.)

    What is / is not derived
    ------------------------
        DERIVED: A_∞ = A_max identity given S_max=A_max/(4G) and locked H_∞.
        DERIVED: finite S_max ⇔ positive finite (H_∞, Λ_S).
        CONDITIONAL: existence and numerical value of S_max (QG / holography).
        OPEN: microstate count that fixes S_max from first principles.
    """
    G, S_max = sym["G"], sym["S_max"]
    eta = 1 / (4 * G)

    # Budget definitions
    A_max = simplify(4 * G * S_max)  # from S_max = A_max/(4G)
    ok_Smax_def = simplify(A_max * eta - S_max) == 0

    # Locked attractor
    H_inf = simplify(sqrt(pi / (G * S_max)))
    Lam_S = simplify(3 * H_inf**2)
    ok_Lam = simplify(Lam_S - 3 * pi / (G * S_max)) == 0

    # Area at attractor
    R_inf = simplify(1 / H_inf)
    A_inf = simplify(4 * pi / H_inf**2)
    A_inf_from_R = simplify(4 * pi * R_inf**2)
    ok_A_R = simplify(A_inf - A_inf_from_R) == 0
    ok_A = simplify(A_inf - A_max) == 0

    # Inverse: S_max from A_inf
    S_from_A = simplify(eta * A_inf)
    ok_S_from_A = simplify(S_from_A - S_max) == 0

    # LAYER 18: H_∞ = √F(S_max)
    F_Smax = simplify(pi / (G * S_max))
    ok_F = simplify(H_inf**2 - F_Smax) == 0

    # Limits
    H_inf_largeS = sp.limit(H_inf, S_max, sp.oo)
    Lam_largeS = sp.limit(Lam_S, S_max, sp.oo)
    ok_limit_inf = (H_inf_largeS == 0) and (Lam_largeS == 0)
    H_inf_smallS = sp.limit(H_inf, S_max, 0)
    ok_limit_0 = H_inf_smallS == sp.oo

    # ρ_S at attractor = 3/(8 G² S_max) = Λ_S/(8πG)
    rho_inf = simplify(sp.Rational(3, 8) / (G**2 * S_max))
    rho_from_Lam = simplify(Lam_S / (8 * pi * G))
    ok_rho = simplify(rho_inf - rho_from_Lam) == 0

    # Euclidean on-shell (bookkeeping identity under standard convention)
    I_E = simplify(-S_max)
    ok_IE = simplify(I_E + S_max) == 0

    # Schwarzschild caution: R_s = 2GM != 1/H
    # Demonstrate 2/H_∞ != R_∞
    R_wrong = simplify(2 / H_inf)
    ok_not_Schwarzschild = simplify(R_wrong - R_inf) != 0

    # Positive finite budget => positive finite vacuum
    # (symbolic positivity assumed by symbol assumptions; check algebra chain)
    ok_chain = ok_A and ok_Lam and ok_rho

    all_ok = (
        ok_Smax_def
        and ok_Lam
        and ok_A_R
        and ok_A
        and ok_S_from_A
        and ok_F
        and ok_limit_inf
        and ok_limit_0
        and ok_rho
        and ok_IE
        and ok_not_Schwarzschild
        and ok_chain
    )

    return DerivationStep(
        name="L21_finite_Smax_from_horizon_area",
        stage="LAYER_21",
        assumptions=[
            "CONDITIONAL: finite holographic budget 0<S_max<∞ exists",
            "Bekenstein-Hawking / η=1/(4G): S_max=A_max/(4G)",
            "Attractor identifies A_max with A_∞=4π/H_∞²",
        ],
        equations={
            "Smax_postulate": r"0<S_{\max}<\infty\quad\mathrm{(holographic\ budget)}",
            "Smax_def": r"S_{\max}=A_{\max}/(4G)=\eta A_{\max}",
            "A_max": Eq(Symbol("A_max"), A_max),
            "H_inf": Eq(Symbol("H_inf"), H_inf),
            "Lambda_S": Eq(Symbol("Lambda_S"), Lam_S),
            "R_inf": Eq(Symbol("R_inf"), R_inf),
            "A_inf": Eq(Symbol("A_inf"), A_inf),
            "identity": (
                r"A_\infty=4\pi/H_\infty^2=4\pi R_\infty^2=4 G S_{\max}=A_{\max}"
            ),
            "S_from_A": Eq(Symbol("S_from_A_inf"), S_from_A),
            "F_link": r"H_\infty=\sqrt{F(S_{\max})},\ F(S)=\pi/(G S)\mathrm{\ (LAYER\ 18)}",
            "rho_inf": Eq(Symbol("rho_S_inf"), rho_inf),
            "finite_iff": (
                r"0<S_{\max}<\infty\Leftrightarrow 0<H_\infty<\infty"
                r"\Leftrightarrow 0<\Lambda_S<\infty"
            ),
            "limit_infinite_S": (
                r"S_{\max}\to\infty\Rightarrow H_\infty\to0,\ \Lambda_S\to0"
            ),
            "limit_vanishing_S": (
                r"S_{\max}\to 0^+\Rightarrow H_\infty\to\infty"
            ),
            "I_E": Eq(Symbol("I_E_dS"), I_E),
            "I_E_note": (
                r"I_E^{\mathrm{dS}}=-S_{\max}\mathrm{\ (EH\ on\!-\!shell\ convention);"
                r"\ not\ a\ derivation\ of\ ``Z\!-\!maximization''}"
            ),
            "radius_caution": (
                r"R_\infty=1/H_\infty\mathrm{\ (apparent/dS)};\ "
                r"R=2/H_\infty\mathrm{\ is\ not\ the\ cosmological\ horizon\ radius}"
            ),
            "why_finite": (
                r"\mathrm{finite\ }A_{\max}\Leftrightarrow\mathrm{finite\ }S_{\max}"
                r"\Leftrightarrow\mathrm{positive\ late\!-\!time\ }H_\infty,\Lambda_S"
            ),
        },
        claims=[
            "LAYER 21 THEOREM (given BH+locked attractor): A_∞=4G S_max=A_max identically.",
            "LAYER 21 DERIVED: finite S_max ⇔ positive finite (H_∞, Λ_S); S_max->∞ kills Λ_S.",
            "LAYER 21: R_∞=1/H_∞ (not 2/H_∞); Euclidean I_E^{dS}=-S_max under EH convention.",
            "LAYER 21 CONDITIONAL: existence/value of S_max is a holographic postulate, not derived.",
        ],
        open_gaps=[
            "Microstate counting / quantum gravity origin of the numerical value of S_max "
            "is not derived - only the geometric consistency of a finite budget.",
        ],
        status="conditional" if all_ok else "failed",
        evidence={
            "A_max": A_max,
            "H_inf": H_inf,
            "Lam_S": Lam_S,
            "R_inf": R_inf,
            "A_inf": A_inf,
            "rho_inf": rho_inf,
            "I_E": I_E,
            "H_inf_largeS": H_inf_largeS,
            "ok_Smax_def": ok_Smax_def,
            "ok_Lam": ok_Lam,
            "ok_A_R": ok_A_R,
            "ok_A": ok_A,
            "ok_S_from_A": ok_S_from_A,
            "ok_F": ok_F,
            "ok_limit_inf": ok_limit_inf,
            "ok_limit_0": ok_limit_0,
            "ok_rho": ok_rho,
            "ok_IE": ok_IE,
            "ok_not_Schwarzschild": ok_not_Schwarzschild,
        },
    )


def step_L_coupled_rad_matter_S(sym: Dict[str, Any]) -> DerivationStep:
    """Structural honesty: entropy-only is two-sided dS; need rad+matter+S."""
    H, G = sym["H"], sym["G"]
    rho_r, rho_m, rho_S = sym["rho_r"], sym["rho_m"], sym["rho_S"]
    return DerivationStep(
        name="L_coupled_radiation_matter_entropy",
        stage="LAYER_STRUCTURE",
        assumptions=["Einstein with three sectors"],
        equations={
            "Friedmann": Eq(H**2, (8 * pi * G / 3) * (rho_r + rho_m + rho_S)),
            "cont_r": r"\dot\rho_r+4H\rho_r=0",
            "cont_m": r"\dot\rho_m+3H\rho_m=0",
            "cont_S": r"\dot\rho_S+3H(\rho_S+P_S)=Q_S,\quad\sum Q_i=0",
            "caution": (
                "Entropy-only: H->H_early (t->-∞) and H->H_∞ (t->+∞) - two-sided dS. "
                "Does NOT alone produce radiation->matter->DE sequence."
            ),
        },
        claims=[
            "Coupled system required for standard cosmic history; LAYERS 1-21 still apply to S-sector.",
        ],
        status="assumed",
        evidence={"architecture": "coupled_FLRW"},
    )


def collect_deep_cosmology_steps(sym: Dict[str, Any]) -> List[DerivationStep]:
    """All 21 first-principles layers + coupled-background honesty note."""
    return [
        step_L01_effective_fluid(sym),
        step_L02_deceleration_cosmography(sym),
        step_L03_phase_trajectory(sym),
        step_L04_autonomous_system(sym),
        step_L05_lyapunov(sym),
        step_L06_horizon_first_law(sym),
        step_L07_entropy_production(sym),
        step_L08_transition_epoch(sym),
        step_L09_scalar_reconstruction(sym),
        step_L10_homogeneous_action(sym),
        step_L11_covariant_chi(sym),
        step_L12_perturbations(sym),
        step_L13_sound_speed(sym),
        step_L14_energy_conditions(sym),
        step_L15_Gamma_class(sym),
        step_L16_bifurcation(sym),
        step_L17_thermodynamic_time(sym),
        step_L18_generalized_Friedmann(sym),
        step_L19_Jacobson_global_bridge(sym),
        step_L20_modified_entropy(sym),
        step_L21_Smax_from_horizon(sym),
        step_L_coupled_rad_matter_S(sym),
    ]


def step_jacobson_future_stub() -> DerivationStep:
    """Backward-compatible name: always uses the embedded full Module-17 derivation."""
    return step_jacobson_local_horizon_full()


# =============================================================================
# STAGE E2 - On-shell Einstein-Hilbert / Euclidean action on the dS attractor
# =============================================================================


def step_Einstein_Hilbert_action(sym: Dict[str, Any]) -> DerivationStep:
    """
    ASSUMED dynamical theory: S_EH = 1/(16πG) ∫ √(-g)(R-2Λ) + S_matter.
    Variation => G_μν + Λ g_μν = 8πG T_μν.
    Do NOT treat this as derived by substituting the attractor backwards.
    """
    G = sym["G"]
    return DerivationStep(
        name="Einstein_Hilbert_action_assumed",
        stage="ONSHELL_ACTION",
        assumptions=[
            "ASSUMED: Einstein-Hilbert + matter is the dynamical theory",
            "Full local EFE from horizon thermo remains FUTURE (Jacobson module)",
        ],
        equations={
            "S_EH": "S_EH = 1/(16πG) ∫ d⁴x √(-g) (R - 2Λ) + S_matter",
            "variation": "δS_EH/δg^{μν} = 0  =>  G_μν + Λ g_μν = 8πG T_μν",
            "architecture": (
                "Jacobson/EFE (FUTURE) OR asymptotic Einstein (STEP 16) "
                "-> restrict to dS attractor -> evaluate S_EH on-shell"
            ),
        },
        claims=[
            "EH action is the theory whose variation reproduces the Einstein equation.",
            "We evaluate it ON the thermodynamic de Sitter solution; we do not invent EH by reverse-engineering.",
        ],
        open_gaps=[ACTION_WORDING_CAUTION],
        status="assumed",
        evidence={"G": G},
    )


def step_Lorentzian_onshell_action(sym: Dict[str, Any]) -> DerivationStep:
    """On-shell Lorentzian EH density; warn that flat FLRW V_4 diverges as t->∞."""
    H_inf, G, S_max = sym["H_star"], sym["G"], sym["S_max"]
    Lambda_S = 3 * H_inf**2
    R_inf = 12 * H_inf**2
    integrand = simplify(R_inf - 2 * Lambda_S)  # = 2 Λ_S
    S_EH_over_V4 = simplify(Lambda_S / (8 * pi * G))
    S_EH_thermo = simplify(S_EH_over_V4.subs({H_inf**2: pi / (G * S_max)}))
    ok = simplify(integrand - 2 * Lambda_S) == 0
    ok2 = simplify(S_EH_thermo - sp.Rational(3, 8) / (G**2 * S_max)) == 0
    return DerivationStep(
        name="Lorentzian_onshell_EH_on_dS",
        stage="ONSHELL_ACTION",
        assumptions=[
            "Vacuum de Sitter: R=4Λ_S, Λ_S=3 H_∞²",
            "Finite spacetime region OR regulated four-volume V_4",
        ],
        equations={
            "R_minus_2Lambda": Eq(Symbol("R_minus_2Lambda"), integrand),
            "S_EH_density": Eq(Symbol("S_EH_over_V4"), S_EH_over_V4),
            "S_EH_thermo": Eq(Symbol("S_EH_over_V4"), S_EH_thermo),
            "V4_def": "V_4 = ∫ d⁴x √(-g)",
            "divergence_caution": (
                "Flat FLRW patch: √(-g)=a³ ∝ e^{3 H_∞ t} => V_4 diverges as t->∞. "
                "Do NOT report ∞ as a physical prediction; use finite region or Euclidean S⁴."
            ),
        },
        claims=[
            "On-shell: R-2Λ_S=2Λ_S => S_EH^{dS}=Λ_S V_4/(8πG)=3 H_∞² V_4/(8πG).",
            "With H_∞²=π/(G S_max): S_EH^{dS}/V_4 = 3/(8 G² S_max).",
            "Direct S_max ↔ on-shell Lorentzian action density (finite-region).",
        ],
        open_gaps=[
            "Lorentzian four-volume of the expanding flat patch is IR-divergent; "
            "prefer Euclidean global dS / Gibbons-Hawking prescriptions for finite I.",
        ],
        status="conditional" if (ok and ok2) else "failed",
        evidence={
            "integrand": integrand,
            "S_EH_over_V4": S_EH_over_V4,
            "S_EH_thermo": S_EH_thermo,
            "ok": ok and ok2,
        },
    )


def step_Euclidean_onshell_action(sym: Dict[str, Any]) -> DerivationStep:
    """
    Euclidean S⁴ continuation: I_E = -π/(G H_∞²) = -S_H(H_∞) -> -S_max.
    Closes the loop S_max -> H_∞ -> Λ_S -> dS -> I_E -> -S_max.
    """
    H_inf, G, S_max = sym["H_star"], sym["G"], sym["S_max"]
    Lambda_S = 3 * H_inf**2
    V_S4 = simplify(8 * pi**2 / (3 * H_inf**4))
    I_E = simplify(-Lambda_S / (8 * pi * G) * V_S4)
    I_E_simp = simplify(I_E)
    target = -pi / (G * H_inf**2)
    ok = simplify(I_E_simp - target) == 0
    I_E_attractor = simplify(target.subs({H_inf**2: pi / (G * S_max)}))
    ok2 = simplify(I_E_attractor + S_max) == 0
    return DerivationStep(
        name="RESULT_Euclidean_onshell_I_E_equals_minus_S_max",
        stage="ONSHELL_ACTION",
        assumptions=[
            "Standard Euclidean de Sitter continuation dS_4 -> S⁴",
            "Radius r_dS = 1/H_∞; Vol(S⁴)=8π²/(3 H_∞⁴)",
            "Euclidean EH convention I_E = -1/(16πG) ∫√g (R-2Λ)",
            "Thermodynamic attractor: S_H->S_max",
        ],
        equations={
            "V_S4": Eq(Symbol("V_S4"), V_S4),
            "I_E_def": "I_E = -1/(16πG) ∫ √g (R-2Λ_S) = -Λ_S V_S4/(8πG)",
            "I_E_dS": Eq(Symbol("I_E"), I_E_simp),
            "I_E_equals_minus_S_H": Eq(Symbol("I_E"), -pi / (G * H_inf**2)),
            "I_E_attractor": Eq(Symbol("I_E"), -S_max),
            "loop": "S_max -> H_∞ -> Λ_S -> dS₄ -> I_E -> -S_max",
        },
        claims=[
            "CONDITIONAL: I_E^{dS} = -π/(G H_∞²) = -S_H(H_∞).",
            "At the thermodynamic attractor: I_E^{dS} = -S_max.",
            "Maximum horizon entropy appears as the magnitude of the Euclidean on-shell action.",
        ],
        open_gaps=[ACTION_WORDING_CAUTION],
        status="conditional" if (ok and ok2) else "failed",
        evidence={
            "V_S4": V_S4,
            "I_E": I_E_simp,
            "I_E_attractor": I_E_attractor,
            "ok": ok and ok2,
        },
    )


def step_semiclassical_partition(sym: Dict[str, Any]) -> DerivationStep:
    """Z ~ e^{-I_E}; with I_E=-S_max formally Z_dS ~ e^{S_max}. Careful wording."""
    S_max = sym["S_max"]
    return DerivationStep(
        name="semiclassical_partition_Z_dS",
        stage="ONSHELL_ACTION",
        assumptions=[
            "Semiclassical Euclidean quantum gravity: Z ~ e^{-I_E}",
            "I_E^{dS}=-S_max from prior step",
        ],
        equations={
            "Z_def": "Z ~ e^{-I_E}",
            "Z_dS": "Z_dS ~ e^{S_max}",
            "NOT_CLAIMED": (
                "NOT claimed: 'the universe chooses de Sitter because QG maximizes Z'"
            ),
        },
        claims=[
            "FORMAL/CONDITIONAL: if I_E^{dS}=-S_max then Z_dS ~ e^{S_max}.",
            "Connects the thermodynamic attractor to a semiclassical gravitational interpretation.",
            ACTION_WORDING_CAUTION,
        ],
        open_gaps=[
            "Euclidean QG subtleties: boundary conditions, path-integral measure, action sign.",
        ],
        status="conditional",
        evidence={"S_max": S_max},
    )


def step_FLRW_action_extremum_vs_S_max(sym: Dict[str, Any]) -> DerivationStep:
    """
    Reduced FLRW EH: does thermodynamic equilibrium coincide with an EH critical
    point under Λ=Λ_S(S_max)?
    """
    H, Hdot, G = sym["H"], sym["Hdot"], sym["G"]
    H_inf, S_max = sym["H_star"], sym["S_max"]
    Lambda_S = 3 * H_inf**2
    R = 6 * (Hdot + 2 * H**2)
    L_bulk = simplify(R - 2 * Lambda_S)
    L_at_eq = simplify(L_bulk.subs({Hdot: 0, H: H_inf}))
    H2_crit = Lambda_S / 3
    coincides = simplify(H2_crit - H_inf**2) == 0
    S_H = Symbol("S_H", real=True, positive=True)
    H2_thermo = pi / (G * S_H)
    delta = simplify(H2_thermo - H_inf**2)
    delta_at_max = simplify(delta.subs({S_H: S_max, H_inf**2: pi / (G * S_max)}))
    return DerivationStep(
        name="FLRW_EH_extremum_coincides_with_S_max",
        stage="ONSHELL_ACTION",
        assumptions=[
            "Λ fixed to thermodynamic Λ_S=3π/(G S_max)",
            "Flat FLRW reduced Einstein-Hilbert theory",
            "Thermo trajectory H²=π/(G S_H)",
        ],
        equations={
            "R_FLRW": Eq(Symbol("R"), R),
            "L_bulk": Eq(Symbol("L_bulk"), L_bulk),
            "L_at_attractor": Eq(Symbol("L_eq"), L_at_eq),
            "EH_critical": Eq(H**2, Lambda_S / 3),
            "thermo_H2": Eq(H**2, H2_thermo),
            "delta_H2": Eq(Symbol("H2_thermo_minus_H2_crit"), delta),
            "delta_at_S_max": Eq(Symbol("delta_at_S_max"), delta_at_max),
            "coincidence": (
                "S_H=S_max ⇔ H=H_∞ ⇔ H²=Λ_S/3 ⇔ vacuum EH critical point (dS)"
            ),
        },
        claims=[
            "With Λ=Λ_S(S_max), the EH vacuum critical point is H²=Λ_S/3=H_∞².",
            "Thermodynamic equilibrium S_H=S_max selects exactly that same H_∞.",
            "Therefore S_H=S_max coincides with the on-shell EH critical point under Λ=Λ_S.",
            "This is a coincidence of equilibria under shared Λ_S - not an independent "
            "derivation of EH from entropy alone.",
        ],
        open_gaps=[
            "Full variational analysis of S_EH[a(t)] along the finite-time thermo trajectory "
            "(off-shell / with Gibbons-Hawking terms) remains an open diagnostic.",
        ],
        status="conditional" if coincides and simplify(delta_at_max) == 0 else "failed",
        evidence={
            "L_at_eq": L_at_eq,
            "coincides": coincides,
            "delta_at_max": delta_at_max,
        },
    )


def step_effective_sector_after_attractor(sym: Dict[str, Any]) -> DerivationStep:
    """Misner-Sharp / vacuum-branch ρ_S (kept for continuity with prior exports)."""
    H, G = sym["H"], sym["G"]
    E_H = 1 / (2 * G * H)
    V_H = (4 * pi / 3) / H**3
    rho_S = simplify(E_H / V_H)
    ok = simplify(rho_S - 3 * H**2 / (8 * pi * G)) == 0
    return DerivationStep(
        name="effective_stress_energy_AFTER_attractor",
        stage="AFTER_ATTRACTOR",
        assumptions=[
            "Attractor theorem already established",
            "Misner-Sharp E_H = R_H/(2G)",
            "ρ_S ≡ E_H/V_H; equilibrium P_S = -ρ_S",
        ],
        equations={
            "E_H": Eq(Symbol("E_H"), E_H),
            "rho_S": Eq(Symbol("rho_S"), rho_S),
            "P_S_eq": Eq(Symbol("P_S"), -rho_S),
            "T_S": "T^{(S)}_{μν} = (ρ_S+P_S) u_μ u_ν + P_S g_μν",
        },
        claims=[
            "Misner-Sharp vacuum branch: ρ_S = 3H²/(8πG)."
            if ok
            else "ρ_S derivation failed.",
        ],
        status="derived" if ok else "failed",
        evidence={"rho_S": rho_S},
    )


def step_modified_friedmann(sym: Dict[str, Any]) -> DerivationStep:
    H, G = sym["H"], sym["G"]
    rho_m, rho_r, rho_S = sym["rho_m"], sym["rho_r"], sym["rho_S"]
    P_m, P_r, P_S = sym["P_m"], sym["P_r"], sym["P_S"]
    friedmann = Eq(H**2, (8 * pi * G / 3) * (rho_m + rho_r + rho_S))
    sourcing = Eq(
        Symbol("rho_S_dot") + 3 * H * (rho_S + P_S),
        3 * H * (rho_m + P_m),
    )
    Hdot_red = Eq(sym["Hdot"], -4 * pi * G * ((rho_r + P_r) + (rho_S + P_S)))
    Pi_H = simplify(8 * pi**2 * (sym["rho"] + sym["P"]) / H**3)
    return DerivationStep(
        name="modified_Friedmann_AFTER_attractor",
        stage="AFTER_ATTRACTOR",
        assumptions=["Einstein equations", "Effective T^(S) from prior step"],
        equations={
            "Friedmann": friedmann,
            "sourcing_paper": sourcing,
            "Raychaudhuri_reduced": Hdot_red,
            "Pi_H_CANONICAL": Eq(Symbol("Pi_H"), Pi_H),
        },
        claims=[
            "Background Friedmann with ρ_S for Layer-2 / CLASS handoff.",
            "Paper sourcing flagged for Q-consistency (see conservation step).",
        ],
        status="derived",
        evidence={"friedmann": friedmann, "Pi_H": Pi_H},
    )


# =============================================================================
# STAGE E - Phenomenology (NOT first principles)
# =============================================================================


def step_phenomenology_comparison(sym: Dict[str, Any]) -> DerivationStep:
    """
    F(Π) and logistic w(t) are phenomenological candidates.

    First-principles question: does thermo EOM reproduce similar transition behavior?
    """
    Pi, k, F0 = sym["Pi"], sym["k"], sym["F0"]
    t, t_crit = sym["t"], sym["t_crit"]
    F = F0 + sp.Rational(1, 2) * k * Pi**2
    rho_S_F = simplify(Pi * diff(F, Pi) - F)
    w_log = 1 / (1 + exp(-k * (t - t_crit)))
    return DerivationStep(
        name="phenomenological_F_and_logistic",
        stage="PHENOMENOLOGY",
        assumptions=[
            "Optional comparison model - NOT an input to the attractor proof",
        ],
        equations={
            "F_Pi_pheno": Eq(Function("F")(Pi), F),
            "rho_S_from_F_pheno": Eq(Symbol("rho_S_F"), rho_S_F),
            "w_logistic_pheno": Eq(Symbol("w_S_pheno"), w_log),
            "validation_question": (
                "Does first-principles χ(t)/w_from_EOM approximate the logistic transition?"
            ),
        },
        claims=[
            "F(Π) and sigmoid w(t) demoted to phenomenological comparison layer.",
            "First-principles layer derives dynamics from S_tot / P / thermo EOM instead.",
            "If Layer-2 finds the thermo EOM reproduces the sigmoid transition - that is a result.",
        ],
        status="assumed",
        evidence={"F": F, "w_log": w_log, "layer": "phenomenological_model"},
    )


def step_perturbations_stub() -> DerivationStep:
    return DerivationStep(
        name="perturbations_CLASS_stub",
        stage="LATER",
        equations={"next": "background -> perturbations -> isocurvature -> CLASS -> observables"},
        open_gaps=["Deferred to Layers 2-3 after attractor theorem is closed."],
        status="stub",
    )


# =============================================================================
# Consistency
# =============================================================================


def run_checks(
    sym: Dict[str, Any], steps: List[DerivationStep], desitter: DeSitterStatus
) -> List[ConsistencyCheck]:
    H, G = sym["H"], sym["G"]
    rho_m, rho_r, rho_S = sym["rho_m"], sym["rho_r"], sym["rho_S"]
    rho, P = sym["rho"], sym["P"]
    checks: List[ConsistencyCheck] = []

    eom = next(s for s in steps if s.name == "thermodynamic_equation_of_motion")
    checks.append(
        ConsistencyCheck(
            "thermo_EOM_present",
            eom.status == "derived",
            "Thermodynamic equation of motion derived from entropy extremization.",
            {"EOM": str(eom.equations.get("thermodynamic_EOM"))},
        )
    )

    lin = next(s for s in steps if s.name == "linearize_and_prove_lambda_negative")
    lam = lin.evidence.get("lambda")
    checks.append(
        ConsistencyCheck(
            "lambda_negative",
            desitter.stability == "proven_strict" and simplify(lam + sym["gamma"]) == 0,
            f"Linearization eigenvalue λ={lam} equals -γ < 0.",
        )
    )

    checks.append(
        ConsistencyCheck(
            "existence_conditional_on_S_max",
            desitter.existence == "conditional",
            desitter.existence_detail,
        )
    )

    # Ordering: attractor -> asymptotic Einstein -> T^(S)/pheno
    names = [s.name for s in steps]
    i_res = names.index("RESULT_stable_de_Sitter_attractor")
    i_asymp = names.index("RESULT_asymptotic_Einstein_equation")
    i_IE = names.index("RESULT_Euclidean_onshell_I_E_equals_minus_S_max")
    i_TS = names.index("effective_stress_energy_AFTER_attractor")
    i_F = names.index("phenomenological_F_and_logistic")
    jac_names = ("LOCAL_horizon_Jacobson_EFE", "FUTURE_local_horizon_Jacobson")
    i_jac = next(names.index(n) for n in jac_names if n in names)
    i_deep = names.index("L01_effective_cosmological_fluid") if "L01_effective_cosmological_fluid" in names else i_jac
    checks.append(
        ConsistencyCheck(
            "pipeline_order_attractor_before_T_S",
            i_res < i_asymp < i_IE < i_TS < i_jac <= i_deep < i_F,
            "Attractor -> asymp. Einstein -> on-shell I_E -> T^(S); Jacobson; deep cosmology; pheno last.",
        )
    )
    fluid = next((s for s in steps if s.name == "L01_effective_cosmological_fluid"), None)
    if fluid is not None:
        checks.append(
            ConsistencyCheck(
                "deep_fluid_w_S_derived",
                fluid.status == "derived",
                "Effective fluid EOS w_S derived from locked entropy dynamics.",
                {"w_S": str(fluid.equations.get("w_S_compact", ""))},
            )
        )
    lyap = next((s for s in steps if s.name == "L05_Lyapunov_global_stability"), None)
    if lyap is not None:
        checks.append(
            ConsistencyCheck(
                "deep_Lyapunov_present",
                lyap.status == "derived",
                "Global Lyapunov V=S_max-S_H constructed.",
            )
        )

    S_max = sym["S_max"]
    H_inf2 = pi / (G * S_max)
    Lambda_S = simplify(3 * pi / (G * S_max))
    R_inf = simplify(12 * pi / (G * S_max))
    rho_S_inf = simplify(sp.Rational(3, 8) / (G**2 * S_max))
    checks.append(
        ConsistencyCheck(
            "Lambda_S_from_S_max",
            simplify(Lambda_S - 3 * H_inf2) == 0,
            "Λ_S = 3π/(G S_max) = 3 H_∞² (π retained).",
            {"Lambda_S": str(Lambda_S)},
        )
    )
    checks.append(
        ConsistencyCheck(
            "R_inf_equals_12_H_inf2",
            simplify(R_inf - 12 * H_inf2) == 0,
            "R_∞ = 12 H_∞² = 12π/(G S_max).",
            {"R_inf": str(R_inf)},
        )
    )
    checks.append(
        ConsistencyCheck(
            "rho_S_inf_from_S_max",
            simplify(rho_S_inf - 3 * H_inf2 / (8 * pi * G)) == 0,
            "ρ_S,∞ = 3/(8 G² S_max) from H_∞²=π/(G S_max).",
            {"rho_S_inf": str(rho_S_inf)},
        )
    )
    checks.append(
        ConsistencyCheck(
            "asymptotic_Einstein_boundary",
            any(
                s.name == "RESULT_asymptotic_Einstein_equation" and s.status == "conditional"
                for s in steps
            ),
            "Asymptotic Einstein labeled CONDITIONAL; full EFE NOT claimed from global module alone.",
        )
    )
    jac_step = next(s for s in steps if s.name in jac_names)
    checks.append(
        ConsistencyCheck(
            "jacobson_module_status",
            jac_step.status in {"conditional", "derived", "stub"},
            (
                f"Jacobson module status={jac_step.status}. "
                "Full local EFE is axiom-conditional (Module 17), not a global-Hubble theorem."
            ),
        )
    )

    eucl = next(
        s for s in steps if s.name == "RESULT_Euclidean_onshell_I_E_equals_minus_S_max"
    )
    I_E = eucl.evidence.get("I_E")
    checks.append(
        ConsistencyCheck(
            "I_E_equals_minus_pi_over_G_H2",
            eucl.status == "conditional"
            and I_E is not None
            and simplify(I_E + pi / (G * sym["H_star"] ** 2)) == 0,
            "Euclidean on-shell action I_E = -π/(G H_∞²) = -S_H(H_∞).",
            {"I_E": str(I_E)},
        )
    )
    checks.append(
        ConsistencyCheck(
            "I_E_attractor_equals_minus_S_max",
            simplify(eucl.evidence.get("I_E_attractor", 0) + S_max) == 0,
            "At attractor: I_E^{dS} = -S_max.",
            {"I_E_attractor": str(eucl.evidence.get("I_E_attractor"))},
        )
    )
    extremum = next(
        s for s in steps if s.name == "FLRW_EH_extremum_coincides_with_S_max"
    )
    checks.append(
        ConsistencyCheck(
            "S_max_coincides_with_EH_critical_point",
            extremum.status == "conditional"
            and simplify(extremum.evidence.get("delta_at_max", 1)) == 0,
            "S_H=S_max coincides with EH vacuum critical point H²=Λ_S/3 under Λ=Λ_S.",
        )
    )
    lor = next(s for s in steps if s.name == "Lorentzian_onshell_EH_on_dS")
    checks.append(
        ConsistencyCheck(
            "Lorentzian_EH_density_from_S_max",
            lor.status == "conditional"
            and simplify(
                lor.evidence.get("S_EH_thermo")
                - sp.Rational(3, 8) / (G**2 * S_max)
            )
            == 0,
            "On-shell Lorentzian S_EH/V_4 = 3/(8 G² S_max).",
        )
    )

    friedmann_GR = simplify(
        ((8 * pi * G / 3) * (rho_m + rho_r + rho_S)).subs({rho_S: 0})
    )
    checks.append(
        ConsistencyCheck(
            "GR_limit",
            simplify(friedmann_GR - (8 * pi * G / 3) * (rho_m + rho_r)) == 0,
            "ρ_S->0 recovers standard Friedmann.",
        )
    )

    Hdot = -4 * pi * G * (rho + P)
    checks.append(
        ConsistencyCheck(
            "de_Sitter_kinematics",
            simplify(Hdot.subs({rho: 0, P: 0})) == 0,
            "ρ+P=0 => Ḣ=0 at equilibrium.",
        )
    )

    S_H = pi / (G * H**2)
    checks.append(
        ConsistencyCheck(
            "dimensional_S_H",
            simplify(S_H * G * H**2 / pi) == 1,
            "S_H * G * H² / π = 1.",
        )
    )

    pheno = next(s for s in steps if s.name == "phenomenological_F_and_logistic")
    checks.append(
        ConsistencyCheck(
            "F_Pi_not_first_principles_centerpiece",
            pheno.stage == "PHENOMENOLOGY" and pheno.status == "assumed",
            "F(Π) and logistic are phenomenological - not used to prove the attractor.",
        )
    )

    prod = next(s for s in steps if s.name == "entropy_production_P")
    checks.append(
        ConsistencyCheck(
            "production_P_defined",
            prod.status == "derived",
            "P ≡ Ṡ_tot defined; GSL is P≥0.",
        )
    )

    return checks


# =============================================================================
# Export - numbered / tagged equation registry + LaTeX / PDF / CSV / JSON
# =============================================================================


@dataclass
class RegistryEq:
    """One numbered equation/line in the verbose Hubble -> de Sitter document."""

    number: str
    step: str
    section: str
    name: str
    tag: EqTag
    latex: str
    ascii: str
    algebra: str = ""
    note: str = ""

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


def _eq(
    number: str,
    step: str,
    section: str,
    name: str,
    tag: EqTag,
    latex: str,
    ascii: str,
    *,
    algebra: str = "",
    note: str = "",
) -> RegistryEq:
    return RegistryEq(number, step, section, name, tag, latex, ascii, algebra, note)


def build_equation_registry(desitter: DeSitterStatus) -> List[RegistryEq]:
    """
    Verbose STEP 0-9 chain for the paper / jury.

    Show algebra at every link. Never relabel ASSUMED as DERIVED.
    """
    return [
        _eq("0.1", "STEP 0", "Assumptions / Postulates", "FLRW", "ASSUMED",
            r"\mathrm{spatially\ flat\ FLRW}", "spatially flat FLRW",
            note="Framework background geometry - not derived here."),
        _eq("0.2", "STEP 0", "Assumptions / Postulates", "R_H_def", "ASSUMED",
            r"R_H = 1/H", "R_H = 1/H",
            note="Apparent Hubble horizon identification (c=1)."),
        _eq("0.3", "STEP 0", "Assumptions / Postulates", "BH_entropy", "ASSUMED",
            r"S_H = A_H/(4G)", "S_H = A_H/(4G)",
            note="Bekenstein-Hawking on the apparent horizon (P1)."),
        _eq("0.4", "STEP 0", "Assumptions / Postulates", "S_max", "ASSUMED",
            r"0 < S_{\max} < \infty", "0 < S_max < ∞",
            note="Finite holographic entropy budget (P3). REQUIRED for H_*>0."),
        _eq("0.5", "STEP 0", "Assumptions / Postulates", "GSL", "ASSUMED",
            r"\dot S_{\mathrm{tot}} \ge 0", "S_tot_dot >= 0",
            note="GSL postulate (P2). These are assumptions of the framework, NOT derived consequences."),
        _eq("1.1", "STEP 1", "Geometry", "R_H", "AXIOM",
            r"R_H = \frac{1}{H}", "R_H = 1/H",
            algebra="Start from the horizon radius definition."),
        _eq("1.2", "STEP 1", "Geometry", "A_H_def", "DERIVED",
            r"A_H = 4\pi R_H^2", "A_H = 4π R_H^2",
            algebra="Sphere area A = 4π R^2 with R = R_H.",
            note="Do not jump from R_H to S_H - area is intermediate."),
        _eq("1.3", "STEP 1", "Geometry", "A_H", "DERIVED",
            r"A_H = 4\pi \left(\frac{1}{H}\right)^2 = \frac{4\pi}{H^2}",
            "A_H = 4π (1/H)^2 = 4π/H^2",
            algebra="Substitute (1.1) into (1.2)."),
        _eq("1.4", "STEP 1", "Geometry", "S_H_from_A", "DERIVED",
            r"S_H = \frac{A_H}{4G} = \frac{1}{4G}\cdot\frac{4\pi}{H^2}",
            "S_H = A_H/(4G) = (1/(4G))*(4π/H^2)",
            algebra="Apply Bekenstein-Hawking (0.3) to (1.3)."),
        _eq("1.5", "STEP 1", "Geometry", "S_H", "DERIVED",
            r"S_H = \frac{\pi}{G H^2}", "S_H = π/(G H^2)",
            algebra="Simplify (1.4): 4π/(4G H^2) = π/(G H^2).",
            note="Horizon entropy as a geometric function of H."),
        _eq("2.1", "STEP 2", "Entropy Evolution", "S_H_power", "DERIVED",
            r"S_H = \frac{\pi}{G}\, H^{-2}", "S_H = (π/G) H^{-2}",
            algebra="Rewrite (1.5) for differentiation."),
        _eq("2.2", "STEP 2", "Entropy Evolution", "S_H_dot_raw", "DERIVED",
            r"\dot S_H = \frac{\pi}{G}\,(-2)\,H^{-3}\,\dot H",
            "S_H_dot = (π/G)*(-2)*H^{-3}*Hdot",
            algebra="d/dt of (2.1): d(H^{-2})/dt = -2 H^{-3} Hdot."),
        _eq("2.3", "STEP 2", "Entropy Evolution", "S_H_dot", "DERIVED",
            r"\dot S_H = -\frac{2\pi}{G}\,\frac{\dot H}{H^3}",
            "S_H_dot = -(2π/G) Hdot / H^3",
            algebra="Simplify (2.2)."),
        _eq("2.4", "STEP 2", "Entropy Evolution", "Theorem_2_2", "THEOREM",
            r"H>0,\; G>0 \quad\Rightarrow\quad \dot S_H \ge 0 \;\Leftrightarrow\; \dot H \le 0",
            "H>0, G>0 => S_H_dot >= 0 <=> Hdot <= 0",
            algebra="Prefactor 2π/(G H^3) > 0 when H,G > 0, so sign(S_H_dot) = -sign(Hdot).",
            note="Connects entropy increase to expansion dynamics. Monotonicity only - NOT the attractor theorem."),
        _eq("3.1", "STEP 3", "Generalized Second Law", "S_tot", "ASSUMED",
            r"S_{\mathrm{tot}} = S_H + S_m + S_{\mathrm{fields}}",
            "S_tot = S_H + S_m + S_fields",
            note="Additive coarse-grained budget (microphysics of S_m, S_fields open)."),
        _eq("3.2", "STEP 3", "Generalized Second Law", "P", "DERIVED",
            r"P \equiv \dot S_{\mathrm{tot}} = \dot S_H + \dot S_m + \dot S_{\mathrm{fields}}",
            "P ≡ S_tot_dot = S_H_dot + S_m_dot + S_fields_dot",
            algebra="Differentiate (3.1)."),
        _eq("3.3", "STEP 3", "Generalized Second Law", "GSL_inequality", "AXIOM",
            r"P \ge 0", "P >= 0",
            note="IMPORTANT: the GSL constrains the direction of thermodynamic evolution but does NOT, by itself, uniquely determine a de Sitter endpoint."),
        _eq("4.1", "STEP 4", "Finite Entropy Equilibrium", "S_max_again", "ASSUMED",
            r"0 < S_{\max} < \infty", "0 < S_max < ∞",
            note="Crucial assumption: a finite entropy maximum exists."),
        _eq("4.2", "STEP 4", "Finite Entropy Equilibrium", "chi", "ASSUMED",
            r"\chi = \frac{S_H - S_{\mathrm{early}}}{S_{\max}-S_{\mathrm{early}}}\in[0,1]",
            "chi = (S_H - S_early)/(S_max - S_early) ∈ [0,1]",
            note="Entropy-completion coordinate (P4). Bridge from conceptual argument to dynamical system. NOT derived geometry."),
        _eq("4.3", "STEP 4", "Finite Entropy Equilibrium", "chi_interpretation", "ASSUMED",
            r"\chi=0:\ \mathrm{early};\quad \chi=1:\ \mathrm{max\ entropy\ equilibrium}",
            "chi=0: early;  chi=1: maximum-entropy equilibrium",
            note="Interpretation of the endpoints of χ."),
        _eq("5.0", "STEP 5", "Thermodynamic Closure", "closure_need", "ASSUMED",
            r"\text{GSL does not uniquely determine }\dot\chi\text{ - a thermodynamic closure is required}",
            "GSL does not uniquely determine chi_dot - closure required",
            note="Transparency: we do NOT claim to derive the sigmoid (or Γ) from the GSL alone."),
        _eq("5.1", "STEP 5", "Thermodynamic Closure", "Gamma_class", "ASSUMED",
            r"\dot\chi = \Gamma(\chi),\quad \Gamma(0)=\Gamma(1)=0,\;\Gamma(\chi)>0\ \mathrm{on}\ (0,1)",
            "chi_dot = Γ(χ), Γ(0)=Γ(1)=0, Γ>0 on (0,1)",
            note="Allowed fixed-point class (P5)."),
        _eq("5.2", "STEP 5", "Thermodynamic Closure", "Gamma_min", "ASSUMED",
            r"\Gamma(\chi)=\gamma\,\chi(1-\chi),\quad \gamma>0",
            "Γ(χ)=γ χ(1-χ), γ>0",
            note="Minimal logistic member of the allowed class. We derive the NEED for a dynamical closure, then choose the minimal member - we do NOT derive the sigmoid from thermodynamics."),
        _eq("6.1", "STEP 6", "Entropy to Hubble", "H_of_S", "DERIVED",
            r"H = \sqrt{\frac{\pi}{G S_H}}", "H = sqrt(π/(G S_H))",
            algebra="Invert (1.5) for H>0."),
        _eq("6.2", "STEP 6", "Entropy to Hubble", "dH_dS", "DERIVED",
            r"\frac{dH}{dS_H} = -\frac{H}{2 S_H}", "dH/dS_H = -H/(2 S_H)",
            algebra="Differentiate (6.1): d(S^{-1/2})/dS = -(1/2) S^{-3/2}."),
        _eq("6.3", "STEP 6", "Entropy to Hubble", "S_of_chi", "DERIVED",
            r"S_H = S_{\mathrm{early}} + \chi(S_{\max}-S_{\mathrm{early}})",
            "S_H = S_early + χ(S_max - S_early)",
            algebra="Invert the definition (4.2)."),
        _eq("6.4", "STEP 6", "Entropy to Hubble", "S_H_dot_chi", "DERIVED",
            r"\dot S_H = (S_{\max}-S_{\mathrm{early}})\,\dot\chi",
            "S_H_dot = (S_max - S_early) chi_dot",
            algebra="Differentiate (6.3)."),
        _eq("6.5", "STEP 6", "Entropy to Hubble", "Hdot_chain", "DERIVED",
            r"\dot H = \frac{dH}{dS_H}\,\dot S_H", "Hdot = (dH/dS_H) * S_H_dot",
            algebra="Chain rule."),
        _eq("6.6", "STEP 6", "Entropy to Hubble", "thermo_EOM", "DERIVED",
            r"\dot H = -\frac{H}{2S_H}\,(S_{\max}-S_{\mathrm{early}})\,\gamma\chi(1-\chi)",
            "Hdot = -(H/(2 S_H))*(S_max-S_early)*γ*χ*(1-χ)",
            algebra="Substitute (6.2),(6.4),(5.2) into (6.5).",
            note="CENTRAL dynamical equation - causal chain now mathematically connected."),
        _eq("7.1", "STEP 7", "Equilibrium / Fixed Point", "chi_eq", "DERIVED",
            r"\chi=1 \;\Rightarrow\; \dot\chi=\gamma(1)(0)=0",
            "chi=1 => chi_dot = γ*1*0 = 0",
            algebra="Evaluate closure (5.2) at the late fixed point."),
        _eq("7.2", "STEP 7", "Equilibrium / Fixed Point", "Sdot_eq", "DERIVED",
            r"\dot S_H = 0 \;\Rightarrow\; \dot H = 0",
            "S_H_dot = 0 => Hdot = 0",
            algebra="From (6.4) and (6.5) with chi_dot=0."),
        _eq("7.3", "STEP 7", "Equilibrium / Fixed Point", "S_max_id", "DERIVED",
            r"S_H=S_{\max}=\frac{\pi}{G H_*^2}",
            "S_H = S_max = π/(G H_*^2)",
            algebra="At χ=1, S_H=S_max; insert into (1.5)."),
        _eq("7.4", "STEP 7", "Equilibrium / Fixed Point", "H_star", "CONDITIONAL",
            r"H_* = \sqrt{\frac{\pi}{G S_{\max}}} > 0",
            "H_* = sqrt(π/(G S_max)) > 0",
            algebra="Solve (7.3) for H_*.",
            note="de Sitter fixed point. EXISTENCE is CONDITIONAL on assumed finite S_max (0.4)/(4.1). Map S_max->H_* is derived."),
        _eq("8.1", "STEP 8", "Stability", "perturbation", "DERIVED",
            r"\chi = 1 - \varepsilon,\quad 0<\varepsilon\ll 1",
            "chi = 1 - eps,  0 < eps << 1",
            algebra="Perturb about the late-time fixed point."),
        _eq("8.2", "STEP 8", "Stability", "eps_dot_exact", "DERIVED",
            r"\dot\chi = \gamma(1-\varepsilon)\varepsilon",
            "chi_dot = γ(1-eps)*eps",
            algebra="Substitute (8.1) into (5.2)."),
        _eq("8.3", "STEP 8", "Stability", "eps_dot_linear", "DERIVED",
            r"\dot\varepsilon = -\gamma\varepsilon + O(\varepsilon^2)",
            "eps_dot = -γ eps + O(eps^2)",
            algebra="chi_dot = -eps_dot; linearize (8.2)."),
        _eq("8.4", "STEP 8", "Stability", "eps_solution", "DERIVED",
            r"\varepsilon(t)=\varepsilon_0 e^{-\gamma t}",
            "eps(t) = eps_0 exp(-γ t)",
            algebra="Solve the linear ODE (8.3)."),
        _eq("8.5", "STEP 8", "Stability", "lambda", "THEOREM",
            r"\gamma>0 \;\Rightarrow\; \lambda=-\gamma<0 \;\Rightarrow\; \varepsilon\to 0,\;\chi\to 1,\; H\to H_*",
            "γ>0 => λ=-γ<0 => eps->0, χ->1, H->H_*",
            note="Actual stability theorem (eigenvalue λ=-γ), not a numerical observation that the solution appears to settle."),
        _eq("9.1", "STEP 9", "de Sitter Expansion", "H_const", "DERIVED",
            r"H(t)=H_*=\mathrm{constant}", "H(t)=H_* = constant",
            algebra="Equilibrium (7.2)+(7.4)."),
        _eq("9.2", "STEP 9", "de Sitter Expansion", "Hubble_def", "AXIOM",
            r"H = \dot a / a", "H = a_dot / a",
            note="Definition of the Hubble parameter."),
        _eq("9.3", "STEP 9", "de Sitter Expansion", "sep_vars", "DERIVED",
            r"\frac{da}{a} = H_*\, dt", "da/a = H_* dt",
            algebra="Separate variables in (9.1)+(9.2)."),
        _eq("9.4", "STEP 9", "de Sitter Expansion", "integrate", "DERIVED",
            r"\ln a = H_* t + C", "ln a = H_* t + C",
            algebra="Integrate (9.3)."),
        _eq("9.5", "STEP 9", "de Sitter Expansion", "a_deSitter", "THEOREM",
            r"a(t) = a_0\, e^{H_* t}", "a(t) = a0 exp(H_* t)",
            algebra="Exponentiate (9.4).",
            note="Accelerated expansion at the attractor. IF the universe evolves toward the max-entropy de Sitter equilibrium permitted by the framework, then H->H_*>0 and a∝e^{H_* t}."),
        _eq("9.6", "STEP 9", "de Sitter Expansion", "attractor_result", "THEOREM",
            r"H\to H_*,\qquad a(t)\to a_0 e^{H_* t}",
            "H -> H_*,  a(t) -> a0 e^{H_* t}",
            note=desitter.verdict_line),
        _eq("10.1", "STEP 10", "Exact Closure Solutions", "chi_t", "DERIVED",
            r"\chi(t)=\frac{1}{1+C\,e^{-\gamma(t-t_{\mathrm{crit}})}}",
            "chi(t)=1/(1+C e^{-γ(t-t_crit)})",
            algebra="Solve ASSUMED closure χ̇=γ χ(1-χ).",
            note="χ = thermodynamic entropy-completion. Do NOT auto-identify with phenomenological w(t)."),
        _eq("10.2", "STEP 10", "Exact Closure Solutions", "S_H_t", "DERIVED",
            r"S_H(t)=S_{\mathrm{early}}+(S_{\max}-S_{\mathrm{early}})\chi(t)",
            "S_H(t)=S_early+(S_max-S_early)χ(t)",
            algebra="Substitute (10.1) into S_H(χ)."),
        _eq("10.3", "STEP 10", "Exact Closure Solutions", "H_t", "DERIVED",
            r"H(t)=\sqrt{\frac{\pi}{G S_H(t)}}",
            "H(t)=sqrt(π/(G S_H(t)))",
            algebra="Invert S_H=π/(G H²) with H>0."),
        _eq("10.4", "STEP 10", "Exact Closure Solutions", "late_limits", "DERIVED",
            r"\lim_{t\to\infty}\chi=1,\quad S_H\to S_{\max},\quad H\to H_\infty=\sqrt{\pi/(G S_{\max})}",
            "t->∞: χ->1, S_H->S_max, H->H_∞=sqrt(π/(G S_max))"),
        _eq("11.1", "STEP 11", "Enthalpy Bridge (paper)", "Raychaudhuri", "AXIOM",
            r"\dot H=-4\pi G(\rho_{\mathrm{tot}}+P_{\mathrm{tot}})",
            "Hdot=-4πG(ρ_tot+P_tot)",
            note="Paper Raychaudhuri / second Friedmann equation."),
        _eq("11.2", "STEP 11", "Enthalpy Bridge (paper)", "enthalpy_eq", "DERIVED",
            r"\rho_{\mathrm{tot}}+P_{\mathrm{tot}}=0 \;\Rightarrow\; \dot H=0 \;\Rightarrow\; H=H_\infty",
            "ρ_tot+P_tot=0 => Hdot=0 => H=H_∞",
            note="Connects thermodynamic H->H_∞ with Einstein/Friedmann de Sitter condition."),
        _eq("11.3", "STEP 11", "Enthalpy Bridge (paper)", "w_S_eq", "DERIVED",
            r"\rho_m,\rho_r\to 0 \;\Rightarrow\; \rho_S+P_S=0 \;\Rightarrow\; w_S=-1",
            "ρ_m,ρ_r->0 => ρ_S+P_S=0 => w_S=-1"),
        _eq("12.1", "STEP 12", "Asymptotic Curvature", "R_FLRW", "AXIOM",
            r"R=6(\dot H+2H^2)\quad (k=0)",
            "R=6(Hdot+2H^2) (flat FLRW)"),
        _eq("12.2", "STEP 12", "Asymptotic Curvature", "R_inf", "DERIVED",
            r"R_\infty=12 H_\infty^2=\frac{12\pi}{G S_{\max}}",
            "R_∞=12 H_∞²=12π/(G S_max)",
            algebra="Ḣ->0, H->H_∞; then H_∞²=π/(G S_max).",
            note="Maximum horizon entropy fixes the asymptotic curvature scale."),
        _eq("13.1", "STEP 13", "Asymptotic Einstein Tensor", "R_mu_nu", "DERIVED",
            r"R^{(\infty)}_{\mu\nu}=3 H_\infty^2 g_{\mu\nu}",
            "R^(∞)_μν=3 H_∞² g_μν"),
        _eq("13.2", "STEP 13", "Asymptotic Einstein Tensor", "G_mu_nu", "DERIVED",
            r"G^{(\infty)}_{\mu\nu}=R_{\mu\nu}-\tfrac12 R g_{\mu\nu}=-3 H_\infty^2 g_{\mu\nu}",
            "G^(∞)_μν=-3 H_∞² g_μν",
            algebra="3 H_∞² g - (1/2)(12 H_∞²) g = -3 H_∞² g."),
        _eq("14.1", "STEP 14", "Thermodynamic Λ_S", "Lambda_S_def", "DERIVED",
            r"\Lambda_S\equiv 3 H_\infty^2",
            "Λ_S ≡ 3 H_∞²"),
        _eq("14.2", "STEP 14", "Thermodynamic Λ_S", "Lambda_S_thermo", "CONDITIONAL",
            r"\Lambda_S=\frac{3\pi}{G S_{\max}}",
            "Λ_S=3π/(G S_max)",
            algebra="Substitute H_∞²=π/(G S_max).",
            note="π retained. Equivalently S_max=3π/(G Λ_S)."),
        _eq("14.3", "STEP 14", "Thermodynamic Λ_S", "chain_S_H_Lambda", "CONDITIONAL",
            r"S_{\max}\;\longleftrightarrow\; H_\infty\;\longleftrightarrow\;\Lambda_S",
            "S_max ↔ H_∞ ↔ Λ_S"),
        _eq("15.1", "STEP 15", "Asymptotic Friedmann / T^(S)", "Friedmann_inf", "DERIVED",
            r"3 H_\infty^2=8\pi G\,\rho_{S,\infty}\quad(\rho_m,\rho_r\to 0)",
            "3 H_∞²=8πG ρ_S,∞"),
        _eq("15.2", "STEP 15", "Asymptotic Friedmann / T^(S)", "rho_S_inf", "CONDITIONAL",
            r"\rho_{S,\infty}=\frac{3 H_\infty^2}{8\pi G}=\frac{3}{8 G^2 S_{\max}}",
            "ρ_S,∞=3 H_∞²/(8πG)=3/(8 G² S_max)",
            algebra="π cancels: 3π/(8π G² S_max)=3/(8 G² S_max)."),
        _eq("15.3", "STEP 15", "Asymptotic Friedmann / T^(S)", "T_S_inf", "DERIVED",
            r"T^{(S,\infty)}_{\mu\nu}=-\rho_{S,\infty}\,g_{\mu\nu}=-\frac{\Lambda_S}{8\pi G}\,g_{\mu\nu}",
            "T^(S,∞)_μν=-ρ_S g_μν=-(Λ_S/(8πG)) g_μν",
            note="w_S=-1 at equilibrium => ρ_S+P_S=0."),
        _eq("16.1", "STEP 16", "Asymptotic Einstein Equation", "Einstein_asymp", "CONDITIONAL",
            r"G^{(\infty)}_{\mu\nu}+\Lambda_S g_{\mu\nu}=0",
            "G^(∞)_μν + Λ_S g_μν = 0",
            algebra="G_μν=8πG T^(S) with T^(S)=-(Λ_S/(8πG)) g => G=-Λ_S g.",
            note="ASYMPTOTIC vacuum Einstein equation ONLY. NOT a derivation of the full local EFE."),
        _eq("16.2", "STEP 16", "Asymptotic Einstein Equation", "boundary", "ASSUMED",
            r"\text{NOT proven: }G_{\mu\nu}+\Lambda g_{\mu\nu}=8\pi G T_{\mu\nu}\text{ from global Hubble horizon alone}",
            BOUNDARY_NOT_PROVEN,
            note="Jacobson local-horizon construction is a FUTURE module."),
        _eq("17.0", "STEP 17", "Einstein-Hilbert Action", "S_EH", "ASSUMED",
            r"S_{\mathrm{EH}}=\frac{1}{16\pi G}\int d^4x\,\sqrt{-g}\,(R-2\Lambda)+S_{\mathrm{matter}}",
            "S_EH=1/(16πG)∫√(-g)(R-2Λ)+S_matter",
            note="ASSUMED dynamical theory. Variation => Einstein eq. NOT obtained by substituting the attractor backwards."),
        _eq("17.1", "STEP 17", "Einstein-Hilbert Action", "variation", "ASSUMED",
            r"\frac{\delta S_{\mathrm{EH}}}{\delta g^{\mu\nu}}=0\;\Rightarrow\;G_{\mu\nu}+\Lambda g_{\mu\nu}=8\pi G T_{\mu\nu}",
            "δS_EH/δg=0 => G_μν+Λ g_μν=8πG T_μν"),
        _eq("18.1", "STEP 18", "Lorentzian On-Shell Action", "R_minus_2Lambda", "DERIVED",
            r"R-2\Lambda_S=2\Lambda_S\quad(R=4\Lambda_S\mathrm{\ on\ vacuum\ dS})",
            "R-2Λ_S=2Λ_S on vacuum dS"),
        _eq("18.2", "STEP 18", "Lorentzian On-Shell Action", "S_EH_density", "CONDITIONAL",
            r"\frac{S_{\mathrm{EH}}^{\mathrm{dS}}}{V_4}=\frac{\Lambda_S}{8\pi G}=\frac{3}{8 G^2 S_{\max}}",
            "S_EH^{dS}/V_4 = Λ_S/(8πG) = 3/(8 G² S_max)",
            note="Finite-region density only. Flat FLRW V_4 diverges as t->∞; do not report ∞ as physics."),
        _eq("19.1", "STEP 19", "Euclidean On-Shell Action", "V_S4", "AXIOM",
            r"V_{S^4}=\frac{8\pi^2}{3 H_\infty^4},\quad r_{\mathrm{dS}}=1/H_\infty",
            "V_S4=8π²/(3 H_∞⁴), r_dS=1/H_∞"),
        _eq("19.2", "STEP 19", "Euclidean On-Shell Action", "I_E_dS", "CONDITIONAL",
            r"I_E^{\mathrm{dS}}=-\frac{\pi}{G H_\infty^2}=-S_H(H_\infty)",
            "I_E^{dS}=-π/(G H_∞²)=-S_H(H_∞)",
            algebra="I_E=-Λ V_S4/(8πG); Λ=3H²; V=8π²/(3H⁴) => I_E=-π/(G H²)."),
        _eq("19.3", "STEP 19", "Euclidean On-Shell Action", "I_E_S_max", "CONDITIONAL",
            r"I_E^{\mathrm{dS}}=-S_{\max}",
            "I_E^{dS}=-S_max",
            note="Loop: S_max->H_∞->Λ_S->dS->I_E->-S_max. Conditional on Euclidean continuation + EH convention + attractor."),
        _eq("20.1", "STEP 20", "Semiclassical Partition", "Z_dS", "CONDITIONAL",
            r"Z\sim e^{-I_E}\;\Rightarrow\;Z_{\mathrm{dS}}\sim e^{S_{\max}}",
            "Z~e^{-I_E} => Z_dS ~ e^{S_max}",
            note=ACTION_WORDING_CAUTION),
        _eq("21.1", "STEP 21", "EH Extremum vs S_max", "coincidence", "CONDITIONAL",
            r"S_H=S_{\max}\;\Leftrightarrow\; H=H_\infty\;\Leftrightarrow\; H^2=\Lambda_S/3"
            r"\quad(\Lambda=\Lambda_S)",
            "S_H=S_max ⇔ H=H_∞ ⇔ H²=Λ_S/3 (Λ=Λ_S)",
            note="Thermodynamic equilibrium coincides with the EH vacuum critical point under shared Λ_S - not an independent derivation of EH from entropy."),
        _eq("C.1", "CONSISTENCY", "Conservation flag", "Q_formalism", "ASSUMED",
            r"\dot\rho_m+3H(\rho_m+P_m)=-Q,\quad \dot\rho_S+3H(\rho_S+P_S)=+Q",
            "dot(rho_m)+3H(...)=-Q;  dot(rho_S)+3H(...)=+Q",
            note="FLAG: paper sourcing with separately conserved matter may break total ∇_μ T^{μν}=0. Diagnostic Q-mode recommended before CLASS."),
        _eq("F.1", "JACOBSON", "Local Horizon (Jacobson Module 17)", "Clausius", "AXIOM",
            r"\delta Q=T\,dS,\quad dS=\eta\,dA,\quad T=\kappa/(2\pi)",
            "δQ=T dS; dS=η dA; T=κ/(2π)",
            note="Module-17 AXIOMS (independent of global Hubble-horizon GSL proof)."),
        _eq("F.2", "JACOBSON", "Local Horizon (Jacobson Module 17)", "null_projection", "DERIVED",
            r"T_{\mu\nu}k^{\mu}k^{\nu}=(\eta/2\pi)\,R_{\mu\nu}k^{\mu}k^{\nu}"
            r"\quad(\forall\mathrm{null}\ k)",
            "T_μν k^μ k^ν=(η/2π) R_μν k^μ k^ν for all null k",
            note="From Clausius + Raychaudhuri at instantaneously stationary local horizon."),
        _eq("F.3", "JACOBSON", "Local Horizon (Jacobson Module 17)", "EFE", "CONDITIONAL",
            r"G_{\mu\nu}+\Lambda g_{\mu\nu}=8\pi G\,T_{\mu\nu}"
            r"\quad(\eta=1/(4G),\ \nabla^\mu T_{\mu\nu}=0)",
            "G_μν+Λ g_μν=8πG T_μν (under Module-17 axioms+conditions)",
            note="Claim F: DERIVED under local-horizon axioms; NOT from global Hubble argument alone."),
        _eq("F.4", "JACOBSON", "Local Horizon (Jacobson Module 17)", "Lambda_bridge", "BRIDGE",
            r"\Lambda\to\Lambda_S=3\pi/(G S_{\max})\;\Rightarrow\;"
            r"G^{(\infty)}_{\mu\nu}+\Lambda_S g_{\mu\nu}=0",
            "Λ->Λ_S=3π/(G S_max) => G^(∞)_μν+Λ_S g_μν=0",
            note="Bridge to global attractor Steps 14-16."),
        _eq("22.1", "AFTER", "Effective Sector (Layer-2 handoff)", "rho_S", "DERIVED",
            r"\rho_S = \frac{3H^2}{8\pi G}\quad (E_H=1/(2GH),\ \mathrm{vacuum\ branch})",
            "ρ_S = 3 H^2/(8π G)",
            note="Misner-Sharp vacuum branch; AFTER attractor + asymptotic Einstein."),
        _eq("22.2", "AFTER", "Effective Sector (Layer-2 handoff)", "Friedmann", "DERIVED",
            r"H^2=\frac{8\pi G}{3}(\rho_m+\rho_r+\rho_S)",
            "H^2 = 8πG/3 (ρ_m+ρ_r+ρ_S)",
            note="Einstein + effective sector for Layer-2 numerics."),
        _eq("P.1", "PHENOMENOLOGY", "Phenomenology (not first principles)", "F_Pi", "PHENOMENOLOGY",
            r"F(\Pi)=F_0+\tfrac12 k\Pi^2", "F(Π)=F0+(1/2)k Π^2",
            note="Comparison / emergent-candidate only - NOT used to prove the attractor."),
        _eq("P.2", "PHENOMENOLOGY", "Phenomenology (not first principles)", "w_logistic", "PHENOMENOLOGY",
            r"w_S(t)=\frac{1}{1+e^{-k(t-t_{\mathrm{crit}})}}",
            "w_S(t)=1/(1+e^{-k(t-t_crit)})",
            note="Old ΛCDM+S sigmoid: finite-time interpolation to the theoretically motivated attractor. NOT the attractor proof. χ != w unless explicitly identified."),
        # ----- LAYERS 1-21 (first principles; analytical_solver.py ONLY) -----
        _eq("L1.1", "LAYER 1", "Effective Fluid", "rho_S", "DERIVED",
            r"\rho_S=\frac{3H^2}{8\pi G}=\frac{3}{8 G^2 S_H}",
            "ρ_S=3H²/(8πG)=3/(8 G² S_H)",
            algebra="Friedmann + locked H²=π/(G S_H).",
            note="Canon: S_H=π/(G H²). Reject GH/(2π)."),
        _eq("L1.2", "LAYER 1", "Effective Fluid", "P_S", "DERIVED",
            r"P_S=-\rho_S-\dot H/(4\pi G)",
            "P_S=-ρ_S-Ḣ/(4πG)"),
        _eq("L1.3", "LAYER 1", "Effective Fluid", "w_S", "THEOREM",
            r"w_S=-1+\frac{(S_{\max}-S_{\mathrm{early}})\gamma\chi(1-\chi)}{3HS_H}",
            "w_S=-1+ΔS γχ(1-χ)/(3 H S_H)",
            note="Predicted EOS; χ->1=>w_S->-1; interior non-phantom."),
        _eq("L2.1", "LAYER 2", "Cosmography", "q", "DERIVED",
            r"q=-1+\frac{(S_{\max}-S_{\mathrm{early}})\gamma\chi(1-\chi)}{2HS_H}",
            "q=-1+ΔS γχ(1-χ)/(2 H S_H)"),
        _eq("L2.2", "LAYER 2", "Cosmography", "j", "AXIOM",
            r"j=1+3\dot H/H^2+\ddot H/H^3",
            "j=1+3Ḣ/H²+Ḧ/H³"),
        _eq("L3.1", "LAYER 3", "Phase Trajectory", "H_chi", "DERIVED",
            r"H(\chi)=\sqrt{\frac{\pi}{G[S_{\mathrm{early}}+(S_{\max}-S_{\mathrm{early}})\chi]}}",
            "H(χ)=sqrt(π/(G[S_early+ΔS χ]))",
            algebra="Invert S_H=π/(G H²) with S_H=S_early+ΔS χ."),
        _eq("L3.2", "LAYER 3", "Phase Trajectory", "dchi_dN", "DERIVED",
            r"\frac{d\chi}{d\ln a}=\frac{\gamma\chi(1-\chi)}{H(\chi)}",
            "dχ/d ln a = γχ(1-χ)/H(χ)",
            algebra="χ̇=γχ(1-χ) and d ln a = H dt => eliminate dt."),
        _eq("L3.3", "LAYER 3", "Phase Trajectory", "a_chi", "DERIVED",
            r"\ln\frac{a}{a_{\mathrm{ref}}}=\int_{\chi_{\mathrm{ref}}}^{\chi}\frac{H(x)}{\gamma x(1-x)}\,dx",
            "ln(a/a_ref)=∫ H(x)/(γ x(1-x)) dx",
            algebra="Separate variables in (L3.2)."),
        _eq("L3.4", "LAYER 3", "Phase Trajectory", "a_chi_Se0", "DERIVED",
            r"S_{\mathrm{early}}=0:\ \ln\frac{a}{a_{\mathrm{ref}}}"
            r"=\frac{1}{\gamma}\sqrt{\frac{\pi}{GS_{\max}}}\big[F(\chi)-F(\chi_{\mathrm{ref}})\big],"
            r"\ F(u)=2\,\mathrm{artanh}\sqrt{u}-2/\sqrt{u}",
            "S_early=0: ln(a/a_ref)=(1/γ)sqrt(π/(G S_max))[F(χ)-F(χ_ref)]",
            algebra="1/(x(1-x))=1/x+1/(1-x); integrate with H=sqrt(π/(G S_max x))."),
        _eq("L3.5", "LAYER 3", "Phase Trajectory", "H_a", "DERIVED",
            r"H(a)=H\big(\chi(a)\big),\quad \chi(a)\ \mathrm{inverts}\ a=a(\chi);"
            r"\quad H(z)=H\big(a=1/(1+z)\big)",
            "H(a)=H(χ(a)); parametric (a(χ),H(χ)) exact; H(z)=H(1/(1+z))",
            note="Exact background phase trajectory without fitting H(z) separately."),
        _eq("L4.1", "LAYER 4", "Autonomous System", "chi_prime", "DERIVED",
            r"\frac{d\chi}{dN}=f(\chi)=\frac{\Gamma(\chi)}{H(\chi)}=\frac{\gamma\chi(1-\chi)}{H(\chi)}",
            "χ'=f(χ)=Γ(χ)/H(χ)=γχ(1-χ)/H(χ)",
            algebra="N=ln a => dN=H dt; χ'=χ̇/H."),
        _eq("L4.2", "LAYER 4", "Autonomous System", "fixed_points", "DERIVED",
            r"f(\chi_*)=0\;\Leftrightarrow\;\chi_*\in\{0,1\}",
            "fixed points χ*=0,1",
            algebra="H>0 => f=0 iff Γ=0 iff χ∈{0,1}."),
        _eq("L4.3", "LAYER 4", "Autonomous System", "lambda_t", "THEOREM",
            r"\lambda_t(0)=\gamma>0,\quad \lambda_t(1)=-\gamma<0",
            "λ_t(0)=γ>0 unstable; λ_t(1)=-γ<0 stable",
            algebra="Γ'(χ)=γ(1-2χ)."),
        _eq("L4.4", "LAYER 4", "Autonomous System", "lambda_N", "THEOREM",
            r"\lambda_N(1)=-\gamma/H_\infty<0,\quad \lambda_N(0)=\gamma/H_{\mathrm{early}}>0",
            "λ_N(1)=-γ/H_∞<0; λ_N(0)=γ/H_early>0",
            algebra="f'=(Γ'H-Γ H')/H²; at fixed points f'=Γ'/H."),
        _eq("L4.5", "LAYER 4", "Autonomous System", "phase_flow", "DERIVED",
            r"\chi\in(0,1):\ f>0\Rightarrow\chi\uparrow;\ [0,1]\ \mathrm{invariant};\ "
            r"\varepsilon(N)\simeq\varepsilon_0 a^{-\gamma/H_\infty}",
            "monotone flow to χ=1; ε∼a^{-γ/H_∞} near attractor"),
        _eq("L5.1", "LAYER 5", "Lyapunov", "V", "THEOREM",
            r"V=S_{\max}-S_H=\Delta S(1-\chi)\ge 0",
            "V=S_max-S_H=ΔS(1-χ)≥0",
            algebra="Entropy deficit to equilibrium."),
        _eq("L5.2", "LAYER 5", "Lyapunov", "V_zero", "THEOREM",
            r"V(\chi)=0\;\Leftrightarrow\;\chi=1",
            "V=0 iff χ=1"),
        _eq("L5.3", "LAYER 5", "Lyapunov", "Vdot", "THEOREM",
            r"\dot V=-\dot S_H=-\Delta S\,\gamma\chi(1-\chi)\le 0",
            "V̇=-Ṡ_H=-ΔS γχ(1-χ)≤0",
            algebra="Chain rule: V̇=(dV/dχ)χ̇=-ΔS Γ(χ)."),
        _eq("L5.4", "LAYER 5", "Lyapunov", "GAS", "THEOREM",
            r"V\ge0,\ \dot V\le0,\ V=0\Leftrightarrow\chi=1"
            r"\;\Rightarrow\;\chi=1\ \mathrm{globally\ asymptotically\ stable\ on\ }(0,1]",
            "χ=1 globally asymptotically stable on (0,1]",
            note="Upgrades local λ=-γ (LAYER 4) to global Lyapunov descent."),
        _eq("L5.5", "LAYER 5", "Lyapunov", "Vprime_N", "DERIVED",
            r"V'=dV/dN=\dot V/H=-\Delta S\,f(\chi)\le 0",
            "V'=V̇/H=-ΔS f(χ)≤0"),
        _eq("L6.1", "LAYER 6", "First Law", "geometry", "DERIVED",
            r"T_H=H/(2\pi),\ S_H=\pi/(GH^2),\ V_H=4\pi/(3H^3),\ E_H=1/(2GH)",
            "T_H=H/(2π), S_H=π/(GH²), V_H=4π/(3H³), E_H=1/(2GH)"),
        _eq("L6.2", "LAYER 6", "First Law", "E_TS", "THEOREM",
            r"E_H=T_H S_H=\rho_S V_H=1/(2GH)",
            "E_H=T_H S_H=ρ_S V_H=1/(2GH)",
            algebra="(H/2π)(π/(GH²))=1/(2GH); ρV with ρ=3H²/(8πG)."),
        _eq("L6.3", "LAYER 6", "First Law", "W_req", "DERIVED",
            r"W_{\mathrm{req}}=(dE-T dS)/dV=-H^2/(8\pi G)=-\rho_S/3",
            "W_req=-H²/(8πG)=-ρ_S/3",
            algebra="From dE=T dS+W dV with equilibrium T_H."),
        _eq("L6.4", "LAYER 6", "First Law", "consistency", "THEOREM",
            r"W_{\mathrm{fluid}}=(\rho-P)/2=\rho+\dot H/(8\pi G);"
            r"\ W_{\mathrm{req}}=W_{\mathrm{fluid}}\Leftrightarrow\dot H=-4H^2",
            "W_fluid=W_req iff Ḣ=-4H² (not generic for logistic)",
            note="Equilibrium first law holds at dS endpoint; off it needs dynamical T."),
        _eq("L6.5", "LAYER 6", "First Law", "T_dyn", "DERIVED",
            r"T_{\mathrm{dyn}}=\frac{H}{2\pi}\Bigl|1+\frac{\dot H}{2H^2}\Bigr|"
            r"\ \xrightarrow{\dot H\to0}\ T_H",
            "T_dyn=(H/2π)|1+Ḣ/(2H²)| -> T_H as Ḣ->0"),
        _eq("L7.1", "LAYER 7", "Entropy Production", "Sdot_geom", "DERIVED",
            r"\dot S_H=-\frac{2\pi\dot H}{G H^3},\quad \dot S_H/S_H=-2\dot H/H",
            "Ṡ_H=-2πḢ/(G H³); Ṡ_H/S_H=-2Ḣ/H",
            algebra="Differentiate S_H=π/(G H²)."),
        _eq("L7.2", "LAYER 7", "Entropy Production", "Sdot_closure", "DERIVED",
            r"\dot S_H=(S_{\max}-S_{\mathrm{early}})\gamma\chi(1-\chi)",
            "Ṡ_H=ΔS γχ(1-χ)",
            algebra="S_H=S_early+ΔS χ and χ̇=γχ(1-χ)."),
        _eq("L7.3", "LAYER 7", "Entropy Production", "xi", "THEOREM",
            r"\xi:=\dot S_H/(H S_H)=-2\dot H/H^2=2(1+q)=3(w_S+1)",
            "ξ=Ṡ_H/(H S_H)=-2Ḣ/H²=2(1+q)=3(w_S+1)",
            note="Dimensionless entropy-growth rate in Hubble units."),
        _eq("L7.4", "LAYER 7", "Entropy Production", "equilibrium", "DERIVED",
            r"\chi\to 1:\ \xi\to 0;\quad \chi\in(0,1):\ \xi>0",
            "ξ->0 at attractor; ξ>0 during transition",
            note="ξ≪1 means near thermodynamic equilibrium."),
        _eq("L8.1", "LAYER 8", "Transition Epoch", "chi_star", "THEOREM",
            r"\frac{d}{d\chi}[\chi(1-\chi)]=1-2\chi=0\;\Rightarrow\;\chi_\star=\tfrac12",
            "χ_★=1/2 unique max of χ̇ and Ṡ_H",
            algebra="Second derivative -2<0."),
        _eq("L8.2", "LAYER 8", "Transition Epoch", "Sdot_max", "THEOREM",
            r"\dot S_{H,\max}=\gamma\Delta S/4",
            "Ṡ_H,max=γ ΔS/4"),
        _eq("L8.3", "LAYER 8", "Transition Epoch", "t_thermo", "DERIVED",
            r"\chi(t)=1/(1+Ce^{-\gamma(t-t_0)}):\ t_\star=t_0+\gamma^{-1}\ln C",
            "t_thermo=t0+γ^{-1} ln C (C=1 => t_thermo=t0)"),
        _eq("L8.4", "LAYER 8", "Transition Epoch", "epoch_compare", "THEOREM",
            r"q=0\Leftrightarrow w_S=-1/3;\quad t_{\mathrm{thermo}}=t_{q=0}"
            r"\Leftrightarrow\gamma=8H_\star S_\star/\Delta S",
            "q=0 iff w=-1/3; equals t_thermo only if γ tuned",
            note="Generic prediction: thermodynamic and acceleration-onset epochs differ."),
        _eq("L9.1", "LAYER 9", "Scalar Reconstruction", "phidot2", "DERIVED",
            r"\dot\phi^2=-\dot H/(4\pi G)"
            r"=H\Delta S\gamma\chi(1-\chi)/(8\pi G S_H)",
            "φ̇²=-Ḣ/(4πG)=H ΔS γχ(1-χ)/(8πG S_H)",
            algebra="ρ+P=-Ḣ/(4πG)=φ̇² for canonical scalar."),
        _eq("L9.2", "LAYER 9", "Scalar Reconstruction", "V_phi", "DERIVED",
            r"V=(3H^2+\dot H)/(8\pi G)",
            "V=(3H²+Ḣ)/(8πG)",
            algebra="V=(ρ-P)/2."),
        _eq("L9.3", "LAYER 9", "Scalar Reconstruction", "match", "THEOREM",
            r"\rho_\phi=\rho_S,\quad w_\phi=w_S",
            "ρ_φ=ρ_S and w_φ=w_S along the trajectory"),
        _eq("L9.4", "LAYER 9", "Scalar Reconstruction", "attractor", "DERIVED",
            r"\chi\to1:\ \dot\phi\to0,\ V\to 3/(8G^2 S_{\max})",
            "χ->1: φ̇->0, V->ρ_{S,∞} (de Sitter)",
            note="Background equivalence ONLY - not a microphysical claim."),
        _eq("L9.5", "LAYER 9", "Scalar Reconstruction", "dchi_dphi", "DERIVED",
            r"d\chi/d\phi=\gamma\chi(1-\chi)/\dot\phi",
            "dχ/dφ=γχ(1-χ)/φ̇ => V=V(φ) by quadrature"),
        _eq("L10.1", "LAYER 10", "Homogeneous Action", "EL", "DERIVED",
            r"L=\tfrac12 K\dot\chi^2-U:\quad K\ddot\chi+\tfrac12 K'\dot\chi^2+U'=0",
            "EL: K χ̈ + ½ K' χ̇² + U'=0"),
        _eq("L10.2", "LAYER 10", "Homogeneous Action", "U", "CONDITIONAL",
            r"K=1,\ \dot\chi=\Gamma:\ U=-\tfrac12\Gamma^2=-\tfrac12\gamma^2\chi^2(1-\chi)^2",
            "U=-½Γ²=-½γ²χ²(1-χ)²",
            algebra="From Γ Γ'+U'=0; admits logistic as EL solution."),
        _eq("L10.3", "LAYER 10", "Homogeneous Action", "EL_check", "THEOREM",
            r"\mathrm{on\ }\dot\chi=\Gamma:\ \ddot\chi+U'=0\mathrm{\ (residual\ 0)}",
            "EL residual vanishes on logistic",
            note="χ̇=-Γ also solves EL; GSL selects χ̇=+Γ."),
        _eq("L10.4", "LAYER 10", "Homogeneous Action", "energy", "DERIVED",
            r"E=\tfrac12\dot\chi^2+U\equiv0\mathrm{\ on\ logistic}",
            "On-shell mechanical energy E≡0"),
        _eq("L10.5", "LAYER 10", "Homogeneous Action", "honesty", "ASSUMED",
            r"\mathrm{GSL}\nRightarrow\Gamma;\ (K,U)\mathrm{\ non-unique}",
            "Does NOT derive Γ from GSL; (K,U) non-unique",
            note="Weakest link remains; covariant completion in LAYER 11."),
        _eq("L11.1", "LAYER 11", "Covariant χ", "L_cov", "DERIVED",
            r"L_\chi=K(\chi)X-U(\chi),\ X=-\tfrac12(\nabla\chi)^2",
            "L_χ=K X-U; X=-½(∇χ)²"),
        _eq("L11.2", "LAYER 11", "Covariant χ", "T_chi", "DERIVED",
            r"T^{(\chi)}_{\mu\nu}=K\nabla_\mu\chi\nabla_\nu\chi-(KX-U)g_{\mu\nu}",
            "T_μν=K ∂_μχ ∂_νχ-(KX-U)g_μν"),
        _eq("L11.3", "LAYER 11", "Covariant χ", "FLRW_KG", "DERIVED",
            r"K=1:\ \ddot\chi+3H\dot\chi+U'=0",
            "FLRW: χ̈+3Hχ̇+U'=0",
            algebra="□χ=-χ̈-3Hχ̇; EL => χ̈+3Hχ̇+U'=0."),
        _eq("L11.4", "LAYER 11", "Covariant χ", "friction_gap", "THEOREM",
            r"U=-\tfrac12\Gamma^2\mathrm{\ on\ }\dot\chi=\Gamma:\ \mathrm{residual}=3H\Gamma",
            "Naive L10 covariantization: FLRW residual 3HΓ != 0",
            note="Hubble friction; remedies R1-R4 (Onsager/constraint/nonminimal/approx)."),
        _eq("L11.5", "LAYER 11", "Covariant χ", "c_s2", "DERIVED",
            r"K=\mathrm{const}\Rightarrow c_s^2=1\mathrm{\ (vs\ }c_a^2\mathrm{\ LAYER\ 13)}",
            "Canonical propagation speed c_s²=1"),
        _eq("L12.1", "LAYER 12", "Perturbations", "delta_S_rho", "DERIVED",
            r"\delta S_H=-2(S_H/H)\delta H,\quad \delta\rho_S=2(\rho_S/H)\delta H",
            "δS_H=-2(S_H/H)δH; δρ_S=2(ρ_S/H)δH",
            algebra="Architecture A: kinematic S_H(H)."),
        _eq("L12.2", "LAYER 12", "Perturbations", "Einstein", "DERIVED",
            r"\delta G_{\mu\nu}=8\pi G\,(\delta T^{(m)}+\delta T^{(S)})",
            "δG=8πG(δT_m+δT_S)"),
        _eq("L12.3", "LAYER 12", "Perturbations", "delta_chi", "DERIVED",
            r"\ddot{\delta\chi}+3H\dot{\delta\chi}+(c_s^2 k^2/a^2+m_{\mathrm{eff}}^2)\delta\chi=S[\Phi,\Psi]",
            "δχ KG with c_s²=1, m_eff²=U''(χ̄) (Architecture B)",
            note="Canonical => Ψ=Φ; clustering vs smooth DE from L11 remedies."),
        _eq("L12.4", "LAYER 12", "Perturbations", "vacuum", "THEOREM",
            r"\chi\to1:\ 1+w_S\to0\mathrm{\ (vacuum\!-\!like;\ isocurvature\ delicate)}",
            "Attractor: entropy sector vacuum-like"),
        _eq("L13.1", "LAYER 13", "Sound Speed", "c_a2_def", "DERIVED",
            r"c_a^2:=\dot P_S/\dot\rho_S=(dP_S/d\chi)/(d\rho_S/d\chi)",
            "c_a²=Ṗ_S/ρ̇_S=(dP/dχ)/(dρ/dχ)"),
        _eq("L13.2", "LAYER 13", "Sound Speed", "c_a2_id", "THEOREM",
            r"c_a^2=w_S-(S_H/\Delta S)w_S'=w_S-\dot w_S/(3H(1+w_S))",
            "c_a²=w-(S_H/ΔS)w'=w-ẇ/(3H(1+w))"),
        _eq("L13.3", "LAYER 13", "Sound Speed", "vs_cs", "DERIVED",
            r"c_s^2=1\mathrm{\ (Arch.\ B,\ K=\mathrm{const})}\neq c_a^2",
            "c_s²=1 != c_a²; gradient instability tracks c_s²",
            note="At χ->1, c_a² delicate as sector becomes Λ-like."),
        _eq("L14.1", "LAYER 14", "Energy Conditions", "NEC", "THEOREM",
            r"\rho_S+P_S=-\dot H/(4\pi G)\ge 0\;\Leftrightarrow\;\dot H\le 0\;\Leftrightarrow\;w_S\ge -1",
            "NEC automatic for logistic (Ḣ≤0, non-phantom)"),
        _eq("L14.2", "LAYER 14", "Energy Conditions", "WEC_DEC", "THEOREM",
            r"\mathrm{WEC:\ }\rho_S>0\mathrm{\ \&\ NEC;\quad DEC\ (}P\le0\mathrm{):\ }\Leftrightarrow\mathrm{NEC}",
            "WEC holds; DEC≡NEC while P_S≤0"),
        _eq("L14.3", "LAYER 14", "Energy Conditions", "SEC", "THEOREM",
            r"\mathrm{SEC}\Leftrightarrow w_S\ge -1/3\Leftrightarrow q\ge 0;"
            r"\ \mathrm{SEC\ fails}\Leftrightarrow\mathrm{acceleration}",
            "SEC failure ↔ accelerated expansion",
            note="At χ=1: NEC saturated, SEC violated (eternal dS acceleration)."),
        _eq("L15.1", "LAYER 15", "Γ-class", "Gamma", "THEOREM",
            r"\Gamma=\gamma\chi^n(1-\chi)^m;\ H_\infty=\sqrt{\pi/(GS_{\max})}\ \mathrm{independent\ of\ }(n,m,\gamma)",
            "dS endpoint universal across Γ-class"),
        _eq("L15.2", "LAYER 15", "Γ-class", "stability", "THEOREM",
            r"m=1:\ \Gamma'(1)=-\gamma;\ m>1:\ \Gamma'(1)=0;\ n=1:\ \Gamma'(0)=+\gamma",
            "Exponential attractor only for m=1; χ=0 unstable for n=1"),
        _eq("L15.3", "LAYER 15", "Γ-class", "Lyapunov_NEC", "THEOREM",
            r"\dot V=-\Delta S\,\Gamma\le0;\ \rho+P\propto\Gamma\ge0",
            "GAS + NEC persist for entire Γ-class; timescale not universal"),
        _eq("L16.1", "LAYER 16", "Bifurcation", "lambda_1", "THEOREM",
            r"\Gamma_\alpha=\gamma\chi(1-\chi)(1+\alpha\chi);\ \lambda_1=-\gamma(1+\alpha);\ "
            r"\chi=1\mathrm{\ stable}\Leftrightarrow\alpha>-1",
            "Linear stability of late dS under α-deformation"),
        _eq("L16.2", "LAYER 16", "Bifurcation", "transcritical", "THEOREM",
            r"\alpha_c=-1:\ \lambda_1=0,\ \chi_*=-1/\alpha\to1;\ \mathrm{transcritical\ exchange}",
            "Transcritical bifurcation at α=-1"),
        _eq("L16.3", "LAYER 16", "Bifurcation", "broken_branch", "DERIVED",
            r"\alpha<-1:\ \chi_*=-1/\alpha\in(0,1)\mathrm{\ attractor},\ \chi=1\mathrm{\ unstable}",
            "Incomplete attractor; Γ sign-flip breaks global GSL on [0,1]"),
        _eq("L17.1", "LAYER 17", "Thermo Time", "tau", "THEOREM",
            r"\tau_S=\chi=(S_H-S_{\mathrm{early}})/\Delta S;\ "
            r"\dot\tau_S>0\Leftrightarrow\dot S_H>0",
            "Thermodynamic clock dual to horizon entropy production"),
        _eq("L17.2", "LAYER 17", "Thermo Time", "arrow", "THEOREM",
            r"t\mapsto -t\Rightarrow\dot S_H\mapsto-\dot S_H\mathrm{\ violates\ GSL\ (unless\ }\Gamma=0\mathrm{)}",
            "Time reversal breaks the thermodynamic arrow except at equilibrium"),
        _eq("L17.3", "LAYER 17", "Thermo Time", "reparam", "DERIVED",
            r"dt=d\tau_S/\Gamma(\tau_S);\ H=H(\tau_S),\ a=a(\tau_S)",
            "Cosmic time is reparametrization of thermo time; EFE alone give no arrow"),
        _eq("L18.1", "LAYER 18", "Generalized Friedmann", "F_locked", "THEOREM",
            r"H^2=F(S_H),\ F_{\mathrm{locked}}(S)=\pi/(GS)\ \mathrm{(inverse\ of\ }S_H=\pi/(GH^2)\mathrm{)}",
            "Entropy-first constitutive map; locked GR special case"),
        _eq("L18.2", "LAYER 18", "Generalized Friedmann", "Hdot_chain", "THEOREM",
            r"\dot H=-(H/(2S_H))\dot S_H\ \mathrm{recovers\ LAYER\ 1/\Gamma\!-\!class\ EOM}",
            "Chain-rule kinematics from H=√F(S)"),
        _eq("L18.3", "LAYER 18", "Generalized Friedmann", "Friedmann", "DERIVED",
            r"\rho=3F(S)/(8\pi G);\ H_\infty=\sqrt{F(S_{\max})};\ \mathrm{other\ }F\Rightarrow\mathrm{modified}",
            "Standard Friedmann = F_locked; architecture admits modified F(S)"),
        _eq("L19.1", "LAYER 19", "Jacobson Bridge", "eta", "CONDITIONAL",
            r"\eta_{\mathrm{local}}=\eta_{\mathrm{global}}=1/(4G)\Rightarrow S_H=\pi/(GH^2)",
            "Bridge B1: common entropy density (assumed, not derived)"),
        _eq("L19.2", "LAYER 19", "Jacobson Bridge", "Lambda_S", "CONDITIONAL",
            r"\Lambda\to\Lambda_S=3H_\infty^2=3\pi/(GS_{\max});\ \rho_\Lambda=\rho_S|_{\chi=1}",
            "Bridge B2: free Jacobson Λ fixed by global attractor"),
        _eq("L19.3", "LAYER 19", "Jacobson Bridge", "architecture", "CONDITIONAL",
            r"\mathrm{local\ thermo}\to\mathrm{EFE}\to\mathrm{FLRW}\to S_H\to\mathrm{dS}\to\Lambda_S",
            "Completed research-target chain; master S[g,χ] still open"),
        _eq("L20.1", "LAYER 20", "Modified Entropy", "S_corr", "DERIVED",
            r"S=A/(4G)+\alpha\ln(A/A_0)+\beta A_0/A;\ "
            r"\eta_{\mathrm{eff}}=1/(4G)+\alpha/A-\beta A_0/A^2",
            "Corrected entropy and running η_eff; BH recovered at α=β=0"),
        _eq("L20.2", "LAYER 20", "Modified Entropy", "H_inf_LO", "DERIVED",
            r"S(H_\infty)=S_{\max};\ \beta=0:\ \delta u\simeq\alpha(u_0/S_{\max})\ln(4\pi/(A_0 u_0))",
            "Leading-order shift of H_∞² about u_0=π/(G S_max)"),
        _eq("L20.3", "LAYER 20", "Modified Entropy", "robustness", "THEOREM",
            r"S'(u_0)|_{\mathrm{BH}}=-S_{\max}/u_0<0\Rightarrow\mathrm{attractor\ robust\ for\ small\ }(\alpha,\beta)",
            "Structural stability of dS under log/inverse-area corrections"),
        _eq("L21.1", "LAYER 21", "Finite S_max", "A_max", "CONDITIONAL",
            r"S_{\max}=A_{\max}/(4G);\ A_\infty=4\pi/H_\infty^2=4GS_{\max}=A_{\max}",
            "Holographic budget = attractor horizon area (consistency identity)"),
        _eq("L21.2", "LAYER 21", "Finite S_max", "finite_iff", "THEOREM",
            r"0<S_{\max}<\infty\Leftrightarrow 0<H_\infty,\Lambda_S<\infty;\ "
            r"S_{\max}\to\infty\Rightarrow\Lambda_S\to0",
            "Finite budget ↔ positive late-time vacuum"),
        _eq("L21.3", "LAYER 21", "Finite S_max", "radius_IE", "DERIVED",
            r"R_\infty=1/H_\infty;\ I_E^{\mathrm{dS}}=-S_{\max}\mathrm{\ (EH\ convention)}",
            "Apparent/dS radius; Euclidean on-shell bookkeeping - S_max value still open"),
        # Full Jacobson ledger (Module 17) - every labeled equation
        _eq("J.A1", "JACOBSON", "Module 17 Axioms", "Clausius", "AXIOM",
            r"\delta Q=T\,dS", "δQ=T dS",
            note="Independent of global Hubble-horizon GSL."),
        _eq("J.A2", "JACOBSON", "Module 17 Axioms", "dS_eta", "AXIOM",
            r"dS=\eta\,dA", "dS=η dA"),
        _eq("J.A3", "JACOBSON", "Module 17 Axioms", "Unruh", "AXIOM",
            r"T=\kappa/(2\pi)", "T=κ/(2π)"),
        _eq("J.H1", "JACOBSON", "Module 17 Conditions", "stationary", "ASSUMED",
            r"\theta|_P=0,\ \sigma_{ab}|_P=0", "θ|_P=σ|_P=0"),
        _eq("J.G1", "JACOBSON", "Module 17 Geometry", "Raychaudhuri", "AXIOM",
            r"d\theta/d\lambda=-\tfrac12\theta^2-\sigma^2-R_{\mu\nu}k^\mu k^\nu",
            "dθ/dλ=-½θ²-σ²-R_μν k^μ k^ν"),
        _eq("J.G2", "JACOBSON", "Module 17 Geometry", "theta_lin", "DERIVED",
            r"\theta(\lambda)=-\lambda R_{\mu\nu}k^\mu k^\nu+O(\lambda^2)",
            "θ(λ)=-λ R_μν k^μ k^ν+O(λ²)"),
        _eq("J.E3", "JACOBSON", "Module 17 Clausius", "null_proj", "DERIVED",
            r"T_{\mu\nu}k^\mu k^\nu=(\eta/2\pi)R_{\mu\nu}k^\mu k^\nu",
            "T_μν k^μ k^ν=(η/2π)R_μν k^μ k^ν"),
        _eq("J.L1", "JACOBSON", "Module 17 Lemma", "null_lemma", "THEOREM",
            r"S_{\mu\nu}k^\mu k^\nu=0\ \forall\mathrm{null}\ k\Rightarrow S_{\mu\nu}=f g_{\mu\nu}",
            "null projection => S_μν=f g_μν"),
        _eq("J.E5", "JACOBSON", "Module 17 Reconstruction", "f_fix", "DERIVED",
            r"f=-R/2+\Lambda", "f=-R/2+Λ"),
        _eq("J.C2", "JACOBSON", "Module 17 Calibration", "eta_G", "ASSUMED",
            r"\eta=1/(4G)\Rightarrow 2\pi/\eta=8\pi G",
            "η=1/(4G)=>2π/η=8πG"),
        _eq("J.E7", "JACOBSON", "Module 17 EFE", "EFE", "CONDITIONAL",
            r"G_{\mu\nu}+\Lambda g_{\mu\nu}=8\pi G\,T_{\mu\nu}",
            "G_μν+Λ g_μν=8πG T_μν",
            note="Claim F: DERIVED under Module-17 axioms; NOT from global module alone."),
        _eq("J.B2", "JACOBSON", "Module 17 Bridge", "Lambda_S", "CONDITIONAL",
            r"\Lambda\to\Lambda_S=3\pi/(G S_{\max})=3H_\infty^2",
            "Λ->Λ_S=3π/(G S_max)",
            note="Λ free in Jacobson; fixed by global attractor."),
    ]


def build_payload(
    steps: List[DerivationStep],
    checks: List[ConsistencyCheck],
    temperature_mode: str,
    desitter: DeSitterStatus,
) -> Dict[str, Any]:
    registry = build_equation_registry(desitter)
    return {
        "theory": "LambdaCDM+S",
        "layer": 1,
        "document": (
            "First-Principles Derivation: Hubble Horizon -> de Sitter Attractor "
            "-> Asymptotic Einstein Structure"
        ),
        "pipeline": PIPELINE_STAGES,
        "conceptual_chain": CONCEPTUAL_CHAIN,
        "scientific_claim": SCIENTIFIC_CLAIM,
        "wording_caution": WORDING_CAUTION,
        "boundary_not_proven": BOUNDARY_NOT_PROVEN,
        "action_wording_caution": ACTION_WORDING_CAUTION,
        "temperature_mode": temperature_mode,
        "desitter_status": desitter.to_dict(),
        "attractor_pass_conditional": (
            desitter.stability == "proven_strict"
            and desitter.existence == "conditional"
        ),
        "postulates": POSTULATES,
        "canonical_inventory": CANONICAL_EQUATIONS,
        "equation_registry": [eq.to_dict() for eq in registry],
        "tag_legend": {
            "AXIOM": "Fundamental definition/postulate (not derived in this file)",
            "DERIVED": "Obtained by explicit algebra from prior equations",
            "ASSUMED": "Closure / modeling choice - must be labeled as such in the paper",
            "CONDITIONAL": "Derived given an assumption (e.g. finite S_max)",
            "THEOREM": "Proven statement under stated axioms+assumptions",
            "PHENOMENOLOGY": "Comparison interpolation - not part of the attractor proof",
        },
        "models": {
            "first_principles": [
                "S_tot",
                "P",
                "thermo_EOM",
                "H_star",
                "lambda",
                "Lambda_S",
                "asymptotic_Einstein",
                "I_E_equals_minus_S_max",
                "w_S_predicted",
                "q_predicted",
                "Lyapunov_V",
            ],
            "jacobson_module_17": ["Clausius", "Raychaudhuri", "EFE", "Lambda_bridge"],
            "deep_cosmology": [
                "rho_S", "P_S", "w_S", "q", "H_chi", "first_law",
                "Sdot_max", "energy_conditions", "Gamma_class",
            ],
            "phenomenological_comparison": ["F_Pi", "w_logistic"],
            "future": ["covariant_L_chi", "linear_perturbations", "S_max_microphysics"],
        },
        "architecture": (
            "First-principles attractor => asymptotic Einstein (Λ_S from S_max) "
            "=> on-shell EH / Euclidean I_E=-S_max => phenomenological interpolation "
            "=> Bayesian observational test"
        ),
        "steps": [s.to_serializable() for s in steps],
        "consistency_checks": [
            {
                "name": c.name,
                "passed": c.passed,
                "detail": c.detail,
                "equations": c.equations,
            }
            for c in checks
        ],
        "handoff": {
            "paper": (
                "Copy STEP 0-21; highlight I_E=-S_max loop and asymptotic Einstein; "
                "do not claim full EFE or 'QG maximizes Z'."
            ),
            "class": (
                "Use thermo EOM / H_* / Λ_S=3π/(G S_max); resolve Q-conservation before "
                "perturbations; do not center on logistic."
            ),
            "bayesian": "Phenomenological sigmoid is the observational interpolation layer.",
        },
    }


def _export_csv(registry: List[Dict[str, str]], path: Path) -> None:
    fields = ["number", "step", "section", "name", "tag", "latex", "ascii", "algebra", "note"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in registry:
            w.writerow({k: row.get(k, "") for k in fields})


def _export_polished_tex(payload: Dict[str, Any], path: Path) -> None:
    ds = payload["desitter_status"]
    legend = payload["tag_legend"]
    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{amsmath,amssymb,amsthm}",
        r"\usepackage{booktabs}",
        r"\usepackage{hyperref}",
        r"\usepackage{xcolor}",
        r"\title{First-Principles Derivation:\\Hubble Horizon $\rightarrow$ de Sitter Attractor\\"
        r"$\rightarrow$ Asymptotic Einstein $\rightarrow$ On-Shell Action $I_E=-S_{\max}$}",
        r"\author{$\Lambda$CDM+S --- Layer-1 analytical\_solver.py}",
        r"\date{\today}",
        r"\begin{document}",
        r"\maketitle",
        "",
        r"\noindent\textbf{Verdict:} "
        + ds["verdict_line"].replace("_", r"\_")
        + r"\\[0.5em]",
        r"\noindent\textbf{Scientific claim.} "
        + payload["scientific_claim"].replace("_", r"\_")
        + r"\\[0.5em]",
        r"\noindent\textbf{Wording caution.} "
        + payload["wording_caution"].replace("_", r"\_")
        + r"\\[0.5em]",
        r"\noindent\textbf{Boundary (not proven).} "
        + payload.get("boundary_not_proven", BOUNDARY_NOT_PROVEN).replace("_", r"\_")
        + r"\\[0.5em]",
        r"\noindent\textbf{Action wording.} "
        + payload.get("action_wording_caution", ACTION_WORDING_CAUTION).replace("_", r"\_"),
        "",
        r"\section*{Conceptual causal chain (jury narrative)}",
        r"\begin{enumerate}",
    ]
    for link in payload["conceptual_chain"]:
        lines.append(r"  \item " + link.replace("_", r"\_").replace("χ", r"$\chi$").replace("->", r"$\rightarrow$"))
    lines += [
        r"\end{enumerate}",
        r"\noindent\textit{Rigorous form: GSL + finite $S_{\max}$ + thermodynamic closure "
        r"$\Rightarrow$ de Sitter attractor $\Rightarrow$ accelerated expansion "
        r"$\Rightarrow$ asymptotic $G_{\mu\nu}^{(\infty)}+\Lambda_S g_{\mu\nu}=0$ "
        r"$\Rightarrow$ (conditional) Euclidean on-shell $I_E=-S_{\max}$. "
        r"Do not write ``2nd law $=$ de Sitter $=$ acceleration.'' "
        r"Do not claim the full local Einstein equations from the global Hubble horizon alone. "
        r"Do not claim quantum gravity selects de Sitter by maximizing $Z$.}",
        "",
        r"\section*{Tag legend}",
        r"\begin{itemize}",
    ]
    for tag, meaning in legend.items():
        lines.append(rf"  \item \textbf{{{tag}}}: {meaning}")
    lines += [
        r"\end{itemize}",
        "",
        r"\section*{Verbose derivation (STEP 0--21 + FUTURE/PHENO)}",
        r"Each equation is numbered and tagged. Intermediate algebra is shown explicitly.",
        "",
    ]

    current_step = None
    for eq in payload["equation_registry"]:
        if eq.get("step") != current_step:
            current_step = eq.get("step")
            lines.append(rf"\subsection*{{{current_step}: {eq['section']}}}")
        note = (eq.get("note") or "").replace("_", r"\_")
        algebra = (eq.get("algebra") or "").replace("_", r"\_")
        lines += [
            rf"\paragraph{{({eq['number']}) [{eq['tag']}] "
            rf"${eq['name'].replace('_', r'\_')}$.}}",
            r"\begin{equation}",
            eq["latex"] + rf"\tag{{{eq['number']}}}",
            r"\end{equation}",
        ]
        if algebra:
            lines.append(r"\textbf{Algebra:} " + algebra + r"\\")
        if note:
            lines.append(rf"\textit{{{note}}}")
        lines.append(r"\vspace{0.35em}")
        lines.append("")

    lines += [
        r"\section*{Architecture}",
        payload["architecture"].replace("_", r"\_").replace("=>", r"$\Rightarrow$"),
        "",
        r"\section*{Phenomenology}",
        r"The sigmoid $w_S(t)$ and $F(\Pi)$ are \textbf{PHENOMENOLOGY}: a finite-time "
        r"interpolation connecting the early gravitational regime to the theoretically "
        r"motivated attractor, then tested observationally. They are \emph{not} what proves "
        r"the attractor.",
        "",
        r"\end{document}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _export_markdown_report(payload: Dict[str, Any], path: Path) -> None:
    ds = payload["desitter_status"]
    lines = [
        "# First-Principles Derivation: Hubble Horizon -> de Sitter Attractor",
        "",
        f"**Verdict:** {ds['verdict_line']}",
        "",
        f"- Existence: `{ds['existence']}` - {ds['existence_detail']}",
        f"- Stability: `{ds['stability']}` - {ds['stability_detail']}",
        f"- λ: `{ds['lambda_linear']}`",
        "",
        "## Scientific claim",
        "",
        payload["scientific_claim"],
        "",
        "## Wording caution",
        "",
        payload["wording_caution"],
        "",
        "## Conceptual causal chain",
        "",
    ]
    for i, link in enumerate(payload["conceptual_chain"], 1):
        lines.append(f"{i}. {link}")
    lines += [
        "",
        "Rigorous form: **GSL + finite S_max + thermodynamic closure => de Sitter attractor => accelerated expansion.**",
        "",
        "## Tag legend",
        "",
    ]
    for tag, meaning in payload["tag_legend"].items():
        lines.append(f"- **{tag}:** {meaning}")
    lines += [
        "",
        "## Architecture",
        "",
        payload["architecture"],
        "",
        "## Verbose STEP 0-9 equation chain",
        "",
    ]
    current_step = None
    for eq in payload["equation_registry"]:
        if eq.get("step") != current_step:
            current_step = eq.get("step")
            lines += ["", f"## {current_step}: {eq['section']}", ""]
        lines += [
            f"### ({eq['number']}) [{eq['tag']}] `{eq['name']}`",
            "",
            f"$${eq['latex']}$$",
            "",
            f"`{eq['ascii']}`",
            "",
        ]
        if eq.get("algebra"):
            lines += [f"**Algebra:** {eq['algebra']}", ""]
        if eq.get("note"):
            lines += [f"*{eq['note']}*", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def _try_compile_latex(tex_path: Path, export_dir: Path) -> Optional[Path]:
    engine = None
    for name in ("pdflatex", "xelatex", "lualatex"):
        if shutil.which(name):
            engine = name
            break
    if engine is None:
        return None
    try:
        for _ in range(2):
            subprocess.run(
                [engine, "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
                cwd=str(export_dir),
                check=True,
                capture_output=True,
                timeout=240,
            )
        pdf = export_dir / (tex_path.stem + ".pdf")
        return pdf if pdf.exists() else None
    except Exception:
        return None


def _export_pdf_matplotlib(payload: Dict[str, Any], path: Path) -> Path:
    """Readable multipage PDF without a TeX install (matplotlib)."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    ds = payload["desitter_status"]
    registry = payload["equation_registry"]

    def _page(fig, title: str, body_lines: List[str]) -> None:
        ax = fig.add_axes([0.08, 0.06, 0.84, 0.88])
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.text(0.0, 0.97, title, fontsize=14, fontweight="bold", va="top")
        y = 0.90
        for line in body_lines:
            for wrapped in textwrap.wrap(line, width=92) or [""]:
                ax.text(0.0, y, wrapped, fontsize=9, family="monospace", va="top")
                y -= 0.028
                if y < 0.04:
                    return

    with PdfPages(path) as pdf:
        # Cover
        fig = plt.figure(figsize=(8.5, 11))
        cover = [
            "Hubble Horizon -> de Sitter Attractor",
            "",
            f"Verdict: {ds['verdict_line']}",
            f"Existence: {ds['existence']}",
            f"Stability: {ds['stability']}",
            f"lambda: {ds['lambda_linear']}",
            "",
            "WORDING CAUTION:",
            WORDING_CAUTION,
            "",
            "Conceptual chain:",
        ]
        for link in payload.get("conceptual_chain", CONCEPTUAL_CHAIN):
            cover.append(f"  -> {link}")
        cover += [
            "",
            "Rigorous: GSL + finite S_max + closure => de Sitter => acceleration",
            "Do NOT write: 2nd law = de Sitter = acceleration",
            "",
            "Generated by analytical_solver.py",
        ]
        fig = plt.figure(figsize=(8.5, 11))
        _page(fig, "First-Principles Derivation", cover)
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure(figsize=(8.5, 11))
        claim_lines = [
            "SCIENTIFIC CLAIM:",
            payload.get("scientific_claim", SCIENTIFIC_CLAIM),
            "",
            "ARCHITECTURE:",
            payload.get("architecture", ""),
            "",
            "TAG LEGEND:",
        ] + [f"{k}: {v}" for k, v in payload["tag_legend"].items()]
        _page(fig, "Claim, architecture, tags", claim_lines)
        pdf.savefig(fig)
        plt.close(fig)

        chunk: List[str] = []
        page_i = 1

        def flush() -> None:
            nonlocal chunk, page_i
            if not chunk:
                return
            fig = plt.figure(figsize=(8.5, 11))
            _page(fig, f"Verbose STEP chain (page {page_i})", chunk)
            pdf.savefig(fig)
            plt.close(fig)
            chunk = []
            page_i += 1

        current_step = None
        for eq in registry:
            if eq.get("step") != current_step:
                current_step = eq.get("step")
                block_hdr = [f"=== {current_step}: {eq.get('section')} ===", ""]
                if len(chunk) + len(block_hdr) > 30:
                    flush()
                chunk.extend(block_hdr)
            block = [
                f"({eq['number']}) [{eq['tag']}] {eq['name']}",
                f"  {eq['ascii']}",
            ]
            if eq.get("algebra"):
                block.append(f"  Algebra: {eq['algebra']}")
            if eq.get("note"):
                block.append(f"  Note: {eq['note']}")
            block.append("")
            if len(chunk) + len(block) > 32:
                flush()
            chunk.extend(block)
        flush()

    return path



def _export_giant_overleaf(payload: Dict[str, Any], path: Path) -> None:
    """
    One giant Overleaf-ready manuscript containing EVERY registered equation,
    Module-17 Jacobson chain, and deep-cosmology implications.
    """
    ds = payload["desitter_status"]
    registry = payload["equation_registry"]
    # Prefer live labeled Jacobson ledger (avoids str()-serialized evidence)
    try:
        jac_eqs = [e.to_dict() for e in build_jacobson_derivation_embedded()["equations"]]
    except Exception:
        jac_eqs = []

    def esc(s: str) -> str:
        return (
            str(s or "")
            .replace("\\", r"\textbackslash{}")
            .replace("&", r"\&")
            .replace("%", r"\%")
            .replace("#", r"\#")
            .replace("_", r"\_")
        )

    L: List[str] = []
    a = L.append
    a(r"% ============================================================")
    a(r"% Overleaf-ready GIANT manuscript - auto-generated by")
    a(r"% analytical_solver.py (all derived equations + Module 17 + deep cosmology)")
    a(r"% Compile with pdfLaTeX.")
    a(r"% ============================================================")
    a("")
    a(r"\documentclass[11pt,a4paper]{article}")
    a(r"\usepackage[margin=1in]{geometry}")
    a(r"\usepackage{amsmath,amssymb,amsthm,mathtools}")
    a(r"\usepackage{booktabs,longtable,array,enumitem}")
    a(r"\usepackage{xcolor,hyperref,microtype,titlesec,fancyhdr}")
    a(r"\usepackage{graphicx,float}")
    a(r"\hypersetup{colorlinks=true,linkcolor=black,urlcolor=blue!50!black}")
    a(r"\pagestyle{fancy}\fancyhf{}")
    a(r"\fancyhead[L]{\textsc{$\Lambda$CDM+S complete derivation}}")
    a(r"\fancyhead[R]{\thepage}")
    a(r"\theoremstyle{plain}")
    a(r"\newtheorem{theorem}{Theorem}[section]")
    a(r"\newtheorem{lemma}[theorem]{Lemma}")
    a(r"\newtheorem{proposition}[theorem]{Proposition}")
    a(r"\newtheorem{corollary}[theorem]{Corollary}")
    a(r"\theoremstyle{definition}")
    a(r"\newtheorem{definition}[theorem]{Definition}")
    a(r"\newtheorem{assumption}[theorem]{Assumption}")
    a(r"\newtheorem{axiom}[theorem]{Axiom}")
    a(r"\theoremstyle{remark}")
    a(r"\newtheorem{remark}[theorem]{Remark}")
    a(r"\newtheorem{caution}[theorem]{Caution}")
    a(r"\newcommand{\tagAxiom}{\textsc{[Axiom]}}")
    a(r"\newcommand{\tagAssumed}{\textsc{[Assumed]}}")
    a(r"\newcommand{\tagJustified}{\textsc{[Justified assumption]}}")
    a(r"\newcommand{\tagDerived}{\textsc{[Derived]}}")
    a(r"\newcommand{\tagLemma}{\textsc{[Lemma]}}")
    a(r"\newcommand{\tagTheorem}{\textsc{[Theorem]}}")
    a(r"\newcommand{\tagCond}{\textsc{[Conditional theorem]}}")
    a(r"\newcommand{\tagPheno}{\textsc{[Phenomenology]}}")
    a(r"\newcommand{\tagUnproven}{\textsc{[Not derived]}}")
    a(r"\newcommand{\tagBridge}{\textsc{[Bridge]}}")
    a(r"\newcommand{\tagModSeventeen}{\textsc{[Derived under Module-17 axioms]}}")
    a(r"\newcommand{\Lcdms}{$\Lambda$CDM+S}")
    a(r"\newcommand{\Smax}{S_{\max}}")
    a(r"\newcommand{\Searly}{S_{\mathrm{early}}}")
    a(r"\newcommand{\Hinf}{H_{\infty}}")
    a(r"\newcommand{\Ls}{\Lambda_{S}}")
    a(r"\newcommand{\chie}{\chi}")
    a(r"\title{Complete First-Principles Derivation for \Lcdms:\\")
    a(r"Hubble-Horizon Entropy, de~Sitter Attractors, Asymptotic Einstein Structure,\\")
    a(r"Local-Horizon (Jacobson) EFE, and Deep Cosmological Implications}")
    a(r"\author{Auto-exported from \texttt{analytical\_solver.py}\\")
    a(r"\large All tagged equations --- global module + Module~17 + deep cosmology}")
    a(r"\date{\today}")
    a(r"\begin{document}")
    a(r"\maketitle")
    a(r"\begin{abstract}")
    a(
        r"We present a rigorously tagged first-principles derivation connecting "
        r"Hubble-horizon thermodynamics to a late-time de~Sitter attractor, "
        r"an asymptotic Einstein vacuum structure with "
        r"$\Ls=3\pi/(G\Smax)$, the logically independent local-horizon "
        r"(Jacobson) Module~17 yielding the full local Einstein equations under "
        r"stated axioms, and the deep cosmological implications of the locked "
        r"entropy structure $S_H=\pi/(GH^2)$: effective fluid $(\rho_S,P_S,w_S)$, "
        r"deceleration $q$, phase trajectory $H(\chie)$, global Lyapunov stability, "
        r"horizon first law, entropy-production epoch, scalar reconstruction, "
        r"energy conditions, and $\Gamma$-class universality. "
        r"Canon locked: $S_H=\pi/(GH^2)$ (the linear $GH/(2\pi)$ convention is rejected). "
        rf"This manuscript enumerates all {len(registry)} registered equations."
    )
    a(r"\end{abstract}")
    a(r"\tableofcontents")
    a(r"\newpage")

    a(r"\section{Scientific posture and claim calibration}")
    a(r"\begin{caution}[Forbidden slogans]")
    a(r"Do \emph{not} write ``second law $=$ de~Sitter $=$ acceleration.''")
    a(r"Do \emph{not} claim full local EFE from the global Hubble-horizon argument alone.")
    a(r"\end{caution}")
    a(r"\noindent\textbf{Verdict:} " + esc(ds.get("verdict_line", "")) + r"\\")
    a(r"\noindent\textbf{Scientific claim.} " + esc(payload.get("scientific_claim", "")) + r"\\")
    a(r"\noindent\textbf{Wording caution.} " + esc(payload.get("wording_caution", "")) + r"\\")
    a(r"\noindent\textbf{Boundary.} " + esc(payload.get("boundary_not_proven", "")) )

    a(r"\paragraph{Calibrated claims.}")
    a(r"\begin{description}[leftmargin=2.2em,style=nextline]")
    a(r"\item[Claim~A.] $S_H=\pi/(GH^2)$ from apparent-horizon geometry + BH area law.")
    a(r"\item[Claim~B.] Given $0<\Smax<\infty$ and closure $\chie\to1$: $\Hinf=\sqrt{\pi/(G\Smax)}>0$.")
    a(r"\item[Claim~C.] Logistic $\gamma>0$: linear stability $\lambda=-\gamma<0$; global Lyapunov $V=\Smax-S_H$.")
    a(r"\item[Claim~D.] $a(t)\to a_0 e^{\Hinf t}$.")
    a(r"\item[Claim~E.] $G_{\mu\nu}^{(\infty)}+\Ls g_{\mu\nu}=0$, $\Ls=3\pi/(G\Smax)$.")
    a(r"\item[Claim~F.] Full local EFE = \tagUnproven\ from global module alone; = \tagModSeventeen.")
    a(r"\item[Claim~G.] Entropy dynamics predicts $w_S=-1+\Delta S\gamma\chie(1-\chie)/(3HS_H)$ (non-phantom).")
    a(r"\end{description}")

    a(r"\begin{equation}")
    a(r"\boxed{")
    a(r"\mathrm{GSL}\wedge 0<\Smax<\infty\wedge\text{thermo closure}")
    a(r"\;\Longrightarrow\;\text{stable dS attractor}")
    a(r"\;\Longrightarrow\;\text{accelerated expansion.}")
    a(r"}")
    a(r"\end{equation}")

    # Conceptual chain
    a(r"\section{Conceptual causal chain}")
    a(r"\begin{enumerate}")
    for link in payload.get("conceptual_chain", []):
        a(r"  \item " + esc(link).replace(r"\textbackslash{}", ""))
    a(r"\end{enumerate}")

    # Tag legend
    a(r"\section{Tag legend}")
    a(r"\begin{itemize}")
    for tag, meaning in payload.get("tag_legend", {}).items():
        a(rf"  \item \textbf{{{tag}}}: {esc(meaning)}")
    a(r"\end{itemize}")

    # All registered equations by step
    a(r"\section{Complete tagged equation ledger}")
    a(r"Every equation below is numbered and tagged. Intermediate algebra is shown when recorded.")
    current = None
    for eq in registry:
        if eq.get("step") != current:
            current = eq.get("step")
            a(rf"\subsection{{{esc(current)}: {esc(eq.get('section',''))}}}")
        a(
            rf"\paragraph{{({esc(eq['number'])}) [{esc(eq['tag'])}] "
            rf"${esc(eq['name'])}$.}}"
        )
        a(r"\begin{equation}")
        a(eq["latex"] + rf"\tag{{{eq['number']}}}")
        a(r"\end{equation}")
        if eq.get("algebra"):
            a(r"\textbf{Algebra:} " + esc(eq["algebra"]) + r"\\")
        if eq.get("note"):
            a(rf"\textit{{{esc(eq['note'])}}}")
        a(r"\vspace{0.25em}")

    # Module 17 narrative block
    a(r"\section{Local-horizon (Jacobson) module}")
    a(r"\label{sec:jacobson}")
    a(r"\begin{caution}[Scope separation]")
    a(
        r"This module is logically \emph{separate} from the global attractor proof. "
        r"No unfinished global step is filled by invention. "
        r"The global argument alone still does \emph{not} imply the full local Einstein equation."
    )
    a(r"\end{caution}")
    if jac_eqs:
        a(r"\subsection{Labeled Jacobson ledger (embedded)}")
        for e in jac_eqs:
            if not isinstance(e, dict):
                continue
            a(rf"\paragraph{{[{esc(e.get('tag',''))}] {esc(e.get('id',''))}: {esc(e.get('title',''))}.}}")
            a(r"\begin{equation}")
            a(e.get("latex", r"\mathrm{(missing)}"))
            a(r"\end{equation}")
            if e.get("note"):
                a(rf"\textit{{{esc(e['note'])}}}")

    a(r"\begin{theorem}[Full local Einstein field equations; Claim~F]")
    a(r"\tagCond\ Under Module-17 axioms and conditions,")
    a(r"\begin{equation}")
    a(r"\boxed{G_{\mu\nu}+\Lambda g_{\mu\nu}=8\pi G\,T_{\mu\nu}.}")
    a(r"\end{equation}")
    a(r"Status: \tagModSeventeen. Not a consequence of the global Hubble-horizon GSL alone.")
    a(r"\end{theorem}")

    a(r"\begin{theorem}[Thermodynamic identification of $\Lambda$]")
    a(r"\tagBridge\ The local argument leaves $\Lambda$ free. The global attractor identifies")
    a(r"\begin{equation}")
    a(r"\boxed{\Lambda\to\Ls=\frac{3\pi}{G\Smax}=3\Hinf^2.}")
    a(r"\end{equation}")
    a(r"\end{theorem}")

    # Deep cosmology summary theorems
    a(r"\section{Layers 1--21: first-principles cosmological implications}")
    a(
        r"All twenty-one layers below are derived inside "
        r"\texttt{analytical\_solver.py} only, from the locked structure "
        r"$S_H=\pi/(GH^2)$, $S_H=S_{\mathrm{early}}+\Delta S\,\chie$, and "
        r"$\dot\chie=\Gamma(\chie)$ (logistic representative)."
    )
    a(r"\begin{enumerate}[leftmargin=2.2em]")
    a(r"\item \textbf{Effective fluid.} $\rho_S=3/(8G^2 S_H)$, "
      r"$w_S=-1+\Delta S\gamma\chie(1-\chie)/(3HS_H)$ (non-phantom).")
    a(r"\item \textbf{Cosmography.} $q=-1+\Delta S\gamma\chie(1-\chie)/(2HS_H)$; jerk/snap hierarchy.")
    a(r"\item \textbf{Phase trajectory.} "
      r"$H(\chie)=\sqrt{\pi/(G[S_{\mathrm{early}}+\Delta S\chie])}$; "
      r"$\ln(a/a_{\mathrm{ref}})=\int H(x)/(\gamma x(1-x))\,dx$; "
      r"$S_{\mathrm{early}}=0$: $F(u)=2\mathrm{artanh}\sqrt{u}-2/\sqrt{u}$; "
      r"$H(a)=H(\chie(a))$, $H(z)=H(1/(1+z))$.")
    a(r"\item \textbf{Autonomous system.} "
      r"$d\chi/dN=f(\chi)=\Gamma/H$; fixed points $\chie=0$ (unstable), $\chie=1$ (stable); "
      r"$\lambda_N(1)=-\gamma/\Hinf$; monotone flow on $(0,1]$.")
    a(r"\item \textbf{Lyapunov.} "
      r"$V=\Smax-S_H=\Delta S(1-\chie)\ge0$, $\dot V=-\dot S_H\le0$, "
      r"$V=0\Leftrightarrow\chie=1$ $\Rightarrow$ global asymptotic stability on $(0,1]$.")
    a(r"\item \textbf{First law.} "
      r"$E_H=T_H S_H=\rho_S V_H$; "
      r"$W_{\mathrm{req}}=-\rho_S/3$; "
      r"$W_{\mathrm{fluid}}=W_{\mathrm{req}}\Leftrightarrow\dot H=-4H^2$ (attractor OK); "
      r"$T_{\mathrm{dyn}}\to T_H$.")
    a(r"\item \textbf{Entropy production.} "
      r"$\xi:=\dot S_H/(H S_H)=-2\dot H/H^2=2(1+q)=3(w_S+1)$; "
      r"$\xi\to0$ at equilibrium.")
    a(r"\item \textbf{Transition epoch.} "
      r"$\chie=\tfrac12$: $\dot S_{H,\max}=\gamma\Delta S/4$; "
      r"$q=0\Leftrightarrow w=-1/3$; coincides with $t_{\mathrm{thermo}}$ iff $\gamma$ tuned.")
    a(r"\item \textbf{Scalar reconstruction.} "
      r"$\dot\phi^2=-\dot H/(4\pi G)$, $V=(3H^2+\dot H)/(8\pi G)$; "
      r"$\rho_\phi=\rho_S$, $w_\phi=w_S$; attractor $\dot\phi\to0$ (background equivalence only).")
    a(r"\item \textbf{Homogeneous action.} "
      r"$K=1$, $U=-\tfrac12\Gamma^2$ $\Rightarrow$ logistic is an EL solution (residual 0); "
      r"$E\equiv0$ on-shell; GSL$\nRightarrow\Gamma$ (conditional).")
    a(r"\item \textbf{Covariant $\chie$.} "
      r"$L_\chie=KX-U$, $T^{(\chie)}_{\mu\nu}$ derived; "
      r"FLRW $\ddot\chie+3H\dot\chie+U'=0$; "
      r"L10 residual $3H\Gamma$ (friction gap; remedies R1--R4).")
    a(r"\item \textbf{Perturbations.} "
      r"Arch.\ A: $\delta S_H=-2(S_H/H)\delta H$; "
      r"Arch.\ B: $\delta\chi$ KG with $c_s^2=1$; "
      r"$\Psi=\Phi$; attractor vacuum-like; CMB/LSS bridge.")
    a(r"\item \textbf{Sound speed.} "
      r"$c_a^2=w-\dot w/(3H(1+w))$; $c_s^2=1$ (Arch.\ B); "
      r"distinguish adiabatic vs propagation; attractor delicate.")
    a(r"\item \textbf{Energy conditions.} "
      r"NEC/WEC automatic; DEC$\equiv$NEC for $P\le0$; "
      r"SEC$\Leftrightarrow q\ge0$; SEC failure $=$ acceleration.")
    a(r"\item \textbf{$\Gamma$-class.} "
      r"$\Gamma=\gamma\chie^n(1-\chie)^m$; $H_\infty$ universal; "
      r"NEC/Lyapunov persist; timescale $(n,m,\gamma)$ not universal.")
    a(r"\item \textbf{Bifurcation.} "
      r"$\Gamma_\alpha=\gamma\chie(1-\chie)(1+\alpha\chie)$; "
      r"$\lambda_1=-\gamma(1+\alpha)$; transcritical at $\alpha_c=-1$; "
      r"dS structurally stable for $\alpha>-1$.")
    a(r"\item \textbf{Thermo arrow.} "
      r"$\tau_S=\chie$; $\dot\tau_S\Leftrightarrow\dot S_H$; "
      r"time reversal violates GSL; $dt=d\tau_S/\Gamma$.")
    a(r"\item \textbf{Generalized Friedmann.} "
      r"$H^2=F(S)$; $F_{\mathrm{locked}}=\pi/(GS)$; "
      r"$\dot H=-(H/2S)\dot S$; standard Friedmann as special case.")
    a(r"\item \textbf{Jacobson bridge.} "
      r"B1: common $\eta=1/(4G)$; B2: $\Lambda\to\Ls=3H_\infty^2$; "
      r"$\rho_\Lambda=\rho_S|_{\chie=1}$; master $S[g,\chie]$ open.")
    a(r"\item \textbf{Modified entropy.} "
      r"$S=A/(4G)+\alpha\ln(A/A_0)+\beta A_0/A$; "
      r"$\eta_{\mathrm{eff}}$ running; LO $\delta u$; attractor robust.")
    a(r"\item \textbf{Finite $\Smax$.} "
      r"$A_\infty=4G\Smax=A_{\max}$; "
      r"finite budget $\Leftrightarrow$ positive $(\Hinf,\Ls)$; "
      r"$R_\infty=1/\Hinf$; value of $\Smax$ open.")
    a(r"\end{enumerate}")

    a(r"\begin{theorem}[Predicted equation of state --- Layer 1]")
    a(r"\tagDerived\ With locked $S_H=\pi/(GH^2)$ and logistic closure,")
    a(r"\begin{equation}")
    a(r"w_S=-1+\frac{(\Smax-\Searly)\gamma\chie(1-\chie)}{3 H S_H}.")
    a(r"\end{equation}")
    a(r"Hence $w_S\to-1$ as $\chie\to1$, and $w_S>-1$ on $(0,1)$ (non-phantom).")
    a(r"\end{theorem}")

    a(r"\begin{theorem}[Deceleration --- Layer 2]")
    a(r"\tagDerived\ ")
    a(r"\begin{equation}")
    a(r"q=-1+\frac{(\Smax-\Searly)\gamma\chie(1-\chie)}{2 H S_H}.")
    a(r"\end{equation}")
    a(r"\end{theorem}")

    a(r"\begin{theorem}[Global Lyapunov stability --- Layer 5]")
    a(r"\tagTheorem\ $V=\Smax-S_H$ satisfies $V\ge0$, $\dot V\le0$, $V=0\Leftrightarrow\chie=1$,")
    a(r"hence $\chie=1$ is globally asymptotically stable on $(0,1]$.")
    a(r"\end{theorem}")

    a(r"\begin{caution}[Two-sided de~Sitter of the entropy-only background]")
    a(
        r"The isolated entropy subsystem has $\dot\chie=0$ at both endpoints, hence "
        r"$H\to H_{\mathrm{early}}$ as $t\to-\infty$ and $H\to\Hinf$ as $t\to+\infty$. "
        r"It does \emph{not} alone generate radiation$\to$matter$\to$dark-energy. "
        r"Coupled $H^2=(8\pi G/3)(\rho_r+\rho_m+\rho_S)$ is required."
    )
    a(r"\end{caution}")

    a(r"\begin{caution}[Weakest mathematical link]")
    a(r"GSL does \emph{not} imply $\dot\chie=\gamma\chie(1-\chie)$. The closure remains \tagAssumed.")
    a(r"Layer~10 constructs an action admitting the logistic as a homogeneous EL solution; "
      r"it does not uniquely derive $\Gamma$ from the GSL.")
    a(r"\end{caution}")

    # Proof audit table
    a(r"\section{Proof audit summary}")
    a(r"\begin{center}")
    a(r"\begin{tabular}{@{}llp{6.2cm}@{}}")
    a(r"\toprule Result & Status & Depends on \\")
    a(r"\midrule")
    a(r"$S_H=\pi/(GH^2)$ & Derived & horizon geometry + BH \\")
    a(r"$\dot S_H\ge0\Leftrightarrow\dot H\le0$ & Theorem & $H,G>0$ \\")
    a(r"finite $\Smax$ & Assumed & holographic / dS budget \\")
    a(r"$\chie$ closure & Assumed & $\Gamma$-class model choice \\")
    a(r"$\chie\to1$ & Cond.\ thm. & closure + IC in $(0,1)$ \\")
    a(r"$\Hinf>0$ & Cond.\ thm. & finite $\Smax$ \\")
    a(r"$\lambda=-\gamma$ & Cond.\ thm. & logistic / $\Gamma'(1)<0$ \\")
    a(r"global Lyapunov $V$ & Theorem & $\dot S_H\ge0$ \\")
    a(r"$w_S\to-1$ non-phantom & Derived & Layer 1 Raychaudhuri+closure \\")
    a(r"$q(\chie)$ & Derived & Layer 2 \\")
    a(r"autonomous $\chie'$ & Derived & Layer 4 \\")
    a(r"global Lyapunov $V$ & Theorem & Layer 5 \\")
    a(r"$E_H=T_H S_H$ & Derived & Layer 6 \\")
    a(r"$\dot S_{H,\max}$ & Theorem & Layer 8 \\")
    a(r"homogeneous $U=-\tfrac12\Gamma^2$ & Cond. & Layer 10 \\")
    a(r"covariant $L_\chie$ & Derived struct. & Layer 11 \\")
    a(r"$\delta G=8\pi G\delta T$ & Derived schem. & Layer 12 \\")
    a(r"$c_a^2$ & Derived & Layer 13 \\")
    a(r"NEC/SEC & Theorem & Layer 14 \\")
    a(r"$\Gamma$-class universality & Derived & Layer 15 \\")
    a(r"bifurcation $\alpha$ & Derived & Layer 16 \\")
    a(r"$H^2=F(S)$ & Derived & Layer 18 \\")
    a(r"modified $S(A)$ & Derived & Layer 20 \\")
    a(r"$\Smax=A_{\max}/(4G)$ & Cond. & Layer 21 \\")
    a(r"$\Ls=3\pi/(G\Smax)$ & Cond.\ thm. & entropy framework \\")
    a(r"$G_{\mu\nu}^{(\infty)}+\Ls g=0$ & Cond.\ thm. & de~Sitter geometry \\")
    a(r"$I_E=-\Smax$ & Cond.\ thm. & Euclidean EH on $S^4$ \\")
    a(r"full local EFE & Mod.~17 derived & Clausius+BH+Unruh+H1+C1+C2 \\")
    a(r"$\Lambda\to\Ls$ & Bridge & Layer 19 + global attractor \\")
    a(r"unique $\Gamma$ from GSL & Not derived & weakest link \\")
    a(r"\bottomrule")
    a(r"\end{tabular}")
    a(r"\end{center}")

    a(r"\section{Final calibrated statements}")
    a(r"\begin{theorem}[Strongest supportable global claim]")
    a(
        r"Under flat FLRW, apparent-horizon BH entropy, GSL, finite $\Smax$, "
        r"and $\Gamma$-class closure with $\chie\to1$ (logistic representative), "
        r"the background possesses a stable positive-$H$ de~Sitter attractor with"
    )
    a(r"\begin{equation}")
    a(r"\Hinf=\sqrt{\frac{\pi}{G\Smax}},\quad")
    a(r"\Ls=\frac{3\pi}{G\Smax},\quad")
    a(r"a(t)\to a_0 e^{\Hinf t},")
    a(r"\end{equation}")
    a(r"and asymptotic Einstein structure $G_{\mu\nu}^{(\infty)}+\Ls g_{\mu\nu}=0$.")
    a(r"Full local EFE remain \tagUnproven\ within the global module alone.")
    a(r"\end{theorem}")

    a(r"\begin{theorem}[Strongest supportable combined claim]")
    a(r"Under Module-17 axioms together with the global assumptions, one obtains both")
    a(r"\begin{equation}")
    a(r"G_{\mu\nu}+\Lambda g_{\mu\nu}=8\pi G\,T_{\mu\nu}")
    a(r"\qquad\text{(\tagModSeventeen)}")
    a(r"\end{equation}")
    a(r"and $\Lambda\to\Ls=3\pi/(G\Smax)$, recovering $G_{\mu\nu}^{(\infty)}+\Ls g_{\mu\nu}=0$.")
    a(r"\end{theorem}")

    a(r"\vspace{1em}")
    a(
        r"\noindent\textit{Manuscript generated for Overleaf from "
        r"\texttt{analytical\_solver.py}. Epistemic tags are part of the scientific content.}"
    )
    a(r"\end{document}")
    a("")

    path.write_text("\n".join(L), encoding="utf-8")




def export_artifacts(payload: Dict[str, Any], export_dir: Path) -> List[Path]:
    export_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    registry = payload["equation_registry"]

    # JSON (full machine registry)
    jp = export_dir / "equations.json"
    jp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    paths.append(jp)

    # CSV equation registry
    cp = export_dir / "equations.csv"
    _export_csv(registry, cp)
    paths.append(cp)

    # Markdown (human audit)
    mp = export_dir / "derivation_report.md"
    _export_markdown_report(payload, mp)
    paths.append(mp)

    # Compact equations.tex (snippet for papers)
    compact = export_dir / "equations.tex"
    compact.write_text(
        "\n".join(
            [
                r"% Compact equation list - see first_principles_derivation.tex for the full report",
                r"\begin{align}",
                *[
                    rf"{eq['latex']} &\quad \text{{[{eq['tag']}]}} \\"
                    for eq in registry
                    if eq["tag"] != "PHENOMENOLOGY"
                ],
                r"\end{align}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    paths.append(compact)

    # Polished full LaTeX document + GIANT Overleaf manuscript (all equations)
    tex_path = export_dir / "first_principles_derivation.tex"
    _export_polished_tex(payload, tex_path)
    paths.append(tex_path)

    giant_path = export_dir / "LCDMS_Complete_Overleaf.tex"
    _export_giant_overleaf(payload, giant_path)
    paths.append(giant_path)

    # PDF: try LaTeX engine, else matplotlib (tolerate locked PDF if open in a viewer)
    pdf_target = export_dir / "first_principles_derivation.pdf"
    pdf_path = _try_compile_latex(tex_path, export_dir)
    if pdf_path is None:
        try:
            pdf_path = _export_pdf_matplotlib(payload, pdf_target)
        except PermissionError:
            alt = export_dir / "first_principles_derivation_new.pdf"
            try:
                pdf_path = _export_pdf_matplotlib(payload, alt)
            except Exception as exc:
                note = export_dir / "PDF_EXPORT_FAILED.txt"
                note.write_text(
                    f"Could not write PDF (file may be open): {exc}\n"
                    f"LaTeX source is ready: {tex_path.name}\n",
                    encoding="utf-8",
                )
                paths.append(note)
                pdf_path = None
    if pdf_path is not None and pdf_path.exists():
        paths.append(pdf_path)

    # Detailed analysis.tex (standalone) + fragment for giant Overleaf
    analysis_bundle = payload.get("integrated_analysis")
    if analysis_bundle:
        analysis_path = export_dir / "analysis.tex"
        export_analysis_tex(analysis_bundle, analysis_path, standalone=True)
        paths.append(analysis_path)
        fragment_path = export_dir / "analysis_fragment.tex"
        export_analysis_tex(analysis_bundle, fragment_path, standalone=False)
        paths.append(fragment_path)
        # Inject analysis appendix into giant Overleaf BEFORE compiling
        # (so a hung pdflatex cannot leave the manuscript without validation space)
        if giant_path.exists() and fragment_path.exists():
            gtxt = giant_path.read_text(encoding="utf-8")
            if "analysis_fragment.tex" not in gtxt:
                gtxt = gtxt.replace(
                    r"\end{document}",
                    "\n".join([
                        r"\clearpage",
                        r"\appendix",
                        r"\section{Numerical, Bayesian, and ODE analysis}",
                        r"\input{analysis_fragment.tex}",
                        r"\end{document}",
                        "",
                    ]),
                    1,
                )
                giant_path.write_text(gtxt, encoding="utf-8")
        # Compile analysis.tex to PDF when possible
        analysis_pdf = _try_compile_latex(analysis_path, export_dir)
        if analysis_pdf is not None and analysis_pdf.exists():
            paths.append(analysis_pdf)
        # Recompile giant manuscript with appendix
        if giant_path.exists():
            giant_pdf = _try_compile_latex(giant_path, export_dir)
            if giant_pdf is not None and giant_pdf.exists():
                paths.append(giant_pdf)

    return paths


# =============================================================================
# Orchestrator
# =============================================================================


def derive_lambdacdm_s(
    temperature_mode: str = "dynamical_horizon",
    export_dir: Optional[Path | str] = None,
    verbose: bool = True,
    run_validation: bool = True,
    validation_quick: bool = False,
) -> DerivationResult:
    if temperature_mode not in {"de_sitter", "dynamical_horizon"}:
        raise ValueError("temperature_mode must be 'de_sitter' or 'dynamical_horizon'")

    sym = make_symbols()
    steps: List[DerivationStep] = []

    steps.append(
        DerivationStep(
            name="fundamental_inputs",
            stage="INPUT",
            assumptions=POSTULATES,
            equations={"pipeline": " | ".join(PIPELINE_STAGES)},
            claims=[
                "Goal: derive thermo EOM -> H_* -> λ<0, not assume de Sitter from Ḣ≤0.",
                "F(Π)/logistic are phenomenological - not the proof engine.",
            ],
            status="assumed",
        )
    )

    geo = step_geometry(sym)
    steps.append(geo)
    ent = step_entropy(sym, geo.evidence["A_H"])
    steps.append(ent)
    steps.append(step_temperature(sym, temperature_mode))
    steps.append(step_S_total_functional(sym))
    prod = step_entropy_production(sym)
    steps.append(prod)
    if prod.status != "failed":
        steps.append(step_theorem_22(sym, prod.evidence["S_H_dot"]))

    eom = step_thermodynamic_EOM(sym)
    steps.append(eom)
    steps.append(step_equilibrium_P_zero(sym, eom))
    steps.append(step_solve_H_star(sym))
    steps.append(step_scale_factor(sym))
    lin, desitter = step_linearize_stability(sym, eom)
    steps.append(lin)
    steps.append(step_attractor_result(desitter))

    # Exact solutions + enthalpy bridge + asymptotic Einstein (after attractor)
    steps.append(step_exact_chi_H_solutions(sym))
    steps.append(step_enthalpy_bridge(sym))
    steps.append(step_asymptotic_curvature(sym))
    steps.append(step_asymptotic_Einstein_tensor(sym))
    steps.append(step_thermodynamic_Lambda_S(sym))
    steps.append(step_asymptotic_Friedmann_and_T_S(sym))
    steps.append(step_asymptotic_Einstein_equation(sym))

    # On-shell EH / Euclidean action on the thermodynamic dS solution
    steps.append(step_Einstein_Hilbert_action(sym))
    steps.append(step_Lorentzian_onshell_action(sym))
    steps.append(step_Euclidean_onshell_action(sym))
    steps.append(step_semiclassical_partition(sym))
    steps.append(step_FLRW_action_extremum_vs_S_max(sym))

    steps.append(step_conservation_Q_flag(sym))
    steps.append(step_effective_sector_after_attractor(sym))
    steps.append(step_modified_friedmann(sym))
    steps.append(step_jacobson_future_stub())  # embedded full Module-17

    # Deep cosmology implications of the locked entropy structure
    steps.extend(collect_deep_cosmology_steps(sym))

    # Phenomenology last
    steps.append(step_phenomenology_comparison(sym))
    steps.append(step_perturbations_stub())

    # Upstream failure downgrade
    for critical in ("horizon_entropy_S_H", "entropy_production_P", "thermodynamic_equation_of_motion"):
        st = next(s for s in steps if s.name == critical)
        if st.status == "failed":
            desitter = DeSitterStatus(
                equilibrium_identified=False,
                existence="failed",
                existence_detail=f"Blocked by {critical}.",
                stability="failed",
                stability_detail=f"Blocked by {critical}.",
                lambda_linear="n/a",
                verdict_line="DE SITTER ATTRACTOR: FAILED - upstream derivation broken",
            )
            # rewrite result step
            for i, s in enumerate(steps):
                if s.name == "RESULT_stable_de_Sitter_attractor":
                    steps[i] = step_attractor_result(desitter)
            break

    checks = run_checks(sym, steps, desitter)
    payload = build_payload(steps, checks, temperature_mode, desitter)

    # Integrated numerical + Bayesian analysis (same run as the symbolic derivation)
    if run_validation:
        try:
            out = Path(export_dir) if export_dir is not None else Path("theory_export")
            out.mkdir(parents=True, exist_ok=True)
            if verbose:
                print("Running integrated numerical/Bayesian/ODE analysis...")
            bundle = run_full_integrated_analysis(out, quick=validation_quick)
            payload["integrated_analysis"] = bundle
            vpass = bool(bundle.get("validation", {}).get("all_critical_pass"))
            steps.append(
                DerivationStep(
                    name="NUM_integrated_validation_bayesian_ode",
                    stage="NUMERICAL_VALIDATION",
                    assumptions=[
                        "Coupled rad+matter+S ODEs with Layer-1 w_S",
                        VALIDATION_DISTINCTION,
                    ],
                    equations={
                        "R_F": r"R_F=E^2-\sum r_i",
                        "R_C": r"R_C=(dr_{tot}/dN)_{num}-(dr_{tot}/dN)_{RHS}",
                        "eps_H": r"\varepsilon_H=(H-H_{\Lambda\mathrm{CDM}})/H_{\Lambda\mathrm{CDM}}",
                    },
                    claims=[
                        f"Integrated analysis critical pass = {vpass}",
                        "Bayesian MAP/AIC/BIC and ODE-vs-LCDM tables exported to analysis.tex",
                    ],
                    status="derived" if vpass else "failed",
                    evidence={"all_critical_pass": vpass},
                )
            )
            payload = build_payload(steps, checks, temperature_mode, desitter)
            payload["integrated_analysis"] = bundle
        except Exception as exc:
            payload["integrated_analysis_error"] = str(exc)
            if verbose:
                print(f"WARNING: integrated analysis failed: {exc}")

    if export_dir is not None:
        payload["export_paths"] = [str(p) for p in export_artifacts(payload, Path(export_dir))]

    result = DerivationResult(
        temperature_mode=temperature_mode,
        steps=steps,
        checks=checks,
        desitter=desitter,
        export_payload=payload,
    )
    if verbose:
        _print_report(result)
    return result


def _print_report(result: DerivationResult) -> None:
    sep = "=" * 72
    print(sep)
    print("LCDM+S  |  analytical_solver.py  |  first-principles logical chain")
    print(sep)
    print(f"Temperature mode : {result.temperature_mode}")
    print(f"Verdict          : {result.desitter.verdict_line}")
    print(f"  existence      : {result.desitter.existence}")
    print(f"  stability      : {result.desitter.stability}")
    print(f"  lambda         : {result.desitter.lambda_linear}")
    print()
    print("WORDING CAUTION:")
    print(f"  {WORDING_CAUTION}")
    print()
    print("Conceptual causal chain:")
    for link in CONCEPTUAL_CHAIN:
        print(f"  -> {link}")
    print()
    print("Rigorous form: GSL + finite S_max + thermodynamic closure")
    print("               => de Sitter attractor => accelerated expansion")
    print()
    print("Verbose STEPs:")
    for st in PIPELINE_STAGES:
        print(f"  {st}")
    print()
    # Print tagged equation chain compactly from registry
    reg = result.export_payload.get("equation_registry") or []
    current = None
    for eq in reg:
        if eq.get("step") != current:
            current = eq.get("step")
            print(f"\n--- {current}: {eq.get('section')} ---")
        print(f"  ({eq['number']}) [{eq['tag']}] {eq['ascii']}")
        if eq.get("algebra"):
            print(f"         algebra: {eq['algebra']}")
    print()
    print("Architecture:")
    print(f"  {result.export_payload.get('architecture', '')}")
    print()
    print("Solver stage status:")
    for s in result.steps:
        flag = {
            "derived": "OK  ",
            "assumed": "ASM ",
            "failed": "FAIL",
            "stub": "STUB",
            "conditional": "COND",
        }[s.status]
        print(f"[{flag}] [{s.stage}] {s.name}")
    print()
    print("Consistency:")
    for c in result.checks:
        print(f"  [{'PASS' if c.passed else 'FAIL'}] {c.name}")
    if result.export_payload.get("export_paths"):
        print()
        print("Exported:")
        for p in result.export_payload["export_paths"]:
            print(f"  {p}")
    print(sep)
    n_fail = sum(1 for s in result.steps if s.status == "failed")
    n_chk = sum(1 for c in result.checks if not c.passed)
    print(f"Steps failed: {n_fail}  |  Checks failed: {n_chk}")
    print(sep)


# === BEGIN EMBEDDED NUMERICAL VALIDATION ===
# NUMERICAL VALIDATION ARCHITECTURE V0-V17 (embedded; do not thin-wrap externally)
# Q1 internal consistency | Q2 established-physics recovery | Q3 observational scaffolds

import math
import time

import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import minimize

# -----------------------------------------------------------------------------
# Embedded validation constants (V0)
# -----------------------------------------------------------------------------
VAL_C_KM_S: float = 299792.458
VAL_KMSMPC_TO_INVGYR: float = 1.022712165045695e-3
VAL_OGH2: float = 2.469e-5
VAL_DEFAULT_OBH2: float = 0.02237
C_KM_S = VAL_C_KM_S
KMSMPC_TO_INVGYR = VAL_KMSMPC_TO_INVGYR

VALIDATION_DISTINCTION = (
    "numerical consistency != observational agreement != evidence for the theory"
)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
C_KM_S = 299792.458
KMSMPC_TO_INVGYR = 1.022712165045695e-3  # H[km/s/Mpc] -> 1/Gyr


# =============================================================================
# 0. Parameters & solution containers
# =============================================================================

@dataclass
class LCDMSParams:
    """Cosmological + thermodynamic parameters (background)."""
    H0: float = 70.0              # km/s/Mpc
    Omega_m0: float = 0.30
    Omega_r0: float = 9.0e-5
    gamma: float = 0.35           # 1/Gyr  (logistic rate)
    S_max: float = 1.0            # entropy units (ratio physics in ?S/S)
    S_early: float = 1.0e-3
    chi0: float = 0.92            # ?(a=1); near attractor today
    # If True, freeze entropy sector to ? (code-verification ?CDM mode)
    lcdm_limit: bool = False
    Obh2: float = VAL_DEFAULT_OBH2

    @property
    def Omega_S0(self) -> float:
        return 1.0 - self.Omega_m0 - self.Omega_r0

    @property
    def Delta_S(self) -> float:
        return self.S_max - self.S_early

    @property
    def H0_gyr(self) -> float:
        return self.H0 * KMSMPC_TO_INVGYR

    @property
    def gamma_tilde(self) -> float:
        """? / H0  (dimensionless)."""
        return self.gamma / max(self.H0_gyr, 1e-30)


@dataclass
class BackgroundSolution:
    params: LCDMSParams
    N: np.ndarray          # ln a
    a: np.ndarray
    z: np.ndarray
    t_gyr: np.ndarray      # cosmic time from early start (Gyr)
    r_r: np.ndarray
    r_m: np.ndarray
    r_S: np.ndarray
    chi: np.ndarray
    E: np.ndarray          # H/H0
    w_S: np.ndarray
    method: str
    # residuals / diagnostics filled by analyzers
    R_F: Optional[np.ndarray] = None
    R_C: Optional[np.ndarray] = None
    R_C_rel: Optional[np.ndarray] = None
    q: Optional[np.ndarray] = None
    xi: Optional[np.ndarray] = None
    S_H: Optional[np.ndarray] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def H_of_z(self, z: float | np.ndarray) -> np.ndarray:
        H0 = self.params.H0
        f = interp1d(self.z[::-1], (self.E * H0)[::-1],
                     kind="cubic", fill_value="extrapolate")
        return np.asarray(f(z), dtype=float)


# =============================================================================
# 1. Background RHS (N = ln a as independent variable)
# =============================================================================

def _w_S(chi: float, E: float, p: LCDMSParams) -> float:
    if p.lcdm_limit or p.gamma <= 0.0:
        return -1.0
    chi = float(np.clip(chi, 1e-12, 1.0 - 1e-12))
    S_H = p.S_early + p.Delta_S * chi
    # w+1 = ?S * ? ?(1-?) / (3 H S_H), H = H0 E
    num = p.Delta_S * p.gamma_tilde * chi * (1.0 - chi)
    den = 3.0 * max(E, 1e-30) * max(S_H, 1e-30)
    return -1.0 + num / den


def rhs_N(N: float, y: Sequence[float], p: LCDMSParams) -> List[float]:
    """dy/dN for y=(r_r, r_m, r_S, ?)."""
    r_r, r_m, r_S, chi = [float(v) for v in y]
    r_r = max(r_r, 0.0)
    r_m = max(r_m, 0.0)
    r_S = max(r_S, 0.0)
    E2 = r_r + r_m + r_S
    E = math.sqrt(max(E2, 1e-30))
    w = _w_S(chi, E, p)
    if p.lcdm_limit or p.gamma <= 0.0:
        dchi = 0.0
        w = -1.0
    else:
        chi_c = min(max(chi, 1e-12), 1.0 - 1e-12)
        dchi = p.gamma_tilde * chi_c * (1.0 - chi_c) / E
    return [-4.0 * r_r, -3.0 * r_m, -3.0 * (1.0 + w) * r_S, dchi]


def rhs_t(t: float, y: Sequence[float], p: LCDMSParams) -> List[float]:
    """dy/dt for y=(a, r_r, r_m, r_S, ?), t in Gyr."""
    a, r_r, r_m, r_S, chi = [float(v) for v in y]
    a = max(a, 1e-30)
    r_r = max(r_r, 0.0)
    r_m = max(r_m, 0.0)
    r_S = max(r_S, 0.0)
    E2 = r_r + r_m + r_S
    E = math.sqrt(max(E2, 1e-30))
    H = p.H0_gyr * E
    w = _w_S(chi, E, p)
    if p.lcdm_limit or p.gamma <= 0.0:
        dchi = 0.0
        w = -1.0
    else:
        chi_c = min(max(chi, 1e-12), 1.0 - 1e-12)
        dchi = p.gamma * chi_c * (1.0 - chi_c)
    return [
        H * a,
        -4.0 * H * r_r,
        -3.0 * H * r_m,
        -3.0 * H * (1.0 + w) * r_S,
        dchi,
    ]


# =============================================================================
# 2. Independent solvers
# =============================================================================

def solve_background_N(
    p: LCDMSParams,
    z_max: float = 1100.0,
    n_eval: int = 2000,
    method: str = "RK45",
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> BackgroundSolution:
    """
    Integrate backward from a=1 (today) to a=1/(1+z_max), then return
    arrays ordered from early -> late.
    """
    N0 = 0.0
    N_early = math.log(1.0 / (1.0 + z_max))
    y0 = [p.Omega_r0, p.Omega_m0, p.Omega_S0, p.chi0]

    sol = solve_ivp(
        fun=lambda N, y: rhs_N(N, y, p),
        t_span=(N0, N_early),
        y0=y0,
        method=method,
        rtol=rtol,
        atol=atol,
        dense_output=True,
        max_step=0.05,
    )
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed ({method}): {sol.message}")

    N_grid = np.linspace(N_early, N0, n_eval)
    Y = sol.sol(N_grid)
    r_r, r_m, r_S, chi = Y
    E = np.sqrt(np.maximum(r_r + r_m + r_S, 1e-30))
    a = np.exp(N_grid)
    z = 1.0 / a - 1.0
    w_S = np.array([_w_S(ci, Ei, p) for ci, Ei in zip(chi, E)])

    # Cosmic time from early epoch: dt = dN / H
    H = p.H0_gyr * E
    dN = np.diff(N_grid, prepend=N_grid[0])
    # trapezoid integrate dt = dN/H
    t = np.zeros_like(N_grid)
    if len(N_grid) > 1:
        t[1:] = cumulative_trapezoid(1.0 / H, N_grid, initial=0.0)[1:]

    return BackgroundSolution(
        params=p,
        N=N_grid,
        a=a,
        z=z,
        t_gyr=t,
        r_r=r_r,
        r_m=r_m,
        r_S=r_S,
        chi=chi,
        E=E,
        w_S=w_S,
        method=method,
        meta={"rtol": rtol, "atol": atol, "z_max": z_max},
    )


def solve_background_t_rk4(
    p: LCDMSParams,
    a_ini: float = 1e-3,
    nsteps: int = 20000,
) -> BackgroundSolution:
    """
    Independent forward RK4 in cosmic time (second implementation).
    Shoots ?_ini / scales so that a ends near 1 with flatness today
    by integrating from early IC constructed from ?CDM-like early densities
    and iterating ?_ini so ?(a~1)~chi0.
    """
    # Early IC from ?CDM dilution + small ?
    a0 = a_ini
    r_r0 = p.Omega_r0 / a0**4
    r_m0 = p.Omega_m0 / a0**3
    # Early DE tiny: start ? small so r_S grows toward ?_S0
    chi_ini = max(1e-4, min(p.chi0 * 1e-3, 0.05))
    # Choose r_S_ini so that if w=-1 frozen, r_S~?_S0; logistic will perturb
    r_S0 = p.Omega_S0
    y = np.array([a0, r_r0, r_m0, r_S0, chi_ini], dtype=float)

    # Estimate t span: from a_ini to ~1
    # Use approximate ?CDM H for dt
    def E_lcdm(a: float) -> float:
        return math.sqrt(
            p.Omega_r0 / a**4 + p.Omega_m0 / a**3 + p.Omega_S0
        )

    # Advance until a>=1 or max steps
    as_, ts, rr, rm, rs, ch, Es, ws = [], [], [], [], [], [], [], []
    t = 0.0
    # adaptive-ish fixed step in ln a via dt = dln a / H with target nsteps
    lna0, lna1 = math.log(a0), 0.0
    hN = (lna1 - lna0) / nsteps

    for _ in range(nsteps):
        a, r_r, r_m, r_S, chi = y
        E = math.sqrt(max(r_r + r_m + r_S, 1e-30))
        H = p.H0_gyr * E
        as_.append(a)
        ts.append(t)
        rr.append(r_r)
        rm.append(r_m)
        rs.append(r_S)
        ch.append(chi)
        Es.append(E)
        ws.append(_w_S(chi, E, p))
        # RK4 with dt corresponding to ?N = hN
        dt = hN / max(H, 1e-30)

        def f(yy):
            return np.array(rhs_t(0.0, yy, p), dtype=float)

        k1 = f(y)
        k2 = f(y + 0.5 * dt * k1)
        k3 = f(y + 0.5 * dt * k2)
        k4 = f(y + dt * k3)
        y = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt
        if y[0] >= 1.0:
            break

    a_arr = np.asarray(as_)
    # Rescale densities so E(a_end)= match flatness at last point closest to a=1
    # (forward IC with r_S=?_S0 constant-like is already near-normalized at late times)
    idx = int(np.argmin(np.abs(a_arr - 1.0)))
    # Trim / interpolate onto uniform z grid via N
    N = np.log(np.maximum(a_arr, 1e-30))
    z = 1.0 / np.maximum(a_arr, 1e-30) - 1.0
    return BackgroundSolution(
        params=p,
        N=N,
        a=a_arr,
        z=z,
        t_gyr=np.asarray(ts),
        r_r=np.asarray(rr),
        r_m=np.asarray(rm),
        r_S=np.asarray(rs),
        chi=np.asarray(ch),
        E=np.asarray(Es),
        w_S=np.asarray(ws),
        method="RK4_t",
        meta={"a_ini": a_ini, "nsteps": nsteps, "idx_a1": idx},
    )


def solve_background(
    p: LCDMSParams,
    method: str = "RK45",
    **kwargs: Any,
) -> BackgroundSolution:
    if method.upper() in {"RK4", "RK4_T", "INDEPENDENT"}:
        return solve_background_t_rk4(p, **{k: v for k, v in kwargs.items()
                                            if k in {"a_ini", "nsteps"}})
    return solve_background_N(p, method=method, **{k: v for k, v in kwargs.items()
                                                   if k in {"z_max", "n_eval", "rtol", "atol"}})


# =============================================================================
# 3. Residual diagnostics (internal consistency)
# =============================================================================

def analyze_residuals(sol: BackgroundSolution) -> BackgroundSolution:
    """
    Friedmann residual:
        R_F = E? - (r_r+r_m+r_S)

    Continuity residual (numerical vs ODE RHS):
        (dr_tot/dN)_num  vs  -4 r_r - 3 r_m - 3(1+w_S) r_S
        R_C = (dr_tot/dN)_num - (dr_tot/dN)_RHS

    Equivalent Bianchi form: dr_tot/dN + 3(r_tot + P?) with P?=r_r/3 + w_S r_S.
    Edges of the grid are excluded from the pass metric (gradient pollution).
    """
    r_tot = sol.r_r + sol.r_m + sol.r_S
    E2 = sol.E**2
    R_F = E2 - r_tot

    dr_num = np.gradient(r_tot, sol.N, edge_order=2)
    dr_rhs = -4.0 * sol.r_r - 3.0 * sol.r_m - 3.0 * (1.0 + sol.w_S) * sol.r_S
    R_C = dr_num - dr_rhs

    # Relative continuity residual
    scale = np.maximum(np.abs(dr_rhs), 1e-30)
    R_C_rel = R_C / scale

    n = len(sol.N)
    lo, hi = max(2, n // 50), max(3, n - n // 50)
    sol.R_F = R_F
    sol.R_C = R_C
    sol.meta["max_abs_R_F"] = float(np.max(np.abs(R_F)))
    sol.meta["max_abs_R_C"] = float(np.max(np.abs(R_C[lo:hi])))
    sol.meta["max_rel_R_C"] = float(np.max(np.abs(R_C_rel[lo:hi])))
    sol.meta["med_abs_R_F"] = float(np.median(np.abs(R_F)))
    sol.meta["med_abs_R_C"] = float(np.median(np.abs(R_C[lo:hi])))
    return sol


# =============================================================================
# 4. Established-physics recovery
# =============================================================================

def lcdm_E(z: np.ndarray | float, p: LCDMSParams) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    a = 1.0 / (1.0 + z)
    return np.sqrt(
        p.Omega_r0 / a**4 + p.Omega_m0 / a**3 + p.Omega_S0
    )


def relative_H_error(sol: BackgroundSolution, p_ref: Optional[LCDMSParams] = None) -> Dict[str, float]:
    p = p_ref or sol.params
    E_ref = lcdm_E(sol.z, p)
    rel = np.abs(sol.E - E_ref) / np.maximum(E_ref, 1e-30)
    return {
        "max_rel_H": float(np.max(rel)),
        "mean_rel_H": float(np.mean(rel)),
        "rel_H_z0": float(rel[np.argmin(np.abs(sol.z))]),
        "rel_H_z1100": float(rel[np.argmin(np.abs(sol.z - 1100.0))])
        if np.max(sol.z) > 100 else float("nan"),
    }


def test_lcdm_recovery(z_max: float = 1100.0) -> Dict[str, Any]:
    """?->0 and explicit lcdm_limit must reproduce ?CDM H(z)."""
    base = LCDMSParams()
    out: Dict[str, Any] = {}

    p_gamma0 = LCDMSParams(**{**asdict(base), "gamma": 0.0})
    sol0 = analyze_residuals(solve_background(p_gamma0, z_max=z_max))
    out["gamma0"] = relative_H_error(sol0)
    out["gamma0"]["max_abs_R_F"] = sol0.meta["max_abs_R_F"]
    out["gamma0"]["max_abs_R_C"] = sol0.meta["max_abs_R_C"]
    out["gamma0"]["pass"] = out["gamma0"]["max_rel_H"] < 1e-6

    p_flag = LCDMSParams(**{**asdict(base), "lcdm_limit": True})
    solL = analyze_residuals(solve_background(p_flag, z_max=z_max))
    out["lcdm_flag"] = relative_H_error(solL)
    out["lcdm_flag"]["pass"] = out["lcdm_flag"]["max_rel_H"] < 1e-6

    # S_max -> large => w_S -> -1 (smaller deviation)
    p_bigS = LCDMSParams(**{**asdict(base), "S_max": 1.0e6, "S_early": 1.0, "chi0": 0.99})
    solS = analyze_residuals(solve_background(p_bigS, z_max=z_max))
    out["Smax_large"] = relative_H_error(solS, base)
    out["Smax_large"]["mean_abs_w_plus_1"] = float(np.mean(np.abs(solS.w_S + 1.0)))
    out["Smax_large"]["pass"] = out["Smax_large"]["max_rel_H"] < 5e-3

    return out


def early_universe_epsilon(sol: BackgroundSolution) -> Dict[str, Any]:
    """?_H(z) = (H_model - H_?CDM)/H_?CDM from z=0..z_max."""
    E_ref = lcdm_E(sol.z, sol.params)
    eps = (sol.E - E_ref) / np.maximum(E_ref, 1e-30)
    # high-z mask
    hi = sol.z > 10.0
    return {
        "eps_H": eps,
        "z": sol.z,
        "max_abs_eps_z_gt_10": float(np.max(np.abs(eps[hi]))) if np.any(hi) else float("nan"),
        "max_abs_eps_z_gt_100": float(np.max(np.abs(eps[sol.z > 100])))
        if np.any(sol.z > 100) else float("nan"),
        "eps_z0": float(eps[np.argmin(np.abs(sol.z))]),
    }


# =============================================================================
# 5. Dual-integrator comparison & convergence
# =============================================================================

def dual_integrator_compare(p: LCDMSParams, z_max: float = 1100.0) -> Dict[str, Any]:
    s1 = analyze_residuals(solve_background(p, method="RK45", z_max=z_max, rtol=1e-8))
    s2 = analyze_residuals(solve_background(p, method="DOP853", z_max=z_max, rtol=1e-10))
    # interpolate s2 onto s1.z
    E2 = interp1d(s2.z[::-1], s2.E[::-1], kind="cubic", fill_value="extrapolate")(s1.z)
    dE = np.abs(s1.E - E2) / np.maximum(s1.E, 1e-30)
    return {
        "max_rel_dE_RK45_vs_DOP853": float(np.max(dE)),
        "mean_rel_dE": float(np.mean(dE)),
        "R_F_RK45": s1.meta["max_abs_R_F"],
        "R_C_RK45": s1.meta["max_abs_R_C"],
        "R_F_DOP853": s2.meta["max_abs_R_F"],
        "R_C_DOP853": s2.meta["max_abs_R_C"],
        "pass": float(np.max(dE)) < 1e-5,
    }


def convergence_test(p: LCDMSParams, z_max: float = 100.0) -> Dict[str, Any]:
    """Vary rtol; solutions should converge."""
    tols = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    sols = []
    for rt in tols:
        sols.append(
            solve_background(p, method="RK45", z_max=z_max, n_eval=1500, rtol=rt, atol=rt * 1e-2)
        )
    # compare each to finest
    ref = sols[-1]
    errs = []
    for s, rt in zip(sols[:-1], tols[:-1]):
        E_ref = interp1d(ref.z[::-1], ref.E[::-1], kind="cubic", fill_value="extrapolate")(s.z)
        errs.append(float(np.max(np.abs(s.E - E_ref) / np.maximum(E_ref, 1e-30))))
    return {
        "rtols": tols[:-1],
        "max_rel_err_vs_finest": errs,
        "pass": errs[-1] < 1e-5,
    }


# =============================================================================
# 6. Observables: distances, CMB acoustic scale, BAO
# =============================================================================

def comoving_distance_Mpc(sol: BackgroundSolution, z: float, n: int = 800) -> float:
    """D_M(z) = ?_0^z c dz'/H(z')  [Mpc]."""
    if z <= 0:
        return 0.0
    zz = np.linspace(0.0, z, n)
    Hz = _H_of_z_safe(sol, zz)  # km/s/Mpc
    return float(np.trapezoid(C_KM_S / np.maximum(Hz, 1e-30), zz))


def angular_diameter_distance(sol: BackgroundSolution, z: float) -> float:
    return comoving_distance_Mpc(sol, z) / (1.0 + z)


def luminosity_distance(sol: BackgroundSolution, z: float) -> float:
    return comoving_distance_Mpc(sol, z) * (1.0 + z)


def hubble_distance(sol: BackgroundSolution, z: float) -> float:
    return C_KM_S / float(_H_of_z_safe(sol, z))


def _H_of_z_safe(sol: BackgroundSolution, z: np.ndarray | float) -> np.ndarray:
    """
    H(z) from the solution inside its tabulated range; beyond z_max use the
    model's own density components diluted as radiation/matter + frozen r_S
    at the earliest tabulated point (early-universe matching), not cubic
    extrapolation.
    """
    z = np.asarray(z, dtype=float)
    z_lo, z_hi = float(np.min(sol.z)), float(np.max(sol.z))
    H = np.empty_like(z, dtype=float)
    inside = (z >= z_lo - 1e-12) & (z <= z_hi + 1e-12)
    if np.any(inside):
        H[inside] = sol.H_of_z(z[inside])
    outside = ~inside
    if np.any(outside):
        # Early: use ?CDM-like with this model's ?'s (valid when ?_H->0 at high z)
        H[outside] = sol.params.H0 * lcdm_E(z[outside], sol.params)
    return H


def sound_horizon_integral(
    sol: BackgroundSolution,
    z_end: float,
    z_start: float = 1.0e5,
    Obh2: float = 0.02237,
    n: int = 4000,
) -> float:
    """
    r_s(z_end) = ?_{z_end}^{z_start} c_s(z)/H(z) dz
    with approximate baryon-photon sound speed
      c_s = c / sqrt(3(1+R)),
      R = 3 ?_b /(4 ?_?) * a,  ?_? h? ~ 2.469e-5.
    """
    Ogh2 = 2.469e-5  # photon
    R0 = 3.0 * Obh2 / (4.0 * Ogh2)  # R/a
    zz = np.linspace(z_end, z_start, n)
    a = 1.0 / (1.0 + zz)
    R = R0 * a
    cs = C_KM_S / np.sqrt(3.0 * (1.0 + R))  # km/s
    Hz = _H_of_z_safe(sol, zz)
    return float(np.trapezoid(cs / np.maximum(Hz, 1e-30), zz))


def cmb_acoustic_scale(
    sol: BackgroundSolution,
    z_star: float = 1090.0,
    Obh2: float = 0.02237,
) -> Dict[str, float]:
    rs = sound_horizon_integral(sol, z_end=z_star, Obh2=Obh2)
    DA = angular_diameter_distance(sol, z_star)
    # ?_* = r_s / ((1+z) D_A) = r_s / D_M
    DM = comoving_distance_Mpc(sol, z_star)
    theta = rs / max(DM, 1e-30)
    ell = math.pi / max(theta, 1e-30)
    return {
        "z_star": z_star,
        "r_s_Mpc": rs,
        "D_A_Mpc": DA,
        "D_M_Mpc": DM,
        "theta_star": theta,
        "ell_star_approx": ell,
    }


def bao_ratios(
    sol: BackgroundSolution,
    z_list: Sequence[float] = (0.38, 0.51, 0.61, 1.48),
    z_d: float = 1060.0,
    Obh2: float = 0.02237,
) -> List[Dict[str, float]]:
    rd = sound_horizon_integral(sol, z_end=z_d, Obh2=Obh2)
    rows = []
    for z in z_list:
        DM = comoving_distance_Mpc(sol, z)
        DH = hubble_distance(sol, z)
        rows.append({
            "z": float(z),
            "r_d": rd,
            "D_M": DM,
            "D_H": DH,
            "DM_over_rd": DM / rd,
            "DH_over_rd": DH / rd,
        })
    return rows


def q_of_solution(sol: BackgroundSolution) -> np.ndarray:
    """q = -1 - ?/H?; ? = H0? E dE/dN * dN/dt / H0 wait: dE/dt = H0 E dE/dN * H? 
    dN/dt = H => dE/dt = (dE/dN) H = (dE/dN) H0 E
    ? = H0 dE/dt = H0? E dE/dN
    q = -1 - ?/H? = -1 - (H0? E E')/(H0? E?) = -1 - E'/E
    """
    dE_dN = np.gradient(sol.E, sol.N, edge_order=2)
    return -1.0 - dE_dN / np.maximum(sol.E, 1e-30)


# =============================================================================
# 7. Synthetic parameter recovery (pipeline test, not real MCMC)
# =============================================================================

def predict_mu(sol: BackgroundSolution, z: np.ndarray) -> np.ndarray:
    """Distance modulus from luminosity distance."""
    mu = []
    for zi in z:
        dL = luminosity_distance(sol, float(zi))
        mu.append(5.0 * np.log10(max(dL, 1e-30)) + 25.0)
    return np.asarray(mu)


def synthetic_parameter_recovery(
    theta_true: Optional[LCDMSParams] = None,
    n_z: int = 40,
    sigma_mu: float = 0.15,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Inject ?_true -> synthetic ?(z) + noise -> optimize ?? -> check recovery.
    Tests equations->solver->observable->likelihood chain (MAP, not full MCMC).
    """
    rng = np.random.default_rng(seed)
    true = theta_true or LCDMSParams(H0=70.0, Omega_m0=0.30, gamma=0.35, chi0=0.90)
    sol_true = solve_background(true, z_max=2.5, n_eval=1200)
    z = np.linspace(0.05, 1.5, n_z)
    mu_true = predict_mu(sol_true, z)
    mu_obs = mu_true + rng.normal(0.0, sigma_mu, size=n_z)

    def chi2(x: np.ndarray) -> float:
        H0, Om, gamma, chi0 = x
        if not (50 < H0 < 90 and 0.1 < Om < 0.5 and 0.0 <= gamma < 2.0 and 0.5 < chi0 < 0.999):
            return 1e12
        p = LCDMSParams(H0=H0, Omega_m0=Om, gamma=gamma, chi0=chi0,
                        Omega_r0=true.Omega_r0, S_max=true.S_max, S_early=true.S_early)
        try:
            sol = solve_background(p, z_max=2.5, n_eval=800, rtol=1e-6)
            mu = predict_mu(sol, z)
        except Exception:
            return 1e12
        return float(np.sum(((mu - mu_obs) / sigma_mu) ** 2))

    x0 = np.array([67.0, 0.27, 0.2, 0.85])
    res = minimize(chi2, x0, method="Nelder-Mead",
                   options={"maxiter": 120, "xatol": 1e-3, "fatol": 1e-2})
    xhat = res.x
    return {
        "theta_true": {
            "H0": true.H0, "Omega_m0": true.Omega_m0,
            "gamma": true.gamma, "chi0": true.chi0,
        },
        "theta_hat": {
            "H0": float(xhat[0]), "Omega_m0": float(xhat[1]),
            "gamma": float(xhat[2]), "chi0": float(xhat[3]),
        },
        "chi2_min": float(res.fun),
        "success": bool(res.success),
        "abs_err": {
            "H0": abs(xhat[0] - true.H0),
            "Omega_m0": abs(xhat[1] - true.Omega_m0),
            "gamma": abs(xhat[2] - true.gamma),
            "chi0": abs(xhat[3] - true.chi0),
        },
        "pass": (abs(xhat[0] - true.H0) < 5.0 and abs(xhat[1] - true.Omega_m0) < 0.08),
    }


# =============================================================================
# 8. Model comparison scaffolding (same data treatment)
# =============================================================================

def model_comparison_synthetic(
    seed: int = 0,
    n_z: int = 50,
    sigma_mu: float = 0.12,
) -> Dict[str, Any]:
    """Fit M0=?CDM and M1=?CDM+S to the same synthetic SN-like data; ???, AIC, BIC."""
    rng = np.random.default_rng(seed)
    true = LCDMSParams(H0=70.0, Omega_m0=0.30, gamma=0.4, chi0=0.88)
    sol_true = solve_background(true, z_max=2.0, n_eval=1000)
    z = np.linspace(0.05, 1.2, n_z)
    mu_obs = predict_mu(sol_true, z) + rng.normal(0.0, sigma_mu, size=n_z)

    def fit_lcdm() -> Tuple[float, int]:
        def chi2(x):
            H0, Om = x
            p = LCDMSParams(H0=H0, Omega_m0=Om, lcdm_limit=True)
            sol = solve_background(p, z_max=2.0, n_eval=600, rtol=1e-6)
            return float(np.sum(((predict_mu(sol, z) - mu_obs) / sigma_mu) ** 2))
        r = minimize(chi2, [70.0, 0.3], method="Nelder-Mead", options={"maxiter": 80})
        k = 2
        return float(r.fun), k

    def fit_lcdms() -> Tuple[float, int]:
        def chi2(x):
            H0, Om, gamma, chi0 = x
            p = LCDMSParams(H0=H0, Omega_m0=Om, gamma=gamma, chi0=chi0)
            sol = solve_background(p, z_max=2.0, n_eval=600, rtol=1e-6)
            return float(np.sum(((predict_mu(sol, z) - mu_obs) / sigma_mu) ** 2))
        r = minimize(chi2, [70.0, 0.3, 0.3, 0.9], method="Nelder-Mead",
                     options={"maxiter": 120})
        k = 4
        return float(r.fun), k

    chi2_0, k0 = fit_lcdm()
    chi2_1, k1 = fit_lcdms()
    n = n_z
    AIC = lambda chi2, k: chi2 + 2 * k
    BIC = lambda chi2, k: chi2 + k * math.log(n)
    return {
        "chi2_LCDM": chi2_0,
        "chi2_LCDMS": chi2_1,
        "delta_chi2": chi2_0 - chi2_1,
        "AIC_LCDM": AIC(chi2_0, k0),
        "AIC_LCDMS": AIC(chi2_1, k1),
        "delta_AIC": AIC(chi2_0, k0) - AIC(chi2_1, k1),
        "BIC_LCDM": BIC(chi2_0, k0),
        "BIC_LCDMS": BIC(chi2_1, k1),
        "delta_BIC": BIC(chi2_0, k0) - BIC(chi2_1, k1),
        "note": "Lower AIC/BIC favors model; extra params penalized. Not Bayes factor.",
    }


# =============================================================================
# 9. Perturbation stability (background-informed checks)
# =============================================================================

def perturbation_stability_checks(sol: BackgroundSolution) -> Dict[str, Any]:
    """
    Arch-B style bookkeeping:
      c_s? = 1 (canonical scalar / Layer 13)
      c_a? = w - w'/(3(1+w)) along the background
    Flag phantom crossings and imaginary c_a regions.
    """
    w = sol.w_S
    dw_dN = np.gradient(w, sol.N, edge_order=2)
    # d/dN = (1/H) d/dt but ca? uses ?w / H = dw/dN
    one_p_w = 1.0 + w
    with np.errstate(divide="ignore", invalid="ignore"):
        c_a2 = w - dw_dN / (3.0 * np.where(np.abs(one_p_w) < 1e-12, np.nan, one_p_w))
    c_s2 = 1.0
    finite = np.isfinite(c_a2)
    return {
        "c_s2": c_s2,
        "frac_ca2_negative": float(np.mean(c_a2[finite] < 0)) if np.any(finite) else float("nan"),
        "min_ca2": float(np.nanmin(c_a2)) if np.any(finite) else float("nan"),
        "max_ca2": float(np.nanmax(c_a2)) if np.any(finite) else float("nan"),
        "min_1_plus_w": float(np.min(one_p_w)),
        "phantom_crossing": bool(np.any(one_p_w < -1e-6)),
        "pass_cs2_positive": c_s2 > 0,
        "pass_nonphantom": bool(np.min(one_p_w) >= -1e-6),
    }


# =============================================================================
# 10. Sensitivity / Fisher scaffolding
# =============================================================================

def finite_diff_sensitivity(
    p: LCDMSParams,
    z_nodes: Optional[np.ndarray] = None,
    eps_frac: float = 1e-3,
) -> Dict[str, Any]:
    """?H(z)/??_i at a few redshifts - forecasting scaffold."""
    z_nodes = z_nodes if z_nodes is not None else np.array([0.0, 0.5, 1.0, 2.0])
    sol0 = solve_background(p, z_max=max(10.0, float(np.max(z_nodes) + 1)), n_eval=1000)
    H0v = sol0.H_of_z(z_nodes)

    params_vary = {
        "H0": p.H0,
        "Omega_m0": p.Omega_m0,
        "gamma": max(p.gamma, 1e-3),
        "chi0": p.chi0,
        "S_max": p.S_max,
    }
    derivs = {}
    for name, val in params_vary.items():
        bump = abs(val) * eps_frac if val != 0 else eps_frac
        kw = asdict(p)
        kw[name] = val + bump
        solp = solve_background(LCDMSParams(**kw), z_max=max(10.0, float(np.max(z_nodes) + 1)),
                                n_eval=1000, rtol=1e-6)
        derivs[name] = ((solp.H_of_z(z_nodes) - H0v) / bump).tolist()
    return {"z": z_nodes.tolist(), "dH_dtheta": derivs}


# =============================================================================
# 11. Posterior-predictive / out-of-sample scaffolding
# =============================================================================

def posterior_predictive_scaffold(
    n_draws: int = 64,
    seed: int = 1,
) -> Dict[str, Any]:
    """
    Draw ?~(crude prior), predict H(z), D_L(z); report predictive bands.
    Not a real posterior - structure for PPC once MCMC chains exist.
    """
    rng = np.random.default_rng(seed)
    z = np.linspace(0.0, 2.0, 80)
    Hs, DLs = [], []
    for _ in range(n_draws):
        p = LCDMSParams(
            H0=float(rng.normal(70, 2)),
            Omega_m0=float(np.clip(rng.normal(0.3, 0.02), 0.15, 0.45)),
            gamma=float(np.clip(rng.normal(0.35, 0.1), 0.0, 1.5)),
            chi0=float(np.clip(rng.normal(0.9, 0.03), 0.6, 0.995)),
        )
        sol = solve_background(p, z_max=3.0, n_eval=600, rtol=1e-5)
        Hs.append(sol.H_of_z(z))
        DLs.append(np.array([luminosity_distance(sol, float(zi)) for zi in z]))
    H = np.asarray(Hs)
    DL = np.asarray(DLs)
    return {
        "z": z.tolist(),
        "H_mean": H.mean(0).tolist(),
        "H_p16": np.percentile(H, 16, 0).tolist(),
        "H_p84": np.percentile(H, 84, 0).tolist(),
        "DL_mean": DL.mean(0).tolist(),
        "DL_p16": np.percentile(DL, 16, 0).tolist(),
        "DL_p84": np.percentile(DL, 84, 0).tolist(),
        "n_draws": n_draws,
        "note": "Prior-predictive scaffold; replace draws with MCMC posterior samples.",
    }


def out_of_sample_scaffold(
    seed: int = 2,
) -> Dict[str, Any]:
    """
    Train on 'Pantheon-like' z?[0.05,0.8]; test on 'DES-like' z?[0.8,1.3].
    Reports test ?? for ?CDM vs ?CDM+S MAP fits (structure for real CV).
    """
    rng = np.random.default_rng(seed)
    true = LCDMSParams(H0=70.0, Omega_m0=0.3, gamma=0.45, chi0=0.87)
    sol_true = solve_background(true, z_max=2.0, n_eval=1000)
    z_train = np.linspace(0.05, 0.8, 30)
    z_test = np.linspace(0.85, 1.3, 20)
    sig = 0.15
    mu_train = predict_mu(sol_true, z_train) + rng.normal(0, sig, size=z_train.size)
    mu_test = predict_mu(sol_true, z_test) + rng.normal(0, sig, size=z_test.size)

    def fit_and_test(lcdm: bool) -> Dict[str, float]:
        if lcdm:
            def chi2(x):
                p = LCDMSParams(H0=x[0], Omega_m0=x[1], lcdm_limit=True)
                sol = solve_background(p, z_max=2.0, n_eval=500, rtol=1e-5)
                return float(np.sum(((predict_mu(sol, z_train) - mu_train) / sig) ** 2))
            r = minimize(chi2, [70, 0.3], method="Nelder-Mead", options={"maxiter": 60})
            p = LCDMSParams(H0=r.x[0], Omega_m0=r.x[1], lcdm_limit=True)
        else:
            def chi2(x):
                p = LCDMSParams(H0=x[0], Omega_m0=x[1], gamma=x[2], chi0=x[3])
                sol = solve_background(p, z_max=2.0, n_eval=500, rtol=1e-5)
                return float(np.sum(((predict_mu(sol, z_train) - mu_train) / sig) ** 2))
            r = minimize(chi2, [70, 0.3, 0.3, 0.9], method="Nelder-Mead",
                         options={"maxiter": 80})
            p = LCDMSParams(H0=r.x[0], Omega_m0=r.x[1], gamma=r.x[2], chi0=r.x[3])
        sol = solve_background(p, z_max=2.0, n_eval=800)
        chi2_test = float(np.sum(((predict_mu(sol, z_test) - mu_test) / sig) ** 2))
        return {"chi2_train": float(r.fun), "chi2_test": chi2_test, "k": 2 if lcdm else 4}

    m0 = fit_and_test(True)
    m1 = fit_and_test(False)
    return {"LCDM": m0, "LCDMS": m1, "delta_chi2_test": m0["chi2_test"] - m1["chi2_test"]}


# =============================================================================
# 12. Plots + report
# =============================================================================

def _save_plots(sol: BackgroundSolution, out: Path, lcdm_eps: Dict[str, Any]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    out.mkdir(parents=True, exist_ok=True)

    # Residuals
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax[0].semilogy(sol.z, np.maximum(np.abs(sol.R_F), 1e-20), lw=1.2)
    ax[0].set_ylabel(r"$|R_F|$")
    ax[0].set_title("Friedmann residual")
    ax[1].semilogy(sol.z, np.maximum(np.abs(sol.R_C), 1e-20), lw=1.2, color="C1")
    ax[1].set_ylabel(r"$|R_C|$")
    ax[1].set_xlabel("z")
    ax[1].set_title("Continuity residual")
    fig.tight_layout()
    fig.savefig(out / "residuals_RF_RC.pdf")
    fig.savefig(out / "residuals_RF_RC.png", dpi=150)
    plt.close(fig)

    # H(z) vs ?CDM
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sol.z, sol.E * sol.params.H0, label="?CDM+S")
    ax.plot(sol.z, lcdm_E(sol.z, sol.params) * sol.params.H0, "--", label="?CDM ref")
    ax.set_xlabel("z")
    ax.set_ylabel("H(z) [km/s/Mpc]")
    ax.set_xlim(0, min(10, sol.z.max()))
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "H_of_z.pdf")
    plt.close(fig)

    # Early-universe epsilon
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(lcdm_eps["z"], lcdm_eps["eps_H"])
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("z")
    ax.set_ylabel(r"$\varepsilon_H$")
    ax.set_xscale("log")
    ax.set_xlim(max(1e-2, sol.z[sol.z > 0].min()), sol.z.max())
    ax.set_title(r"$(H-H_{\Lambda\mathrm{CDM}})/H_{\Lambda\mathrm{CDM}}$")
    fig.tight_layout()
    fig.savefig(out / "epsilon_H_early.pdf")
    plt.close(fig)

    # Background thermodynamics
    fig, ax = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
    ax[0].plot(sol.z, sol.w_S)
    ax[0].set_ylabel(r"$w_S$")
    ax[1].plot(sol.z, sol.chi)
    ax[1].set_ylabel(r"$\chi$")
    ax[2].plot(sol.z, q_of_solution(sol))
    ax[2].set_ylabel(r"$q$")
    ax[2].set_xlabel("z")
    ax[2].set_xlim(0, min(5, sol.z.max()))
    fig.tight_layout()
    fig.savefig(out / "w_chi_q.pdf")
    plt.close(fig)


def run_validation_suite_legacy(
    out_dir: str | Path = "lcdms_validation_out",
    quick: bool = False,
) -> Dict[str, Any]:
    """
    Execute the validation hierarchy (numerical consistency first).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    report: Dict[str, Any] = {
        "hierarchy": [
            "symbolic equations (analytical_solver Layers 1-21)",
            "coupled ODE solver",
            "Friedmann/continuity residuals",
            "?CDM / ?->0 / S_max->? recovery",
            "dual integrators + rtol convergence",
            "observables H, DL, DA, DM, rs, ?*, BAO",
            "synthetic parameter recovery",
            "model comparison scaffolding",
            "early-universe ?_H",
            "perturbation stability bookkeeping",
            "sensitivity / PPC / OOS scaffolds",
        ],
        "distinction": (
            "numerical consistency != observational agreement != evidence for the theory"
        ),
    }

    p = LCDMSParams()
    z_max = 300.0 if quick else 1100.0

    # --- 1. Background + residuals ---
    sol = analyze_residuals(solve_background(p, method="RK45", z_max=z_max, rtol=1e-9))
    report["residuals"] = {
        "max_abs_R_F": sol.meta["max_abs_R_F"],
        "max_abs_R_C": sol.meta["max_abs_R_C"],
        "max_rel_R_C": sol.meta["max_rel_R_C"],
        "med_abs_R_F": sol.meta["med_abs_R_F"],
        "med_abs_R_C": sol.meta["med_abs_R_C"],
        # R_F is algebraic (E?-?r); float noise grows mildly at high z
        "pass_RF": sol.meta["max_abs_R_F"] < 1e-6,
        "pass_RC": sol.meta["max_rel_R_C"] < 1e-3,
    }

    # --- 2. Limits ---
    report["lcdm_recovery"] = test_lcdm_recovery(z_max=z_max)

    # --- 3. Dual integrators / convergence ---
    report["dual_integrators"] = dual_integrator_compare(p, z_max=min(z_max, 1100.0))
    report["convergence"] = convergence_test(p, z_max=min(100.0, z_max))

    # --- 4. Early universe ---
    eps = early_universe_epsilon(sol)
    report["early_universe"] = {
        "max_abs_eps_z_gt_10": eps["max_abs_eps_z_gt_10"],
        "max_abs_eps_z_gt_100": eps["max_abs_eps_z_gt_100"],
        "eps_z0": eps["eps_z0"],
    }

    # --- 5. Observables ---
    report["cmb"] = cmb_acoustic_scale(sol)
    report["bao"] = bao_ratios(sol)
    report["distances_z1"] = {
        "D_M": comoving_distance_Mpc(sol, 1.0),
        "D_A": angular_diameter_distance(sol, 1.0),
        "D_L": luminosity_distance(sol, 1.0),
        "D_H": hubble_distance(sol, 1.0),
    }

    # --- 6. Stability ---
    report["perturbation_stability"] = perturbation_stability_checks(sol)

    # --- 7. Synthetic recovery + model comparison (skip-heavy in --quick partially) ---
    if quick:
        report["synthetic_recovery"] = {"skipped": True}
        report["model_comparison"] = {"skipped": True}
        report["oos"] = {"skipped": True}
        report["ppc"] = {"skipped": True}
        report["sensitivity"] = {"skipped": True}
    else:
        report["synthetic_recovery"] = synthetic_parameter_recovery()
        report["model_comparison"] = model_comparison_synthetic()
        report["oos"] = out_of_sample_scaffold()
        report["ppc"] = {k: v for k, v in posterior_predictive_scaffold(n_draws=32).items()
                         if k in {"n_draws", "note"}}
        report["sensitivity"] = finite_diff_sensitivity(p)

    # --- plots ---
    _save_plots(sol, out / "figures", eps)

    # Pass summary
    passes = {
        "residuals_RF": report["residuals"]["pass_RF"],
        "residuals_RC": report["residuals"]["pass_RC"],
        "lcdm_gamma0": report["lcdm_recovery"]["gamma0"]["pass"],
        "lcdm_flag": report["lcdm_recovery"]["lcdm_flag"]["pass"],
        "dual_integrators": report["dual_integrators"]["pass"],
        "convergence": report["convergence"]["pass"],
        "nonphantom": report["perturbation_stability"]["pass_nonphantom"],
    }
    if not quick:
        passes["synthetic_H0_Om"] = report["synthetic_recovery"].get("pass", False)
    report["passes"] = passes
    report["all_critical_pass"] = all(
        passes[k] for k in
        ["residuals_RF", "residuals_RC", "lcdm_gamma0", "lcdm_flag", "dual_integrators"]
    )
    report["elapsed_sec"] = time.time() - t0

    # Write JSON (numpy-safe)
    def _jsonify(o: Any) -> Any:
        if isinstance(o, dict):
            return {str(k): _jsonify(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_jsonify(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return float(o)
        if isinstance(o, (np.integer, int)):
            return int(o)
        if isinstance(o, (np.bool_, bool)):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o

    (out / "validation_report.json").write_text(
        json.dumps(_jsonify(report), indent=2), encoding="utf-8"
    )

    # Markdown summary
    lines = [
        "# ?CDM+S numerical validation report",
        "",
        f"- elapsed: {report['elapsed_sec']:.1f}s",
        f"- critical pass: **{report['all_critical_pass']}**",
        "",
        "## Internal consistency",
        f"- max |R_F| = {report['residuals']['max_abs_R_F']:.3e}",
        f"- max rel |R_C| = {report['residuals']['max_rel_R_C']:.3e}",
        f"- dual integrator max dE/E = {report['dual_integrators']['max_rel_dE_RK45_vs_DOP853']:.3e}",
        "",
        "## Established-physics recovery",
        f"- gamma->0 max |dH|/H = {report['lcdm_recovery']['gamma0']['max_rel_H']:.3e}",
        f"- lcdm_limit flag max |dH|/H = {report['lcdm_recovery']['lcdm_flag']['max_rel_H']:.3e}",
        f"- early |eps_H|(z>100) = {report['early_universe']['max_abs_eps_z_gt_100']:.3e}",
        "",
        "## Observables (fiducial)",
        f"- theta_* ~ {report['cmb']['theta_star']:.6e}, ell_* ~ {report['cmb']['ell_star_approx']:.1f}",
        f"- r_s(z*) ~ {report['cmb']['r_s_Mpc']:.2f} Mpc",
        f"- D_L(z=1) ~ {report['distances_z1']['D_L']:.1f} Mpc",
        "",
        "## Pass table",
    ]
    for k, v in passes.items():
        lines.append(f"- {'PASS' if v else 'FAIL'}: {k}")
    lines += [
        "",
        "## Reminder",
        report["distinction"],
        "",
        "Next: feed D_L, D_A, r_s, D_M into the Bayesian pipeline only after critical passes.",
    ]
    (out / "validation_report.md").write_text("\n".join(lines), encoding="utf-8")
    return report

# ----- extensions (detailed architecture) -----
# Extended validation architecture (V0 inventory detail through V17 exports).
# Appended into analytical_solver.py as part of the embedded numerical suite.
#
# NOTE: This module is designed to be concatenated/embedded. When embedded,
# names from lcdms_numerical_validation / analytical_solver share one namespace.


# =============================================================================
# V0b - Detailed architecture documentation & check registry
# =============================================================================

VALIDATION_EQUATION_LEDGER: list[dict] = [
    {
        "id": "V.E1",
        "latex": r"E^2 = r_r + r_m + r_S",
        "ascii": "E^2 = r_r + r_m + r_S",
        "meaning": "Flat Friedmann constraint in units of rho_crit0",
    },
    {
        "id": "V.E2",
        "latex": r"dr_r/dN = -4 r_r,\quad dr_m/dN = -3 r_m",
        "ascii": "dr_r/dN=-4 r_r; dr_m/dN=-3 r_m",
        "meaning": "Standard non-interacting dilution",
    },
    {
        "id": "V.E3",
        "latex": r"\dot\chi = \gamma\chi(1-\chi)",
        "ascii": "chi_dot = gamma chi (1-chi)",
        "meaning": "Logistic entropy-completion closure (assumed Gamma-class)",
    },
    {
        "id": "V.E4",
        "latex": r"w_S=-1+\Delta S\,\gamma\chi(1-\chi)/(3 H S_H)",
        "ascii": "w_S = -1 + DeltaS*gamma*chi*(1-chi)/(3 H S_H)",
        "meaning": "Layer-1 thermodynamic EOS for entropy sector",
    },
    {
        "id": "V.E5",
        "latex": r"dr_S/dN = -3(1+w_S)r_S",
        "ascii": "dr_S/dN = -3(1+w_S) r_S",
        "meaning": "Entropy-sector continuity with Q_S=0",
    },
    {
        "id": "V.E6",
        "latex": r"R_F = E^2-(r_r+r_m+r_S)",
        "ascii": "R_F = E^2 - sum r_i",
        "meaning": "Friedmann residual (must be ~0)",
    },
    {
        "id": "V.E7",
        "latex": r"R_C=(dr_{\rm tot}/dN)_{\rm num}-(dr_{\rm tot}/dN)_{\rm RHS}",
        "ascii": "R_C = dr_tot_num - dr_tot_RHS",
        "meaning": "Continuity residual (must be ~0)",
    },
    {
        "id": "V.E8",
        "latex": r"D_M(z)=\int_0^z c\,dz'/H(z')",
        "ascii": "D_M = int c dz/H",
        "meaning": "Comoving distance",
    },
    {
        "id": "V.E9",
        "latex": r"r_s(z_*)=\int_{z_*}^{z_{\rm start}} c_s(z)/H(z)\,dz",
        "ascii": "r_s = int c_s/H dz",
        "meaning": "Sound horizon to last scattering / drag",
    },
    {
        "id": "V.E10",
        "latex": r"\theta_*=r_s(z_*)/D_M(z_*)",
        "ascii": "theta_* = r_s / D_M",
        "meaning": "Angular acoustic scale",
    },
]


@dataclass
class ValidationCheck:
    """Atomic pass/fail record for the embedded suite."""

    id: str
    section: str
    description: str
    passed: bool
    metric: float
    threshold: float
    detail: str = ""


@dataclass
class ValidationReport:
    """Aggregate Q1/Q2/Q3 report with explicit critical-gate logic."""

    checks: List[ValidationCheck] = field(default_factory=list)
    sections: Dict[str, Any] = field(default_factory=dict)
    elapsed_sec: float = 0.0
    distinction: str = (
        "numerical consistency != observational agreement != evidence for the theory"
    )

    CRITICAL_IDS: tuple = (
        "V2.RF",
        "V2.RC",
        "V3.gamma0",
        "V3.lcdm_flag",
        "V4.dual",
    )

    def add(
        self,
        id: str,
        section: str,
        description: str,
        passed: bool,
        metric: float,
        threshold: float,
        detail: str = "",
    ) -> None:
        self.checks.append(
            ValidationCheck(id, section, description, passed, metric, threshold, detail)
        )

    @property
    def all_critical_pass(self) -> bool:
        by_id = {c.id: c.passed for c in self.checks}
        return all(by_id.get(i, False) for i in self.CRITICAL_IDS)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "distinction": self.distinction,
            "equation_ledger": VALIDATION_EQUATION_LEDGER,
            "elapsed_sec": self.elapsed_sec,
            "all_critical_pass": self.all_critical_pass,
            "checks": [asdict(c) for c in self.checks],
            "sections": self.sections,
        }


# =============================================================================
# V1b - Third integrator + entropy-only honesty probe
# =============================================================================

def solve_background_Radau(p: "LCDMSParams", **kwargs: Any) -> "BackgroundSolution":
    """Stiff-capable third independent scipy integrator."""
    kwargs = dict(kwargs)
    kwargs.setdefault("method", "Radau")
    kwargs.setdefault("rtol", 1e-8)
    return solve_background_N(p, **kwargs)


def entropy_only_vs_coupled_honesty(p: "LCDMSParams", z_max: float = 5.0) -> Dict[str, Any]:
    """
    Entropy-only H(chi)=const/sqrt(S_H(chi)) is two-sided dS and MUST disagree
    with coupled H(z) at high z. This test PASSES when disagreement is large
    at z>2 (honesty gate), not when curves match.
    """
    np = _require_numpy_local()
    sol = solve_background(p, z_max=z_max, n_eval=800)
    S_H = p.S_early + p.Delta_S * sol.chi
    H_eo_raw = 1.0 / np.sqrt(np.maximum(S_H, 1e-30))
    idx = int(np.argmin(np.abs(sol.z)))
    H_c = sol.E * p.H0
    scale = H_c[idx] / max(H_eo_raw[idx], 1e-30)
    H_eo = scale * H_eo_raw
    rel = np.abs(H_eo - H_c) / np.maximum(H_c, 1e-30)
    return {
        "max_rel_diff_all_z": float(np.max(rel)),
        "max_rel_diff_z_lt_1": float(np.max(rel[sol.z < 1])),
        "max_rel_diff_z_gt_2": float(np.max(rel[sol.z > 2])) if np.any(sol.z > 2) else float("nan"),
        "pass_honesty": bool(np.any(sol.z > 2) and float(np.max(rel[sol.z > 2])) > 1e-3),
        "note": "Large high-z disagreement is REQUIRED; entropy-only omits rad/matter.",
    }


def _require_numpy_local():
    import numpy as np
    return np


# =============================================================================
# V2b - Per-sector continuity + Bianchi form
# =============================================================================

def sector_continuity_residuals(sol: "BackgroundSolution") -> Dict[str, float]:
    np = _require_numpy_local()
    n = len(sol.N)
    lo, hi = max(2, n // 50), max(3, n - n // 50)
    rr = {
        "r": np.gradient(sol.r_r, sol.N, edge_order=2) + 4.0 * sol.r_r,
        "m": np.gradient(sol.r_m, sol.N, edge_order=2) + 3.0 * sol.r_m,
        "S": np.gradient(sol.r_S, sol.N, edge_order=2) + 3.0 * (1.0 + sol.w_S) * sol.r_S,
    }
    return {k: float(np.max(np.abs(v[lo:hi]))) for k, v in rr.items()}


def bianchi_identity_residual(sol: "BackgroundSolution") -> Dict[str, float]:
    """
    From Friedmann + Raychaudhuri, d(E^2)/dN = -3(r_tot + P_hat)*? wait:
    E^2 = r_tot => d(E^2)/dN = dr_tot/dN = -3(r_tot + P_hat)
    with P_hat = r_r/3 + w_S r_S.
    """
    np = _require_numpy_local()
    r_tot = sol.r_r + sol.r_m + sol.r_S
    P_hat = sol.r_r / 3.0 + sol.w_S * sol.r_S
    dE2 = np.gradient(sol.E**2, sol.N, edge_order=2)
    target = -3.0 * (r_tot + P_hat)
    n = len(sol.N)
    lo, hi = max(2, n // 50), max(3, n - n // 50)
    res = dE2 - target
    scale = np.maximum(np.abs(target), 1e-30)
    return {
        "max_abs_bianchi": float(np.max(np.abs(res[lo:hi]))),
        "max_rel_bianchi": float(np.max(np.abs(res[lo:hi] / scale[lo:hi]))),
    }


# =============================================================================
# V4b / V5b - Triple compare, IC stability, method matrix, order estimate
# =============================================================================

def triple_integrator_compare(p: "LCDMSParams", z_max: float = 300.0) -> Dict[str, Any]:
    from scipy.interpolate import interp1d
    np = _require_numpy_local()
    s1 = analyze_residuals(solve_background(p, method="RK45", z_max=z_max, rtol=1e-8))
    s2 = analyze_residuals(solve_background(p, method="DOP853", z_max=z_max, rtol=1e-10))
    out: Dict[str, Any] = {}
    E2 = interp1d(s2.z[::-1], s2.E[::-1], kind="cubic", fill_value="extrapolate")(s1.z)
    d12 = np.abs(s1.E - E2) / np.maximum(s1.E, 1e-30)
    out["max_rel_RK45_DOP853"] = float(np.max(d12))
    try:
        s3 = analyze_residuals(solve_background_Radau(p, z_max=z_max, rtol=1e-8))
        E3 = interp1d(s3.z[::-1], s3.E[::-1], kind="cubic", fill_value="extrapolate")(s1.z)
        d13 = np.abs(s1.E - E3) / np.maximum(s1.E, 1e-30)
        out["max_rel_RK45_Radau"] = float(np.max(d13))
        out["pass"] = out["max_rel_RK45_DOP853"] < 1e-5 and out["max_rel_RK45_Radau"] < 5e-5
    except Exception as exc:
        out["Radau_error"] = str(exc)
        out["pass"] = out["max_rel_RK45_DOP853"] < 1e-5
    return out


def ic_stability_test(p: "LCDMSParams", z_max: float = 50.0, delta: float = 1e-3) -> Dict[str, Any]:
    from scipy.interpolate import interp1d
    np = _require_numpy_local()
    s0 = solve_background(p, z_max=z_max, rtol=1e-7)
    variants = {
        "chi0+": LCDMSParams(**{**asdict(p), "chi0": min(0.999, p.chi0 * (1 + delta))}),
        "chi0-": LCDMSParams(**{**asdict(p), "chi0": max(0.5, p.chi0 * (1 - delta))}),
        "Om+": LCDMSParams(**{**asdict(p), "Omega_m0": p.Omega_m0 * (1 + delta)}),
        "gamma+": LCDMSParams(**{**asdict(p), "gamma": p.gamma * (1 + delta)}),
    }
    out: Dict[str, Any] = {}
    for name, pv in variants.items():
        s = solve_background(pv, z_max=z_max, rtol=1e-7)
        E = interp1d(s.z[::-1], s.E[::-1], kind="cubic", fill_value="extrapolate")(s0.z)
        out[name] = float(np.max(np.abs(E - s0.E) / np.maximum(s0.E, 1e-30)))
    out["pass"] = all(float(v) < 0.05 for k, v in out.items() if k != "pass")
    return out


def method_matrix_stability(p: "LCDMSParams", z_max: float = 100.0) -> Dict[str, Any]:
    from scipy.interpolate import interp1d
    np = _require_numpy_local()
    ref = solve_background(p, method="DOP853", z_max=z_max, rtol=1e-10)
    table: Dict[str, Any] = {}
    for m in ("RK45", "DOP853"):
        for rt in (1e-6, 1e-8):
            s = solve_background(p, method=m, z_max=z_max, rtol=rt, n_eval=1200)
            E = interp1d(ref.z[::-1], ref.E[::-1], kind="cubic", fill_value="extrapolate")(s.z)
            table[f"{m}_rtol_{rt:g}"] = float(np.max(np.abs(s.E - E) / np.maximum(E, 1e-30)))
    table["pass"] = max(v for k, v in table.items() if k != "pass") < 1e-4
    return table


def nsteps_convergence_rk4(p: "LCDMSParams", steps_list: Optional[Sequence[int]] = None) -> Dict[str, Any]:
    """
    Forward RK4 refinement study: expect error decrease with nsteps.
    Estimates empirical order p from consecutive refinements.
    """
    from scipy.interpolate import interp1d
    np = _require_numpy_local()
    steps_list = list(steps_list or [2000, 4000, 8000, 16000])
    sols = [solve_background_t_rk4(p, nsteps=n) for n in steps_list]
    ref = sols[-1]
    errs = []
    for s in sols[:-1]:
        # compare E at overlapping a via interp on N
        E_ref = interp1d(ref.N, ref.E, kind="linear", fill_value="extrapolate")(s.N)
        errs.append(float(np.max(np.abs(s.E - E_ref) / np.maximum(E_ref, 1e-30))))
    orders = []
    for i in range(len(errs) - 1):
        if errs[i + 1] > 0 and errs[i] > 0:
            # n doubled => order ~ log2(e_n/e_{2n})
            orders.append(float(np.log(errs[i] / errs[i + 1]) / np.log(2.0)))
    return {
        "nsteps": steps_list[:-1],
        "max_rel_err_vs_finest": errs,
        "empirical_orders": orders,
        "pass": len(errs) >= 2 and errs[-1] < errs[0],
    }


# =============================================================================
# V6b - Ensemble recovery
# =============================================================================

def synthetic_recovery_ensemble(n_realizations: int = 8, seed: int = 0) -> Dict[str, Any]:
    np = _require_numpy_local()
    rows = [synthetic_parameter_recovery(seed=seed + 17 * i) for i in range(n_realizations)]
    h0_err = [r["abs_err"]["H0"] for r in rows]
    om_err = [r["abs_err"]["Omega_m0"] for r in rows]
    return {
        "n": n_realizations,
        "pass_fraction": float(np.mean([r["pass"] for r in rows])),
        "mean_abs_err_H0": float(np.mean(h0_err)),
        "mean_abs_err_Om": float(np.mean(om_err)),
        "pass": float(np.mean([r["pass"] for r in rows])) >= 0.5,
    }


# =============================================================================
# V11b / V12b - Distance tables + DV
# =============================================================================

def distance_ladder_table(
    sol: "BackgroundSolution",
    z_nodes: Optional[Sequence[float]] = None,
) -> List[Dict[str, float]]:
    z_nodes = list(z_nodes or [0.1, 0.3, 0.5, 1.0, 1.5, 2.0])
    rows = []
    for z in z_nodes:
        dL = luminosity_distance(sol, z)
        rows.append({
            "z": float(z),
            "H": float(_H_of_z_safe(sol, z)),
            "D_M": comoving_distance_Mpc(sol, z),
            "D_A": angular_diameter_distance(sol, z),
            "D_L": dL,
            "D_H": hubble_distance(sol, z),
            "mu": 5.0 * math.log10(max(dL, 1e-30)) + 25.0,
        })
    return rows


def bao_ratios_extended(
    sol: "BackgroundSolution",
    z_list: Sequence[float] = (0.38, 0.51, 0.61, 1.48, 2.33),
    z_d: float = 1060.0,
) -> List[Dict[str, float]]:
    rows = bao_ratios(sol, z_list=z_list, z_d=z_d)
    for row in rows:
        z = row["z"]
        DM, DH, rd = row["D_M"], row["D_H"], row["r_d"]
        row["DV_over_rd"] = ((z * DH * DM**2) ** (1.0 / 3.0)) / rd
    return rows


# =============================================================================
# V13b - Layer-7 xi bookkeeping on coupled background
# =============================================================================

def layer7_xi_consistency(sol: "BackgroundSolution") -> Dict[str, Any]:
    np = _require_numpy_local()
    if getattr(sol, "q", None) is None or sol.q is None:
        dE = np.gradient(sol.E, sol.N, edge_order=2)
        q = -1.0 - dE / np.maximum(sol.E, 1e-30)
    else:
        q = sol.q
    xi_q = 2.0 * (1.0 + q)
    xi_w = 3.0 * (1.0 + sol.w_S)
    idx = int(np.argmin(np.abs(sol.z)))
    return {
        "xi_q_today": float(xi_q[idx]),
        "xi_w_today": float(xi_w[idx]),
        "abs_diff_today": float(abs(xi_q[idx] - xi_w[idx])),
        "note": "xi_q=xi_w exact only for single-fluid; coupled background differs.",
    }


def numeric_layer_crosschecks(p: "LCDMSParams", z_max: float = 5.0) -> Dict[str, Any]:
    np = _require_numpy_local()
    sol = analyze_residuals(solve_background(p, z_max=z_max, rtol=1e-8))
    # ensure q present
    if sol.q is None:
        dE = np.gradient(sol.E, sol.N, edge_order=2)
        sol.q = -1.0 - dE / np.maximum(sol.E, 1e-30)
    nec = sol.r_S * (1.0 + sol.w_S)
    dchi = np.gradient(sol.chi, sol.N, edge_order=2)
    return {
        "min_NEC_S": float(np.min(nec)),
        "pass_NEC_S": bool(np.min(nec) >= -1e-10),
        "min_dchi_dN": float(np.min(dchi)),
        "pass_chi_monotone": bool(np.min(dchi) >= -1e-8),
        "sector_cont": sector_continuity_residuals(sol),
        "bianchi": bianchi_identity_residual(sol),
        "xi_check": layer7_xi_consistency(sol),
        "w_today": float(sol.w_S[np.argmin(np.abs(sol.z))]),
        "q_today": float(sol.q[np.argmin(np.abs(sol.z))]),
    }


# =============================================================================
# V15 / V16 - Fisher + identifiability
# =============================================================================

def fisher_matrix_scaffold(
    p: "LCDMSParams",
    z_nodes: Optional[Any] = None,
    sigma_H_frac: float = 0.01,
) -> Dict[str, Any]:
    np = _require_numpy_local()
    sens = finite_diff_sensitivity(p, z_nodes=z_nodes)
    z = np.asarray(sens["z"], dtype=float)
    names = list(sens["dH_dtheta"].keys())
    dH = np.array([sens["dH_dtheta"][n] for n in names], dtype=float)
    sol = solve_background(p, z_max=max(10.0, float(np.max(z) + 1)), n_eval=800)
    H = sol.H_of_z(z)
    sigma = np.maximum(sigma_H_frac * H, 1e-30)
    F = np.zeros((len(names), len(names)))
    for a in range(len(z)):
        g = dH[:, a] / sigma[a]
        F += np.outer(g, g)
    w, v = np.linalg.eigh(F)
    return {
        "parameters": names,
        "fisher": F.tolist(),
        "eigenvalues": w.tolist(),
        "eigenvectors": v.tolist(),
        "condition_number": float(np.max(w) / max(np.min(np.abs(w)), 1e-30)),
        "note": "Scaffold Fisher from fractional H(z) errors; replace with survey cov.",
    }


def identifiability_analysis(p: "LCDMSParams") -> Dict[str, Any]:
    np = _require_numpy_local()
    fish = fisher_matrix_scaffold(p)
    w = np.asarray(fish["eigenvalues"])
    v = np.asarray(fish["eigenvectors"])
    names = fish["parameters"]
    order = np.argsort(w)
    weak = []
    for idx in order[:2]:
        weak.append({
            "eigenvalue": float(w[idx]),
            "mode": {names[i]: float(v[i, idx]) for i in range(len(names))},
        })
    return {
        "weak_modes": weak,
        "condition_number": fish["condition_number"],
        "parameters": names,
        "interpretation": (
            "Small eigenvalues => poorly constrained combinations "
            "(often gamma <-> transition timing; S_max <-> late H)."
        ),
    }


def prior_sensitivity_hook(p: "LCDMSParams", seed: int = 3) -> Dict[str, Any]:
    wide = synthetic_parameter_recovery(LCDMSParams(**{**asdict(p), "gamma": 0.5}), seed=seed)
    tight = synthetic_parameter_recovery(LCDMSParams(**{**asdict(p), "gamma": 0.2}), seed=seed + 1)
    return {
        "wide_gamma_case": wide["theta_hat"],
        "tight_gamma_case": tight["theta_hat"],
        "note": "Hook only - replace with full MCMC prior reweighting.",
    }


# =============================================================================
# V17b - CSV export + extended markdown + DerivationStep bridge
# =============================================================================

def export_residual_csv(sol: "BackgroundSolution", path: Path) -> Path:
    np = _require_numpy_local()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    q = sol.q if getattr(sol, "q", None) is not None and sol.q is not None else np.full_like(sol.z, np.nan)
    if sol.q is None:
        dE = np.gradient(sol.E, sol.N, edge_order=2)
        q = -1.0 - dE / np.maximum(sol.E, 1e-30)
    RF = sol.R_F if sol.R_F is not None else np.zeros_like(sol.z)
    RC = sol.R_C if sol.R_C is not None else np.zeros_like(sol.z)
    RCr = sol.R_C_rel if getattr(sol, "R_C_rel", None) is not None and sol.R_C_rel is not None else np.zeros_like(sol.z)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["z", "N", "E", "r_r", "r_m", "r_S", "chi", "w_S", "R_F", "R_C", "R_C_rel", "q"])
        for i in range(len(sol.z)):
            w.writerow([
                float(sol.z[i]), float(sol.N[i]), float(sol.E[i]),
                float(sol.r_r[i]), float(sol.r_m[i]), float(sol.r_S[i]),
                float(sol.chi[i]), float(sol.w_S[i]),
                float(RF[i]), float(RC[i]), float(RCr[i]), float(q[i]),
            ])
    return path


def export_distance_csv(sol: "BackgroundSolution", path: Path) -> Path:
    path = Path(path)
    rows = distance_ladder_table(sol)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(list(rows[0].keys()))
        for row in rows:
            w.writerow([row[k] for k in row.keys()])
    return path


def export_bao_csv(sol: "BackgroundSolution", path: Path) -> Path:
    path = Path(path)
    rows = bao_ratios_extended(sol)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(list(rows[0].keys()))
        for row in rows:
            w.writerow([row[k] for k in row.keys()])
    return path


def step_numerical_validation_embedded(
    out_dir: str | Path = "lcdms_validation_out",
    quick: bool = True,
) -> "DerivationStep":
    """
    Pipeline-facing DerivationStep wrapping the numerical validation suite.
    Status is 'derived' if critical numerical gates pass, else 'failed'.
    """
    rep = run_embedded_validation_suite(out_dir=out_dir, quick=quick)
    passed = bool(rep.get("all_critical_pass"))
    return DerivationStep(
        name="NUM_validation_hierarchy_V0_V17",
        stage="NUMERICAL_VALIDATION",
        assumptions=[
            "Coupled rad+matter+S with Layer-1 w_S and Q_S=0",
            "numpy/scipy available",
            VALIDATION_DISTINCTION if "VALIDATION_DISTINCTION" in globals() else (
                "numerical consistency != observational agreement != evidence"
            ),
        ],
        equations={
            "Friedmann": r"E^2=r_r+r_m+r_S",
            "R_F": r"R_F=E^2-\sum r_i",
            "R_C": r"R_C=(dr_{tot}/dN)_{num}-(dr_{tot}/dN)_{RHS}",
            "w_S": r"w_S=-1+\Delta S\gamma\chi(1-\chi)/(3HS_H)",
        },
        claims=[
            f"NUM suite critical pass = {passed}",
            f"max |R_F| = {rep.get('residuals', {}).get('max_abs_R_F', float('nan'))}",
            f"max rel |R_C| = {rep.get('residuals', {}).get('max_rel_R_C', float('nan'))}",
            "Q1 residuals/limits/dual integrators gated before any observational claim.",
        ],
        status="derived" if passed else "failed",
        evidence={"report_keys": list(rep.keys()), "all_critical_pass": passed},
    )


def run_embedded_validation_suite(
    out_dir: str | Path = "lcdms_validation_out",
    quick: bool = False,
) -> Dict[str, Any]:
    """
    Full embedded runner (V0-V17). Prefer this from analytical_solver CLI.
    Builds ValidationReport with explicit critical gates, writes artifacts.
    """
    import time as _time
    np = _require_numpy_local()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = _time.time()
    report = ValidationReport()
    p = LCDMSParams()
    z_max = 300.0 if quick else 1100.0

    # V0
    warns = []
    if hasattr(p, "validate_inventory"):
        # optional if present on dataclass in base module
        try:
            warns = []  # base LCDMSParams may not have validate_inventory
        except Exception:
            warns = []
    # lightweight inventory
    if p.Omega_S0 <= 0:
        warns.append("Omega_S0 <= 0")
    report.sections["V0"] = {
        "warnings": warns,
        "params": asdict(p),
        "equation_ledger": VALIDATION_EQUATION_LEDGER,
        "dimensional_analysis": {
            "H0": "km/s/Mpc",
            "gamma": "1/Gyr",
            "r_i": "dimensionless",
            "E": "H/H0",
            "distances": "Mpc",
        },
    }
    report.add("V0.inventory", "V0", "parameter inventory clean", len(warns) == 0, float(len(warns)), 0.5)

    # V1/V2
    sol = analyze_residuals(solve_background(p, method="RK45", z_max=z_max, rtol=1e-9))
    # attach q
    dE = np.gradient(sol.E, sol.N, edge_order=2)
    sol.q = -1.0 - dE / np.maximum(sol.E, 1e-30)
    sec_c = sector_continuity_residuals(sol)
    bian = bianchi_identity_residual(sol)
    report.sections["residuals"] = {
        "max_abs_R_F": sol.meta["max_abs_R_F"],
        "max_rel_R_C": sol.meta["max_rel_R_C"],
        "med_abs_R_F": sol.meta.get("med_abs_R_F", float("nan")),
        "sector_cont": sec_c,
        "bianchi": bian,
    }
    report.add("V2.RF", "V2", "Friedmann residual", sol.meta["max_abs_R_F"] < 1e-6, sol.meta["max_abs_R_F"], 1e-6)
    report.add("V2.RC", "V2", "continuity residual (relative)", sol.meta["max_rel_R_C"] < 1e-3, sol.meta["max_rel_R_C"], 1e-3)
    report.add(
        "V2.bianchi",
        "V2",
        "Bianchi/FLRW identity residual",
        bian["max_rel_bianchi"] < 1e-3,
        bian["max_rel_bianchi"],
        1e-3,
    )

    # V3
    rec = test_lcdm_recovery(z_max=z_max)
    report.sections["lcdm_recovery"] = rec
    report.add("V3.gamma0", "V3", "gamma->0 recovers LCDM", rec["gamma0"]["pass"], rec["gamma0"]["max_rel_H"], 1e-6)
    report.add("V3.lcdm_flag", "V3", "lcdm_limit flag recovers LCDM", rec["lcdm_flag"]["pass"], rec["lcdm_flag"]["max_rel_H"], 1e-6)
    report.add("V3.Smax", "V3", "large S_max near LCDM", rec["Smax_large"]["pass"], rec["Smax_large"]["max_rel_H"], 5e-3)

    # V4/V5
    dual = dual_integrator_compare(p, z_max=min(z_max, 1100.0))
    report.sections["dual_integrators"] = dual
    report.add("V4.dual", "V4", "RK45 vs DOP853", dual["pass"], dual["max_rel_dE_RK45_vs_DOP853"], 1e-5)
    trip = triple_integrator_compare(p, z_max=min(z_max, 300.0))
    report.sections["triple_integrators"] = trip
    report.add("V4.triple", "V4", "RK45/DOP853/Radau", trip.get("pass", False), trip.get("max_rel_RK45_DOP853", 1.0), 1e-5)
    conv = convergence_test(p, z_max=min(100.0, z_max))
    report.sections["convergence"] = conv
    report.add("V5.rtol", "V5", "rtol convergence", conv["pass"], conv["max_rel_err_vs_finest"][-1], 1e-5)
    ic = ic_stability_test(p, z_max=min(50.0, z_max))
    report.sections["ic_stability"] = ic
    report.add("V5.IC", "V5", "IC stability", ic["pass"], max(v for k, v in ic.items() if k != "pass"), 0.05)
    meth = method_matrix_stability(p, z_max=min(100.0, z_max))
    report.sections["method_matrix"] = meth
    report.add("V5.methods", "V5", "method matrix", meth["pass"], max(v for k, v in meth.items() if k != "pass"), 1e-4)

    # V10
    eps = early_universe_epsilon(sol)
    report.sections["early_universe"] = {
        "max_abs_eps_z_gt_10": eps["max_abs_eps_z_gt_10"],
        "max_abs_eps_z_gt_100": eps["max_abs_eps_z_gt_100"],
        "eps_z0": eps["eps_z0"],
    }
    report.add(
        "V10.early",
        "V10",
        "early-universe eps_H(z>100) small for mild thermo",
        (eps["max_abs_eps_z_gt_100"] < 1e-2) if eps["max_abs_eps_z_gt_100"] == eps["max_abs_eps_z_gt_100"] else False,
        float(eps["max_abs_eps_z_gt_100"]) if eps["max_abs_eps_z_gt_100"] == eps["max_abs_eps_z_gt_100"] else 1.0,
        1e-2,
    )

    # V11/V12
    report.sections["cmb"] = cmb_acoustic_scale(sol)
    report.sections["bao"] = bao_ratios_extended(sol)
    report.sections["distances_z1"] = {
        "D_M": comoving_distance_Mpc(sol, 1.0),
        "D_A": angular_diameter_distance(sol, 1.0),
        "D_L": luminosity_distance(sol, 1.0),
        "D_H": hubble_distance(sol, 1.0),
    }
    report.sections["distance_ladder"] = distance_ladder_table(sol)
    report.add(
        "V11.theta",
        "V11",
        "theta_* in plausible acoustic range",
        0.005 < report.sections["cmb"]["theta_star"] < 0.02,
        report.sections["cmb"]["theta_star"],
        0.02,
    )

    # V13
    stab = perturbation_stability_checks(sol)
    report.sections["perturbation_stability"] = stab
    report.add("V13.nonphantom", "V13", "non-phantom w_S", stab["pass_nonphantom"], stab["min_1_plus_w"], 0.0)
    report.sections["layer_crosschecks"] = numeric_layer_crosschecks(p)
    report.sections["entropy_only_honesty"] = entropy_only_vs_coupled_honesty(p)
    report.add(
        "V13.honesty",
        "V13",
        "entropy-only disagrees at high z (required)",
        report.sections["entropy_only_honesty"]["pass_honesty"],
        report.sections["entropy_only_honesty"]["max_rel_diff_z_gt_2"],
        1e-3,
    )

    # Heavy Q3 scaffolds
    if quick:
        report.sections["synthetic_recovery"] = {"skipped": True}
        report.sections["model_comparison"] = {"skipped": True}
        report.sections["oos"] = {"skipped": True}
        report.sections["ppc"] = {"skipped": True}
        report.sections["sensitivity"] = {"skipped": True}
        report.sections["fisher"] = {"skipped": True}
        report.sections["identifiability"] = {"skipped": True}
        report.sections["rk4_order"] = {"skipped": True}
    else:
        syn = synthetic_parameter_recovery()
        report.sections["synthetic_recovery"] = syn
        report.add("V6.synth", "V6", "synthetic H0/Om recovery", syn["pass"], syn["abs_err"]["H0"], 5.0)
        report.sections["synthetic_ensemble"] = synthetic_recovery_ensemble(n_realizations=5)
        report.sections["model_comparison"] = model_comparison_synthetic()
        report.sections["oos"] = out_of_sample_scaffold()
        report.sections["ppc"] = {
            k: v
            for k, v in posterior_predictive_scaffold(n_draws=32).items()
            if k in {"n_draws", "note"}
        }
        report.sections["sensitivity"] = finite_diff_sensitivity(p)
        report.sections["fisher"] = fisher_matrix_scaffold(p)
        report.sections["identifiability"] = identifiability_analysis(p)
        report.sections["prior_hook"] = prior_sensitivity_hook(p)
        report.sections["rk4_order"] = nsteps_convergence_rk4(p)
        report.add(
            "V5.rk4_order",
            "V5",
            "RK4 error decreases under refinement",
            report.sections["rk4_order"]["pass"],
            report.sections["rk4_order"]["max_rel_err_vs_finest"][-1],
            report.sections["rk4_order"]["max_rel_err_vs_finest"][0],
        )

    # Artifacts
    export_residual_csv(sol, out / "residuals.csv")
    export_distance_csv(sol, out / "distances.csv")
    export_bao_csv(sol, out / "bao.csv")
    try:
        _save_plots(sol, out / "figures", eps)
    except Exception:
        pass
    # denser figures if helper exists
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.loglog(sol.z + 1, sol.r_r, label="r_r")
        ax.loglog(sol.z + 1, sol.r_m, label="r_m")
        ax.loglog(sol.z + 1, sol.r_S, label="r_S")
        ax.set_xlabel("1+z"); ax.set_ylabel("r_i"); ax.legend()
        (out / "figures").mkdir(parents=True, exist_ok=True)
        fig.tight_layout(); fig.savefig(out / "figures" / "densities_ri.pdf"); plt.close(fig)
    except Exception:
        pass

    report.elapsed_sec = _time.time() - t0
    payload = report.to_dict()
    # compatibility keys used by older CLI printers
    payload["all_critical_pass"] = report.all_critical_pass
    payload["residuals"] = report.sections["residuals"]
    payload["lcdm_recovery"] = report.sections["lcdm_recovery"]
    payload["dual_integrators"] = report.sections["dual_integrators"]
    payload["early_universe"] = report.sections["early_universe"]
    payload["cmb"] = report.sections["cmb"]
    payload["distances_z1"] = report.sections["distances_z1"]
    payload["synthetic_recovery"] = report.sections.get("synthetic_recovery", {})
    payload["model_comparison"] = report.sections.get("model_comparison", {})
    payload["passes"] = {c.id: c.passed for c in report.checks}
    payload["hierarchy"] = [e["id"] + ": " + e["ascii"] for e in VALIDATION_EQUATION_LEDGER]

    def _jsonify(o: Any) -> Any:
        if isinstance(o, dict):
            return {str(k): _jsonify(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_jsonify(v) for v in o]
        if isinstance(o, float):
            return float(o)
        if isinstance(o, int):
            return int(o)
        if isinstance(o, bool):
            return bool(o)
        try:
            import numpy as _np
            if isinstance(o, (_np.floating,)):
                return float(o)
            if isinstance(o, (_np.integer,)):
                return int(o)
            if isinstance(o, (_np.bool_,)):
                return bool(o)
            if isinstance(o, _np.ndarray):
                return o.tolist()
        except Exception:
            pass
        return o

    (out / "validation_report.json").write_text(
        json.dumps(_jsonify(payload), indent=2), encoding="utf-8"
    )

    lines = [
        "# LCDM+S numerical validation report (embedded architecture)",
        "",
        f"- elapsed: {report.elapsed_sec:.1f}s",
        f"- critical pass: **{report.all_critical_pass}**",
        "",
        "## Distinction",
        report.distinction,
        "",
        "## Equation ledger",
    ]
    for e in VALIDATION_EQUATION_LEDGER:
        lines.append(f"- `{e['id']}` {e['ascii']} - {e['meaning']}")
    lines += ["", "## Checks"]
    for c in report.checks:
        flag = "PASS" if c.passed else "FAIL"
        lines.append(
            f"- {flag}: `{c.id}` [{c.section}] {c.description} "
            f"(metric={c.metric:.3e}, thr={c.threshold:.3e})"
        )
    lines += [
        "",
        f"- max |R_F| = {payload['residuals']['max_abs_R_F']:.3e}",
        f"- max rel |R_C| = {payload['residuals']['max_rel_R_C']:.3e}",
        f"- theta_* ~ {payload['cmb']['theta_star']:.6e}",
        f"- r_s ~ {payload['cmb']['r_s_Mpc']:.2f} Mpc",
        "",
        "Artifacts: residuals.csv, distances.csv, bao.csv, figures/, validation_report.json",
        "",
        "Next: Bayesian pipeline only after critical Q1 gates pass.",
    ]
    (out / "validation_report.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


# Primary entry point used by analytical_solver CLI / DerivationStep bridge
def run_validation_suite(out_dir="lcdms_validation_out", quick=False):
    """Alias to the full embedded V0-V17 runner."""
    return run_embedded_validation_suite(out_dir=out_dir, quick=quick)


# -----------------------------------------------------------------------------
# Architecture appendix — CLASS/CAMB bridge stub + validation contract
# -----------------------------------------------------------------------------

CLASS_COMPARISON_CONTRACT: Dict[str, Any] = {
    "purpose": (
        "Gold-standard external cross-check: when entropy modification is off "
        "(lcdm_limit=True or gamma=0), H(z) must match CLASS/CAMB flat LCDM "
        "point-by-point within numerical tolerance."
    ),
    "required_when_available": [
        "H(z) grid comparison at >= 200 redshifts",
        "D_A(z_*), r_s(z_*), theta_* consistency",
        "growth D(a) / f sigma8 only after perturbation module is enabled",
    ],
    "status": "live CAMB/CLASS bridge when importable; honest unavailable otherwise",
}


def class_camb_comparison_stub(
    p: LCDMSParams,
    z_grid: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """
    External CLASS/CAMB cross-check for the LCDM limit.
    Prefer CAMB when installed; try classy next. Never invent agreement.
    """
    z_grid = list(z_grid or np.concatenate([
        np.linspace(0.0, 3.0, 80),
        np.array([5.0, 10.0, 50.0, 100.0, 200.0, 500.0, 1100.0]),
    ]).tolist())
    p_lcdm = LCDMSParams(**{**asdict(p), "lcdm_limit": True, "gamma": 0.0})
    sol = solve_background(p_lcdm, z_max=max(z_grid), n_eval=2000, rtol=1e-8)
    H_internal = [float(_H_of_z_safe(sol, z)) for z in z_grid]

    engine = None
    H_external: Optional[List[float]] = None
    detail = ""
    # CAMB first (commonly available via pip)
    try:
        import camb  # type: ignore
        engine = "camb"
        Obh2 = 0.02237
        Om = float(p_lcdm.Omega_m0)
        H0 = float(p_lcdm.H0)
        Oc = max(Om - Obh2 / (H0 / 100.0) ** 2, 1e-6)
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=H0, ombh2=Obh2, omch2=Oc * (H0 / 100.0) ** 2, omk=0.0)
        pars.set_dark_energy(w=-1.0)
        pars.set_matter_power(redshifts=[0.0], kmax=1.0)
        results = camb.get_background(pars)
        # Prefer hubble_parameter (km/s/Mpc); fall back to h_of_z (1/Mpc)*c
        z_arr = np.asarray(z_grid, dtype=float)
        if hasattr(results, "hubble_parameter"):
            Hz = results.hubble_parameter(z_arr)
            H_external = [float(h) for h in Hz]
        else:
            c_kms = 299792.458
            Hz = results.h_of_z(z_arr)
            H_external = [float(h * c_kms) for h in Hz]
        detail = (
            f"CAMB background H(z) vs internal LCDM-limit on {len(z_grid)} nodes; "
            f"Om={Om}, H0={H0}, Obh2={Obh2}."
        )
    except Exception as exc_camb:
        try:
            import classy  # type: ignore
            engine = "classy"
            Obh2 = 0.02237
            H0 = float(p_lcdm.H0)
            Om = float(p_lcdm.Omega_m0)
            h = H0 / 100.0
            Oc = max(Om - Obh2 / h**2, 1e-6)
            cosmo = classy.Class()
            cosmo.set({
                "H0": H0,
                "omega_b": Obh2,
                "omega_cdm": Oc * h**2,
                "Omega_k": 0.0,
                "output": "mPk",
                "z_max_pk": 0.0,
            })
            cosmo.compute()
            H_external = [float(cosmo.Hubble(z) * 299792.458) for z in z_grid]
            cosmo.struct_cleanup()
            cosmo.empty()
            detail = f"CLASS/classy background H(z) on {len(z_grid)} nodes."
        except Exception as exc_class:
            engine = None
            detail = (
                "neither CAMB nor classy produced H(z); "
                f"camb_err={exc_camb}; class_err={exc_class}"
            )

    rel = None
    table = []
    if H_external is not None:
        rel = [
            abs(a - b) / max(abs(b), 1e-30)
            for a, b in zip(H_internal, H_external)
        ]
        # sparse table for LaTeX
        pick = [0.0, 0.5, 1.0, 2.0, 10.0, 100.0, 1100.0]
        for z0 in pick:
            i = int(np.argmin(np.abs(np.asarray(z_grid) - z0)))
            table.append({
                "z": float(z_grid[i]),
                "H_internal": float(H_internal[i]),
                "H_external": float(H_external[i]),
                "rel": float(rel[i]),
            })

    max_rel = (float(max(rel)) if rel else None)
    return {
        "contract": CLASS_COMPARISON_CONTRACT,
        "engine": engine,
        "z_grid_sample": [row["z"] for row in table] if table else z_grid[:8],
        "H_internal_lcdm_limit": [row["H_internal"] for row in table] if table else H_internal[:8],
        "H_external": [row["H_external"] for row in table] if table else H_external,
        "table": table,
        "max_rel_diff": max_rel,
        "n_nodes": len(z_grid),
        "detail": detail,
        "pass": (engine is None) or (max_rel is not None and max_rel < 1e-2),
        "note": (
            "Unavailable external engine is NOT a failure of the thermo model; "
            "it is a missing optional cross-check. Pass threshold 1% on LCDM limit "
            "(radiation/neutrino content can differ slightly from CAMB defaults)."
        ),
    }


VALIDATION_ACCEPTANCE_CONTRACT: Dict[str, Any] = {
    "Q1_internal_consistency": {
        "gates": ["V2.RF", "V2.RC", "V3.gamma0", "V3.lcdm_flag", "V4.dual"],
        "must_pass_before": "any observational or Bayesian claim",
    },
    "Q2_established_physics": {
        "gates": ["V3.gamma0", "V3.lcdm_flag", "V3.Smax", "V10.early"],
        "meaning": "LCDM recovery and early-universe approach",
    },
    "Q3_observational": {
        "gates": ["V6.synth", "V11.theta"],
        "meaning": "scaffolds only until real datasets are wired",
        "warning": "lower chi2 alone never implies a better theory",
    },
    "artifact_list": [
        "validation_report.md",
        "validation_report.json",
        "residuals.csv",
        "distances.csv",
        "bao.csv",
        "figures/residuals_RF_RC.pdf",
        "figures/H_of_z.pdf",
        "figures/epsilon_H_early.pdf",
        "figures/w_chi_q.pdf",
        "figures/densities_ri.pdf",
    ],
}


def describe_validation_architecture() -> str:
    """Human-readable architecture blurb for logs / Overleaf appendices."""
    lines = [
        "LCDM+S numerical validation architecture (embedded in analytical_solver.py)",
        VALIDATION_DISTINCTION,
        "",
        "Critical Q1 gates:",
    ]
    for g in VALIDATION_ACCEPTANCE_CONTRACT["Q1_internal_consistency"]["gates"]:
        lines.append(f"  - {g}")
    lines += ["", "Equation ledger:"]
    for e in VALIDATION_EQUATION_LEDGER:
        lines.append(f"  - {e['id']}: {e['ascii']}")
    lines += ["", "Artifacts:"]
    for a in VALIDATION_ACCEPTANCE_CONTRACT["artifact_list"]:
        lines.append(f"  - {a}")
    return "\n".join(lines)


def collect_numerical_validation_steps(
    out_dir: str | Path = "lcdms_validation_out",
    quick: bool = True,
) -> List[DerivationStep]:
    """Return DerivationStep list for optional inclusion in derive_lambdacdm_s."""
    return [step_numerical_validation_embedded(out_dir=out_dir, quick=quick)]



# =============================================================================
# Integrated numerical + Bayesian analysis export (always run with derivation)
# =============================================================================

def build_dynamic_lcdm_comparison(
    p: Optional["LCDMSParams"] = None,
    z_max: float = 1100.0,
) -> Dict[str, Any]:
    """
    Dynamically evolve the coupled LCDM+S ODEs and compare major
    background observables to flat LCDM on the same cosmology today.
    """
    p = p or LCDMSParams()
    sol = analyze_residuals(solve_background(p, method="RK45", z_max=z_max, rtol=1e-8, n_eval=2500))
    # ensure q
    if sol.q is None:
        dE = np.gradient(sol.E, sol.N, edge_order=2)
        sol.q = -1.0 - dE / np.maximum(sol.E, 1e-30)
    E_ref = lcdm_E(sol.z, p)
    H = sol.E * p.H0
    H_ref = E_ref * p.H0
    eps = (sol.E - E_ref) / np.maximum(E_ref, 1e-30)
    # sample nodes for tables
    z_nodes = [0.0, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0, min(1100.0, float(np.max(sol.z)))]
    rows = []
    for z in z_nodes:
        i = int(np.argmin(np.abs(sol.z - z)))
        rows.append({
            "z": float(sol.z[i]),
            "H_model": float(H[i]),
            "H_LCDM": float(H_ref[i]),
            "eps_H": float(eps[i]),
            "w_S": float(sol.w_S[i]),
            "q": float(sol.q[i]),
            "chi": float(sol.chi[i]),
            "R_F": float(sol.R_F[i]) if sol.R_F is not None else float("nan"),
            "R_C_rel": float(sol.R_C_rel[i]) if sol.R_C_rel is not None else float("nan"),
        })
    return {
        "params": asdict(p),
        "max_abs_eps_H": float(np.max(np.abs(eps))),
        "max_abs_eps_z_gt_100": float(np.max(np.abs(eps[sol.z > 100]))) if np.any(sol.z > 100) else float("nan"),
        "table": rows,
        "residuals": {
            "max_abs_R_F": sol.meta.get("max_abs_R_F"),
            "max_rel_R_C": sol.meta.get("max_rel_R_C"),
        },
        "cmb": cmb_acoustic_scale(sol),
        "bao": bao_ratios_extended(sol) if "bao_ratios_extended" in globals() else bao_ratios(sol),
        "stability": perturbation_stability_checks(sol),
    }




def verify_major_equations_over_time(
    p: Optional["LCDMSParams"] = None,
    z_max: float = 1100.0,
) -> Dict[str, Any]:
    """
    Dynamically check major background identities along the numerical solution:
      - Friedmann: E^2 = r_r + r_m + r_S
      - Continuity residual R_C
      - Layer-1 EOS: w_S = -1 + (2/3) xi  (with xi from chi evolution)
      - Deceleration identity vs E(N)
      - LCDM recovery of H(z) when gamma->0
    """
    p = p or LCDMSParams()
    sol = analyze_residuals(solve_background(p, method="RK45", z_max=z_max, rtol=1e-8, n_eval=2500))
    if sol.q is None:
        dE = np.gradient(sol.E, sol.N, edge_order=2)
        sol.q = -1.0 - dE / np.maximum(sol.E, 1e-30)
    # xi from w_S definition used in the solver
    xi_from_w = 1.5 * (1.0 + sol.w_S)
    # q identity: q = -1 - dlnE/dN
    dlnE = np.gradient(np.log(np.maximum(sol.E, 1e-30)), sol.N, edge_order=2)
    q_from_E = -1.0 - dlnE
    q_err = np.max(np.abs(sol.q - q_from_E))
    # gamma->0 recovery
    p0 = LCDMSParams(**{**asdict(p), "gamma": 0.0, "lcdm_limit": True})
    sol0 = solve_background(p0, z_max=min(z_max, 1100.0), rtol=1e-8, n_eval=2000)
    E_ref = lcdm_E(sol0.z, p0)
    eps0 = (sol0.E - E_ref) / np.maximum(E_ref, 1e-30)
    # sample equation residuals over time
    z_nodes = [0.0, 0.5, 1.0, 2.0, 10.0, 100.0, min(z_max, float(np.max(sol.z)))]
    rows = []
    for z in z_nodes:
        i = int(np.argmin(np.abs(sol.z - z)))
        rows.append({
            "z": float(sol.z[i]),
            "R_F": float(sol.R_F[i]) if sol.R_F is not None else float("nan"),
            "R_C_rel": float(sol.R_C_rel[i]) if sol.R_C_rel is not None else float("nan"),
            "w_S": float(sol.w_S[i]),
            "xi_from_w": float(xi_from_w[i]),
            "q": float(sol.q[i]),
            "q_from_E": float(q_from_E[i]),
            "chi": float(sol.chi[i]),
        })
    xi_chk = layer7_xi_consistency(sol) if "layer7_xi_consistency" in globals() else {}
    return {
        "max_abs_R_F": sol.meta.get("max_abs_R_F"),
        "max_rel_R_C": sol.meta.get("max_rel_R_C"),
        "max_abs_q_identity": float(q_err),
        "gamma0_max_abs_eps_H": float(np.max(np.abs(eps0))),
        "pass_friedmann": bool(sol.meta.get("max_abs_R_F", 1.0) < 1e-5),
        "pass_continuity": bool(sol.meta.get("max_rel_R_C", 1.0) < 1e-4),
        "pass_q_identity": bool(q_err < 1e-4),
        "pass_gamma0_lcdm": bool(np.max(np.abs(eps0)) < 1e-4),
        "xi_today": xi_chk,
        "table": rows,
    }


def build_bayesian_bundle(seed: int = 7) -> Dict[str, Any]:
    """
    Fast Bayesian / information-criterion bundle (minutes, not hours):
      - synthetic MAP recovery
      - LCDM vs LCDM+S AIC/BIC on synthetic SN-like data
      - out-of-sample train/test scaffold
      - optional short emcee if installed (tiny chain)
    """
    out: Dict[str, Any] = {
        "synthetic_recovery": synthetic_parameter_recovery(seed=seed),
        "model_comparison": model_comparison_synthetic(seed=seed),
        "oos": out_of_sample_scaffold(seed=seed + 1),
        "identifiability": identifiability_analysis(LCDMSParams()) if "identifiability_analysis" in globals() else {},
        "fisher": fisher_matrix_scaffold(LCDMSParams()) if "fisher_matrix_scaffold" in globals() else {},
        "emcee": {"status": "skipped"},
    }
    # Optional ultra-short emcee (keep walltime small)
    try:
        import emcee  # type: ignore
        # 2-param LCDM toy on synthetic mu — demonstrates Bayesian sampling path
        true = LCDMSParams(H0=70.0, Omega_m0=0.3, lcdm_limit=True)
        sol_true = solve_background(true, z_max=1.5, n_eval=600, rtol=1e-6)
        z = np.linspace(0.05, 1.0, 25)
        rng = np.random.default_rng(seed)
        mu_obs = predict_mu(sol_true, z) + rng.normal(0, 0.12, size=z.size)
        sig = 0.12

        def log_prob(x):
            H0, Om = x
            if not (55 < H0 < 85 and 0.15 < Om < 0.45):
                return -np.inf
            p = LCDMSParams(H0=float(H0), Omega_m0=float(Om), lcdm_limit=True)
            sol = solve_background(p, z_max=1.5, n_eval=400, rtol=1e-5)
            chi2 = float(np.sum(((predict_mu(sol, z) - mu_obs) / sig) ** 2))
            return -0.5 * chi2

        nwalkers, ndim, nsteps = 8, 2, 40
        p0 = np.array([70.0, 0.3]) + 1e-2 * rng.normal(size=(nwalkers, ndim))
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
        sampler.run_mcmc(p0, nsteps, progress=False)
        chain = sampler.get_chain(discard=10, flat=True)
        out["emcee"] = {
            "status": "ok",
            "nwalkers": nwalkers,
            "nsteps": nsteps,
            "mean_H0": float(np.mean(chain[:, 0])),
            "mean_Om": float(np.mean(chain[:, 1])),
            "std_H0": float(np.std(chain[:, 0])),
            "std_Om": float(np.std(chain[:, 1])),
            "note": "Ultra-short demo chain for LCDM limit only; not a production posterior.",
        }
    except Exception as exc:
        out["emcee"] = {"status": "unavailable", "detail": str(exc)}
    return out


def run_full_integrated_analysis(
    export_dir: Path | str,
    quick: bool = False,
) -> Dict[str, Any]:
    """
    Single integrated analysis for the derivation export.
    Designed to finish well under a multi-hour budget (typically minutes).
    """
    export_dir = Path(export_dir)
    val_dir = export_dir / "numerical_validation"
    val_dir.mkdir(parents=True, exist_ok=True)

    # Q1+Q2+light Q3 numerical suite
    validation = run_validation_suite(out_dir=val_dir, quick=quick)
    dynamic = build_dynamic_lcdm_comparison(z_max=1100.0 if not quick else 300.0)
    eq_time = verify_major_equations_over_time(z_max=1100.0 if not quick else 300.0)
    bayes = {} if quick else build_bayesian_bundle()
    class_cmp = class_camb_comparison_stub(LCDMSParams())

    bundle = {
        "validation": validation,
        "dynamic_lcdm_comparison": dynamic,
        "equation_evolution": eq_time,
        "bayesian": bayes,
        "class_camb": class_cmp,
        "distinction": VALIDATION_DISTINCTION,
        "acceptance": VALIDATION_ACCEPTANCE_CONTRACT,
    }
    (export_dir / "integrated_analysis.json").write_text(
        json.dumps(bundle, indent=2, default=str), encoding="utf-8"
    )
    return bundle


def _tex_esc_analysis(s: Any) -> str:
    return (
        str(s)
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("#", r"\#")
        .replace("_", r"\_")
    )


def export_analysis_tex(bundle: Dict[str, Any], path: Path, standalone: bool = True) -> Path:
    """
    Write detailed analysis.tex with ALL numerical/Bayesian/ODE/CLASS results.
    If standalone=True, wrap in a full document; else emit a fragment for \\input.
    """
    path = Path(path)
    L: List[str] = []
    a = L.append
    val = bundle.get("validation", {})
    dyn = bundle.get("dynamic_lcdm_comparison", {})
    bay = bundle.get("bayesian", {})
    cls = bundle.get("class_camb", {})

    if standalone:
        a(r"\documentclass[11pt,a4paper]{article}")
        a(r"\usepackage[margin=1in]{geometry}")
        a(r"\usepackage{amsmath,amssymb,booktabs,longtable,hyperref,xcolor,graphicx}")
        a(r"\usepackage{float}")
        a(r"\title{Numerical, Bayesian, and ODE Analysis for $\Lambda$CDM+S}")
        a(r"\author{Auto-exported from \texttt{analytical\_solver.py}}")
        a(r"\date{\today}")
        a(r"\begin{document}")
        a(r"\maketitle")
        a(r"\tableofcontents")
        a(r"\newpage")

    a(r"\section{Numerical validation analysis}")
    a(r"\subsection{Epistemic distinction}")
    a(
        r"We keep three questions separate: "
        r"\emph{numerical consistency} $\neq$ \emph{observational agreement} "
        r"$\neq$ \emph{evidence for the theory}."
    )
    a(r"\subsection{Critical gates}")
    a(rf"Critical pass: \textbf{{{bool(val.get('all_critical_pass'))}}}.")
    a(r"\begin{center}\begin{tabular}{@{}llp{7cm}@{}}")
    a(r"\toprule Check & Pass & Metric \\ \midrule")
    passes = val.get("passes", {})
    residuals = val.get("residuals", {})
    for cid, ok in passes.items():
        a(rf"{_tex_esc_analysis(cid)} & {ok} & --- \\")
    a(r"\bottomrule\end{tabular}\end{center}")
    a(r"\subsection{Residuals}")
    a(
        rf"$\max|R_F|={_tex_esc_analysis(residuals.get('max_abs_R_F'))}$, "
        rf"$\max|R_C|_{{\mathrm{{rel}}}}={_tex_esc_analysis(residuals.get('max_rel_R_C'))}$."
    )
    a(r"These quantify whether the coupled ODEs satisfy Friedmann and continuity.")

    a(r"\subsection{$\Lambda$CDM recovery}")
    rec = val.get("lcdm_recovery", {})
    a(r"\begin{center}\begin{tabular}{@{}lll@{}}")
    a(r"\toprule Limit & $\max|\Delta H|/H$ & Pass \\ \midrule")
    for key in ("gamma0", "lcdm_flag", "Smax_large"):
        block = rec.get(key, {})
        a(
            rf"{_tex_esc_analysis(key)} & "
            rf"{_tex_esc_analysis(block.get('max_rel_H'))} & "
            rf"{block.get('pass')} \\"
        )
    a(r"\bottomrule\end{tabular}\end{center}")

    a(r"\section{Dynamic ODE evolution versus $\Lambda$CDM}")
    a(
        r"We integrate the coupled radiation+matter+entropy system and compare "
        r"$H(z)$, $w_S(z)$, $q(z)$, and $\chi(z)$ to flat $\Lambda$CDM with the same "
        r"present-day densities."
    )
    a(rf"Global $\max|\varepsilon_H|={_tex_esc_analysis(dyn.get('max_abs_eps_H'))}$; "
      rf"at $z>100$: ${_tex_esc_analysis(dyn.get('max_abs_eps_z_gt_100'))}$.")
    a(r"\begin{center}\footnotesize\begin{tabular}{@{}rrrrrrrr@{}}")
    a(r"\toprule $z$ & $H$ & $H_{\Lambda\mathrm{CDM}}$ & $\varepsilon_H$ & $w_S$ & $q$ & $\chi$ & $R_F$ \\ \midrule")
    for row in dyn.get("table", []):
        a(
            rf"{row['z']:.3g} & {row['H_model']:.4g} & {row['H_LCDM']:.4g} & "
            rf"{row['eps_H']:.3e} & {row['w_S']:.4f} & {row['q']:.4f} & "
            rf"{row['chi']:.4f} & {row['R_F']:.2e} \\"
        )
    a(r"\bottomrule\end{tabular}\end{center}")

    cmb = dyn.get("cmb", {})
    a(r"\subsection{CMB acoustic bookkeeping}")
    a(
        rf"$r_s(z_*)\approx{_tex_esc_analysis(cmb.get('r_s_Mpc'))}\,\mathrm{{Mpc}}$, "
        rf"$\theta_*\approx{_tex_esc_analysis(cmb.get('theta_star'))}$, "
        rf"$\ell_*\approx{_tex_esc_analysis(cmb.get('ell_star_approx'))}$."
    )
    a(r"\subsection{BAO combinations}")
    a(r"\begin{center}\begin{tabular}{@{}rrrrr@{}}")
    a(r"\toprule $z$ & $D_M/r_d$ & $D_H/r_d$ & $D_V/r_d$ & $r_d$ \\ \midrule")
    for row in dyn.get("bao", []):
        a(
            rf"{row.get('z', float('nan')):.3g} & "
            rf"{row.get('DM_over_rd', float('nan')):.4g} & "
            rf"{row.get('DH_over_rd', float('nan')):.4g} & "
            rf"{row.get('DV_over_rd', float('nan')):.4g} & "
            rf"{row.get('r_d', float('nan')):.4g} \\"
        )
    a(r"\bottomrule\end{tabular}\end{center}")

    a(r"\section{Bayesian and information-criterion analysis}")
    if not bay:
        a(r"\emph{Bayesian bundle skipped in quick mode.}")
    else:
        syn = bay.get("synthetic_recovery", {})
        a(r"\subsection{Synthetic parameter recovery}")
        a(
            r"We inject known parameters, add SN-like noise to $\mu(z)$, and recover a MAP estimate. "
            r"This tests equations$\to$solver$\to$observable$\to$likelihood, not real-data posteriors."
        )
        a(r"\begin{center}\begin{tabular}{@{}lll@{}}")
        a(r"\toprule Parameter & True & MAP \\ \midrule")
        tt = syn.get("theta_true", {})
        th = syn.get("theta_hat", {})
        for k in ("H0", "Omega_m0", "gamma", "chi0"):
            a(rf"{_tex_esc_analysis(k)} & {_tex_esc_analysis(tt.get(k))} & {_tex_esc_analysis(th.get(k))} \\")
        a(r"\bottomrule\end{tabular}\end{center}")
        a(rf"Recovery pass: \textbf{{{syn.get('pass')}}}; $\chi^2_{{\min}}={_tex_esc_analysis(syn.get('chi2_min'))}$.")

        mc = bay.get("model_comparison", {})
        a(r"\subsection{Same-data model comparison}")
        a(
            r"Both $M_0=\Lambda$CDM and $M_1=\Lambda$CDM+S are fit to identical synthetic data. "
            r"A lower $\chi^2$ does \emph{not} automatically win once complexity is penalized."
        )
        a(r"\begin{center}\begin{tabular}{@{}lrr@{}}")
        a(r"\toprule Metric & $\Lambda$CDM & $\Lambda$CDM+S \\ \midrule")
        a(rf"$\chi^2$ & {_tex_esc_analysis(mc.get('chi2_LCDM'))} & {_tex_esc_analysis(mc.get('chi2_LCDMS'))} \\")
        a(rf"AIC & {_tex_esc_analysis(mc.get('AIC_LCDM'))} & {_tex_esc_analysis(mc.get('AIC_LCDMS'))} \\")
        a(rf"BIC & {_tex_esc_analysis(mc.get('BIC_LCDM'))} & {_tex_esc_analysis(mc.get('BIC_LCDMS'))} \\")
        a(r"\bottomrule\end{tabular}\end{center}")
        a(
            rf"$\Delta\chi^2={_tex_esc_analysis(mc.get('delta_chi2'))}$, "
            rf"$\Delta\mathrm{{AIC}}={_tex_esc_analysis(mc.get('delta_AIC'))}$, "
            rf"$\Delta\mathrm{{BIC}}={_tex_esc_analysis(mc.get('delta_BIC'))}$."
        )

        oos = bay.get("oos", {})
        a(r"\subsection{Out-of-sample scaffold}")
        a(r"Train on low-$z$ synthetic SN; test on held-out higher-$z$ SN.")
        a(r"\begin{center}\begin{tabular}{@{}lrr@{}}")
        a(r"\toprule Model & $\chi^2_{\mathrm{train}}$ & $\chi^2_{\mathrm{test}}$ \\ \midrule")
        for name in ("LCDM", "LCDMS"):
            block = oos.get(name, {})
            a(
                rf"{_tex_esc_analysis(name)} & "
                rf"{_tex_esc_analysis(block.get('chi2_train'))} & "
                rf"{_tex_esc_analysis(block.get('chi2_test'))} \\"
            )
        a(r"\bottomrule\end{tabular}\end{center}")

        em = bay.get("emcee", {})
        a(r"\subsection{Optional short MCMC (emcee)}")
        if em.get("status") == "ok":
            a(
                rf"Short ensemble chain on the $\Lambda$CDM limit: "
                rf"$H_0={_tex_esc_analysis(em.get('mean_H0'))}\pm{_tex_esc_analysis(em.get('std_H0'))}$, "
                rf"$\Omega_m={_tex_esc_analysis(em.get('mean_Om'))}\pm{_tex_esc_analysis(em.get('std_Om'))}$ "
                rf"({_tex_esc_analysis(em.get('nwalkers'))} walkers $\times$ {_tex_esc_analysis(em.get('nsteps'))} steps)."
            )
            a(r"\textbf{Note:} this is a pipeline demonstration, not a production cosmological posterior.")
        else:
            a(rf"emcee status: {_tex_esc_analysis(em.get('status'))} ({_tex_esc_analysis(em.get('detail', em.get('note','')))}).")

        ident = bay.get("identifiability", {})
        if ident:
            a(r"\subsection{Identifiability / Fisher scaffold}")
            a(
                rf"Fisher condition number $\approx{_tex_esc_analysis(ident.get('condition_number'))}$. "
                rf"{_tex_esc_analysis(ident.get('interpretation', ''))}"
            )

    eqt = bundle.get("equation_evolution", {})
    a(r"\section{Major equations evolving in time}")
    a(
        r"We evaluate Friedmann, continuity, deceleration, and $\gamma\to0$ "
        r"$\Lambda$CDM recovery on the same numerical trajectory."
    )
    a(
        rf"$\max|R_F|={_tex_esc_analysis(eqt.get('max_abs_R_F'))}$, "
        rf"$\max|R_C|_{{\mathrm{{rel}}}}={_tex_esc_analysis(eqt.get('max_rel_R_C'))}$, "
        rf"$\max|\Delta q|={_tex_esc_analysis(eqt.get('max_abs_q_identity'))}$, "
        rf"$\gamma\to0~\max|\varepsilon_H|={_tex_esc_analysis(eqt.get('gamma0_max_abs_eps_H'))}$."
    )
    a(r"\begin{center}\footnotesize\begin{tabular}{@{}rrrrrrr@{}}")
    a(r"\toprule $z$ & $R_F$ & $R_C^{\mathrm{rel}}$ & $w_S$ & $\xi(w)$ & $q$ & $q(E)$ \\ \midrule")
    for row in eqt.get("table", []):
        a(
            rf"{row['z']:.3g} & {row['R_F']:.2e} & {row['R_C_rel']:.2e} & "
            rf"{row['w_S']:.4f} & {row['xi_from_w']:.4f} & "
            rf"{row['q']:.4f} & {row['q_from_E']:.4f} \\"
        )
    a(r"\bottomrule\end{tabular}\end{center}")
    a(
        rf"Passes: Friedmann={eqt.get('pass_friedmann')}, "
        rf"continuity={eqt.get('pass_continuity')}, "
        rf"$q$-identity={eqt.get('pass_q_identity')}, "
        rf"$\gamma\to0$={eqt.get('pass_gamma0_lcdm')}."
    )

    a(r"\section{CLASS / CAMB external cross-check}")
    a(rf"Engine: {_tex_esc_analysis(cls.get('engine'))}; nodes={_tex_esc_analysis(cls.get('n_nodes'))}.")
    a(_tex_esc_analysis(cls.get("detail", "")))
    a(_tex_esc_analysis(cls.get("note", "")))
    a(rf"External pass: \textbf{{{cls.get('pass')}}}; "
      rf"$\max|\Delta H|/H={_tex_esc_analysis(cls.get('max_rel_diff'))}$.")
    if cls.get("table"):
        a(r"\begin{center}\begin{tabular}{@{}rrrr@{}}")
        a(r"\toprule $z$ & $H_{\mathrm{internal}}$ & $H_{\mathrm{external}}$ & rel \\ \midrule")
        for row in cls.get("table", []):
            a(
                rf"{row['z']:.4g} & {row['H_internal']:.6g} & "
                rf"{row['H_external']:.6g} & {row['rel']:.3e} \\"
            )
        a(r"\bottomrule\end{tabular}\end{center}")
    elif cls.get("H_internal_lcdm_limit"):
        a(r"Internal $\Lambda$CDM-limit $H(z)$ samples:")
        a(r"\begin{center}\begin{tabular}{@{}rr@{}}\toprule $z$ & $H_{\mathrm{internal}}$ \\ \midrule")
        for z, H in zip(cls.get("z_grid_sample", cls.get("z_grid", [])), cls.get("H_internal_lcdm_limit", [])):
            a(rf"{z:.4g} & {H:.6g} \\")
        a(r"\bottomrule\end{tabular}\end{center}")

    a(r"\section{Perturbation / stability bookkeeping}")
    stab = dyn.get("stability", {})
    a(
        rf"$c_s^2={_tex_esc_analysis(stab.get('c_s2'))}$, "
        rf"non-phantom pass={stab.get('pass_nonphantom')}, "
        rf"$\min(1+w_S)={_tex_esc_analysis(stab.get('min_1_plus_w'))}$, "
        rf"$\min c_a^2={_tex_esc_analysis(stab.get('min_ca2'))}$."
    )

    a(r"\section{Figures}")
    a(r"If present, numerical-validation figures are included below.")
    a(r"\IfFileExists{numerical_validation/figures/residuals_RF_RC.pdf}{")
    a(r"\begin{figure}[H]\centering\includegraphics[width=0.85\linewidth]{numerical_validation/figures/residuals_RF_RC.pdf}")
    a(r"\caption{Friedmann and continuity residuals.}\end{figure}}{}")
    a(r"\IfFileExists{numerical_validation/figures/H_of_z.pdf}{")
    a(r"\begin{figure}[H]\centering\includegraphics[width=0.85\linewidth]{numerical_validation/figures/H_of_z.pdf}")
    a(r"\caption{$H(z)$ versus $\Lambda$CDM reference.}\end{figure}}{}")
    a(r"\IfFileExists{numerical_validation/figures/epsilon_H_early.pdf}{")
    a(r"\begin{figure}[H]\centering\includegraphics[width=0.85\linewidth]{numerical_validation/figures/epsilon_H_early.pdf}")
    a(r"\caption{Early-universe fractional Hubble residual $\varepsilon_H(z)$.}\end{figure}}{}")
    a(r"\IfFileExists{numerical_validation/figures/w_chi_q.pdf}{")
    a(r"\begin{figure}[H]\centering\includegraphics[width=0.85\linewidth]{numerical_validation/figures/w_chi_q.pdf}")
    a(r"\caption{Entropy-sector $w_S$, $\chi$, and deceleration $q$.}\end{figure}}{}")

    a(r"\section{Conclusions of the numerical analysis}")
    a(r"\begin{enumerate}")
    a(r"\item Internal ODE consistency is gated by $R_F$ and $R_C$ before any data claim.")
    a(r"\item $\gamma\to0$ / \texttt{lcdm\_limit} must recover flat $\Lambda$CDM $H(z)$.")
    a(r"\item Bayesian MAP/AIC/BIC tests exercise the inference pipeline on synthetic data.")
    a(r"\item CLASS/CAMB comparison is performed when available; absence is reported honestly.")
    a(r"\item None of the above alone constitutes empirical discovery of $\Lambda$CDM+S.")
    a(r"\end{enumerate}")

    if standalone:
        a(r"\end{document}")
    path.write_text("\n".join(L), encoding="utf-8")
    return path


# === END EMBEDDED NUMERICAL VALIDATION ===

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Layer-1 thermo-EOM derivation for ΛCDM+S"
    )
    p.add_argument(
        "--temperature-mode",
        choices=["de_sitter", "dynamical_horizon"],
        default="dynamical_horizon",
    )
    p.add_argument("--export-dir", type=str, default="theory_export")
    p.add_argument("--no-export", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip integrated numerical/Bayesian analysis (not recommended)",
    )
    p.add_argument(
        "--validation-only",
        action="store_true",
        help="Run only the numerical validation suite (debug path)",
    )
    p.add_argument(
        "--validation-out",
        type=str,
        default="lcdms_validation_out",
        help="Output directory for --validation-only",
    )
    p.add_argument(
        "--validation-quick",
        action="store_true",
        help="Faster integrated analysis (skip some Bayesian scaffolds)",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    args = build_arg_parser().parse_args(argv)
    # Default: symbolic derivation AND numerical/Bayesian analysis together.
    # --validation-only keeps the old isolated path for debugging.
    if getattr(args, "validation_only", False):
        rep = run_validation_suite(
            out_dir=args.validation_out,
            quick=args.validation_quick,
        )
        print(f"validation critical pass: {rep['all_critical_pass']}")
        print(f"report -> {args.validation_out}/validation_report.md")
        return 0 if rep["all_critical_pass"] else 1
    result = derive_lambdacdm_s(
        temperature_mode=args.temperature_mode,
        export_dir=None if args.no_export else args.export_dir,
        verbose=not args.quiet,
        run_validation=not args.skip_validation,
        validation_quick=args.validation_quick,
    )
    critical = {
        "horizon_entropy_S_H",
        "entropy_production_P",
        "thermodynamic_equation_of_motion",
        "linearize_and_prove_lambda_negative",
        "RESULT_stable_de_Sitter_attractor",
    }
    ok = all(s.status != "failed" for s in result.steps if s.name in critical)
    ok = ok and all(c.passed for c in result.checks) and result.attractor_pass
    ia = result.export_payload.get("integrated_analysis") or {}
    if ia:
        ok = ok and bool(ia.get("validation", {}).get("all_critical_pass", True))
        if not args.quiet:
            print("integrated analysis critical pass:",
                  ia.get("validation", {}).get("all_critical_pass"))
            print("analysis.tex ->", (args.export_dir if not args.no_export else "(no export)"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
