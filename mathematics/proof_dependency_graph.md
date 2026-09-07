# Proof dependency graph

```mermaid
flowchart TD
  FLRW[G0.1 ASSUMED FLRW]
  RH[G0.3 JUSTIFIED R_H=1/H]
  BH[G0.4 JUSTIFIED S_H=A/4G]
  Smax[G0.5 ASSUMED finite S_max]
  GSL[G0.6 ASSUMED GSL]
  SH[G1.5 DERIVED S_H=π/(G H²)]
  MON[G2.2 THEOREM monotonicity]
  CLS[G5.2 ASSUMED logistic closure]
  GCL[G5.1 ASSUMED Gamma-class]
  Hinf[G7.2 CONDITIONAL H_∞]
  STAB[G8.2 CONDITIONAL λ=-γ]
  ADS[G9.1 CONDITIONAL a~e^{H t}]
  FR[G11.2 STANDARD Raychaudhuri]
  LAM[G14.2 CONDITIONAL Λ_S]
  AE[G16.1 CONDITIONAL asymp Einstein]
  GFE[G16.2 NOT from Module G]
  IE[G19.3 CONDITIONAL I_E=-S_max]
  CLAU[J1.2 Clausius]
  RAY[J2.3 Raychaudhuri eq]
  NULL[J3.5 null projection]
  EFE[J5.1 CONDITIONAL full EFE]
  FLRW --> RH --> SH
  BH --> SH --> MON
  GSL -.->|insufficient alone| Hinf
  Smax --> Hinf
  GCL --> Hinf
  CLS --> STAB --> ADS
  SH --> Hinf --> LAM --> AE
  FR --> AE
  AE -.-> GFE
  Hinf --> IE
  CLAU --> NULL
  RAY --> NULL --> EFE
  EFE -.->|FLRW specialize| SH
```

Module J (local) and Module G (global) are separate; EFE is conditional in J.
