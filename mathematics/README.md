# Mathematics index

This folder contains the **first-principles derivation materials that
already existed** next to `analytical_solver.py`. Nothing here is a new
theorem invented during packaging.

| File | Origin |
| --- | --- |
| `mathematical_framework.md` | equations as implemented in the two Python sources |
| `analytical_derivation.md` | export from the solver (`proof_export`) |
| `assumption_ledger.md` | tagged assumptions / not-derived claims |
| `counterexample_report.md` | GSL-alone does not imply de Sitter |
| `proof_dependency_graph.md` | module dependency notes |
| `referee_report.md` | internal referee-style notes |
| `LambdaCDM_S_Complete_Derivation.tex` | Overleaf-ready derivation |
| `LambdaCDM_S_First_Principles_Derivation.tex` | same family |

A compiled `derivations.pdf` is **not** committed (no binary was
provided). Generate one by compiling the TeX or by running

```text
python scripts/run_solver.py --skip-validation --export-dir theory_export
```

and using the matplotlib/TeX artifacts the solver writes.
