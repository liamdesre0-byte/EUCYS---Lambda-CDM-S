# Figures

Commit **code that generates figures**, not fabricated result plots.

```text
python scripts/reproduce_figures.py --output figures --verbose
```

writes fiducial ΛCDM vs ΛCDM+S background curves at Table 1 means.
Captions are stored in `generated_captions.md` (gitignored together
with PNG/PDF).

MCMC trace/corner/GetDist figures appear under `results/figures/` after
`python scripts/run_validation.py` with artifacts enabled. Those files
describe **that sampling run only**.
