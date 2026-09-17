# Paper materials

This directory is for the written report associated with the software.

| File | Status |
| --- | --- |
| `LCDMS_CWSF_Written_Report.tex` | source already present with the project |
| `EUCYS_Final_Paper.pdf` | **not included** — no PDF was supplied with the two Python files |

To add a camera-ready PDF, compile your EUCYS/CWSF manuscript and save
it here as `EUCYS_Final_Paper.pdf`.

The packaged MCMC default matches that paper: 48 walkers, 50,000
production steps, 2,000 burn-in, 3 affine ensembles, ESS 75,000,
7,200,000 posterior samples, entropy sector ON. Do not treat prints
from `python scripts/run_validation.py --quick` (or the unmodified
`original/Bayesian_Validationn.py` frozen-sector path) as those numbers.

The TeX author line in the bundled report names **Liam Desre**.
