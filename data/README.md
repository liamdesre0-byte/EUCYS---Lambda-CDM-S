# Data

Compressed inference vectors from public survey releases. Formats are the
loader's native ones (`#`-comment headers).

| File | Contents | Role |
| :--- | :--- | :--- |
| `pantheon_plus.csv` | 1465 SNe: `z mu sigma` (Pantheon+, SH0ES calibrators removed) | training |
| `des_sny5.csv` | 1820 unique SNe: `CID z mu sigma_total` (1623 DES + 197 external low-z; σ = √(MUERR² + VPEC² + SYS²)) | **held out** — never enters the training likelihood |
| `shoes_h0.csv` | SH0ES local H₀ constraint | training |

Counts are verified by the loader at run time (`n_unique_sn`,
`n_measurements` in dataset metadata). DESI DR2 / BOSS DR12 BAO and the
Planck 2018 compressed vector are built into
`src/lcdm_plus_s/bayesian_validation.py` with their covariances.

Credits: Pantheon+ (Brout et al. 2022; Scolnic et al. 2022), DES-SN5YR
Dovekie HD (DES Collaboration 2024), SH0ES (Riess et al. 2022).
