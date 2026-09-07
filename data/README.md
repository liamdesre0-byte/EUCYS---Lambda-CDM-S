# Survey data (not bundled)

Production MCMC in this package uses **compressed literature vectors
already stored in** `src/lcdm_plus_s/bayesian_validation.py`
(DESI DR2 BAO, BOSS DR12 BAO/RSD, Planck 2018-style distance priors,
SH0ES \(H_0\)). Those numbers are small public summary statistics.

Do **not** commit full Pantheon+, DES-SN Y5, SPARC, or WMAP releases
into git.

## Optional supernova CSVs

If you have licensed catalogs, place them here (any of the names the
loader accepts):

```text
pantheon_plus.csv
des_sny5.csv
```

Then:

```text
python scripts/run_validation.py --data-dir data --require-sn --entropy-sector
```

Official starting points (retrieve yourself, respect licenses):

- Pantheon+: [github.com/PantheonPlusSH0ES](https://github.com/PantheonPlusSH0ES/DataRelease)
- DES-SN Y5: DES collaboration data releases
- SPARC: [astroweb.cwru.edu/SPARC](http://astroweb.cwru.edu/SPARC/)
- Planck: [pla.esac.esa.int](https://pla.esac.esa.int/)
- DESI BAO: DESI collaboration papers / data releases cited in the
  source (`DESI Collaboration 2024/2025`)
