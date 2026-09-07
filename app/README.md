# Interactive model explorer

This Streamlit app calls the **same** `lcdm_plus_s` package used by the
scripts. It does not contain a second copy of the Friedmann equation.

```text
python -m pip install -e ".[app]"
python -m streamlit run app/streamlit_app.py
```

Then open http://localhost:8501.

What you can change: \(H_0\), \(\Omega_\Lambda\equiv\Omega_{S,0}\), \(k\),
\(t_{\mathrm{crit}}\). What you see: \(H(z)\), \(\chi(t)\), \(w_S(t)\),
and \(\mu(z)\) for ΛCDM+S versus the `lcdm_limit=True` ΛCDM limit.

What this app does **not** do: it does not run production MCMC, does not
load Pantheon+/DES catalogs, and does not display fabricated posterior
intervals. For inference see `python scripts/run_validation.py`.
