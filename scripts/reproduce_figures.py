#!/usr/bin/env python3
"""Generate background comparison figures from the packaged model.

This script does not replay unpublished MCMC posteriors. It draws the
fiducial ΛCDM vs ΛCDM+S expansion history using ``solve_background``.
Full paper MCMC figures are produced by ``scripts/run_validation.py``
when artifacts are enabled.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _cli_utils import REPO_ROOT, add_common_args, default_config_path, load_yaml


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write fiducial background figures for ΛCDM vs ΛCDM+S",
    )
    add_common_args(parser)
    args = parser.parse_args(argv)

    cfg_path = args.config
    if cfg_path is None and default_config_path().is_file():
        cfg_path = default_config_path()
    cfg = load_yaml(cfg_path)
    fid = cfg.get("fiducial") or {}
    paths = cfg.get("paths") or {}
    outdir = Path(args.output or paths.get("figures") or (REPO_ROOT / "figures"))
    outdir.mkdir(parents=True, exist_ok=True)

    from lcdm_plus_s.bayesian_validation import (
        BackgroundParams,
        FIDUCIAL_H0,
        FIDUCIAL_K_GYR,
        FIDUCIAL_OMEGA_LAMBDA,
        FIDUCIAL_T_CRIT_GYR,
        KMSMPC_TO_INVGYR,
        background_plot_data,
        render_background_plots,
    )

    params = BackgroundParams(
        H0_kms_mpc=float(fid.get("H0", FIDUCIAL_H0)),
        Omega_Lambda=float(fid.get("Omega_Lambda", FIDUCIAL_OMEGA_LAMBDA)),
        k_gyr=float(fid.get("k", FIDUCIAL_K_GYR)),
        t_crit_gyr=float(fid.get("t_crit", FIDUCIAL_T_CRIT_GYR)),
        omega_r0=float(fid.get("omega_r0", 9.0e-5)),
        lcdm_limit=False,
    )
    lcdm = BackgroundParams(
        H0_kms_mpc=params.H0_kms_mpc,
        Omega_Lambda=params.Omega_Lambda,
        k_gyr=params.k_gyr,
        t_crit_gyr=params.t_crit_gyr,
        omega_r0=params.omega_r0,
        lcdm_limit=True,
    )
    data = background_plot_data(params)
    written = render_background_plots(data, outdir=str(outdir))
    # Always also record H0 conversion used on the Hubble axis.
    readme = outdir / "generated_captions.md"
    readme.write_text(
        "\n".join(
            [
                "# Generated background figures",
                "",
                "Fiducial θ from configs/default.yaml (Table 1 means in code):",
                f"- H0 = {params.H0_kms_mpc} km s⁻¹ Mpc⁻¹ "
                f"({params.H0_gyr:.6f} Gyr⁻¹ using KMSMPC_TO_INVGYR={KMSMPC_TO_INVGYR})",
                f"- Ω_Λ ≡ Ω_S,0 = {params.Omega_Lambda}",
                f"- k = {params.k_gyr} Gyr⁻¹",
                f"- t_crit = {params.t_crit_gyr} Gyr",
                "",
                "ΛCDM comparison uses the same (H0, Ω_Λ) with `lcdm_limit=True`.",
                "These are model curves at the code fiducial, not posterior means.",
                "",
                "Files:",
                *[f"- {name}: {prod.path}" for name, prod in (written or {}).items()
                  if getattr(prod, "path", None)],
            ]
        ),
        encoding="utf-8",
    )
    if args.verbose:
        print(f"wrote figures under {outdir}", flush=True)
        print(f"captions → {readme}", flush=True)
    del lcdm
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
