#!/usr/bin/env python3
"""Run the Layer-1 analytical solver (symbolic derivation + optional numerics).

Examples
--------
python scripts/run_solver.py
python scripts/run_solver.py --quick
python scripts/run_solver.py --skip-validation --export-dir theory_export
python scripts/run_solver.py --validation-only --validation-quick
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _cli_utils import (
    REPO_ROOT,
    add_common_args,
    default_config_path,
    load_yaml,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ΛCDM+S Layer-1 analytical solver (analytical_solver.py)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_args(parser)
    parser.add_argument(
        "--temperature-mode",
        choices=["de_sitter", "dynamical_horizon"],
        default=None,
    )
    parser.add_argument("--export-dir", type=str, default=None)
    parser.add_argument("--no-export", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--validation-out", type=str, default=None)
    parser.add_argument("--validation-quick", action="store_true")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Alias for --validation-quick (faster integrated analysis)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg_path = args.config
    if cfg_path is None and default_config_path().is_file():
        cfg_path = default_config_path()
    cfg = load_yaml(cfg_path)
    solver_cfg = cfg.get("analytical_solver") or {}
    paths = cfg.get("paths") or {}

    temperature = args.temperature_mode or solver_cfg.get(
        "temperature_mode", "dynamical_horizon"
    )
    export_dir = args.export_dir or (
        str(args.output) if args.output else paths.get("export_dir", "theory_export")
    )
    validation_out = args.validation_out or paths.get(
        "validation_out", "lcdms_validation_out"
    )
    quick = bool(args.validation_quick or args.quick)
    quiet = bool(args.quiet and not args.verbose)

    forwarded: list[str] = ["--temperature-mode", str(temperature)]
    if args.no_export:
        forwarded.append("--no-export")
    else:
        forwarded.extend(["--export-dir", str(Path(export_dir))])
    if quiet:
        forwarded.append("--quiet")
    if args.skip_validation:
        forwarded.append("--skip-validation")
    if args.validation_only:
        forwarded.append("--validation-only")
        forwarded.extend(["--validation-out", str(validation_out)])
    if quick:
        forwarded.append("--validation-quick")

    if args.verbose:
        print("run_solver forwarding:", " ".join(forwarded), flush=True)

    from lcdm_plus_s.analytical_solver import main as solver_main

    # Export paths are resolved relative to the repository root.
    return solver_main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
