"""Shared CLI helpers for scripts/ (config YAML, repo sys.path)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def load_yaml(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit(
            "PyYAML is required for --config. Install with: pip install pyyaml"
        ) from exc
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"config {path} must be a mapping")
    return data


def default_config_path() -> Path:
    return REPO_ROOT / "configs" / "default.yaml"


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML config (default: configs/default.yaml when the file exists)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Master RNG seed (overrides config mcmc.seed)",
    )
    parser.add_argument(
        "--output",
        "--outdir",
        dest="output",
        type=Path,
        default=None,
        help="Output directory for artifacts",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print extra progress (default for these wrappers)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce logging",
    )
