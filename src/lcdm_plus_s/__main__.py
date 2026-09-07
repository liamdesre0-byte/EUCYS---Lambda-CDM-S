"""Package CLI: python -m lcdm_plus_s"""

from __future__ import annotations


def main() -> int:
    print("ΛCDM+S package. Use:")
    print("  python scripts/run_solver.py")
    print("  python scripts/run_validation.py")
    print("  python -m streamlit run app/streamlit_app.py")
    print("  python -m pytest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
