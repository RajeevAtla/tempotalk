"""Thin wrapper that makes the package CLI importable from the repo root."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def run() -> int:
    """Runs the package CLI after bootstrapping the import path.

    Returns:
        Process exit code from the package CLI.
    """
    from tempus_copilot.cli import main

    return main()


if __name__ == "__main__":
    raise SystemExit(run())
