from __future__ import annotations

import argparse
from pathlib import Path

from tempus_copilot.config import Settings, load_settings
from tempus_copilot.pipeline import run_pipeline


def run(settings: Settings) -> int:
    run_pipeline(settings)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Tempus Sales Copilot CLI")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/defaults.toml"),
        help="Path to TOML config file",
    )
    args = parser.parse_args()
    settings = load_settings(args.config)
    return run(settings)


if __name__ == "__main__":
    raise SystemExit(main())
