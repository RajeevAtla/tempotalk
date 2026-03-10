"""Command-line entrypoints for running and validating TempoTalk output."""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from tempus_copilot.config import Settings, load_settings
from tempus_copilot.output_schema import validate_run_outputs
from tempus_copilot.pipeline import run_pipeline


def run(
    settings: Settings,
    strict_citations: bool = False,
    fail_on_low_confidence: float | None = None,
) -> int:
    """Executes the pipeline for one configuration.

    Args:
        settings: Loaded application settings.
        strict_citations: Whether to filter unsupported citations.
        fail_on_low_confidence: Optional minimum confidence threshold.

    Returns:
        Process exit code.
    """
    run_pipeline(
        settings,
        strict_citations=strict_citations,
        fail_on_low_confidence=fail_on_low_confidence,
    )
    return 0


def main() -> int:
    """Parses CLI arguments and dispatches the requested command.

    Returns:
        Process exit code.
    """
    load_dotenv()
    parser = argparse.ArgumentParser(description="Tempus Sales Copilot CLI")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run pipeline")
    run_parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/defaults.toml"),
        help="Path to TOML config file",
    )
    run_parser.add_argument(
        "--strict-citations",
        action="store_true",
        help="Enforce retrieved citation whitelist on generated outputs",
    )
    run_parser.add_argument(
        "--fail-on-low-confidence",
        type=float,
        default=None,
        help="Fail run if any generated confidence is below threshold (0.0-1.0)",
    )

    validate_parser = subparsers.add_parser(
        "validate-output",
        help="Validate output schema/checksum",
    )
    validate_parser.add_argument("run_dir", type=Path, help="Run directory containing TOML outputs")

    args = parser.parse_args()
    if args.command in (None, "run"):
        config_path = getattr(args, "config", Path("config/defaults.toml"))
        strict = bool(getattr(args, "strict_citations", False))
        threshold = getattr(args, "fail_on_low_confidence", None)
        settings = load_settings(config_path)
        return run(
            settings,
            strict_citations=strict,
            fail_on_low_confidence=threshold,
        )

    if args.command == "validate-output":
        errors = validate_run_outputs(args.run_dir)
        if errors:
            for err in errors:
                print(f"ERROR: {err}")
            return 1
        print("Output validation passed")
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
