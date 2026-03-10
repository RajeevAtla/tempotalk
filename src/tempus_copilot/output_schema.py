"""Validation helpers for schema-versioned pipeline outputs."""

from __future__ import annotations

import tomllib
from hashlib import sha256
from pathlib import Path

REQUIRED_TOP_LEVEL = {
    "ranked_providers.toml": ["schema_version", "providers"],
    "objection_handlers.toml": ["schema_version", "objections"],
    "meeting_scripts.toml": ["schema_version", "scripts"],
    "retrieval_debug.toml": ["schema_version", "retrieval_debug"],
    "run_metadata.toml": [
        "schema_version",
        "output_checksum_sha256",
        "baml_schema_sha256",
        "baml_prompt_sha256",
    ],
}


def parse_toml(path: Path) -> dict[str, object]:
    """Parses a TOML file into a plain mapping.

    Args:
        path: Path to the TOML file.

    Returns:
        Parsed top-level payload.
    """
    return tomllib.loads(path.read_text(encoding="utf-8"))


def compute_output_checksum(run_dir: Path) -> str:
    """Computes the canonical checksum for user-facing run artifacts.

    Args:
        run_dir: Directory containing pipeline output files.

    Returns:
        SHA-256 checksum over ranked, objection, and script outputs.
    """
    # The checksum intentionally tracks only the user-facing generated artifacts.
    ranked = (run_dir / "ranked_providers.toml").read_text(encoding="utf-8")
    objections = (run_dir / "objection_handlers.toml").read_text(encoding="utf-8")
    scripts = (run_dir / "meeting_scripts.toml").read_text(encoding="utf-8")
    return sha256("|".join([ranked, objections, scripts]).encode("utf-8")).hexdigest()


def validate_run_outputs(run_dir: Path) -> list[str]:
    """Validates required files, top-level keys, and checksum integrity.

    Args:
        run_dir: Directory containing pipeline output files.

    Returns:
        Validation error messages. An empty list means the run is valid.
    """
    errors: list[str] = []
    for file_name, keys in REQUIRED_TOP_LEVEL.items():
        path = run_dir / file_name
        if not path.exists():
            errors.append(f"Missing output file: {file_name}")
            continue
        payload = parse_toml(path)
        for key in keys:
            if key not in payload:
                errors.append(f"{file_name} missing key: {key}")
    metadata_path = run_dir / "run_metadata.toml"
    if metadata_path.exists():
        metadata = parse_toml(metadata_path)
        actual = compute_output_checksum(run_dir)
        expected = str(metadata.get("output_checksum_sha256", ""))
        if actual != expected:
            errors.append("Checksum mismatch between metadata and outputs")
    return errors
