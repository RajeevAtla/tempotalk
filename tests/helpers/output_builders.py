from __future__ import annotations

from pathlib import Path

import tomli_w

from tempus_copilot.output_schema import compute_output_checksum


def write_minimal_outputs(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "ranked_providers.toml").write_text(
        'schema_version = "1.0.0"\nproviders = []\n',
        encoding="utf-8",
    )
    (run_dir / "objection_handlers.toml").write_text(
        'schema_version = "1.0.0"\nobjections = []\n',
        encoding="utf-8",
    )
    (run_dir / "meeting_scripts.toml").write_text(
        'schema_version = "1.0.0"\nscripts = []\n',
        encoding="utf-8",
    )
    (run_dir / "retrieval_debug.toml").write_text(
        'schema_version = "1.0.0"\nretrieval_debug = []\n',
        encoding="utf-8",
    )


def write_valid_run_dir(run_dir: Path) -> None:
    write_minimal_outputs(run_dir)
    checksum = compute_output_checksum(run_dir)
    metadata = {
        "schema_version": "1.0.0",
        "output_checksum_sha256": checksum,
        "baml_schema_sha256": "x",
        "baml_prompt_sha256": "y",
    }
    (run_dir / "run_metadata.toml").write_bytes(tomli_w.dumps(metadata).encode("utf-8"))
