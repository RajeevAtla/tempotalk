from pathlib import Path

import tomli_w

from tempus_copilot.output_schema import compute_output_checksum, parse_toml, validate_run_outputs


def _write_minimal_outputs(run_dir: Path) -> None:
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


def test_validate_run_outputs_detects_checksum_mismatch(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    _write_minimal_outputs(run_dir)
    (run_dir / "run_metadata.toml").write_text(
        'schema_version = "1.0.0"\noutput_checksum_sha256 = "bad"\n',
        encoding="utf-8",
    )
    errors = validate_run_outputs(run_dir)
    assert "Checksum mismatch between metadata and outputs" in errors


def test_validate_run_outputs_without_metadata_has_no_checksum_error(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    _write_minimal_outputs(run_dir)
    errors = validate_run_outputs(run_dir)
    assert "Checksum mismatch between metadata and outputs" not in errors
    assert "Missing output file: run_metadata.toml" in errors


def test_parse_toml_and_checksum_happy_path(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    _write_minimal_outputs(run_dir)
    checksum = compute_output_checksum(run_dir)
    metadata = {
        "schema_version": "1.0.0",
        "output_checksum_sha256": checksum,
        "baml_schema_sha256": "a",
        "baml_prompt_sha256": "b",
    }
    (run_dir / "run_metadata.toml").write_bytes(tomli_w.dumps(metadata).encode("utf-8"))
    parsed = parse_toml(run_dir / "ranked_providers.toml")
    assert parsed["schema_version"] == "1.0.0"
    assert validate_run_outputs(run_dir) == []
