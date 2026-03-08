from pathlib import Path

from tempus_copilot.output_schema import validate_run_outputs


def test_validate_run_outputs_detects_checksum_mismatch(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
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
    (run_dir / "run_metadata.toml").write_text(
        'schema_version = "1.0.0"\noutput_checksum_sha256 = "bad"\n',
        encoding="utf-8",
    )
    errors = validate_run_outputs(run_dir)
    assert "Checksum mismatch between metadata and outputs" in errors
