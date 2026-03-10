import sys
import runpy
from pathlib import Path

import pytest
import tomli_w

from tempus_copilot.cli import main, run
from tempus_copilot.config import load_settings
from tempus_copilot.output_schema import compute_output_checksum


def test_cli_run_returns_zero(tmp_path: Path) -> None:
    settings = load_settings(Path("config/defaults.toml"))
    settings = settings.model_copy(update={"output_dir": tmp_path})
    code = run(settings)
    assert code == 0


def test_cli_validate_output_command_returns_zero(tmp_path: Path) -> None:
    settings = load_settings(Path("config/defaults.toml")).model_copy(
        update={"output_dir": tmp_path}
    )
    run(settings)
    run_dir = max(tmp_path.iterdir(), key=lambda p: p.name)
    original = sys.argv
    try:
        sys.argv = ["tempus-copilot", "validate-output", str(run_dir)]
        code = main()
    finally:
        sys.argv = original
    assert code == 0


def test_cli_fail_on_low_confidence_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:1")
    settings = load_settings(Path("config/defaults.toml")).model_copy(
        update={"output_dir": tmp_path}
    )
    with pytest.raises(ValueError):
        run(settings, fail_on_low_confidence=0.95)


def test_cli_main_defaults_to_run_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = load_settings(Path("config/defaults.toml")).model_copy(
        update={"output_dir": tmp_path}
    )
    monkeypatch.setattr("tempus_copilot.cli.load_settings", lambda _: settings)
    original = sys.argv
    try:
        sys.argv = ["tempus-copilot"]
        code = main()
    finally:
        sys.argv = original
    assert code == 0


def test_cli_validate_output_command_prints_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    original = sys.argv
    try:
        sys.argv = ["tempus-copilot", "validate-output", str(tmp_path)]
        code = main()
    finally:
        sys.argv = original
    out = capsys.readouterr().out
    assert code == 1
    assert "ERROR: Missing output file:" in out


def test_cli_returns_two_for_unexpected_command_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Args:
        command = "unexpected"

    monkeypatch.setattr(
        "tempus_copilot.cli.argparse.ArgumentParser.parse_args",
        lambda self: _Args(),
    )
    code = main()
    assert code == 2


def test_cli_module_main_guard_executes_and_exits_zero(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "ranked_providers.toml").write_text(
        'schema_version = "1.0.0"\nproviders = []\n', encoding="utf-8"
    )
    (run_dir / "objection_handlers.toml").write_text(
        'schema_version = "1.0.0"\nobjections = []\n', encoding="utf-8"
    )
    (run_dir / "meeting_scripts.toml").write_text(
        'schema_version = "1.0.0"\nscripts = []\n', encoding="utf-8"
    )
    (run_dir / "retrieval_debug.toml").write_text(
        'schema_version = "1.0.0"\nretrieval_debug = []\n', encoding="utf-8"
    )
    checksum = compute_output_checksum(run_dir)
    metadata = {
        "schema_version": "1.0.0",
        "output_checksum_sha256": checksum,
        "baml_schema_sha256": "x",
        "baml_prompt_sha256": "y",
    }
    (run_dir / "run_metadata.toml").write_bytes(tomli_w.dumps(metadata).encode("utf-8"))

    original = sys.argv
    try:
        sys.argv = ["tempus-copilot", "validate-output", str(run_dir)]
        with pytest.raises(SystemExit) as exc:
            runpy.run_module("tempus_copilot.cli", run_name="__main__")
    finally:
        sys.argv = original
    assert exc.value.code == 0
