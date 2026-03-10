import runpy
import sys
from pathlib import Path

import pytest

from tempus_copilot.cli import main, run
from tempus_copilot.config import load_settings
from tests.helpers.output_builders import write_valid_run_dir


def test_cli_run_returns_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = load_settings(Path("config/defaults.toml"))
    settings = settings.model_copy(update={"output_dir": tmp_path})
    monkeypatch.setattr("tempus_copilot.cli.run_pipeline", lambda *_, **__: None)
    code = run(settings)
    assert code == 0


def test_cli_validate_output_command_returns_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run_pipeline(*_: object, **__: object) -> None:
        write_valid_run_dir(tmp_path / "run_20260309_000000")

    monkeypatch.setattr("tempus_copilot.cli.run_pipeline", fake_run_pipeline)
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


def test_cli_fail_on_low_confidence_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run_pipeline(*_: object, **__: object) -> None:
        raise ValueError("Confidence threshold violated")

    monkeypatch.setattr("tempus_copilot.cli.run_pipeline", fake_run_pipeline)
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
    monkeypatch.setattr("tempus_copilot.cli.run_pipeline", lambda *_, **__: None)
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
    write_valid_run_dir(run_dir)

    original = sys.argv
    try:
        sys.argv = ["tempus-copilot", "validate-output", str(run_dir)]
        with pytest.raises(SystemExit) as exc:
            runpy.run_module("tempus_copilot.cli", run_name="__main__")
    finally:
        sys.argv = original
    assert exc.value.code == 0
