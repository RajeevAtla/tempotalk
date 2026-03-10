import sys
from pathlib import Path

import pytest

from tempus_copilot.cli import main, run
from tempus_copilot.config import load_settings


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
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    settings = load_settings(Path("config/defaults.toml")).model_copy(
        update={"output_dir": tmp_path}
    )
    with pytest.raises(ValueError):
        run(settings, fail_on_low_confidence=0.95)
