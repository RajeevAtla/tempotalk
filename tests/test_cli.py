import sys
from pathlib import Path

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
