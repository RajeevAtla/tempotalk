from pathlib import Path

from tempus_copilot.cli import run
from tempus_copilot.config import load_settings


def test_cli_run_returns_zero(tmp_path: Path) -> None:
    settings = load_settings(Path("config/defaults.toml"))
    settings = settings.model_copy(update={"output_dir": tmp_path})
    code = run(settings)
    assert code == 0
