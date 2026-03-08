from pathlib import Path

from tempus_copilot.config import load_settings
from tempus_copilot.pipeline import run_pipeline


def test_pipeline_writes_toml_outputs(tmp_path: Path) -> None:
    settings = load_settings(Path("config/defaults.toml"))
    settings = settings.model_copy(update={"output_dir": tmp_path})
    result = run_pipeline(settings)
    assert result.ranked_providers_path.exists()
    assert result.objection_handlers_path.exists()
    assert result.meeting_scripts_path.exists()
