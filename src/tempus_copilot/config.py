"""Configuration models and loaders for TempoTalk."""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel

from tempus_copilot.models import RankingCalibration, RankingWeights


class ModelSettings(BaseModel):
    """Defines generation and embedding runtime settings."""

    generation_provider: str
    generation_model: str
    embedding_provider: str
    embedding_model: str


class RagSettings(BaseModel):
    """Configures chunking, retrieval, and embedding retry behavior."""

    chunk_size: int
    chunk_overlap: int
    top_k: int
    embedding_dimension: int
    request_retries: int
    backoff_seconds: float


class OutputSettings(BaseModel):
    """Controls output validation and citation policy."""

    strict_citations: bool = False


class Settings(BaseModel):
    """Collects the full application configuration surface."""

    market_csv: Path
    crm_csv: Path
    kb_markdown: Path
    output_dir: Path
    models: ModelSettings
    ranking_weights: RankingWeights
    ranking_calibration: RankingCalibration
    rag: RagSettings
    output: OutputSettings
    mock_seed: int
    mock_scale: int


def load_settings(path: Path) -> Settings:
    """Loads typed settings from a TOML file.

    Args:
        path: Path to the TOML configuration file.

    Returns:
        Parsed application settings.
    """
    # Keep the loader explicit so config key changes fail near their source.
    with path.open("rb") as fh:
        raw = tomllib.load(fh)
    paths = raw["paths"]
    models = raw["models"]
    ranking = raw["ranking"]["weights"]
    calibration = raw["ranking"]["calibration"]
    rag = raw["rag"]
    output = raw.get("output", {"strict_citations": False})
    mock = raw["mock"]
    return Settings(
        market_csv=Path(paths["market_csv"]),
        crm_csv=Path(paths["crm_csv"]),
        kb_markdown=Path(paths["kb_markdown"]),
        output_dir=Path(paths["output_dir"]),
        models=ModelSettings(**models),
        ranking_weights=RankingWeights(**ranking),
        ranking_calibration=RankingCalibration(**calibration),
        rag=RagSettings(**rag),
        output=OutputSettings(**output),
        mock_seed=int(mock["seed"]),
        mock_scale=int(mock["scale"]),
    )
