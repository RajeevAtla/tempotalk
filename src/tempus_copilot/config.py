from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel

from tempus_copilot.models import RankingCalibration, RankingWeights


class ModelSettings(BaseModel):
    generation_provider: str
    generation_model: str
    embedding_provider: str
    embedding_model: str


class RagSettings(BaseModel):
    chunk_size: int
    chunk_overlap: int
    top_k: int
    embedding_dimension: int


class Settings(BaseModel):
    market_csv: Path
    crm_csv: Path
    kb_markdown: Path
    output_dir: Path
    models: ModelSettings
    ranking_weights: RankingWeights
    ranking_calibration: RankingCalibration
    rag: RagSettings
    mock_seed: int
    mock_scale: int


def load_settings(path: Path) -> Settings:
    with path.open("rb") as fh:
        raw = tomllib.load(fh)
    paths = raw["paths"]
    models = raw["models"]
    ranking = raw["ranking"]["weights"]
    calibration = raw["ranking"]["calibration"]
    rag = raw["rag"]
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
        mock_seed=int(mock["seed"]),
        mock_scale=int(mock["scale"]),
    )
