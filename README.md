# TempoTalk (CLI)

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![ty](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json)](https://github.com/astral-sh/ty)


## Quickstart

```bash
uv sync --dev
uv run baml-cli generate
uv run pytest -q
uv run pytest -q tests/test_golden_run.py
uv run ruff check .
uv run ty check
uv run python -m scripts.run_cli run --config config/defaults.toml --strict-citations --fail-on-low-confidence 0.6
uv run python -m scripts.run_cli validate-output outputs/run_YYYYMMDD_HHMMSS
```

Run optional live Ollama integration tests:

```bash
uv run pytest -q -m integration
```

## Ollama Setup

Generation uses Ollama Cloud and embeddings use local Ollama (`embeddinggemma`).
Set environment variables (or `.env`):

```bash
OLLAMA_API_KEY=your_ollama_cloud_api_key
OLLAMA_BASE_URL=https://ollama.com
OLLAMA_EMBED_BASE_URL=http://127.0.0.1:11434
```

Pull the local embedding model:

```bash
ollama pull embeddinggemma
```

Only embeddings run locally. Generation remains cloud-only.

## Mock Data

```bash
uv run python -m scripts.generate_mock_data
```

This writes:

- `data/mock/market_intelligence.csv`
- `data/mock/crm_notes.csv`
- `data/mock/product_kb.md`

## Outputs

Running the CLI writes TOML output files to `outputs/run_YYYYMMDD_HHMMSS/`.
Outputs include `schema_version` and metadata checksum (`output_checksum_sha256`).
