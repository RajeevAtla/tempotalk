# TempoTalk

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![ty](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json)](https://github.com/astral-sh/ty)


## Quickstart

```bash
uv sync --dev
uv run pytest -q
uv run pytest -q tests/test_golden_run.py
uv run ruff check .
uv run ty check
uv run python -m scripts.run_cli run --config config/defaults.toml --strict-citations --fail-on-low-confidence 0.6
uv run python -m scripts.run_cli validate-output outputs/run_YYYYMMDD_HHMMSS
uv run streamlit run streamlit_app.py
```

Regenerate committed BAML artifacts only when `baml_src/sales_copilot.baml` changes:

```bash
uv run baml-cli generate
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

## Streamlit Frontend

Launch the operator UI:

```bash
uv run streamlit run streamlit_app.py
```

The Streamlit app is a thin frontend over the same typed backend used by the CLI. It does not shell
out to the CLI for normal runs. Instead it calls the library layer directly through
`src/tempus_copilot/ui_service.py`.

The frontend includes:

- a guided run form for launching the pipeline
- a run browser for reviewing ranked providers, objections, scripts, and retrieval traces
- an output validation view for schema/checksum checks
- system notes for runtime boundaries and env configuration

Operator notes:

- the sidebar `Config Path` selects which TOML settings file is loaded at startup
- the run form applies overrides in memory only; it does not rewrite `config/defaults.toml`
- if the configured input files are missing, the backend generates mock inputs before running
- if the configured `outputs/` directory has no `run_*` folders yet, the browse/validate views show
  empty-state guidance instead of failing
- the run browser and validator read the same TOML artifacts the CLI writes and the validator checks
  for required keys plus checksum integrity

## Runtime Architecture

The active runtime path uses the handwritten adapter in
`src/tempus_copilot/llm/baml_adapter.py`.

`baml_src/` is the human-edited prompt/schema source of truth used for metadata hashing and for
regenerating `baml_client/`.

`baml_client/` is committed generated code. It is kept for inspection and sync validation, but the
pipeline does not call it at runtime.

## Mock Data

```bash
uv run python -m scripts.generate_mock_data
```

This writes:

- `data/mock/market_intelligence.csv`
- `data/mock/crm_notes.csv`
- `data/mock/product_kb.md`

If the configured input files are missing, the pipeline auto-generates mock data in the configured
input directory before running.

`data/` and `outputs/` are local artifacts and are gitignored. The committed output contract lives
under `tests/fixtures/`, especially `tests/fixtures/golden/`.

## Outputs

Running the CLI writes TOML output files to `outputs/run_YYYYMMDD_HHMMSS/`.
Outputs include `schema_version` and metadata checksum (`output_checksum_sha256`).
