# Tempus Sales Copilot (CLI)

## Quickstart

```bash
uv sync --dev
uv run baml-cli generate
uv run pytest -q
uv run ruff check .
uv run ty check
uv run python -m scripts.run_cli run --config config/defaults.toml --strict-citations
uv run python -m scripts.run_cli validate-output outputs/run_YYYYMMDD_HHMMSS
```

Run optional live Gemini integration tests:

```bash
uv run pytest -q -m integration
```

## Gemini Setup

Set `GOOGLE_API_KEY` to enable:

- BAML generation via `gemini-2.5-flash`
- Gemini embeddings via `gemini-embedding-001`

If `GOOGLE_API_KEY` is not set, the CLI falls back to deterministic local stubs for generation and embeddings.

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
