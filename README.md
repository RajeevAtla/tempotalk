# Tempus Sales Copilot (CLI)

## Quickstart

```bash
uv sync --dev
uv run pytest -q
uv run ruff check .
uv run ty check
uv run python -m scripts.run_cli --config config/defaults.toml
```

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
