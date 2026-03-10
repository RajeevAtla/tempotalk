# AGENTS.md

## Typing Rule

- Do not use the `Any` type anywhere in this repository.
- Treat `Any` as disallowed in application code, tests, scripts, and tooling code.

## Preferred Alternatives

- Use `TypedDict` for structured dictionaries.
- Use `Protocol` for interface-style typing.
- Use `TypeVar`/generics for reusable typed helpers.
- Use `object` only at unavoidable dynamic boundaries, then narrow via `isinstance`, attribute checks, or `cast`.
- Prefer precise container types (`Mapping`, `Sequence`, concrete models) over broad `dict`/`list` with loose values.

## Generated Code

- Avoid manually editing generated files unless explicitly required.
- If generated code introduces `Any`, address it via regeneration settings/templates or wrapper layers in typed code.

## Python and Tooling

- Use `uv` for all Python commands.
- Target Python `3.13` only.
- Keep source and tests type-check clean under `ty`.
- Keep lint clean under `ruff`.

### Standard Commands

- Install/sync: `uv sync --dev`
- Lint: `uv run ruff check .`
- Typecheck: `uv run ty check`
- Tests: `uv run pytest -q`
- Run CLI: `uv run python -m scripts.run_cli run --config config/defaults.toml`
- Validate outputs: `uv run python -m scripts.run_cli validate-output <run_dir>`

## Testing Expectations

- Prefer test-first changes whenever practical.
- Add or update unit tests for every behavioral change.
- Keep integration tests optional and environment-gated.
- Do not rely on network calls in unit tests; mock external calls.

## Runtime Model Policy

- Generation must use Ollama Cloud (no local generation model usage).
- Embeddings must use local Ollama only.
- Current defaults:
  - generation model: `ministral-3:8b`
  - embedding model: `embeddinggemma`
- Keep provider restrictions enforced in code (fail fast on unsupported providers/hosts).

### Environment Variables

- `OLLAMA_API_KEY` for Ollama Cloud generation auth.
- `OLLAMA_BASE_URL` for cloud generation endpoint (default `https://ollama.com`).
- `OLLAMA_EMBED_BASE_URL` for local embedding endpoint (default `http://127.0.0.1:11434`).
- `.env` is local-only and must remain gitignored.

## Data and Output Contracts

- Input defaults are from `config/defaults.toml`.
- Output artifacts must remain TOML and schema-versioned.
- Preserve output filenames and required fields used by validators/tests.
- Keep `run_metadata.toml` checksum behavior intact for reproducibility.

## RAG and Retrieval

- Keep FAISS index usage for vector retrieval.
- Keep chunking/embedding/retrieval deterministic where possible.
- Maintain citation handling and strict-citation enforcement paths.

## Config and Schema Changes

- If config keys change, update:
  - defaults in `config/defaults.toml`
  - loaders/parsers in `src/tempus_copilot/config.py`
  - tests that assert config values
  - README usage docs
- If output schema changes, update validators and fixtures together.

## Documentation Expectations

- Update README when setup, env vars, model choices, or commands change.
- Keep instructions copy-pasteable and accurate for Windows PowerShell and `uv`.

## Change Safety

- Do not silently weaken typing, tests, or CI checks.
- Avoid introducing hidden fallbacks that violate runtime policy.
- Prefer explicit errors over implicit fallback behavior for unsupported paths.
