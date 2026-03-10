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
