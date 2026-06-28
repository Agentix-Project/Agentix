# Development Notes

[Docs Home](index.md) | [中文](development.zh-CN.md)

## Source Policy

This package is a public wrapper around Miles' work. The implementation should
reuse the upstream logic wherever possible.

Maintenance rules:

- Preserve Miles attribution and upstream notices.
- Keep the vendored source audit current.
- Prefer wrapper code over changing TITO algorithms.
- Keep copied upstream tests as the compatibility contract.
- Do not weaken negative tests.

## Public Documentation Standard

Public docs should make the relationship clear:

- Miles owns the underlying TITO design and implementation.
- This package makes that work importable and runnable as a standalone gateway.
- Optional heavy verification depends on the Miles/SGLang training stack.

## Local Setup

Editable installs are for repository development:

```bash
pip install -e '.[test]'
pytest tests/package -q
```

Public users should install the package distribution instead:

```bash
pip install tito-gateway
pip install 'tito-gateway[verify]'
```

## Build A Distribution

```bash
python -m pip install build
python -m build
python -m pip install dist/tito_gateway-0.1.0-py3-none-any.whl
tito-gateway --help
```

Releases should be tagged as `vX.Y.Z`. The GitHub Actions workflow builds on
every push and publishes to PyPI only from version tags, using PyPI trusted
publishing for the `pypi` environment.

## Pre-Push Checklist

1. Run CPU-fast tests.
2. Run CLI help smoke checks.
3. Confirm staged files do not include cache, model weights, credentials, or local env files.
4. Run a secret scan over staged and outgoing changes.
5. Check `git diff --check`.

## Current Secret Guard

The `guard-secret` skill can be used before pushing. If installed, run it before
`git push` and only push when it reports `SAFE_TO_PUSH`.
