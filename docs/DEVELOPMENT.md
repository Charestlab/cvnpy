# Development

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[dev,docs]"
```

## Test

```bash
.venv/bin/ruff check src examples tests
.venv/bin/python -m pytest -q
```

## Build Docs

```bash
.venv/bin/mkdocs build
```

## Build Package

```bash
.venv/bin/python -m build
```

## Publish

Publishing is configured for PyPI Trusted Publishing through GitHub Actions.

1. Create a PyPI project named `cvnpy`.
2. Add a Trusted Publisher for the GitHub repository and workflow `.github/workflows/publish.yml`.
3. Push a tag matching `v*`, for example `v0.1.0`.
