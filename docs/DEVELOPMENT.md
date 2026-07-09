# Development

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,docs]"
```

## Test

```bash
ruff check src examples tests
python -m pytest -q
```

## Build Docs

```bash
mkdocs build
```

## Build Package

```bash
python -m build
```

## Publish

Publishing is configured for PyPI Trusted Publishing through GitHub Actions.

1. Create a PyPI project named `cvnpy`.
2. Add a Trusted Publisher for the GitHub repository and workflow `.github/workflows/publish.yml`.
3. Push a tag matching `v*`, for example `v0.1.0`.
