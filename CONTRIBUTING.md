# Contributing

## Local Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[dev,docs]"
```

## Checks

```bash
.venv/bin/ruff check src examples tests
.venv/bin/python -m pytest -q
```

## Documentation

```bash
.venv/bin/mkdocs serve
```

Read the Docs builds from `mkdocs.yml` using `.readthedocs.yaml`.

## Release

Releases are intended to be published by GitHub Actions using PyPI Trusted Publishing.

1. Update `CHANGELOG.md`.
2. Update the version in `pyproject.toml`.
3. Commit the release changes.
4. Tag the commit with `vX.Y.Z`.
5. Push the tag to GitHub.

The `publish.yml` workflow builds the sdist/wheel and publishes to PyPI.
