# Development Guide

## Local checks

```bash
cd python
pytest
ruff check
black .
```

## Documentation locally

```bash
cd python
uv pip install -e .[dev]
mkdocs serve
```

## Build docs static site

```bash
cd python
mkdocs build
```
