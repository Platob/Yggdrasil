# Yggdrasil Python documentation

This directory is the entry point for all Python-facing documentation. Use the links below to navigate between guides, templates, and per-module reference pages.

## Table of contents
- [Module overview](modules.md) – summary of each subpackage and when to use it.
- [Module index](modules/README.md) – direct links to detailed pages for every submodule.
- [Developer templates](developer-templates.md) – copy/paste snippets for common tasks.
- [Python utility reference](pyutils.md) – concurrency and retry helpers.
- [Serialization guide](ser.md) – dependency inspection and serialization utilities.

## Prerequisites
- Python **3.10+**
- Install Yggdrasil from the `python/` directory:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e .[dev]
```

Install extras for the engines you plan to use (`.[polars]`, `.[pandas]`, `.[spark]`, `.[databricks]`).

## Getting started
Looking for a minimal example? Start with the **developer templates** or jump straight to [`yggdrasil.dataclasses`](modules/dataclasses/README.md) to learn how to build Arrow-aware dataclasses.
