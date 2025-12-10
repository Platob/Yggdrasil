# Yggdrasil

Utilities for schema-aware data interchange, conversions, and platform integrations. The repository currently focuses on the Python package in `python/`.

## Contents
- [Python package](python/README.md): Installation, quickstart, and module documentation for the `yggdrasil` Python utilities.
- [Module docs](python/docs): Per-module guides (e.g., dataclasses, types, pyutils, requests, libs, databricks, ser).
- [Tests](python/tests): Pytest suite for validating conversions, dataclasses, requests, and platform helpers.

## Getting started
Change into the Python project and follow its README for setup and usage:

```bash
cd python
uv venv .venv
source .venv/bin/activate
uv pip install -e .[dev]
```

Then explore the quickstart examples in [`python/README.md`](python/README.md).
