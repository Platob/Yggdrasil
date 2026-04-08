# Yggdrasil Python Documentation

Yggdrasil (`ygg` on PyPI, `yggdrasil` in imports) is a schema-aware data interchange library centered on an Arrow-first converter registry.

## Why Yggdrasil

- **Registry-driven conversion** across Python values, dataclasses, Arrow, Polars, pandas, Spark, and Databricks.
- **Arrow schema contract** that keeps fields, nullability, and metadata consistent.
- **Optional dependency guards** so base installs stay lightweight.
- **Production-friendly IO and HTTP** via `BytesIO`, URL resources, and `HTTPSession`.

## Documentation structure

- **Getting Started**: install, first conversions, and minimum working examples.
- **Guides**: architecture, casting design, HTTP/IO stack, Databricks, and development.
- **Module Walkthrough**: curated markdown pages already maintained in `python/docs/modules/`.
- **API Reference**: generated docs for the Python package tree under `src/yggdrasil`.

## Quick links

- [Getting Started](getting-started.md)
- [Architecture guide](guides/architecture.md)
- [Module index](modules.md)
- [Generated API reference](api/index.md)
