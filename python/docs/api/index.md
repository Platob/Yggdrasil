# API Reference

Auto-generated from `python/src/yggdrasil` using **mkdocstrings**. The pages are emitted on every docs build by [`docs/_scripts/gen_ref_pages.py`](https://github.com/Platob/Yggdrasil/blob/main/python/docs/_scripts/gen_ref_pages.py).

## Browse

Open the **API Reference** branch in the navigation. Each Python module gets a page; submodules nest under their parent.

The full literate nav is generated into `reference/SUMMARY.md` and rendered by `mkdocs-literate-nav`.

## High-traffic entry points

| Path | What it covers |
|---|---|
| `yggdrasil.data.cast.registry` | `convert`, `register_converter` — the central registry |
| `yggdrasil.data.cast.options` | `CastOptions`, `CastOptions.check` |
| `yggdrasil.arrow.cast` | `cast_arrow_tabular`, `cast_arrow_record_batch_reader` |
| `yggdrasil.polars.cast` | `cast_polars_dataframe`, `cast_polars_lazyframe`, round-trip helpers |
| `yggdrasil.pandas.cast` | `cast_pandas_dataframe` |
| `yggdrasil.spark.cast` | `cast_spark_dataframe` |
| `yggdrasil.http_` | `HTTPSession`, `PreparedRequest`, `Response` |
| `yggdrasil.io` | `URL`, `SendConfig`, `SendManyConfig` |
| `yggdrasil.databricks` | `DatabricksClient` |
| `yggdrasil.pyutils` | `retry`, `parallelize` |
| `yggdrasil.concurrent` | `Job`, `JobPoolExecutor` |

## Curated module pages

For hand-written, copy-paste-friendly examples, see the [Module walkthrough](../modules.md).
