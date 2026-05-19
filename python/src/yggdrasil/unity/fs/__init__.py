"""Filesystem-backed :mod:`yggdrasil.unity` backend.

Quick start::

    from yggdrasil.io.path import LocalPath
    from yggdrasil.unity.fs import FSEngine

    engine = FSEngine(base=LocalPath("/tmp/warehouse"))
    catalog = engine.create_catalog("main")
    schema = catalog.create_schema("default")
    table = schema.create_table(
        "sales", schema=my_schema, format="parquet",
    )
    table.write_arrow_batches(batches)
    arrow = engine["main"]["default"]["sales"].read_arrow_table()

The engine takes any :class:`yggdrasil.io.path.Path` as its base — a
:class:`LocalPath`, a registered remote path (S3, Databricks, …), or
anything else satisfying the :class:`Path` contract. Metadata is JSON
sidecars under ``_yggdrasil/``; row data is multi-file Parquet / Arrow
IPC under ``data/`` driven by :class:`FolderIO`, so partitioning,
upsert, predicate pushdown, and schema collection all reuse the
project's existing tabular machinery.
"""

from yggdrasil.unity.fs.catalog import FSCatalog
from yggdrasil.unity.fs.engine import FSEngine
from yggdrasil.unity.fs.schema import FSSchema
from yggdrasil.unity.fs.table import FSTable
from yggdrasil.unity.fs.view import FSView

__all__ = [
    "FSEngine",
    "FSCatalog",
    "FSSchema",
    "FSTable",
    "FSView",
]
