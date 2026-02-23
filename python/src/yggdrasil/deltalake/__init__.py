"""
deltalake — Delta Lake transaction log reader and writer.

Parses Delta log files from any PyArrow-supported filesystem (S3, GCS, Azure
Blob, local) and writes new commits back to the log without a Spark runtime.

Quick start
-----------
**Open an existing table** ::

    from pyarrow.fs import S3FileSystem
    from deltalake import DeltaTable

    fs  = S3FileSystem(region="us-east-1")
    tbl = DeltaTable(fs=fs, storage_location="s3://bucket/trading/crude_oil/")

    files   = tbl.select(partition_filter={"commodity": "crude_oil"})
    dataset = tbl.to_arrow_dataset(files, schema=arrow_schema)

**Initialise a new table** ::

    tbl = DeltaTable.init(
        fs=fs,
        storage_location="s3://bucket/trading/crude_oil/",
        schema=pa.schema([
            pa.field("trade_date", pa.date32(),        nullable=False),
            pa.field("commodity",  pa.string(),         nullable=False),
            pa.field("price",      pa.float64()),
            pa.field("notional",   pa.decimal128(18, 6)),
        ]),
        partition_columns=["trade_date", "commodity"],
    )

**Write Arrow data** ::

    version = tbl.write_arrow_dataset(
        pa_table,
        mode="append",
        partition_by=["trade_date", "commodity"],
    )

**Schema serialisation** ::

    from deltalake import arrow_schema_to_schema_string

    schema_json = arrow_schema_to_schema_string(pa_schema)

Package layout
--------------
``_uri.py``    URI scheme detection and stripping (internal).
``models.py``  Value objects: ``DeltaProtocol``, ``DeletionVector``,
               ``DeltaMetadata``, ``DeltaStats``, ``DeltaFile``.
``schema.py``  PyArrow → Delta ``schemaString`` serialisation.
``logs.py``    ``DeltaTable`` class (transaction log reader / writer).
"""

from __future__ import annotations

from .logs import DeltaTable, DeltaLog
from .models import (
    DeltaProtocol,
    DeletionVector,
    DeltaMetadata,
    DeltaStats,
    DeltaFile,
)
from .schema import arrow_schema_to_schema_string

__all__ = [
    # Primary class
    "DeltaTable",
    # Backward-compatibility alias
    "DeltaLog",
    # Value objects
    "DeltaProtocol",
    "DeletionVector",
    "DeltaMetadata",
    "DeltaStats",
    "DeltaFile",
    # Schema utility
    "arrow_schema_to_schema_string",
]
