"""
PySpark serialization support.

Wire-tag range: 700–799  (PYSPARK_BASE = 700)

Objects covered
---------------
pyspark.sql.DataFrame          -> PYSPARK_DATAFRAME  (700)  via PyArrow IPC file
pyspark.sql.Row                -> PYSPARK_ROW        (701)  via Arrow record-batch
pyspark.sql.types.StructType   -> PYSPARK_SCHEMA     (702)  via Arrow IPC schema
pyspark.sql.types.DataType     -> PYSPARK_DATATYPE   (703)  via JSON repr
pyspark.sql.Column             -> PYSPARK_COLUMN     (704)  via pickle/cloudpickle
pyspark.RDD                    -> PYSPARK_RDD        (705)  collect() → Arrow IPC
pyspark.sql.SparkSession       -> PYSPARK_SESSION    (706)  not reconstructed;
                                                              serialised as a
                                                              metadata-only stub

DataFrame / RDD payloads delegate to pyarrow.  All other PySpark objects that
cannot be naturally expressed with Arrow fall back to cloudpickle (if available)
or standard pickle.

Decoding note
-------------
SparkSession references are *not* reconstructed on load.  ``value`` returns
``None`` for PYSPARK_SESSION because a live Spark context cannot be round-
tripped through bytes.  The serialized form serves as documentation / provenance
only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Generic, Mapping, TypeVar

from yggdrasil.pickle.ser.constants import CODEC_NONE
from yggdrasil.pickle.ser.pyarrow import (
    _merge_metadata,
    _schema_file_metadata,
    _table_to_ipc_file_buffer,
    _table_from_ipc_file_buffer,
    _record_batch_to_ipc_file_buffer,
    _schema_to_ipc_file_buffer,
    _schema_from_ipc_file_buffer,
)
from yggdrasil.pickle.ser.serialized import Serialized
from yggdrasil.pickle.ser.tags import Tags

__all__ = [
    "TPySpark",
    "PySparkSerialized",
    "PySparkDataFrameSerialized",
    "PySparkRowSerialized",
    "PySparkSchemaSerialized",
    "PySparkDataTypeSerialized",
    "PySparkColumnSerialized",
    "PySparkRDDSerialized",
    "PySparkSessionSerialized",
]

TPySpark = TypeVar("TPySpark", bound=object)

# ---------------------------------------------------------------------------
# Lazy PySpark imports
# ---------------------------------------------------------------------------

def _pyspark_types():
    """Return pyspark.sql.types (raises ImportError if PySpark is not installed)."""
    import pyspark.sql.types as _t
    return _t


def _pyspark_sql():
    """Return pyspark.sql (raises ImportError if PySpark is not installed)."""
    import pyspark.sql as _sql
    return _sql


def _pa():
    """Return pyarrow (already a hard dep via pyarrow.py in this package)."""
    import pyarrow as pa
    return pa


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _row_to_record_batch(row: Any) -> Any:
    """Convert a ``pyspark.sql.Row`` to a ``pyarrow.RecordBatch``."""
    pa = _pa()
    if hasattr(row, "__fields__"):
        fields = row.__fields__
        arrays = [pa.array([row[f]]) for f in fields]
        schema = pa.schema([pa.field(f, arr.type) for f, arr in zip(fields, arrays)])
        return pa.record_batch(arrays, schema=schema)
    # Unnamed row – use positional column names
    values = list(row)
    arrays = [pa.array([v]) for v in values]
    schema = pa.schema([pa.field(f"col_{i}", arr.type) for i, arr in enumerate(arrays)])
    return pa.record_batch(arrays, schema=schema)


def _record_batch_to_row(batch: Any) -> Any:
    """Convert a ``pyarrow.RecordBatch`` to a ``pyspark.sql.Row``."""
    from pyspark.sql import Row
    if batch.num_rows == 0:
        return Row(**{name: None for name in batch.schema.names})
    row_dict = {name: batch.column(i)[0].as_py()
                for i, name in enumerate(batch.schema.names)}
    return Row(**row_dict)


def _schema_to_arrow_schema(schema: Any) -> Any:
    """Convert a ``pyspark.sql.types.StructType`` to a ``pyarrow.Schema``."""
    from yggdrasil.spark.cast import spark_schema_to_arrow_schema
    return spark_schema_to_arrow_schema(schema)


def _arrow_schema_to_spark_schema(arrow_schema: Any) -> Any:
    """Convert a ``pyarrow.Schema`` to a ``pyspark.sql.types.StructType``."""
    from yggdrasil.spark.cast import any_to_spark_schema
    return any_to_spark_schema(arrow_schema)


def _datatype_to_json(dtype: Any) -> str:
    """Return the JSON representation of a ``pyspark.sql.types.DataType``."""
    return dtype.json()


def _datatype_from_json(json_str: str) -> Any:
    """Reconstruct a ``pyspark.sql.types.DataType`` from its JSON representation."""
    from pyspark.sql.types import _parse_datatype_json_string  # type: ignore[attr-defined]
    return _parse_datatype_json_string(json_str)


def _pickle_column(col: Any) -> bytes:
    """Pickle a ``pyspark.sql.Column`` using cloudpickle if available, else pickle."""
    try:
        import cloudpickle
        return cloudpickle.dumps(col)
    except ImportError:
        import pickle
        return pickle.dumps(col)


def _unpickle_column(data: bytes) -> Any:
    """Unpickle a ``pyspark.sql.Column``."""
    try:
        import cloudpickle
        return cloudpickle.loads(data)
    except ImportError:
        import pickle
        return pickle.loads(data)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class PySparkSerialized(Serialized[TPySpark], Generic[TPySpark]):
    """Abstract base for all PySpark-flavoured Serialized subclasses."""

    TAG: ClassVar[int]

    @property
    def value(self) -> TPySpark:
        raise NotImplementedError

    def as_python(self) -> TPySpark:
        return self.value

    def decode_arrow_buffer(self) -> Any:
        """Return payload as a pyarrow.Buffer for IPC reads."""
        pa = _pa()
        if self.codec == CODEC_NONE:
            return pa.py_buffer(self.to_bytes())
        return pa.py_buffer(self.decode())

    @classmethod
    def from_python_object(
        cls,
        obj: object,
        *,
        metadata: Mapping[bytes, bytes] | None = None,
        codec: int | None = None,
    ) -> "Serialized[object] | None":
        try:
            import pyspark.sql as _sql
            import pyspark.sql.types as _types
        except ImportError:
            return None

        if isinstance(obj, _sql.DataFrame):
            return PySparkDataFrameSerialized.from_value(obj, metadata=metadata, codec=codec)

        if isinstance(obj, _sql.Row):
            return PySparkRowSerialized.from_value(obj, metadata=metadata, codec=codec)

        if isinstance(obj, _types.StructType):
            return PySparkSchemaSerialized.from_value(obj, metadata=metadata, codec=codec)

        if isinstance(obj, _types.DataType):
            return PySparkDataTypeSerialized.from_value(obj, metadata=metadata, codec=codec)

        if isinstance(obj, _sql.Column):
            return PySparkColumnSerialized.from_value(obj, metadata=metadata, codec=codec)

        try:
            import pyspark  # type: ignore[import]
            if isinstance(obj, pyspark.RDD):
                return PySparkRDDSerialized.from_value(obj, metadata=metadata, codec=codec)
        except ImportError:
            pass

        try:
            from pyspark.sql import SparkSession  # type: ignore[import]
            if isinstance(obj, SparkSession):
                return PySparkSessionSerialized.from_value(obj, metadata=metadata, codec=codec)
        except ImportError:
            pass

        return None


# ---------------------------------------------------------------------------
# DataFrame  (700) – Arrow IPC file via pyarrow fallback
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class PySparkDataFrameSerialized(PySparkSerialized[Any]):
    """
    Serialize a ``pyspark.sql.DataFrame`` by collecting it to the driver and
    storing the result as an Arrow IPC file.

    On deserialization a ``pyarrow.Table`` is returned (not a DataFrame) because
    there is no live SparkSession available at load time.  Call
    ``.to_pyspark(spark)`` to re-wrap if a session is available.
    """

    TAG: ClassVar[int] = Tags.PYSPARK_DATAFRAME

    @property
    def value(self) -> Any:
        """Return the collected data as a ``pyarrow.Table``."""
        return _table_from_ipc_file_buffer(self.decode_arrow_buffer())

    def to_pyspark(self, spark: Any = None) -> Any:
        """
        Re-create a ``pyspark.sql.DataFrame`` from the stored Arrow data.

        Uses :func:`~yggdrasil.spark.cast.arrow_table_to_spark_dataframe` which
        requires an active ``SparkSession`` (obtained via
        ``SparkSession.getActiveSession()`` when *spark* is ``None``).

        Parameters
        ----------
        spark:
            An active ``pyspark.sql.SparkSession``, or ``None`` to use the
            currently-active session.
        """
        from yggdrasil.spark.cast import arrow_table_to_spark_dataframe
        table = self.value
        return arrow_table_to_spark_dataframe(table)

    @classmethod
    def from_value(
        cls,
        df: Any,
        *,
        metadata: Mapping[bytes, bytes] | None = None,
        codec: int | None = None,
    ) -> "Serialized[object]":
        from yggdrasil.spark.cast import spark_dataframe_to_arrow_table
        table = spark_dataframe_to_arrow_table(df)
        merged = _schema_file_metadata(
            metadata,
            arrow_object=b"pyspark_dataframe",
            arrow_encoding=b"ipc_file",
        )
        buf = _table_to_ipc_file_buffer(table, metadata=merged)
        return cls.build(tag=cls.TAG, data=buf, metadata=merged, codec=codec)


# ---------------------------------------------------------------------------
# Row  (701) – single-row Arrow RecordBatch
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class PySparkRowSerialized(PySparkSerialized[Any]):
    """Serialize a ``pyspark.sql.Row`` as a one-row Arrow RecordBatch."""

    TAG: ClassVar[int] = Tags.PYSPARK_ROW

    @property
    def value(self) -> Any:
        """Return the data as a ``pyspark.sql.Row`` (requires PySpark on load)."""
        pa = _pa()
        reader = pa.ipc.open_file(pa.BufferReader(self.decode_arrow_buffer()))
        if reader.num_record_batches != 1:
            raise ValueError(
                f"PYSPARK_ROW payload must contain exactly 1 batch, "
                f"got {reader.num_record_batches}"
            )
        batch = reader.get_batch(0)
        return _record_batch_to_row(batch)

    @classmethod
    def from_value(
        cls,
        row: Any,
        *,
        metadata: Mapping[bytes, bytes] | None = None,
        codec: int | None = None,
    ) -> "Serialized[object]":
        batch = _row_to_record_batch(row)
        merged = _schema_file_metadata(
            metadata,
            arrow_object=b"pyspark_row",
        )
        buf = _record_batch_to_ipc_file_buffer(batch, metadata=merged)
        return cls.build(tag=cls.TAG, data=buf, metadata=merged, codec=codec)


# ---------------------------------------------------------------------------
# Schema / StructType  (702) – Arrow IPC schema
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class PySparkSchemaSerialized(PySparkSerialized[Any]):
    """
    Serialize a ``pyspark.sql.types.StructType`` as an Arrow IPC schema.

    On load, if PySpark is available the Arrow schema is converted back to a
    ``StructType``; otherwise a ``pyarrow.Schema`` is returned.
    """

    TAG: ClassVar[int] = Tags.PYSPARK_SCHEMA

    @property
    def value(self) -> Any:
        arrow_schema = _schema_from_ipc_file_buffer(self.decode_arrow_buffer())
        try:
            return _arrow_schema_to_spark_schema(arrow_schema)
        except Exception:
            # No PySpark or converter not available – return the Arrow schema.
            return arrow_schema

    @classmethod
    def from_value(
        cls,
        schema: Any,
        *,
        metadata: Mapping[bytes, bytes] | None = None,
        codec: int | None = None,
    ) -> "Serialized[object]":
        try:
            arrow_schema = _schema_to_arrow_schema(schema)
        except RuntimeError:
            # Converter not found: fall back to storing the JSON form as a
            # single-field Arrow schema with a string column.
            pa = _pa()
            json_str = _datatype_to_json(schema)
            arrow_schema = pa.schema(
                [pa.field("__pyspark_schema_json__", pa.utf8())],
                metadata={b"pyspark_schema_json": json_str.encode()},
            )

        merged = _schema_file_metadata(
            metadata,
            arrow_object=b"pyspark_schema",
        )
        buf = _schema_to_ipc_file_buffer(arrow_schema, metadata=merged)
        return cls.build(tag=cls.TAG, data=buf, metadata=merged, codec=codec)


# ---------------------------------------------------------------------------
# DataType  (703) – JSON-encoded string payload
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class PySparkDataTypeSerialized(PySparkSerialized[Any]):
    """Serialize a ``pyspark.sql.types.DataType`` via its JSON representation."""

    TAG: ClassVar[int] = Tags.PYSPARK_DATATYPE

    @property
    def value(self) -> Any:
        json_str = self.decode().decode("utf-8")
        return _datatype_from_json(json_str)

    @classmethod
    def from_value(
        cls,
        dtype: Any,
        *,
        metadata: Mapping[bytes, bytes] | None = None,
        codec: int | None = None,
    ) -> "Serialized[object]":
        json_str = _datatype_to_json(dtype)
        raw = json_str.encode("utf-8")
        merged = _merge_metadata(
            metadata,
            {b"pyspark_object": b"datatype"},
        )
        return cls.build(tag=cls.TAG, data=raw, metadata=merged, codec=codec)


# ---------------------------------------------------------------------------
# Column  (704) – cloudpickle / pickle payload
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class PySparkColumnSerialized(PySparkSerialized[Any]):
    """
    Serialize a ``pyspark.sql.Column`` expression.

    Column objects are opaque JVM-backed objects.  We use cloudpickle (preferred)
    or standard pickle as the storage strategy.  Note that the deserialized Column
    requires the same PySpark / JVM version to be available.
    """

    TAG: ClassVar[int] = Tags.PYSPARK_COLUMN

    @property
    def value(self) -> Any:
        return _unpickle_column(self.decode())

    @classmethod
    def from_value(
        cls,
        col: Any,
        *,
        metadata: Mapping[bytes, bytes] | None = None,
        codec: int | None = None,
    ) -> "Serialized[object]":
        raw = _pickle_column(col)
        merged = _merge_metadata(
            metadata,
            {b"pyspark_object": b"column"},
        )
        return cls.build(tag=cls.TAG, data=raw, metadata=merged, codec=codec)


# ---------------------------------------------------------------------------
# RDD  (705) – collect → Arrow IPC file
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class PySparkRDDSerialized(PySparkSerialized[Any]):
    """
    Serialize a ``pyspark.RDD`` by collecting all elements to the driver.

    Elements must be homogeneous and Arrow-convertible (e.g. Rows, dicts, tuples,
    or primitives).  The collected data is stored as an Arrow IPC file.

    On deserialization, a ``pyarrow.Table`` is returned.
    """

    TAG: ClassVar[int] = Tags.PYSPARK_RDD

    @property
    def value(self) -> Any:
        """Return collected RDD data as a ``pyarrow.Table``."""
        return _table_from_ipc_file_buffer(self.decode_arrow_buffer())

    @classmethod
    def from_value(
        cls,
        rdd: Any,
        *,
        metadata: Mapping[bytes, bytes] | None = None,
        codec: int | None = None,
    ) -> "Serialized[object]":
        pa = _pa()

        rows = rdd.collect()

        if not rows:
            table = pa.table({})
        else:
            first = rows[0]
            if hasattr(first, "__fields__"):
                # pyspark.sql.Row
                fields = first.__fields__
                columns: dict[str, list] = {f: [] for f in fields}
                for row in rows:
                    for f in fields:
                        columns[f].append(row[f])
                table = pa.table(columns)
            elif isinstance(first, dict):
                keys = list(first.keys())
                columns = {k: [r[k] for r in rows] for k in keys}
                table = pa.table(columns)
            else:
                table = pa.table({"value": pa.array(rows)})

        merged = _schema_file_metadata(
            metadata,
            arrow_object=b"pyspark_rdd",
            arrow_encoding=b"ipc_file",
        )
        buf = _table_to_ipc_file_buffer(table, metadata=merged)
        return cls.build(tag=cls.TAG, data=buf, metadata=merged, codec=codec)


# ---------------------------------------------------------------------------
# SparkSession  (706) – metadata stub, no live reconstruction
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class PySparkSessionSerialized(PySparkSerialized[None]):
    """
    Provenance stub for a ``pyspark.sql.SparkSession``.

    A live SparkSession cannot be serialized in a meaningful way – it is a
    client handle to a running JVM.  We store app-name, master URL, and Spark
    version as metadata, and return ``None`` on deserialization.
    """

    TAG: ClassVar[int] = Tags.PYSPARK_SESSION

    @property
    def value(self) -> None:
        return None

    @classmethod
    def from_value(
        cls,
        spark: Any,
        *,
        metadata: Mapping[bytes, bytes] | None = None,
        codec: int | None = None,
    ) -> "Serialized[object]":
        try:
            conf = spark.sparkContext.getConf()
            app_name = conf.get("spark.app.name", "").encode("utf-8")
            master   = conf.get("spark.master", "").encode("utf-8")
            version  = spark.version.encode("utf-8") if hasattr(spark, "version") else b""
        except Exception:
            app_name = master = version = b""

        merged = _merge_metadata(
            metadata,
            {
                b"pyspark_object":  b"sparksession",
                b"spark_app_name":  app_name,
                b"spark_master":    master,
                b"spark_version":   version,
            },
        )
        # Payload is intentionally empty.
        return cls.build(tag=cls.TAG, data=b"", metadata=merged, codec=codec)


# ---------------------------------------------------------------------------
# Register all subclasses with the tag registry
# ---------------------------------------------------------------------------

for _cls in PySparkSerialized.__subclasses__():
    Tags.register_class(_cls, tag=_cls.TAG)

