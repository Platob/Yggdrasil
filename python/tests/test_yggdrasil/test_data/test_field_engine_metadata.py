"""Engine-export metadata contracts on ``Field``.

What this file pins:

* **Arrow** — ``to_arrow_field`` does NOT attach a ``type_json`` blob
  by default. Arrow preserves struct / list / map child fields with
  their own metadata recursively, so the dtype intent round-trips
  natively. ``dump_json=True`` is opt-in for the rare callers that
  need to recover the exact yggdrasil :class:`DataType` subclass.
* **Spark struct / primitive** — ``to_pyspark_field`` likewise skips
  the blob; Spark's :class:`StructType` carries child metadata.
* **Spark Map / Array** — Spark's :class:`MapType` /
  :class:`ArrayType` keep only the element Spark types and lose any
  per-field metadata, so we dump ``type_json`` for those (and only
  those) to recover the original yggdrasil dtype on read.

Tests that need a live ``SparkSession`` subclass
:class:`SparkTestCase` so they auto-skip when PySpark isn't
installed.
"""
from __future__ import annotations

import pyarrow as pa

from yggdrasil.data.data_field import Field, _TYPE_JSON_METADATA_KEY
from yggdrasil.data.types import (
    ArrayType,
    IntegerType,
    MapType,
    StringType,
    StructType,
)
from yggdrasil.spark.tests import SparkTestCase


_TYPE_JSON_STR = _TYPE_JSON_METADATA_KEY.decode("utf-8")


# ===========================================================================
# Arrow side — pure-arrow tests, no Spark dependency.
# ===========================================================================


class TestArrowFieldNoBlobByDefault:

    def test_primitive_field_has_no_type_json(self) -> None:
        f = Field("qty", IntegerType(byte_size=8), nullable=False)
        out = f.to_arrow_field()

        assert out.metadata is None or _TYPE_JSON_METADATA_KEY not in out.metadata

    def test_struct_field_inner_metadata_recurses_through_arrow(self) -> None:
        """Arrow's struct type carries each child's metadata, so an
        outer struct ``Field`` can rely on it instead of dumping a
        JSON blob.
        """
        struct_dtype = StructType(
            fields=(
                Field("id", IntegerType(byte_size=8), metadata={"role": "pk"}),
                Field("name", StringType(), metadata={"comment": "label"}),
            )
        )
        f = Field("user", struct_dtype, metadata={"comment": "outer"})

        out = f.to_arrow_field()

        assert out.metadata is not None
        assert _TYPE_JSON_METADATA_KEY not in out.metadata
        assert out.metadata.get(b"comment") == b"outer"

        inner_id = out.type.field(0)
        assert inner_id.metadata is not None
        assert inner_id.metadata.get(b"role") == b"pk"
        assert _TYPE_JSON_METADATA_KEY not in inner_id.metadata

        inner_name = out.type.field(1)
        assert inner_name.metadata is not None
        assert inner_name.metadata.get(b"comment") == b"label"

    def test_dump_json_attaches_blob(self) -> None:
        f = Field("qty", IntegerType(byte_size=8))
        out = f.to_arrow_field(dump_json=True)

        assert out.metadata is not None
        assert _TYPE_JSON_METADATA_KEY in out.metadata

    def test_round_trip_recovers_dtype_without_blob(self) -> None:
        """``from_arrow_field`` falls back through
        :meth:`DataType.from_arrow_type` when the blob is missing,
        so the round trip stays lossless for the common cases.
        """
        src = Field("qty", IntegerType(byte_size=8), nullable=False)
        out = Field.from_arrow_field(src.to_arrow_field())

        assert out.name == "qty"
        assert out.arrow_type == pa.int64()
        assert out.nullable is False

    def test_to_arrow_field_cache_is_no_blob_path(self) -> None:
        """The cached arrow field is the no-blob shape; opting in
        once does not poison the cache for subsequent default calls.
        """
        f = Field("qty", IntegerType(byte_size=8))
        with_blob = f.to_arrow_field(dump_json=True)
        no_blob = f.to_arrow_field()

        assert _TYPE_JSON_METADATA_KEY in (with_blob.metadata or {})
        assert no_blob.metadata is None or _TYPE_JSON_METADATA_KEY not in no_blob.metadata


# ===========================================================================
# Spark side — needs a live SparkSession; auto-skipped without PySpark.
# ===========================================================================


class TestSparkFieldNoBlobForStructAndPrimitive(SparkTestCase):

    def test_primitive_field_has_no_type_json(self) -> None:
        f = Field("qty", IntegerType(byte_size=8), nullable=False)
        out = f.to_pyspark_field()

        assert _TYPE_JSON_STR not in (out.metadata or {})

    def test_struct_field_inner_metadata_round_trips_natively(self) -> None:
        struct_dtype = StructType(
            fields=(
                Field("id", IntegerType(byte_size=8), metadata={"role": "pk"}),
                Field("name", StringType(), metadata={"comment": "label"}),
            )
        )
        f = Field("user", struct_dtype, metadata={"comment": "outer"})

        out = f.to_pyspark_field()

        # Outer struct field carries no blob, just the user metadata.
        assert out.metadata.get("comment") == "outer"
        assert _TYPE_JSON_STR not in out.metadata

        # Spark's StructType preserves the children's metadata.
        inner_id = out.dataType.fields[0]
        assert inner_id.metadata.get("role") == "pk"
        assert _TYPE_JSON_STR not in inner_id.metadata

    def test_round_trip_preserves_dtype_without_blob(self) -> None:
        src = Field("qty", IntegerType(byte_size=8), nullable=False)
        out = Field.from_spark_field(src.to_pyspark_field())

        assert out.name == "qty"
        assert out.nullable is False
        assert out.dtype == IntegerType(byte_size=8)


class TestSparkFieldDumpsBlobForMapAndArray(SparkTestCase):

    def test_array_of_struct_dumps_type_json(self) -> None:
        """Spark's :class:`ArrayType` keeps only the element Spark
        type, so the inner struct's per-field metadata would be lost
        without the JSON blob round-trip.
        """
        item = Field(
            "item",
            StructType(
                fields=(
                    Field("id", IntegerType(byte_size=8), metadata={"role": "pk"}),
                )
            ),
        )
        array_dtype = ArrayType(item_field=item)
        f = Field("rows", array_dtype, metadata={"comment": "outer"})

        out = f.to_pyspark_field()

        # Outer user metadata still present.
        assert out.metadata.get("comment") == "outer"
        # Map/Array path dumps the dtype JSON to recover inner intent.
        assert _TYPE_JSON_STR in out.metadata

    def test_map_dumps_type_json(self) -> None:
        key = Field("key", StringType())
        value = Field("value", IntegerType(byte_size=8), metadata={"unit": "USD"})
        f = Field(
            "prices",
            MapType.from_key_value(key, value),
        )

        out = f.to_pyspark_field()

        assert _TYPE_JSON_STR in out.metadata

    def test_array_round_trip_recovers_inner_metadata(self) -> None:
        """The blob is what makes the round trip lossless for
        Map/Array — without it the inner field metadata would be
        gone after decoding from Spark.
        """
        item = Field(
            "item",
            StructType(
                fields=(
                    Field(
                        "id",
                        IntegerType(byte_size=8),
                        metadata={"role": "pk"},
                    ),
                )
            ),
        )
        src = Field("rows", ArrayType(item_field=item))

        recovered = Field.from_spark_field(src.to_pyspark_field())

        # The recovered dtype is an ArrayType again with its inner
        # struct child metadata intact.
        assert recovered.type_id == src.type_id
        item_dtype = recovered.dtype.item_field.dtype
        recovered_id = item_dtype.fields[0]
        assert recovered_id.metadata.get(b"role") == b"pk"

    def test_from_spark_field_strips_type_json_from_metadata(self) -> None:
        """The blob is internal — :meth:`from_spark_field` must
        strip it off so user-visible metadata stays clean.
        """
        item = Field("item", StructType(fields=(Field("id", IntegerType(byte_size=8)),)))
        src = Field("rows", ArrayType(item_field=item), metadata={"comment": "outer"})

        recovered = Field.from_spark_field(src.to_pyspark_field())

        meta = recovered.metadata or {}
        assert b"comment" in meta
        assert _TYPE_JSON_METADATA_KEY not in meta
