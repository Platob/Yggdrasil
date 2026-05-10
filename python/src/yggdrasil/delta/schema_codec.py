"""Back-compat shim — schema codec lives at :mod:`yggdrasil.io.nested.delta.schema_codec`."""

from __future__ import annotations

from yggdrasil.io.nested.delta.schema_codec import (
    arrow_schema_to_spark_json,
    schema_to_spark_json,
    spark_json_to_arrow_schema,
    spark_json_to_schema,
)

__all__ = [
    "arrow_schema_to_spark_json",
    "schema_to_spark_json",
    "spark_json_to_arrow_schema",
    "spark_json_to_schema",
]
