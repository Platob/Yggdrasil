import json
from typing import Any

from databricks.sdk.service.catalog import ColumnInfo as CatalogColumnInfo, ColumnTypeName
from databricks.sdk.service.sql import ColumnInfo as SQLColumnInfo, ColumnInfoTypeName

from yggdrasil.data import Field, DataType, field
from yggdrasil.data.types import ObjectType, NullType, BinaryType, BooleanType, StringType, TimestampType, ArrayType, \
    IntegerType, DateType, DecimalType, FloatingPointType, DurationType, MapType, StructType

__all__ = [
    "parse_databricks_field",
]


COLUMN_TYPE_MAP = {
    ColumnTypeName.ARRAY: ArrayType.from_item(Field.make_default_field()),
    ColumnTypeName.BINARY: BinaryType(),
    ColumnTypeName.BOOLEAN: BooleanType(),
    ColumnTypeName.BYTE: IntegerType(byte_size=1),
    ColumnTypeName.CHAR: StringType(byte_size=1),
    ColumnTypeName.DATE: DateType(),
    ColumnTypeName.DECIMAL: DecimalType(),
    ColumnTypeName.DOUBLE: FloatingPointType(byte_size=8),
    ColumnTypeName.FLOAT: FloatingPointType(byte_size=4),
    ColumnTypeName.GEOGRAPHY: ObjectType(),
    ColumnTypeName.GEOMETRY: ObjectType(),
    ColumnTypeName.INT: IntegerType(byte_size=4),
    ColumnTypeName.INTERVAL: DurationType(),
    ColumnTypeName.LONG: IntegerType(byte_size=8),
    ColumnTypeName.MAP: MapType.from_key_value(Field.make_default_field(), Field.make_default_field()),
    ColumnTypeName.NULL: NullType(),
    ColumnTypeName.SHORT: FloatingPointType(byte_size=2),
    ColumnTypeName.STRING: StringType(),
    ColumnTypeName.STRUCT: StructType.empty(),
    ColumnTypeName.TABLE_TYPE: StructType.empty(),
    ColumnTypeName.TIMESTAMP: TimestampType(unit="us", tz="UTC"),
    ColumnTypeName.TIMESTAMP_NTZ: TimestampType(unit="us"),
    ColumnTypeName.USER_DEFINED_TYPE: ObjectType(),
    ColumnTypeName.VARIANT: ObjectType(),
}


COLUMN_INFO_TYPE_MAP = {
    ColumnInfoTypeName.ARRAY: ArrayType.from_item(Field.make_default_field()),
    ColumnInfoTypeName.BINARY: BinaryType(),
    ColumnInfoTypeName.BOOLEAN: BooleanType(),
    ColumnInfoTypeName.BYTE: IntegerType(byte_size=1),
    ColumnInfoTypeName.CHAR: StringType(byte_size=1),
    ColumnInfoTypeName.DATE: DateType(),
    ColumnInfoTypeName.DECIMAL: DecimalType(),
    ColumnInfoTypeName.DOUBLE: FloatingPointType(byte_size=8),
    ColumnInfoTypeName.FLOAT: FloatingPointType(byte_size=4),
    ColumnInfoTypeName.INT: IntegerType(byte_size=4),
    ColumnInfoTypeName.INTERVAL: DurationType(),
    ColumnInfoTypeName.LONG: IntegerType(byte_size=8),
    ColumnInfoTypeName.MAP: MapType.from_key_value(Field.make_default_field(), Field.make_default_field()),
    ColumnInfoTypeName.NULL: NullType(),
    ColumnInfoTypeName.SHORT: FloatingPointType(byte_size=2),
    ColumnInfoTypeName.STRING: StringType(),
    ColumnInfoTypeName.STRUCT: StructType.empty(),
    ColumnInfoTypeName.TIMESTAMP: TimestampType(unit="us", tz="UTC"),
    ColumnInfoTypeName.USER_DEFINED_TYPE: ObjectType(),
}


REPLACE_TIMEZONES = {
    "Etc/UTC": "UTC",
    "CET": "Europe/Zurich",
    "EST": "America/New_York",
    "MST": "America/Denver",
    "PST": "America/Los_Angeles",
    "GMT": "UTC",
}


def parse_databricks_field(obj: Any) -> Field:
    if isinstance(obj, Field):
        return obj
    if isinstance(obj, dict):
        return parse_field_dict(obj)
    if isinstance(obj, CatalogColumnInfo):
        return parse_catalog_column_info_field(obj)
    if isinstance(obj, SQLColumnInfo):
        return parse_sql_column_info(obj)

    if isinstance(obj, str):
        if obj.startswith("{") and obj.endswith("}"):
            return parse_field_dict(json.loads(obj))

        try:
            dtype = ColumnTypeName[obj.upper()]
            mapping = COLUMN_TYPE_MAP
        except KeyError:
            dtype = ColumnInfoTypeName[obj.upper()]
            mapping = COLUMN_INFO_TYPE_MAP

        return Field(
            name="",
            dtype=mapping.get(dtype, ObjectType.instance()),
        )

    raise TypeError(f"Cannot parse field from {obj!r}")


def parse_sql_column_info(obj: SQLColumnInfo) -> Field:
    name = obj.name or ""
    dtype = COLUMN_INFO_TYPE_MAP.get(obj.type_name, ObjectType.instance())

    if isinstance(dtype, DecimalType):
        precision = 38 if obj.type_precision is None else obj.type_precision
        scale = 18 if obj.type_scale is None else obj.type_scale
        if precision is not None and scale is not None:
            dtype = DecimalType(precision=precision, scale=scale)

    if isinstance(dtype, TimestampType):
        if dtype.tz:
            if dtype.tz in REPLACE_TIMEZONES.keys():
                tz = REPLACE_TIMEZONES.get(dtype.tz, dtype.tz)
                dtype = TimestampType(unit=dtype.unit, tz=tz)

    if dtype.type_id.is_nested:
        dtype = dtype.merge_with(DataType.from_str(obj.type_text))

    metadata = {}
    if obj.position:
        metadata[b"position"] = str(obj.position).encode()

    return Field(name=name, dtype=dtype, metadata=metadata)


def parse_catalog_column_info_field(obj: CatalogColumnInfo) -> Field:
    name = obj.name or ""
    nullable = bool(obj.nullable) if obj.nullable is not None else True
    dtype = ObjectType()

    if isinstance(dtype, DecimalType):
        precision = 38 if obj.type_precision is None else obj.type_precision
        scale = 18 if obj.type_scale is None else obj.type_scale
        if precision is not None and scale is not None:
            dtype = DecimalType(precision=precision, scale=scale)

    metadata = {}
    partition_by = False

    if obj.comment:
        metadata[b"comment"] = obj.comment.encode()

    if obj.partition_index is not None:
        partition_by = True
        metadata[b"partition_index"] = str(obj.partition_index).encode()

    if obj.position is not None:
        metadata[b"position"] = str(obj.position).encode()

    if obj.type_name:
        dtype = dtype.merge_with(COLUMN_TYPE_MAP.get(obj.type_name, ObjectType.instance()))

    if obj.type_json:
        parsed = parse_field_dict(obj.type_json)
        name = parsed.name or name
        dtype = dtype.merge_with(parsed.dtype)
        metadata.update(parsed.metadata or {})
        partition_by = partition_by or parsed.partition_by

    if obj.type_text and dtype.type_id.is_any_or_null:
        dtype = dtype.merge_with(DataType.from_str(obj.type_text))

    if isinstance(dtype, TimestampType):
        if dtype.tz:
            if dtype.tz in REPLACE_TIMEZONES.keys():
                tz = REPLACE_TIMEZONES.get(dtype.tz, dtype.tz)
                dtype = TimestampType(unit=dtype.unit, tz=tz)

    return Field(
        name=name,
        dtype=dtype,
        nullable=nullable,
        metadata=metadata
    ).with_partition_by(partition_by)


def parse_field_dict(obj: Any) -> Field:
    obj = _safe_dict(obj)
    name = obj.get("name", None) or ""
    nullable = obj.get("nullable", None)
    metadata = _safe_dict(obj.get("metadata", None))
    comment = metadata.get("comment", None)
    if comment:
        metadata[b"comment"] = comment

    dtype = obj.get("type", None)
    if dtype:
        dtype = parse_databricks_field(dtype).dtype
    else:
        dtype = ObjectType()

    fields = obj.get("fields", None)
    if fields:
        fields = [parse_databricks_field(_) for _ in fields]

        if isinstance(dtype, StructType):
            dtype = StructType(fields)

    contains_null = obj.get("containsNull", True)
    element_type = obj.get("elementType", None)
    if element_type:
        element_field = parse_databricks_field(element_type).with_nullable(contains_null)
        if not element_field.name:
            element_field = element_field.with_name("item")

        if isinstance(dtype, ArrayType):
            dtype = ArrayType.from_item(element_field)

    if isinstance(dtype, MapType):
        value_contains_null = obj.get("valueContainsNull", True)
        key_type = obj.get("keyType", None)
        key_field = parse_databricks_field(key_type).with_nullable(False) if key_type else dtype.key_field
        value_type = obj.get("valueType", None)
        value_field = parse_databricks_field(value_type).with_nullable(value_contains_null) if value_type else dtype.value_field

        dtype = MapType.from_key_value(key_field, value_field)

    return field(
        name=name,
        dtype=dtype,
        nullable=bool(nullable) if nullable is not None else True,
        metadata=metadata,
    )


def _safe_dict(obj: Any) -> dict:
    if obj is None:
        return {}

    if isinstance(obj, dict):
        return obj

    if isinstance(obj, (str, bytes)):
        return _safe_dict(json.loads(obj))

    raise TypeError(f"Cannot parse dict from {type(obj)}")
