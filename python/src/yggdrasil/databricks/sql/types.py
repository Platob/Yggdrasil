"""Type utilities for Databricks SQL metadata and Arrow."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import pyarrow as pa
from yggdrasil.data import Schema


@dataclass(frozen=True, slots=True)
class PrimaryKeySpec:
    columns: list[str]
    constraint_name: str | None = None
    rely: bool = True
    timeseries: str | None = None

    @classmethod
    def from_schema(
        cls,
        schema: Schema,
        *,
        constraint_name: str | None = None,
        rely: bool | None = None,
        timeseries: str | None = None
    ) -> PrimaryKeySpec | None:
        rely = True if rely is None else bool(rely)
        info = Schema.from_any(schema)
        fields = [f for f in info.fields if f.name in info.primary_key_names]

        if not fields:
            return None

        if timeseries is None:
            for f in fields:
                arrow_type = getattr(f, "type", None)
                if arrow_type is None and hasattr(f, "dtype"):
                    to_arrow = getattr(f.dtype, "to_arrow", None)
                    arrow_type = to_arrow() if callable(to_arrow) else None
                if arrow_type is not None and pa.types.is_timestamp(arrow_type):
                    timeseries = f.name
                    break

        return cls(
            columns=[f.name for f in fields],
            constraint_name=constraint_name,
            rely=rely,
            timeseries=timeseries,
        )

    @classmethod
    def from_str(
        cls,
        value: str,
        *,
        constraint_name: str | None = None,
        rely: bool | None = None,
        timeseries: str | None = None,
    ) -> PrimaryKeySpec:
        rely = True if rely is None else bool(rely)
        value = str(value).strip()
        if not value:
            raise ValueError("Primary key column string cannot be empty")

        return cls(
            columns=[value],
            constraint_name=constraint_name,
            rely=rely,
            timeseries=timeseries,
        )

    @classmethod
    def from_any(
        cls,
        value: Any = None,
        *,
        schema: Any | None = None,
        constraint_name: str | None = None,
        rely: bool | None = None,
        timeseries: str | None = None,
    ) -> PrimaryKeySpec | None:
        rely = True if rely is None else bool(rely)

        if isinstance(value, cls):
            return value

        if isinstance(value, pa.Schema) or hasattr(value, "to_arrow_schema"):
            return cls.from_schema(
                value,
                constraint_name=constraint_name,
                rely=rely,
                timeseries=timeseries,
            )

        if value is None:
            return (
                cls.from_schema(
                    schema,
                    constraint_name=constraint_name,
                    rely=rely,
                    timeseries=timeseries,
                )
                if schema is not None
                else None
            )

        if isinstance(value, str):
            return cls.from_str(
                value,
                constraint_name=constraint_name,
                rely=bool(rely),
                timeseries=timeseries,
            )

        if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
            columns = [str(item).strip() for item in value if str(item).strip()]
            if not columns:
                return None
            return cls(
                columns=columns,
                constraint_name=constraint_name,
                rely=bool(rely),
                timeseries=timeseries,
            )

        raise TypeError(f"Cannot build {cls.__name__} from {type(value).__name__}")

    @classmethod
    def from_(
        cls,
        value: Any = None,
        **kwargs: Any,
    ) -> PrimaryKeySpec | None:
        return cls.from_any(value, **kwargs)


@dataclass(frozen=True, slots=True)
class ForeignKeySpec:
    column: str
    ref: str
    constraint_name: str | None = None
    rely: bool = False
    match_full: bool = False
    on_update_no_action: bool = False
    on_delete_no_action: bool = False

    @classmethod
    def from_schema(cls, schema: Schema) -> list[ForeignKeySpec]:
        info = Schema.from_any(schema)
        return [
            cls(column=column, ref=ref)
            for column, ref in info.foreign_key_refs.items()
        ]

    @classmethod
    def from_str(
        cls,
        value: str,
        *,
        constraint_name: str | None = None,
        rely: bool = False,
        match_full: bool = False,
        on_update_no_action: bool = False,
        on_delete_no_action: bool = False,
    ) -> ForeignKeySpec:
        raw = str(value).strip()
        if not raw:
            raise ValueError("Foreign key spec string cannot be empty")

        for delimiter in ("->", "=", ":"):
            if delimiter in raw:
                column, ref = raw.split(delimiter, 1)
                column = column.strip()
                ref = ref.strip()
                if not column or not ref:
                    break
                return cls(
                    column=column,
                    ref=ref,
                    constraint_name=constraint_name,
                    rely=rely,
                    match_full=match_full,
                    on_update_no_action=on_update_no_action,
                    on_delete_no_action=on_delete_no_action,
                )

        raise ValueError(
            "Foreign key spec string must look like 'column=ref', "
            "'column:ref', or 'column->ref'"
        )

    @classmethod
    def from_any(
        cls,
        value: Any = None,
        *,
        schema: Any | None = None,
    ) -> list[ForeignKeySpec]:
        if isinstance(value, cls):
            return [value]

        if isinstance(value, pa.Schema) or hasattr(value, "to_arrow_schema"):
            return cls.from_schema(value)

        if value is None:
            return cls.from_schema(schema) if schema is not None else []

        if isinstance(value, str):
            return [cls.from_str(value)]

        if isinstance(value, Mapping):
            return [
                cls(column=str(column), ref=str(ref))
                for column, ref in value.items()
                if str(column).strip() and str(ref).strip()
            ]

        if isinstance(value, tuple) and len(value) == 2:
            column, ref = value
            if str(column).strip() and str(ref).strip():
                return [cls(column=str(column).strip(), ref=str(ref).strip())]
            return []

        if isinstance(value, list) and all(isinstance(item, cls) for item in value):
            return value

        if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
            specs: list[ForeignKeySpec] = []
            for item in value:
                specs.extend(cls.from_any(item))
            return specs

        raise TypeError(f"Cannot build {cls.__name__} from {type(value).__name__}")

    @classmethod
    def from_(
        cls,
        value: Any = None,
        **kwargs: Any,
    ) -> list[ForeignKeySpec]:
        return cls.from_any(value, **kwargs)

