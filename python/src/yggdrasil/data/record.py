"""Single-row materialization with a singleton :class:`Schema` reference.

A :class:`Record` is a :class:`collections.abc.Mapping` over one row's
values, keyed by field name. The schema is *shared by reference*
across sibling rows — a stream of N records carries one
:class:`Schema` and N value tuples, not N (Schema, values) pairs.
That keeps row materialization cheap when the same schema repeats
across millions of rows (the only real use case).

:class:`Record` is the natural unit returned by
:meth:`TabularIO.read_records` and consumed by
:meth:`TabularIO.write_records`. Subclasses with a richer row shape
(SQL row, Spark Row, etc.) should still satisfy the Mapping contract
so callers don't need to know the concrete origin.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Union

import pyarrow as pa

if TYPE_CHECKING:
    from .data_field import Field
    from .schema import Schema


__all__ = ["Record", "RecordValues"]


# Anything we accept as the per-row value carrier. Tuples / lists go
# in positional, mappings get re-aligned to the schema's field order.
RecordValues = Union[tuple, list, Mapping[str, Any]]


class Record(Mapping[str, Any]):
    """Single row, keyed by field name, sharing a :class:`Schema`.

    The Mapping protocol gives callers ``record[name]``,
    ``record.get(name, default)``, ``in``, ``keys()``, ``values()``,
    ``items()``, and ``len()`` without any extra surface. Positional
    integer access (``record[0]``) is supported as a convenience for
    fast-path callers that already know the field index.

    Construction:

    - ``Record((v0, v1, v2), schema)`` — values aligned with
      ``schema.fields`` order.
    - ``Record({"a": 1, "b": 2}, schema)`` — dict re-aligned to
      ``schema.fields`` order; missing keys land as ``None``.

    The schema is taken by reference — pass the same :class:`Schema`
    instance across a stream of rows to keep allocation flat.
    """

    __slots__ = ("_values", "_schema")

    def __init__(
        self,
        values: RecordValues,
        schema: "Schema",
    ) -> None:
        if isinstance(values, Mapping):
            values = tuple(values.get(f.name) for f in schema.fields)
        else:
            values = tuple(values)
        if len(values) != len(schema.fields):
            raise ValueError(
                f"Record values length {len(values)} does not match "
                f"schema field count {len(schema.fields)}. "
                f"Schema fields: {[f.name for f in schema.fields]!r}."
            )
        self._values = values
        self._schema = schema

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def values(self) -> tuple:  # noqa: D401 — Mapping has a `values()` method
        """Positional value tuple, aligned with ``schema.fields``."""
        return self._values

    @property
    def schema(self) -> "Schema":
        """The shared :class:`Schema`. Same instance across sibling rows."""
        return self._schema

    @property
    def fields(self) -> list["Field"]:
        return self._schema.fields

    # ------------------------------------------------------------------
    # Mapping protocol
    # ------------------------------------------------------------------

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, int):
            return self._values[key]
        if isinstance(key, str):
            for i, f in enumerate(self._schema.fields):
                if f.name == key:
                    return self._values[i]
            raise KeyError(
                f"No field named {key!r}. Available: "
                f"{[f.name for f in self._schema.fields]!r}."
            )
        raise TypeError(
            f"Record key must be str or int; got {type(key).__name__}: {key!r}"
        )

    def __iter__(self) -> Iterator[str]:
        return (f.name for f in self._schema.fields)

    def __len__(self) -> int:
        return len(self._values)

    def __contains__(self, key: object) -> bool:
        if isinstance(key, str):
            return any(f.name == key for f in self._schema.fields)
        return False

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Record):
            return (
                self._schema is other._schema
                and self._values == other._values
            )
        if isinstance(other, Mapping):
            return Mapping.__eq__(self, other)
        return NotImplemented

    # Mapping is intentionally unhashable; restate so __slots__ doesn't
    # quietly inherit object.__hash__.
    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        body = ", ".join(
            f"{f.name}={self._values[i]!r}"
            for i, f in enumerate(self._schema.fields)
        )
        return f"Record({body})"

    # ------------------------------------------------------------------
    # Conversions
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {f.name: v for f, v in zip(self._schema.fields, self._values)}

    def to_tuple(self) -> tuple:
        return self._values

    # ------------------------------------------------------------------
    # Streaming helpers — share one Schema across a whole stream
    # ------------------------------------------------------------------

    @classmethod
    def from_arrow_batches(
        cls,
        batches: Iterable[pa.RecordBatch],
        *,
        schema: "Schema | None" = None,
    ) -> Iterator["Record"]:
        """Yield :class:`Record`\\ s from an Arrow-batch stream.

        The first batch's schema becomes the singleton :class:`Schema`
        all yielded records share, unless one is passed explicitly.
        Per-row values are materialized via ``column[i].as_py()`` —
        cheap for primitive columns, expensive for nested types. If
        you only need a few columns, project the batch first.
        """
        from .schema import Schema as _Schema

        shared = schema
        for batch in batches:
            if shared is None:
                shared = _Schema.from_arrow(batch.schema)
            cols = [batch.column(i) for i in range(batch.num_columns)]
            for i in range(batch.num_rows):
                yield cls(tuple(c[i].as_py() for c in cols), shared)
