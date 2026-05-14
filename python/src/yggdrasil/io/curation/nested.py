""":class:`NestedCurator` — recursively curate struct / list / map arrays.

Nested columns are common at integration boundaries: a JSON HTTP body
that has a ``list<string>`` of numeric strings, a Databricks payload
shaped as ``struct<id: string, when: string>``, a CSV-derived
``map<string, string>`` where every value is actually a datetime. The
recursive curator descends through each layer, picks the right
:class:`Curator` subclass for the leaf, and rebuilds the parent array
preserving offsets + validity.

Layered behaviour (one rule per Arrow nested family):

* **struct**  — curate every child field. The child's name + nullability
  flow into the new :class:`StructType`.
* **list** (incl. ``large_list``, ``fixed_size_list``) — curate the
  flat values, rebuild with the original offsets + null mask. Empty
  lists and null lists round-trip exactly.
* **map** — curate the keys and items independently. Keys stay
  non-nullable per Arrow's map contract.

Children whose dtype no Curator subclass handles pass through
unchanged — same contract as :meth:`Curator.curate_arrow_tabular`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pyarrow as pa

from .base import ArrayLike, CurationResult, Curator

if TYPE_CHECKING:
    from yggdrasil.data.types import DataType


__all__ = ["NestedCurator"]


class NestedCurator(Curator):
    """Recursively curate ``struct`` / ``list`` / ``map`` Arrow arrays.

    Constructor kwargs are forwarded to every inner curator picked for
    a child field / list value / map key+item — so a single
    ``NestedCurator(target_tz="Europe/Paris")`` propagates the timezone
    choice all the way down to whatever :class:`StringCurator` ends up
    curating a leaf timestamp column.
    """

    def __init__(self, **inner_kwargs: Any) -> None:
        # Plain class (not a frozen dataclass) because ``Curator.pick``
        # forwards arbitrary curator kwargs in here, and we want to
        # capture them as-is rather than declaring a field for each.
        self._inner_kwargs = inner_kwargs

    # ===================================================== Curator surface

    @classmethod
    def handles(cls, dtype: pa.DataType) -> bool:
        return (
            pa.types.is_struct(dtype)
            or pa.types.is_list(dtype)
            or pa.types.is_large_list(dtype)
            or pa.types.is_fixed_size_list(dtype)
            or pa.types.is_map(dtype)
        )

    def infer(self, array: ArrayLike) -> "DataType":
        return self.curate(array).dtype

    def curate(self, array: ArrayLike) -> CurationResult:
        flat = self._flatten_chunks(array)
        dtype = flat.type

        if pa.types.is_struct(dtype):
            new_array, new_type = self._curate_struct(flat)
        elif pa.types.is_map(dtype):
            new_array, new_type = self._curate_map(flat)
        else:
            # Catch-all for the list family — covers list, large_list,
            # fixed_size_list. The shape of the rebuild is the same;
            # only the constructor differs.
            new_array, new_type = self._curate_list(flat)

        from yggdrasil.data.types import DataType

        ygg_dtype = DataType.from_arrow_type(new_type)

        if isinstance(array, pa.ChunkedArray):
            new_array = pa.chunked_array([new_array])
        return CurationResult(array=new_array, dtype=ygg_dtype)

    # ===================================================== Per-family ops

    def _curate_struct(self, array: pa.StructArray) -> tuple[pa.Array, pa.DataType]:
        new_children: list[pa.Array] = []
        new_fields: list[pa.Field] = []
        for i in range(array.type.num_fields):
            field = array.type.field(i)
            child = array.field(i)
            curated_child, curated_type = self._curate_child(child)
            new_children.append(curated_child)
            new_fields.append(
                pa.field(field.name, curated_type, nullable=field.nullable)
            )

        # Preserve the parent struct's validity. ``mask=`` takes
        # "True means null", same convention as ``array.is_null()``.
        mask = array.is_null() if array.null_count else None
        new_array = pa.StructArray.from_arrays(
            new_children, fields=new_fields, mask=mask
        )
        return new_array, pa.struct(new_fields)

    def _curate_list(self, array: pa.Array) -> tuple[pa.Array, pa.DataType]:
        values = array.values
        curated_values, curated_value_type = self._curate_child(values)

        # The list value field carries its own name + nullability;
        # keep them on the rebuild so schema round-trips don't drop
        # the "item" / "element" alias the source picked.
        value_field = array.type.value_field.with_type(curated_value_type)

        mask = array.is_null() if array.null_count else None
        if pa.types.is_large_list(array.type):
            new_array = pa.LargeListArray.from_arrays(
                array.offsets, curated_values, mask=mask
            )
            new_type = pa.large_list(value_field)
        elif pa.types.is_fixed_size_list(array.type):
            # FixedSizeListArray has no ``from_arrays(offsets, values)`` —
            # construct via ``pa.FixedSizeListArray.from_arrays(values, list_size)``
            # which uses the flat values + the type's known list_size.
            new_array = pa.FixedSizeListArray.from_arrays(
                curated_values, array.type.list_size
            )
            if mask is not None:
                # FixedSizeListArray doesn't take a mask kwarg; rebuild
                # via the Array buffer to attach the null bitmap.
                new_array = self._apply_null_mask(new_array, mask)
            new_type = pa.list_(value_field, list_size=array.type.list_size)
        else:
            new_array = pa.ListArray.from_arrays(
                array.offsets, curated_values, mask=mask
            )
            new_type = pa.list_(value_field)
        return new_array, new_type

    def _curate_map(self, array: pa.MapArray) -> tuple[pa.Array, pa.DataType]:
        # Map's keys must stay non-nullable per Arrow's map contract,
        # so we curate keys + items independently and keep the keys
        # field non-nullable on the rebuild.
        curated_keys, curated_key_type = self._curate_child(array.keys)
        curated_items, curated_item_type = self._curate_child(array.items)

        mask = array.is_null() if array.null_count else None
        new_array = pa.MapArray.from_arrays(
            array.offsets, curated_keys, curated_items, mask=mask
        )
        # Field names follow the Arrow-canonical "key" / "value" so the
        # MapType round-trip through ``DataType.from_arrow_type`` finds
        # them where it expects.
        new_type = pa.map_(
            curated_key_type,
            pa.field("value", curated_item_type, nullable=True),
        )
        return new_array, new_type

    # ============================================================ helpers

    def _curate_child(self, child: pa.Array) -> tuple[pa.Array, pa.DataType]:
        """Pick the right curator for *child* and return (array, type).

        Falls back to the original child + dtype when no Curator
        subclass claims the child's type — the as-is contract that the
        tabular and engine wrappers also honour.
        """
        try:
            curator = Curator.pick(child, **self._inner_kwargs)
        except TypeError:
            return child, child.type
        result = curator.curate(child)
        flat = (
            result.array.combine_chunks()
            if isinstance(result.array, pa.ChunkedArray)
            else result.array
        )
        return flat, flat.type

    @staticmethod
    def _flatten_chunks(array: ArrayLike) -> pa.Array:
        if isinstance(array, pa.ChunkedArray):
            return array.combine_chunks() if array.num_chunks != 1 else array.chunk(0)
        return array

    @staticmethod
    def _apply_null_mask(array: pa.Array, mask: pa.BooleanArray) -> pa.Array:
        """Attach *mask* (True = null) to *array* via the Array C API.

        Used for ``FixedSizeListArray`` where ``from_arrays`` doesn't
        accept a mask kwarg directly. Everywhere else we let
        ``from_arrays(..., mask=...)`` handle it.
        """
        # ``pa.Array.from_buffers`` is the canonical way to swap the
        # validity buffer without copying the values. We invert the
        # mask because Arrow stores "1 = valid".
        import pyarrow.compute as pc

        valid_bits = pc.invert(mask).cast(pa.bool_())
        # Build a new array with the validity buffer. Easiest portable
        # path: round-trip via ``pa.array(array.to_pylist(), …)`` would
        # be a per-row hop. Instead reach for ``pa.Array.from_buffers``.
        n = len(array)
        validity_buffer = pa.array(valid_bits).buffers()[1]
        buffers = list(array.buffers())
        buffers[0] = validity_buffer
        return pa.Array.from_buffers(array.type, n, buffers, null_count=-1)
