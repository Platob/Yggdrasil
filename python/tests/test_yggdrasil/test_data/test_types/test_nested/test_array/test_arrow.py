"""Arrow-side casts for :class:`ArrayType`.

Two cast surfaces under test:

* :func:`cast_arrow_list_array` — list / large_list / fixed_size_list
  / list_view / large_list_view / chunked_list source → list /
  large_list / fixed_size_list / list_view / large_list_view target.
* :func:`cast_arrow_map_array_to_list` — map<k,v> →
  list<struct<key,value>>; the inverse direction lives in
  ``test_map``.

Plus the rejection paths: mismatched source kind and struct items
that don't fit the (key, value) 2-ary contract.
"""
from __future__ import annotations

import pyarrow as pa
import pytest

from yggdrasil.data.data_field import Field
from yggdrasil.data.options import CastOptions
from yggdrasil.data.types.nested.array import (
    cast_arrow_list_array,
    cast_arrow_map_array_to_list,
)


# ---------------------------------------------------------------------------
# list → list
# ---------------------------------------------------------------------------


class TestListToList:

    def test_changes_item_dtype(
        self,
        source_array_field: Field,
        target_array_field: Field,
    ) -> None:
        array = pa.array(
            [[1, 2], [3, None], None],
            type=pa.list_(pa.int64()),
        )

        result = cast_arrow_list_array(
            array,
            CastOptions(
                source=source_array_field,
                target=target_array_field,
            ),
        )

        assert isinstance(result, pa.ListArray)
        assert result.type == pa.list_(pa.string())
        assert result.to_pylist() == [["1", "2"], ["3", None], None]

    def test_widens_to_large_list(
        self,
        source_array_field: Field,
        target_large_array_field: Field,
    ) -> None:
        array = pa.array([[1, 2], [3, None], None], type=pa.list_(pa.int64()))

        result = cast_arrow_list_array(
            array,
            CastOptions(
                source=source_array_field,
                target=target_large_array_field,
            ),
        )

        assert isinstance(result, pa.LargeListArray)
        assert result.type == pa.large_list(pa.string())
        assert result.to_pylist() == [["1", "2"], ["3", None], None]

    def test_narrows_to_fixed_size_list(
        self,
        source_array_field: Field,
        target_fixed_array_field: Field,
    ) -> None:
        array = pa.array([[1, 2], [3, None], None], type=pa.list_(pa.int64()))

        result = cast_arrow_list_array(
            array,
            CastOptions(
                source=source_array_field,
                target=target_fixed_array_field,
            ),
        )

        assert isinstance(result, pa.FixedSizeListArray)
        assert result.type == pa.list_(pa.string(), 2)
        # FixedSizeListArray with a mask collapses null rows out — visible
        # rows match what made it through the cast.
        assert result.to_pylist() == [["1", "2"], ["3", None]]

    def test_chunked_input_keeps_chunked_shape(
        self,
        source_array_field: Field,
        target_array_field: Field,
    ) -> None:
        chunk_1 = pa.array([[1, 2], None], type=pa.list_(pa.int64()))
        chunk_2 = pa.array([[3]], type=pa.list_(pa.int64()))
        array = pa.chunked_array([chunk_1, chunk_2], type=pa.list_(pa.int64()))

        result = cast_arrow_list_array(
            array,
            CastOptions(
                source=source_array_field,
                target=target_array_field,
            ),
        )

        assert isinstance(result, pa.ChunkedArray)
        assert result.type == pa.list_(pa.string())
        assert result.to_pylist() == [["1", "2"], None, ["3"]]

    def test_preserves_null_mask_for_outer_null_and_empty_list(
        self,
        source_array_field: Field,
        target_array_field: Field,
    ) -> None:
        array = pa.array(
            [None, [], [1, None, 3]],
            type=pa.list_(pa.int64()),
        )

        result = cast_arrow_list_array(
            array,
            CastOptions(
                source=source_array_field,
                target=target_array_field,
            ),
        )

        assert result.to_pylist() == [None, [], ["1", None, "3"]]
        assert result.is_null().to_pylist() == [True, False, False]


class TestListShortCircuits:

    def test_target_none_returns_input_identity(
        self, source_array_field: Field
    ) -> None:
        array = pa.array([[1, 2]], type=pa.list_(pa.int64()))

        out = cast_arrow_list_array(
            array,
            CastOptions(source=source_array_field, target=None),
        )

        assert out is array


class TestListRejections:

    def test_non_array_source_raises(
        self,
        source_map_field: Field,
        target_array_field: Field,
    ) -> None:
        array = pa.array(
            [[("a", 1)]], type=pa.map_(pa.string(), pa.int64())
        )

        with pytest.raises(pa.ArrowInvalid, match="Cannot cast"):
            cast_arrow_list_array(
                array,
                CastOptions(
                    source=source_map_field,
                    target=target_array_field,
                ),
            )

class TestListViewSource:
    """ListView / LargeListView source coverage.

    Pyarrow's ``pc.cast(list_view → list)`` is broken: it reuses the
    raw offsets buffer and silently truncates rows whose ``(offset,
    size)`` pairs don't pack into monotone List offsets. We materialise
    via ``flatten()`` + ``cumulative_sum(sizes)`` instead, which keeps
    every row including out-of-order and overlapping layouts.
    """

    def test_list_view_source_packs_to_list(
        self,
        target_array_field: Field,
    ) -> None:
        array = pa.array(
            [[1, 2], [3], [4, 5, 6], None, [7]],
            type=pa.list_view(pa.int64()),
        )

        result = cast_arrow_list_array(
            array,
            CastOptions(target=target_array_field),
        )

        assert isinstance(result, pa.ListArray)
        assert result.type == pa.list_(pa.string())
        assert result.to_pylist() == [
            ["1", "2"], ["3"], ["4", "5", "6"], None, ["7"],
        ]

    def test_list_view_source_handles_out_of_order_offsets(
        self,
        target_array_field: Field,
    ) -> None:
        # Out-of-order layout: row 0 reads from index 3, row 1 from
        # index 0, row 2 from index 2 — exactly the shape pyarrow's
        # ``pc.cast`` mishandles by reusing raw offsets.
        array = pa.ListViewArray.from_arrays(
            offsets=pa.array([3, 0, 2], type=pa.int32()),
            sizes=pa.array([3, 2, 1], type=pa.int32()),
            values=pa.array([10, 20, 30, 40, 50, 60], type=pa.int64()),
        )

        result = cast_arrow_list_array(
            array,
            CastOptions(target=target_array_field),
        )

        assert isinstance(result, pa.ListArray)
        assert result.to_pylist() == [
            ["40", "50", "60"], ["10", "20"], ["30"],
        ]

    def test_large_list_view_source_packs_to_list(
        self,
        target_array_field: Field,
    ) -> None:
        array = pa.array(
            [[1, 2], None, [3, 4, 5]],
            type=pa.large_list_view(pa.int64()),
        )

        result = cast_arrow_list_array(
            array,
            CastOptions(target=target_array_field),
        )

        assert result.to_pylist() == [["1", "2"], None, ["3", "4", "5"]]


class TestListViewTarget:
    """``view=True`` target builds a ListView / LargeListView from the
    cast result instead of raising.
    """

    def test_list_to_list_view(
        self,
        source_array_field: Field,
        target_view_array_field: Field,
    ) -> None:
        array = pa.array(
            [[1, 2], [3, None], None, [4, 5, 6]],
            type=pa.list_(pa.int64()),
        )

        result = cast_arrow_list_array(
            array,
            CastOptions(
                source=source_array_field,
                target=target_view_array_field,
            ),
        )

        assert isinstance(result, pa.ListViewArray)
        assert result.type == pa.list_view(pa.string())
        assert result.to_pylist() == [
            ["1", "2"], ["3", None], None, ["4", "5", "6"],
        ]

    def test_list_view_to_list_view(
        self,
        target_view_array_field: Field,
    ) -> None:
        array = pa.array(
            [[1, 2], [3], [4, 5, 6]],
            type=pa.list_view(pa.int64()),
        )

        result = cast_arrow_list_array(
            array,
            CastOptions(target=target_view_array_field),
        )

        assert isinstance(result, pa.ListViewArray)
        assert result.to_pylist() == [["1", "2"], ["3"], ["4", "5", "6"]]


class TestListViewOfStructWide:
    """Wide list_view<struct{...}> with many items per row.

    Production payloads (event arrays, search hits, audit trails) often
    arrive shaped as ``list_view<struct{...}>`` with dozens of struct
    items per row across thousands of rows. The cast path has to walk
    every nested struct field while honouring the view's
    out-of-order / overlapping / null-row layout.
    """

    @staticmethod
    def _wide_struct(int_byte_size: int = 8) -> pa.StructType:
        # 16 fields — half int, half string — mirrors the "wide list of
        # struct" shape the bench exercises end to end.
        int_t = pa.int64() if int_byte_size == 8 else pa.int32()
        return pa.struct([
            (f"i{k:02d}", int_t) for k in range(8)
        ] + [
            (f"s{k:02d}", pa.string()) for k in range(8)
        ])

    @staticmethod
    def _wide_payload(items_per_row: int, rows: int) -> list:
        return [
            None if (r % 11 == 0) else [
                {
                    **{f"i{k:02d}": r * items_per_row + k for k in range(8)},
                    **{f"s{k:02d}": f"r{r}-k{k}" for k in range(8)},
                }
                for _ in range(items_per_row)
            ]
            for r in range(rows)
        ]

    def test_wide_list_view_of_struct_widens_inner_int(self) -> None:
        from yggdrasil.data.types.nested import ArrayType, StructType
        from yggdrasil.data.types.primitive import IntegerType, StringType

        items_per_row, rows = 50, 32
        src_struct = self._wide_struct(int_byte_size=8)
        src = pa.array(
            self._wide_payload(items_per_row, rows),
            type=pa.list_view(src_struct),
        )

        item_field = Field(
            "item",
            StructType(fields=tuple([
                Field(f"i{k:02d}", IntegerType(byte_size=4, signed=True))
                for k in range(8)
            ] + [
                Field(f"s{k:02d}", StringType()) for k in range(8)
            ])),
        )
        target = Field("rows", ArrayType.from_item(item_field))

        result = cast_arrow_list_array(
            src, CastOptions(target=target),
        )

        assert isinstance(result, pa.ListArray)
        # int columns were narrowed from int64 to int32 — the inner
        # struct rebuild must have fired across every list element.
        result_inner = result.type.value_type
        assert result_inner.field("i00").type == pa.int32()
        assert result_inner.field("s00").type == pa.string()
        # Roundtrip preserves null rows and item count.
        py = result.to_pylist()
        assert len(py) == rows
        assert py[0] is None  # r=0, r%11==0
        assert len(py[1]) == items_per_row
        # Field values survive end-to-end.
        assert py[1][0]["i00"] == 1 * items_per_row + 0
        assert py[1][0]["s00"] == "r1-k0"

    def test_wide_list_view_of_struct_out_of_order_offsets(self) -> None:
        from yggdrasil.data.types.nested import ArrayType, StructType
        from yggdrasil.data.types.primitive import IntegerType, StringType

        # Build a list_view whose offsets explicitly point in reverse —
        # the regular ``pc.cast`` path drops rows here.
        struct_t = pa.struct([("k", pa.int64()), ("v", pa.string())])
        flat_values = pa.array(
            [{"k": i, "v": f"v{i}"} for i in range(12)], type=struct_t,
        )
        lv = pa.ListViewArray.from_arrays(
            offsets=pa.array([8, 4, 0], type=pa.int32()),
            sizes=pa.array([4, 4, 4], type=pa.int32()),
            values=flat_values,
        )

        item_field = Field("item", StructType(fields=(
            Field("k", IntegerType(byte_size=4, signed=True)),
            Field("v", StringType()),
        )))
        target = Field("rows", ArrayType.from_item(item_field))

        result = cast_arrow_list_array(
            lv, CastOptions(target=target),
        )

        py = result.to_pylist()
        assert len(py) == 3
        # Row 0 reads from index 8 → values 8..11; row 2 from 0..3.
        assert [r["k"] for r in py[0]] == [8, 9, 10, 11]
        assert [r["k"] for r in py[2]] == [0, 1, 2, 3]
        # Inner cast actually applied.
        assert result.type.value_type.field("k").type == pa.int32()

    def test_wide_list_view_of_struct_round_trips_via_parquet(
        self, tmp_path,
    ) -> None:
        """End-to-end: cast list_view<struct> source to list, write
        Parquet, read back — the realistic ingest flow.

        Parquet has no native list_view encoding (Arrow's writer raises
        ``ArrowNotImplemented`` on list_view / large_list_view). Down-
        stream callers cast to a regular list target before writing —
        this test exercises that exact pipeline against a wide struct
        item with many rows of out-of-order list_view input.
        """
        import pyarrow.parquet as pq

        from yggdrasil.data.types.nested import ArrayType, StructType
        from yggdrasil.data.types.primitive import IntegerType, StringType

        items_per_row, rows = 24, 64
        src_struct = self._wide_struct(int_byte_size=8)
        src = pa.array(
            self._wide_payload(items_per_row, rows),
            type=pa.list_view(src_struct),
        )

        item_field = Field(
            "item",
            StructType(fields=tuple([
                Field(f"i{k:02d}", IntegerType(byte_size=4, signed=True))
                for k in range(8)
            ] + [
                Field(f"s{k:02d}", StringType()) for k in range(8)
            ])),
        )
        target = Field("rows", ArrayType.from_item(item_field))
        casted = cast_arrow_list_array(src, CastOptions(target=target))

        path = tmp_path / "list_view_struct.parquet"
        pq.write_table(pa.table({"rows": casted}), path)

        roundtripped = pq.read_table(path)["rows"].combine_chunks()
        assert roundtripped.to_pylist() == casted.to_pylist()
        assert pa.types.is_list(roundtripped.type)

    def test_large_list_view_target_then_cast_back_writes_to_parquet(
        self, tmp_path,
    ) -> None:
        """Verify the full view round-trip: list -> large_list_view ->
        large_list and parquet round-trip.

        Parquet rejects ``large_list_view`` directly; the realistic
        flow is "cast to view for in-memory analytics, cast back to
        list before persistence". Both legs of that chain go through
        :func:`cast_arrow_list_array`.
        """
        import pyarrow.parquet as pq

        from yggdrasil.data.types.nested import ArrayType
        from yggdrasil.data.types.primitive import IntegerType

        src = pa.array(
            [[1, 2], None, [3, 4, 5]] * 100,
            type=pa.list_(pa.int64()),
        )

        view_target = Field(
            "vals",
            ArrayType.from_item(
                Field("item", IntegerType(byte_size=8, signed=True)),
                large=True, view=True,
            ),
        )
        list_target = Field(
            "vals",
            ArrayType.from_item(
                Field("item", IntegerType(byte_size=8, signed=True)),
                large=True,
            ),
        )

        as_view = cast_arrow_list_array(
            src, CastOptions(target=view_target),
        )
        assert isinstance(as_view, pa.LargeListViewArray)

        as_list = cast_arrow_list_array(
            as_view, CastOptions(target=list_target),
        )
        assert isinstance(as_list, pa.LargeListArray)

        path = tmp_path / "large_list_view_round_trip.parquet"
        pq.write_table(pa.table({"vals": as_list}), path)
        roundtripped = pq.read_table(path)["vals"].combine_chunks()
        assert roundtripped.to_pylist() == as_view.to_pylist() == src.to_pylist()


# ---------------------------------------------------------------------------
# map → list<struct<key, value>>
# ---------------------------------------------------------------------------


class TestMapToListEntries:

    def test_materialises_each_entry_as_struct(
        self,
        source_map_field: Field,
        target_entries_array_field: Field,
    ) -> None:
        array = pa.array(
            [
                [("a", 1), ("b", 2)],
                [("c", None)],
                None,
            ],
            type=pa.map_(pa.string(), pa.int64()),
        )

        result = cast_arrow_map_array_to_list(
            array,
            CastOptions(
                source=source_map_field,
                target=target_entries_array_field,
            ),
        )

        assert isinstance(result, pa.ListArray)
        assert result.type == pa.list_(
            pa.struct(
                [
                    pa.field("key", pa.string(), nullable=False),
                    pa.field("value", pa.string()),
                ]
            )
        )
        assert result.to_pylist() == [
            [{"key": "a", "value": "1"}, {"key": "b", "value": "2"}],
            [{"key": "c", "value": None}],
            None,
        ]

    def test_widens_to_large_list(
        self,
        source_map_field: Field,
        target_entries_large_array_field: Field,
    ) -> None:
        array = pa.array(
            [[("a", 1), ("b", 2)], [("c", None)], None],
            type=pa.map_(pa.string(), pa.int64()),
        )

        result = cast_arrow_map_array_to_list(
            array,
            CastOptions(
                source=source_map_field,
                target=target_entries_large_array_field,
            ),
        )

        assert isinstance(result, pa.LargeListArray)

    def test_chunked_input_kept_chunked(
        self,
        source_map_field: Field,
        target_entries_array_field: Field,
    ) -> None:
        chunk_1 = pa.array(
            [[("a", 1)], None],
            type=pa.map_(pa.string(), pa.int64()),
        )
        chunk_2 = pa.array(
            [[("b", 2), ("c", 3)]],
            type=pa.map_(pa.string(), pa.int64()),
        )
        array = pa.chunked_array(
            [chunk_1, chunk_2], type=pa.map_(pa.string(), pa.int64())
        )

        result = cast_arrow_map_array_to_list(
            array,
            CastOptions(
                source=source_map_field,
                target=target_entries_array_field,
            ),
        )

        assert isinstance(result, pa.ChunkedArray)
        assert result.to_pylist() == [
            [{"key": "a", "value": "1"}],
            None,
            [{"key": "b", "value": "2"}, {"key": "c", "value": "3"}],
        ]


class TestMapToListShortCircuits:

    def test_target_none_returns_input_identity(
        self, source_map_field: Field
    ) -> None:
        array = pa.array(
            [[("a", 1)]], type=pa.map_(pa.string(), pa.int64())
        )

        out = cast_arrow_map_array_to_list(
            array,
            CastOptions(source=source_map_field, target=None),
        )

        assert out is array


class TestMapToListRejections:

    def test_non_map_source_raises(
        self,
        source_array_field: Field,
        target_entries_array_field: Field,
    ) -> None:
        array = pa.array([[1, 2]], type=pa.list_(pa.int64()))

        with pytest.raises(pa.ArrowInvalid, match="Cannot cast"):
            cast_arrow_map_array_to_list(
                array,
                CastOptions(
                    source=source_array_field,
                    target=target_entries_array_field,
                ),
            )

    def test_target_item_must_be_struct(
        self,
        source_map_field: Field,
        invalid_target_entries_scalar_array_field: Field,
    ) -> None:
        array = pa.array(
            [[("a", 1)]], type=pa.map_(pa.string(), pa.int64())
        )

        with pytest.raises(pa.ArrowInvalid, match="Cannot cast"):
            cast_arrow_map_array_to_list(
                array,
                CastOptions(
                    source=source_map_field,
                    target=invalid_target_entries_scalar_array_field,
                ),
            )

    def test_target_struct_must_be_two_ary(
        self,
        source_map_field: Field,
        invalid_target_entries_struct_one_field_array_field: Field,
    ) -> None:
        array = pa.array(
            [[("a", 1)]], type=pa.map_(pa.string(), pa.int64())
        )

        with pytest.raises(pa.ArrowInvalid, match="Cannot cast"):
            cast_arrow_map_array_to_list(
                array,
                CastOptions(
                    source=source_map_field,
                    target=invalid_target_entries_struct_one_field_array_field,
                ),
            )
