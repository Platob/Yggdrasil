"""Arrow-side casts for :class:`ArrayType`.

Two cast surfaces under test:

* :func:`cast_arrow_list_array` — list / large_list / fixed_size_list
  / chunked_list source → list / large_list / fixed_size_list target.
* :func:`cast_arrow_map_array_to_list` — map<k,v> →
  list<struct<key,value>>; the inverse direction lives in
  ``test_map``.

Plus the rejection paths: bad target shape (list_view target),
mismatched source kind, struct items that don't fit the (key, value)
2-ary contract.
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

    def test_list_view_target_rejected(
        self,
        source_array_field: Field,
        target_view_array_field: Field,
    ) -> None:
        array = pa.array([[1, 2]], type=pa.list_(pa.int64()))

        with pytest.raises(pa.ArrowInvalid, match="Cannot cast"):
            cast_arrow_list_array(
                array,
                CastOptions(
                    source=source_array_field,
                    target=target_view_array_field,
                ),
            )


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
