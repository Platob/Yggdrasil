"""Unit tests for :class:`yggdrasil.io.buffer.TabularUpsertBatch`.

The carrier is pure data — these tests cover construction, type
checking, iteration order, and the :meth:`apply` routing without
touching any specific format backend.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from yggdrasil.io.buffer import TabularUpsertBatch
from yggdrasil.io.buffer.memory import MemoryArrowIO


def _holder(*rows: dict[str, int]) -> MemoryArrowIO:
    """In-memory holder pre-populated with a single Arrow batch.

    Empty input gives a schema-less empty holder so we can exercise
    the apply path without forcing a particular schema.
    """
    h = MemoryArrowIO()
    if rows:
        h.write_arrow_table(pa.Table.from_pylist(list(rows)))
    return h


class TestTabularUpsertBatch:
    def test_defaults_are_empty_dicts(self):
        # The carrier is a pure data holder — both sides default to
        # empty insertion-ordered dicts so callers can stage rows
        # incrementally via setdefault / direct assignment.
        parent = MemoryArrowIO()
        batch = TabularUpsertBatch(parent=parent)
        assert batch.parent is parent
        assert batch.match == {}
        assert batch.new == {}
        assert batch.match_keys == []
        assert batch.new_keys == []
        assert batch.is_empty()
        assert list(batch) == []

    def test_iteration_walks_match_then_new_in_insertion_order(self):
        parent = MemoryArrowIO()
        m_a = _holder({"k": 1})
        m_b = _holder({"k": 2})
        n_c = _holder({"k": 3})
        batch = TabularUpsertBatch(
            parent=parent,
            match={"src_a": m_a, "src_b": m_b},
            new={"src_c": n_c},
        )
        assert list(batch) == [m_a, m_b, n_c]
        assert batch.match_keys == ["src_a", "src_b"]
        assert batch.new_keys == ["src_c"]
        assert not batch.is_empty()

    def test_mapping_inputs_are_copied_into_owned_dicts(self):
        # Hand the constructor a non-dict mapping (a MappingProxyType)
        # so we can verify the post-init copy: external mutation must
        # not leak through, and the attribute must be a real dict so
        # ``setdefault`` and ordered insertion keep working.
        from types import MappingProxyType

        parent = MemoryArrowIO()
        m_src = {"src_a": _holder({"k": 1})}
        proxy = MappingProxyType(m_src)
        batch = TabularUpsertBatch(parent=parent, match=proxy)
        assert isinstance(batch.match, dict)
        assert batch.match == dict(proxy)
        # Mutating the original source mapping after construction does
        # not affect the carrier — the carrier owns its container.
        m_src["src_b"] = _holder({"k": 2})
        assert "src_b" not in batch.match

    def test_non_tabular_holder_is_rejected(self):
        # Strict on meaning: only TabularIO holders go in the buckets,
        # otherwise downstream ``apply`` would fail in a confusing
        # place. Reject up front in ``__post_init__``.
        parent = MemoryArrowIO()
        with pytest.raises(TypeError, match="must be a TabularIO"):
            TabularUpsertBatch(
                parent=parent,
                match={"bad": [1, 2, 3]},  # type: ignore[dict-item]
            )

    def test_apply_routes_match_via_upsert_and_new_via_append(self):
        # Sanity-check the apply contract: every staged holder gets
        # forwarded to the parent in pipeline order — match holders
        # under Mode.UPSERT, new holders under Mode.APPEND. We capture
        # the (rows, mode) pairs the parent would see by stubbing
        # write_arrow_batches.
        from yggdrasil.io.enums import Mode
        from yggdrasil.data.options import CastOptions

        captured: list[tuple[list[dict], Mode]] = []

        class CapturingParent(MemoryArrowIO):
            def write_arrow_batches(self, batches, options=None, **kwargs):
                table = pa.Table.from_batches(list(batches))
                captured.append(
                    (table.to_pylist(), options.mode if options else Mode.AUTO),
                )

        parent = CapturingParent()
        batch = TabularUpsertBatch(
            parent=parent,
            match={"src_match": _holder({"k": 1, "v": "old"})},
            new={"src_new": _holder({"k": 9, "v": "fresh"})},
        )

        batch.apply(options=CastOptions(match_by_names=["k"]))

        assert captured == [
            ([{"k": 1, "v": "old"}], Mode.UPSERT),
            ([{"k": 9, "v": "fresh"}], Mode.APPEND),
        ]


class TestTabularIOUpsertHook:
    def test_default_upsert_hook_raises_not_implemented(self):
        # ``_upsert(options)`` is the canonical engine hook for native
        # MERGE — the base implementation surfaces a clear error so
        # callers know to land on a subclass that implements it (or
        # take the rewrite path explicitly).
        from yggdrasil.data.options import CastOptions

        holder = MemoryArrowIO()
        with pytest.raises(NotImplementedError, match="_upsert is not implemented"):
            holder._upsert(CastOptions(match_by_names=["k"]))
