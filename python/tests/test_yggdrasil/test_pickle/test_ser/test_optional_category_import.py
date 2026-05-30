"""Regression tests: an absent optional SDK must not break core pickling.

The serializer registry resolves a Python object's wire tag by sweeping
serializer categories 0..8 (:meth:`Tags.get_class_from_type`). Categories
7 (pyspark) and 8 (databricks) cover *optional* third-party SDKs whose
serializer modules import the SDK at module load. If one of those imports
is allowed to propagate, serializing **any** object — a plain Arrow
table, an int — dies with ``ModuleNotFoundError`` even though the object
has nothing to do with the missing SDK. (This was the real cause of a
batch of ``test_pickle_file`` failures.)

The contract pinned here: :meth:`Tags._ensure_category_imported`
swallows an optional category's ``ImportError`` (marking it
resolved-as-empty so it isn't retried), while a non-import failure still
propagates.

Two of these tests run unconditionally; the rest only when an optional
SDK is genuinely absent (the CI image here ships without
``databricks`` / ``pyspark``), which is exactly the regression scenario.
"""
from __future__ import annotations

import importlib.util

import pyarrow as pa
import pytest

from yggdrasil.pickle.ser import dumps, loads
from yggdrasil.pickle.ser.tags import Tags


def _missing(mod: str) -> bool:
    return importlib.util.find_spec(mod) is None


_DATABRICKS_MISSING = _missing("databricks")
_PYSPARK_MISSING = _missing("pyspark")


# ---------------------------------------------------------------------------
# Always-on: core serialization works regardless of optional SDK state
# ---------------------------------------------------------------------------


def test_dumps_arrow_table_roundtrips():
    table = pa.table({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    back = loads(dumps(table))
    assert back.num_rows == 3
    assert back.equals(table)


def test_dumps_primitives_roundtrip():
    assert loads(dumps(42)) == 42
    assert loads(dumps("hello")) == "hello"
    assert loads(dumps({"k": [1, 2, 3]})) == {"k": [1, 2, 3]}


def test_get_class_from_type_resolves_core_types():
    # The 0..8 sweep must resolve a core type without raising even when an
    # optional category's import fails partway through.
    assert Tags.get_class_from_type(int) is not None
    assert Tags.get_class_from_type(str) is not None
    assert Tags.get_class_from_type(dict) is not None


# ---------------------------------------------------------------------------
# The regression scenario itself: the optional SDK is actually absent
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _DATABRICKS_MISSING,
    reason="databricks SDK is installed; the absent-SDK path can't be exercised here",
)
def test_absent_databricks_category_resolved_as_empty():
    # With the SDK missing, the sweep has run during the dumps above and
    # marked category 8 imported-but-empty: no databricks types leaked in,
    # yet the category is recorded so it isn't retried on every call.
    Tags.get_class_from_type(int)  # ensure the sweep has executed
    assert 8 in Tags._IMPORTED_CATEGORIES
    assert not any("databricks" in str(t).lower() for t in Tags.TYPES)


@pytest.mark.skipif(
    not _DATABRICKS_MISSING,
    reason="databricks SDK is installed",
)
def test_ensure_category_imported_is_idempotent_when_sdk_absent():
    # Calling it directly must not raise, and must leave the category
    # marked resolved so a second call is a cheap no-op.
    Tags._ensure_category_imported(8 * Tags.CATEGORY_SIZE)
    assert 8 in Tags._IMPORTED_CATEGORIES
    # idempotent
    Tags._ensure_category_imported(8 * Tags.CATEGORY_SIZE)
    assert 8 in Tags._IMPORTED_CATEGORIES


# ---------------------------------------------------------------------------
# A non-ImportError from a category import must still propagate
# ---------------------------------------------------------------------------


def test_non_importerror_from_category_propagates(monkeypatch):
    # Pick a category id that is NOT yet cached so the import path runs.
    cid = 8
    was_cached = cid in Tags._IMPORTED_CATEGORIES
    monkeypatch.setattr(Tags, "_IMPORTED_CATEGORIES", set(Tags._IMPORTED_CATEGORIES))
    Tags._IMPORTED_CATEGORIES.discard(cid)

    real_import_category = Tags._import_category.__func__

    def boom(cls, c):
        if c == cid:
            raise RuntimeError("not an import problem")
        return real_import_category(cls, c)

    monkeypatch.setattr(Tags, "_import_category", classmethod(boom))
    with pytest.raises(RuntimeError, match="not an import problem"):
        Tags._ensure_category_imported(cid * Tags.CATEGORY_SIZE)

    # A RuntimeError is not swallowed, so the category must NOT be marked
    # resolved — a later call (after the transient fault clears) retries.
    assert cid not in Tags._IMPORTED_CATEGORIES
    _ = was_cached  # documented: monkeypatch restores the real cache after
