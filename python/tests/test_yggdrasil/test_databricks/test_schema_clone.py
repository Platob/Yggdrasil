"""Unit tests for :meth:`UCSchema.clone` — parallel fan-out over children.

``clone`` ensures the target schema exists, then clones every child in
parallel: a child already present in the target is skipped, a missing one is
created (``CREATE TABLE IF NOT EXISTS … CLONE``). ``replace=True`` overwrites,
``include_views=False`` drops view-shaped children, and a single child's
failure is recorded without aborting the batch.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from yggdrasil.databricks.schema.schema import UCSchema


def _child(name: str, *, exists: bool = False, is_view: bool = False) -> MagicMock:
    """A stand-in source :class:`Table` whose ``clone`` records its kwargs."""
    t = MagicMock(name=f"src:{name}")
    t.table_name = name
    t.is_view = is_view
    t.full_name.return_value = f"src.s.{name}"
    return t


def _schema_with_children(children: list[MagicMock], *, target_present: set[str] | None = None):
    """Build a source ``UCSchema`` whose ``tables()`` yields *children* and
    whose target-side ``table(name)`` reports presence per *target_present*."""
    present = target_present or set()
    svc = MagicMock()
    svc.catalog_name = "src"
    svc.schema_name = "s"
    svc.client.base_url.host = "example.cloud.databricks.com"

    src = UCSchema(service=svc, catalog_name="src", schema_name="s")

    # Target-side table handles: exists() answers from *present*; clone is a no-op.
    def _target_table(tname: str):
        h = MagicMock(name=f"dst:{tname}")
        h.exists.return_value = tname in present
        h.full_name.return_value = f"dst.s.{tname}"
        return h

    # Patch navigation: tables() → children; target schema build → a stub whose
    # .table(name) hands back per-name target handles and ensure_created is a no-op.
    tgt = MagicMock(name="tgt-schema")
    tgt.table.side_effect = _target_table
    tgt.full_name.return_value = "dst.s"
    return src, tgt, children


def test_clone_creates_missing_and_skips_present():
    a, b, c = _child("a"), _child("b"), _child("c")
    src, tgt, children = _schema_with_children([a, b, c], target_present={"b"})

    # Build the target via the real ``type(self)(...)`` → a real UCSchema;
    # redirect its ``.table`` to our per-name stub and no-op ``ensure_created``.
    with patch.object(UCSchema, "tables", return_value=iter(children)), \
         patch("yggdrasil.databricks.schema.schema.UCSchema.ensure_created"), \
         patch.object(UCSchema, "table", side_effect=tgt.table):
        result = src.clone(schema_name="dst")

    assert result == {"a": "created", "b": "skipped", "c": "created"}
    a.clone.assert_called_once()
    c.clone.assert_called_once()
    b.clone.assert_not_called()
    # missing-create path uses IF NOT EXISTS (missing_ok=True, replace=False)
    _, kwargs = a.clone.call_args
    assert kwargs["missing_ok"] is True and kwargs["replace"] is False


def test_clone_replace_overwrites_even_when_present():
    a = _child("a")
    src, tgt, children = _schema_with_children([a], target_present={"a"})
    with patch.object(UCSchema, "tables", return_value=iter(children)), \
         patch("yggdrasil.databricks.schema.schema.UCSchema.ensure_created"), \
         patch.object(type(src), "table", side_effect=tgt.table):
        result = src.clone(schema_name="dst", replace=True)
    assert result == {"a": "created"}
    _, kwargs = a.clone.call_args
    assert kwargs["replace"] is True and kwargs["missing_ok"] is False


def test_clone_excludes_views_when_requested():
    tbl, view = _child("t"), _child("v", is_view=True)
    src, tgt, children = _schema_with_children([tbl, view])
    with patch.object(UCSchema, "tables", return_value=iter(children)), \
         patch("yggdrasil.databricks.schema.schema.UCSchema.ensure_created"), \
         patch.object(type(src), "table", side_effect=tgt.table):
        result = src.clone(schema_name="dst", include_views=False)
    assert result == {"t": "created"}
    view.clone.assert_not_called()


def test_clone_records_per_child_failure_without_aborting():
    good, bad = _child("good"), _child("bad")
    bad.clone.side_effect = RuntimeError("permission denied")
    src, tgt, children = _schema_with_children([good, bad])
    with patch.object(UCSchema, "tables", return_value=iter(children)), \
         patch("yggdrasil.databricks.schema.schema.UCSchema.ensure_created"), \
         patch.object(type(src), "table", side_effect=tgt.table):
        result = src.clone(schema_name="dst")
    assert result["good"] == "created"
    assert result["bad"].startswith("failed: ")
    good.clone.assert_called_once()


def test_clone_onto_itself_is_rejected():
    src, _tgt, _children = _schema_with_children([])
    try:
        src.clone(schema_name="s")  # same catalog + schema as source
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "onto itself" in str(exc)
