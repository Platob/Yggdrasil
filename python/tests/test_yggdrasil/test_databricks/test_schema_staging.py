"""Unit tests for :meth:`UCSchema.staging_location`.

The schema records its yggdrasil staging root (``<root>/uc/tables``) under the
``ygg.staging_path`` property so a table reads it straight from the schema
metadata instead of re-deriving it from the storage location every time.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from yggdrasil.databricks.schema.schema import UCSchema


def _schema() -> UCSchema:
    svc = MagicMock()
    svc.client.base_url.host = "example.cloud.databricks.com"
    return UCSchema(service=svc, catalog_name="cat", schema_name="sch")


def _info(*, storage_location=None, properties=None):
    return SimpleNamespace(storage_location=storage_location, properties=properties)


def test_returns_existing_property_without_deriving_or_writing():
    s = _schema()
    info = _info(
        storage_location="s3://bkt/meta/__unitystorage/...",
        properties={"ygg.staging_path": "s3://custom/staging"},
    )
    with patch.object(UCSchema, "read_infos", return_value=info), \
            patch.object(UCSchema, "update") as upd:
        assert s.staging_location() == "s3://custom/staging"
    upd.assert_not_called()  # already recorded → no re-derive, no ALTER


def test_derives_from_storage_location_and_stamps_property():
    s = _schema()
    info = _info(
        storage_location="s3://bkt/meta/__unitystorage/catalogs/x/schemas/y",
        properties={"existing": "keep"},
    )
    with patch.object(UCSchema, "read_infos", return_value=info), \
            patch.object(UCSchema, "update") as upd:
        out = s.staging_location()

    assert out == "s3://bkt/meta/uc/tables"
    # Stamped onto the schema properties, merged with the existing ones.
    upd.assert_called_once()
    props = upd.call_args.kwargs["properties"]
    assert props == {"existing": "keep", "ygg.staging_path": "s3://bkt/meta/uc/tables"}


def test_create_false_does_not_write():
    s = _schema()
    info = _info(storage_location="s3://bkt/meta/__unitystorage/x", properties={})
    with patch.object(UCSchema, "read_infos", return_value=info), \
            patch.object(UCSchema, "update") as upd:
        assert s.staging_location(create=False) == "s3://bkt/meta/uc/tables"
    upd.assert_not_called()


def test_none_when_no_storage_location():
    s = _schema()
    with patch.object(UCSchema, "read_infos", return_value=_info(storage_location=None)), \
            patch.object(UCSchema, "update") as upd:
        assert s.staging_location() is None
    upd.assert_not_called()


def test_none_when_schema_missing():
    s = _schema()
    with patch.object(UCSchema, "read_infos", return_value=None):
        assert s.staging_location() is None


def test_stamp_failure_is_best_effort():
    s = _schema()
    info = _info(storage_location="s3://bkt/meta/__unitystorage/x", properties=None)
    with patch.object(UCSchema, "read_infos", return_value=info), \
            patch.object(UCSchema, "update", side_effect=RuntimeError("no ALTER grant")):
        # The derived value still comes back even when the property write fails.
        assert s.staging_location() == "s3://bkt/meta/uc/tables"
