"""Unit tests for :meth:`UCSchema.staging_location`.

Pure derivation of the schema staging root (``<root-before-__unitystorage>/
uc/tables``); the resolved per-table staging root is recorded on the table
(``ygg.staging_root`` TBLPROPERTY), not in schema metadata.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from yggdrasil.databricks.schema.schema import UCSchema


def _schema() -> UCSchema:
    svc = MagicMock()
    svc.client.base_url.host = "example.cloud.databricks.com"
    return UCSchema(service=svc, catalog_name="cat", schema_name="sch")


def _info(*, storage_location=None):
    return SimpleNamespace(storage_location=storage_location)


def test_derives_uc_tables_root_before_unitystorage():
    s = _schema()
    info = _info(storage_location="s3://bkt/meta/__unitystorage/catalogs/x/schemas/y")
    with patch.object(UCSchema, "read_infos", return_value=info):
        assert s.staging_location() == "s3://bkt/meta/uc/tables"


def test_derives_from_plain_storage_location():
    s = _schema()
    with patch.object(UCSchema, "read_infos",
                      return_value=_info(storage_location="s3://bkt/root/")):
        assert s.staging_location() == "s3://bkt/root/uc/tables"


def test_none_when_no_storage_location():
    s = _schema()
    with patch.object(UCSchema, "read_infos", return_value=_info(storage_location=None)):
        assert s.staging_location() is None


def test_none_when_schema_missing():
    s = _schema()
    with patch.object(UCSchema, "read_infos", return_value=None):
        assert s.staging_location() is None
