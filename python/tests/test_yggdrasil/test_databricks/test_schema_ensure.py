"""Unit tests for idempotent, parent-ensuring ``create`` on UC schema/catalog.

``create`` reads first (idempotent — a successful read means it exists, and
reads never auto-create) and, on a not-found create error, materialises the
missing parent and retries once. For a schema the parent is its catalog; a
catalog is top-level, so it has no parent to create.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from yggdrasil.databricks.catalog.catalog import UCCatalog
from yggdrasil.databricks.schema.schema import UCSchema


class _Boom(Exception):
    pass


def _schema() -> UCSchema:
    svc = MagicMock()
    svc.client.base_url.host = "example.cloud.databricks.com"
    return UCSchema(service=svc, catalog_name="cat", schema_name="sch")


def test_schema_create_is_idempotent_when_it_exists():
    s = _schema()
    uc = s.client.workspace_client().schemas
    with patch.object(UCSchema, "read_infos", return_value=object()):
        assert s.create() is s
    uc.create.assert_not_called()


def test_schema_create_ensures_catalog_on_not_found_and_retries():
    s = _schema()
    uc = s.client.workspace_client().schemas
    uc.create.side_effect = [_Boom("Catalog does not exist"), object()]
    with patch.object(UCSchema, "read_infos", return_value=None), \
         patch.object(UCCatalog, "ensure_created") as catalog_ensure:
        assert s.create() is s
    catalog_ensure.assert_called_once()
    assert uc.create.call_count == 2


def test_schema_create_does_not_ensure_parent_when_create_succeeds():
    s = _schema()
    uc = s.client.workspace_client().schemas
    uc.create.return_value = object()
    with patch.object(UCSchema, "read_infos", return_value=None), \
         patch.object(UCCatalog, "ensure_created") as catalog_ensure:
        s.create()
    catalog_ensure.assert_not_called()
    uc.create.assert_called_once()


def test_catalog_create_is_idempotent_and_has_no_parent():
    svc = MagicMock()
    svc.client.base_url.host = "example.cloud.databricks.com"
    c = UCCatalog(service=svc, catalog_name="cat")
    uc = c.client.workspace_client().catalogs
    with patch.object(UCCatalog, "read_infos", return_value=object()):
        assert c.create() is c
    uc.create.assert_not_called()
