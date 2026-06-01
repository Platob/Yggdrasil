"""Unit tests for :meth:`UCSchema.ensure_created` parent cascade (no cluster).

``ensure_created`` is the single seam the volume / volume-path auto-create
recovery leans on to materialise missing parents, so it must ensure the parent
**catalog** before creating the schema — one call brings up the whole
catalog → schema chain.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from yggdrasil.databricks.catalog.catalog import UCCatalog
from yggdrasil.databricks.schema.schema import UCSchema


def _schema() -> UCSchema:
    svc = MagicMock()
    svc.client.base_url.host = "example.cloud.databricks.com"
    return UCSchema(service=svc, catalog_name="cat", schema_name="sch")


def test_ensure_created_cascades_to_catalog_when_missing():
    s = _schema()
    with patch.object(UCSchema, "exists", return_value=False), \
         patch.object(UCSchema, "create") as create, \
         patch.object(UCCatalog, "ensure_created") as catalog_ensure:
        assert s.ensure_created() is s
    # catalog ensured first, then the schema created
    catalog_ensure.assert_called_once()
    create.assert_called_once()


def test_ensure_created_is_noop_when_schema_exists():
    s = _schema()
    with patch.object(UCSchema, "exists", return_value=True), \
         patch.object(UCSchema, "create") as create, \
         patch.object(UCCatalog, "ensure_created") as catalog_ensure:
        assert s.ensure_created() is s
    create.assert_not_called()
    catalog_ensure.assert_not_called()
