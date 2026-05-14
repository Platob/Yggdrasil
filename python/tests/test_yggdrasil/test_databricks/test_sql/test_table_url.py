"""Behaviors for :class:`Table` as a :class:`Holder` / :class:`URLBased`.

Pins:

* registration under :attr:`Scheme.DATABRICKS_TABLE` (``dbfs+table``)
  so ``URLBased.dispatch`` can route ``dbfs+table://...`` URLs to a
  :class:`Table`;
* :meth:`Table.from_url` reads the catalog/schema/table from the URL
  path and, when no ``service`` / ``client`` is passed, infers the
  underlying :class:`DatabricksClient` from the URL via
  :meth:`DatabricksClient.from_url` (userinfo + query params);
* :meth:`Table.to_url` round-trips back to a URL of the same shape;
* the byte-level :class:`Holder` primitives intentionally raise — a
  SQL table is not a positional byte buffer.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from yggdrasil.data.enums import Scheme
from yggdrasil.databricks import DatabricksClient
from yggdrasil.databricks.table.table import Table
from yggdrasil.io.url import URL, URLBased


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _service(host: str = "https://ws.example.com") -> MagicMock:
    """Build a mock :class:`Tables` service so :class:`Table` can be
    constructed without touching a live workspace."""
    service = MagicMock()
    service.client = DatabricksClient(host=host)
    return service


# ---------------------------------------------------------------------------
# Scheme registration
# ---------------------------------------------------------------------------


class TestSchemeRegistration:

    def test_class_scheme(self) -> None:
        assert Table.scheme is Scheme.DATABRICKS_TABLE

    def test_for_scheme_resolves_to_table(self) -> None:
        assert URLBased.for_scheme(Scheme.DATABRICKS_TABLE) is Table

    def test_alias_resolution(self) -> None:
        assert URLBased.for_scheme("dbfs+table") is Table


# ---------------------------------------------------------------------------
# from_url
# ---------------------------------------------------------------------------


class TestFromUrl:

    def test_path_drives_catalog_schema_table(self) -> None:
        service = _service()
        t = Table.from_url(
            "dbfs+table://ws.example.com/catalog/schema/orders",
            service=service,
        )
        assert t.catalog_name == "catalog"
        assert t.schema_name == "schema"
        assert t.table_name == "orders"
        assert t.service is service

    def test_explicit_kwargs_override_url_path(self) -> None:
        service = _service()
        t = Table.from_url(
            "dbfs+table://ws.example.com/cat/sch/tbl",
            service=service,
            table_name="other",
        )
        assert t.table_name == "other"
        assert t.catalog_name == "cat"

    def test_inferred_client_from_url_when_no_service(self) -> None:
        """No ``service`` / ``client`` passed → :meth:`from_url` parses
        a :class:`DatabricksClient` from the URL itself (host +
        userinfo + query items)."""
        t = Table.from_url(
            "dbfs+table://:dapi-secret@ws.example.com/cat/sch/tbl?profile=dev",
        )
        assert t.client.host == "https://ws.example.com"
        assert t.client.token == "dapi-secret"
        assert t.client.profile == "dev"
        assert t.catalog_name == "cat"

    def test_dispatch_via_urlbased(self) -> None:
        t = URLBased.dispatch(
            "dbfs+table://abc:xyz@ws.example.com/cat/sch/orders?account_id=acct-1",
        )
        assert isinstance(t, Table)
        assert t.client.client_id == "abc"
        assert t.client.client_secret == "xyz"
        assert t.client.account_id == "acct-1"
        assert (t.catalog_name, t.schema_name, t.table_name) == (
            "cat", "sch", "orders",
        )


# ---------------------------------------------------------------------------
# to_url
# ---------------------------------------------------------------------------


class TestToUrl:

    def test_round_trip_with_inferred_client(self) -> None:
        url = "dbfs+table://:dapi-1@ws.example.com/cat/sch/orders?profile=dev"
        rebuilt = Table.from_url(url).to_url()
        # Token rides in the userinfo, the path keeps the
        # cat/sch/tbl layout, and ``profile=dev`` survives in the
        # query string.
        assert rebuilt.path == "/cat/sch/orders"
        assert rebuilt.password == "dapi-1"
        items = dict(rebuilt.query_items())
        assert items.get("profile") == "dev"
        assert rebuilt.scheme == "dbfs+table"

    def test_explicit_construction_renders_url(self) -> None:
        service = _service("https://ws.example.com")
        t = Table(
            service=service,
            catalog_name="cat",
            schema_name="sch",
            table_name="tbl",
        )
        url = t.to_url()
        assert url.path == "/cat/sch/tbl"
        assert url.host == "ws.example.com"
        assert url.scheme == "dbfs+table"


# ---------------------------------------------------------------------------
# Holder surface
# ---------------------------------------------------------------------------


class TestHolderSurface:

    def test_predicates(self) -> None:
        t = Table(service=_service(), catalog_name="c", schema_name="s", table_name="t")
        assert not t.is_memory
        assert not t.is_local_path
        assert not t.is_remote_path

    def test_byte_primitives_raise(self) -> None:
        t = Table(service=_service(), catalog_name="c", schema_name="s", table_name="t")
        with pytest.raises(NotImplementedError):
            t._read_mv(8, 0)
        with pytest.raises(NotImplementedError):
            t._write_mv(memoryview(b""), 0)
        with pytest.raises(NotImplementedError):
            t.truncate(0)
        with pytest.raises(NotImplementedError):
            t._clear()

    def test_size_and_url_via_holder(self) -> None:
        t = Table(service=_service(), catalog_name="c", schema_name="s", table_name="t")
        # ``size`` is 0 (no positional byte concept) but ``url`` comes
        # from the Holder base, not the legacy explore-deep-link.
        assert t.size == 0
        assert t.url.scheme == "dbfs+table"
        assert t.url.path == "/c/s/t"

    def test_explore_url_still_available(self) -> None:
        t = Table(service=_service(), catalog_name="c", schema_name="s", table_name="t")
        deep_link = t.explore_url
        assert "/explore/data/c/s/t" in deep_link.path
