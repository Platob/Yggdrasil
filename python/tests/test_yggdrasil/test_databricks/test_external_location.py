"""Unity Catalog external-location resource + service (mock-driven).

No live workspace: a MagicMock stands in for ``workspace_client().external_locations``
and the tests pin that the service maps SDK calls to ExternalLocation handles
and the resource exposes the metadata / storage path / Console link.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from databricks.sdk.errors import NotFound
from databricks.sdk.service.catalog import ExternalLocationInfo

from yggdrasil.databricks.client import DatabricksClient
from yggdrasil.databricks.external.location import ExternalLocation, ExternalLocations
from yggdrasil.url import URL


@pytest.fixture
def store():
    return {
        "raw_zone": ExternalLocationInfo(
            name="raw_zone", url="s3://my-bucket/raw/", credential_name="prod-cred",
            read_only=False, comment="raw landing", owner="data-eng",
        ),
        "ro_zone": ExternalLocationInfo(
            name="ro_zone", url="s3://other/ro/", credential_name="ro-cred", read_only=True,
        ),
    }


@pytest.fixture
def service(store):
    client = MagicMock(spec=DatabricksClient)
    client.base_url = URL.from_("https://dbc-x.cloud.databricks.com")
    api = client.workspace_client.return_value.external_locations

    def _get(name, **k):
        if name not in store:
            raise NotFound(f"external location {name} not found")
        return store[name]

    def _create(*, name, url, credential_name, **k):
        store[name] = ExternalLocationInfo(name=name, url=url, credential_name=credential_name, **k)
        return store[name]

    def _update(name, **changes):
        cur = store[name].as_dict()
        cur.update(changes)
        store[name] = ExternalLocationInfo.from_dict(cur)
        return store[name]

    api.get.side_effect = _get
    api.list.side_effect = lambda **k: list(store.values())
    api.create.side_effect = _create
    api.update.side_effect = _update
    api.delete.side_effect = lambda name, **k: store.pop(name, None)
    return ExternalLocations(client=client)


class TestService:
    def test_getitem_is_lazy(self, service, store):
        el = service["raw_zone"]
        assert isinstance(el, ExternalLocation) and el.name == "raw_zone"
        service.client.workspace_client.return_value.external_locations.get.assert_not_called()

    def test_get_fetches(self, service):
        el = service.get("raw_zone")
        assert el.url == "s3://my-bucket/raw/" and el.credential_name == "prod-cred"

    def test_list_and_names(self, service):
        assert sorted(e.name for e in service.list()) == ["raw_zone", "ro_zone"]
        assert sorted(service.names()) == ["raw_zone", "ro_zone"]

    def test_exists(self, service):
        assert service.exists("raw_zone") is True
        assert service.exists("ghost") is False

    def test_create(self, service, store):
        el = service.create("new_zone", "s3://b/new/", "c", comment="hi")
        assert el.name == "new_zone" and el.url == "s3://b/new/" and el.comment == "hi"
        assert "new_zone" in store

    def test_update(self, service):
        el = service.update("raw_zone", comment="updated")
        assert el.comment == "updated"

    def test_delete(self, service, store):
        service.delete("raw_zone")
        assert "raw_zone" not in store


class TestResource:
    def test_metadata(self, service):
        el = service.get("raw_zone")
        assert el.url == "s3://my-bucket/raw/"
        assert el.read_only is False and el.owner == "data-eng"
        assert service.get("ro_zone").read_only is True

    def test_lazy_info_single_fetch(self, service):
        el = service["raw_zone"]  # lazy, no fetch
        api = service.client.workspace_client.return_value.external_locations
        api.get.assert_not_called()
        _ = el.url  # triggers exactly one GET
        _ = el.comment  # cached
        api.get.assert_called_once()

    def test_path_is_an_s3path(self, service):
        from yggdrasil.aws.fs.path import S3Path

        p = service.get("raw_zone").path
        assert isinstance(p, S3Path)
        assert p.bucket == "my-bucket" and p.key == "raw/"

    def test_explore_url_and_clickable_repr(self, service):
        el = service["raw_zone"]
        assert str(el.explore_url) == "https://dbc-x.cloud.databricks.com/explore/location/raw_zone"
        assert repr(el) == f"ExternalLocation({el.explore_url!r})"
        assert el._repr_html_().startswith('<a href="https://dbc-x.cloud.databricks.com/explore/location/raw_zone"')

    def test_instance_update_refreshes_cache(self, service):
        el = service.get("raw_zone")
        el.update(comment="patched")
        assert el.comment == "patched"

    def test_refresh_refetches(self, service, store):
        el = service.get("raw_zone")
        store["raw_zone"] = ExternalLocationInfo(name="raw_zone", url="s3://my-bucket/raw2/", credential_name="prod-cred")
        assert el.refresh().url == "s3://my-bucket/raw2/"


def test_client_external_locations_property_is_cached():
    client = MagicMock(spec=DatabricksClient)
    # bind the real property to the mock so .external_locations runs the impl
    type(client).external_locations = DatabricksClient.external_locations
    a = client.external_locations
    assert isinstance(a, ExternalLocations)
    assert client.external_locations is a  # cached on the instance
