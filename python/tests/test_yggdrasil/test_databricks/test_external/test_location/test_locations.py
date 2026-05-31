"""Service: yggdrasil.databricks.external.location.locations.ExternalLocations."""
from __future__ import annotations

from unittest.mock import MagicMock

from yggdrasil.databricks.client import DatabricksClient
from yggdrasil.databricks.external.location import ExternalLocation, ExternalLocations


def test_getitem_is_lazy(service):
    el = service["raw_zone"]
    assert isinstance(el, ExternalLocation) and el.name == "raw_zone"
    service.client.workspace_client.return_value.external_locations.get.assert_not_called()


def test_get_fetches(service):
    el = service.get("raw_zone")
    assert el.url == "s3://my-bucket/raw/" and el.credential_name == "prod-cred"


def test_list_and_names(service):
    assert sorted(e.name for e in service.list()) == ["raw_zone", "ro_zone"]
    assert sorted(service.names()) == ["raw_zone", "ro_zone"]


def test_exists(service):
    assert service.exists("raw_zone") is True
    assert service.exists("ghost") is False


def test_create(service, store):
    el = service.create("new_zone", "s3://b/new/", "c", comment="hi")
    assert el.name == "new_zone" and el.url == "s3://b/new/" and el.comment == "hi"
    assert "new_zone" in store


def test_update(service):
    assert service.update("raw_zone", comment="updated").comment == "updated"


def test_delete(service, store):
    service.delete("raw_zone")
    assert "raw_zone" not in store


def test_client_external_locations_property_is_cached():
    client = MagicMock(spec=DatabricksClient)
    type(client).external_locations = DatabricksClient.external_locations
    a = client.external_locations
    assert isinstance(a, ExternalLocations)
    assert client.external_locations is a
