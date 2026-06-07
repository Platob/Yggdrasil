"""Service: yggdrasil.databricks.external.location.service.ExternalLocations
(incl. its MutableMapping surface)."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from yggdrasil.databricks.client import DatabricksClient
from yggdrasil.databricks.external.location import ExternalLocation, ExternalLocations


def test_location_finder_is_lazy(service):
    el = service.location("raw_zone")  # lazy handle — no fetch
    assert isinstance(el, ExternalLocation) and el.name == "raw_zone"
    service.client.workspace_client.return_value.external_locations.get.assert_not_called()
    assert service.resolve(el) is el  # a handle passes through


def test_get_fetches(service):
    el = service.get("raw_zone")
    assert el.url == "s3://my-bucket/raw/" and el.credential_name == "prod-cred"


def test_list_and_names(service):
    assert sorted(e.name for e in service.list()) == ["raw_zone", "ro_zone"]
    assert sorted(service.names()) == ["raw_zone", "ro_zone"]


def test_exists(service):
    assert service.exists("raw_zone") is True
    assert service.exists("ghost") is False


# --- MutableMapping surface ------------------------------------------------
def test_getitem_fetches_and_missing_raises_keyerror(service):
    assert service["raw_zone"].url == "s3://my-bucket/raw/"
    with pytest.raises(KeyError):
        service["ghost"]


def test_contains_len_iter(service):
    assert "raw_zone" in service and "ghost" not in service
    assert len(service) == 2
    assert sorted(service) == ["raw_zone", "ro_zone"]


def test_setitem_creates_then_updates(service, store):
    service["new_zone"] = {"url": "s3://b/new/", "credential_name": "c", "comment": "hi"}
    assert store["new_zone"].url == "s3://b/new/" and store["new_zone"].comment == "hi"
    service["new_zone"] = {"comment": "patched"}  # exists → update
    assert store["new_zone"].comment == "patched"


def test_setitem_create_requires_url_and_credential(service):
    with pytest.raises(ValueError):
        service["bad"] = {"comment": "no url"}


def test_delitem_and_pop(service, store):
    del service["raw_zone"]
    assert "raw_zone" not in store
    with pytest.raises(KeyError):
        del service["ghost"]
    popped = service.pop("ro_zone")
    assert popped.name == "ro_zone" and "ro_zone" not in store


def test_get_and_keys_values_items(service):
    assert service.get("ghost") is None
    assert sorted(service.keys()) == ["raw_zone", "ro_zone"]
    assert all(isinstance(v, ExternalLocation) for v in service.values())


def test_clear_is_refused(service):
    with pytest.raises(NotImplementedError):
        service.clear()


def test_create(service, store):
    el = service.create("new_zone", "s3://b/new/", "c", comment="hi")
    assert el.name == "new_zone" and el.url == "s3://b/new/" and el.comment == "hi"
    assert "new_zone" in store


def test_update(service):
    assert service.update("raw_zone", comment="updated").comment == "updated"


def test_delete(service, store):
    service.delete("raw_zone")
    assert "raw_zone" not in store


# --- find_url + list caching ----------------------------------------------
def test_find_url_longest_prefix_match(service):
    el = service.find_url("s3://my-bucket/raw/sub/f.parquet")
    assert el is not None and el.name == "raw_zone"   # covered by raw_zone's root
    assert service.find_url("s3://my-bucket/raw") is not None      # the root itself
    assert service.find_url("s3://nowhere/x") is None             # uncovered


def test_list_is_cached_within_ttl(service):
    api = service.client.workspace_client.return_value.external_locations
    service.find_url("s3://my-bucket/raw/x")
    service.find_url("s3://other/ro/y")
    list(service.list())
    # All served from one underlying list() — external locations are near-static.
    assert api.list.call_count == 1


def test_refresh_forces_a_relist(service):
    api = service.client.workspace_client.return_value.external_locations
    service.find_url("s3://my-bucket/raw/x")
    service.find_url("s3://my-bucket/raw/x", refresh=True)
    assert api.list.call_count == 2


def test_writes_bust_the_cache(service):
    api = service.client.workspace_client.return_value.external_locations
    service.find_url("s3://my-bucket/raw/x")          # list #1, populates cache
    service.create("new_zone", "s3://b/new/", "c")    # write → invalidate
    found = service.find_url("s3://b/new/sub/f")       # list #2, sees the new one
    assert found is not None and found.name == "new_zone"
    assert api.list.call_count == 2


def test_client_external_locations_property_is_cached():
    # ``client.external_locations`` is a flat alias onto the
    # ``client.external`` umbrella (centralized external-data services).
    client = MagicMock(spec=DatabricksClient)
    type(client).external = DatabricksClient.external
    type(client).external_locations = DatabricksClient.external_locations
    a = client.external_locations
    assert isinstance(a, ExternalLocations)
    assert client.external_locations is a          # cached on the umbrella
    assert client.external.locations is a          # same instance both ways
