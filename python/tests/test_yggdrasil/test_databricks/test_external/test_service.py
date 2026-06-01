"""Service: yggdrasil.databricks.external.service.DatabricksExternal.

The external-data umbrella that groups external locations + storage
credentials under ``client.external``.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from yggdrasil.databricks.client import DatabricksClient
from yggdrasil.databricks.credentials import Credentials
from yggdrasil.databricks.external import DatabricksExternal
from yggdrasil.databricks.external.location import ExternalLocations


def test_sub_services_are_typed_and_cached():
    ext = DatabricksExternal(client=MagicMock(spec=DatabricksClient))
    assert isinstance(ext.locations, ExternalLocations)
    assert isinstance(ext.credentials, Credentials)
    assert ext.locations is ext.locations        # lazy + cached
    assert ext.credentials is ext.credentials


def test_sub_services_share_the_client():
    client = MagicMock(spec=DatabricksClient)
    ext = DatabricksExternal(client=client)
    assert ext.locations.client is client
    assert ext.credentials.client is client


def test_client_external_property_is_cached():
    client = MagicMock(spec=DatabricksClient)
    type(client).external = DatabricksClient.external
    a = client.external
    assert isinstance(a, DatabricksExternal)
    assert client.external is a


def test_flat_aliases_resolve_through_the_umbrella():
    client = MagicMock(spec=DatabricksClient)
    type(client).external = DatabricksClient.external
    type(client).external_locations = DatabricksClient.external_locations
    type(client).credentials = DatabricksClient.credentials
    assert client.external_locations is client.external.locations
    assert client.credentials is client.external.credentials
