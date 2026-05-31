"""Shared fixtures: an ExternalLocations service over a mocked workspace
client, plus its in-memory store."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from databricks.sdk.errors import NotFound
from databricks.sdk.service.catalog import ExternalLocationInfo

from yggdrasil.databricks.client import DatabricksClient
from yggdrasil.databricks.external.location import ExternalLocations
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
