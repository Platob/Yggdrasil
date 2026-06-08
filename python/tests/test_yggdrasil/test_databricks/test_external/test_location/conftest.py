"""Shared fixtures: an ExternalLocations service over a mocked workspace
client, plus its in-memory store."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

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
    from tests.test_yggdrasil.test_aws._fake_s3 import FakeS3, wire_s3_path, reset_s3_singletons

    reset_s3_singletons()
    fake = FakeS3()
    client = MagicMock(spec=DatabricksClient)
    client.base_url = URL.from_("https://dbc-x.cloud.databricks.com")
    api = client.workspace_client.return_value.external_locations

    # The inner storage path is vended by the location's *path* credential
    # (UC ``generate_temporary_path_credentials`` scoped to the URL — the
    # only endpoint that works for an external location's STORAGE
    # credential; the service-credential endpoint rejects it). The resource
    # builds it via ``AWSDatabricksPathCredentials(url, client=...)
    # .aws_client(mode, region).s3.path(url)``. Patch the provider's
    # ``aws_client`` to hand back a fake-backed S3 client so delegation hits
    # an in-memory S3 without a live workspace round-trip.
    def _s3_path(url):
        bucket = url.split("://", 1)[1].split("/", 1)[0]
        return wire_s3_path(fake, url, bucket=bucket)

    aws = MagicMock()
    aws.s3.path.side_effect = _s3_path

    path_cred_calls: list = []

    def _aws_client(self, *, mode=None, region=None):
        path_cred_calls.append((self.url, mode, region))
        return aws

    patcher = patch(
        "yggdrasil.databricks.aws.AWSDatabricksPathCredentials.aws_client",
        new=_aws_client,
    )
    patcher.start()

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
    svc = ExternalLocations(client=client)
    svc._fake = fake  # the in-memory S3 backing the inner storage paths
    svc._path_cred_calls = path_cred_calls  # (url, mode, region) per vend
    try:
        yield svc
    finally:
        patcher.stop()
