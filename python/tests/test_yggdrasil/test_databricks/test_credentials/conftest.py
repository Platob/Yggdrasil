"""Shared fixture: a Credentials service over a mocked workspace client."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from databricks.sdk.errors import NotFound
from databricks.sdk.service.catalog import (
    AwsCredentials as SDKAws,
    AwsIamRole,
    CredentialInfo,
    CredentialPurpose,
    TemporaryCredentials,
)

from yggdrasil.databricks.client import DatabricksClient
from yggdrasil.databricks.credentials import Credentials
from yggdrasil.url import URL


@pytest.fixture
def service():
    """A :class:`Credentials` bound to an in-memory mock of
    ``workspace_client().credentials`` (one seeded read-only credential)."""
    client = MagicMock(spec=DatabricksClient)
    client.host = "https://dbc-x.cloud.databricks.com"
    client.base_url = URL.from_("https://dbc-x.cloud.databricks.com")
    api = client.workspace_client.return_value.credentials
    store: "dict[str, CredentialInfo]" = {
        "ro_cred": CredentialInfo(
            name="ro_cred", id="c-0", purpose=CredentialPurpose.SERVICE,
            aws_iam_role=AwsIamRole(role_arn="arn:aws:iam::123:role/RO"), read_only=True,
        ),
    }

    def _create(*, name, aws_iam_role, purpose, **k):
        store[name] = CredentialInfo(
            name=name, id="c-1", aws_iam_role=aws_iam_role, purpose=purpose,
            comment=k.get("comment"), read_only=k.get("read_only", False),
        )
        return store[name]

    def _get(name):
        if name not in store:
            raise NotFound(name)
        return store[name]

    counter = {"n": 0}

    def _temp(name):
        counter["n"] += 1
        return TemporaryCredentials(
            aws_temp_credentials=SDKAws(
                access_key_id=f"AKIA{counter['n']}", secret_access_key="sk",
                session_token="tok", access_point="ap",
            ),
            expiration_time=1893456000000,  # 2030-01-01Z
        )

    api.create_credential.side_effect = _create
    api.get_credential.side_effect = _get
    api.list_credentials.side_effect = lambda **k: list(store.values())
    api.delete_credential.side_effect = lambda name, **k: store.pop(name, None)
    api.generate_temporary_service_credential.side_effect = _temp

    svc = Credentials(client=client)
    svc._store, svc._counter = store, counter
    return svc
