"""Unity Catalog credentials resource + service + refreshable AWS (mock-driven).

No live workspace: a MagicMock stands in for ``workspace_client().credentials``.
Tests pin the easy-create path, the STS → AwsCredentials mapping, and that the
provider re-mints on each refresh cycle.
"""
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
from yggdrasil.databricks.credentials import Credential, Credentials, DatabricksCredentialAwsProvider
from yggdrasil.url import URL


@pytest.fixture
def service():
    client = MagicMock(spec=DatabricksClient)
    client.host = "https://dbc-x.cloud.databricks.com"
    client.base_url = URL.from_("https://dbc-x.cloud.databricks.com")
    api = client.workspace_client.return_value.credentials
    store: dict[str, CredentialInfo] = {
        "ro_cred": CredentialInfo(
            name="ro_cred", id="c-0", purpose=CredentialPurpose.SERVICE,
            aws_iam_role=AwsIamRole(role_arn="arn:aws:iam::123:role/RO"), read_only=True,
        ),
    }

    def _create(*, name, aws_iam_role, purpose, **k):
        store[name] = CredentialInfo(name=name, id="c-1", aws_iam_role=aws_iam_role,
                                     purpose=purpose, comment=k.get("comment"), read_only=k.get("read_only", False))
        return store[name]

    def _get(name):
        if name not in store:
            raise NotFound(name)
        return store[name]

    counter = {"n": 0}

    def _temp(name):
        counter["n"] += 1
        return TemporaryCredentials(
            aws_temp_credentials=SDKAws(access_key_id=f"AKIA{counter['n']}", secret_access_key="sk",
                                        session_token="tok", access_point="ap"),
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


class TestService:
    def test_create_aws_easy_path(self, service):
        cred = service.create_aws("prod_s3", "arn:aws:iam::123:role/Reader", comment="prod")
        assert cred.aws_role_arn == "arn:aws:iam::123:role/Reader"
        assert cred.purpose == CredentialPurpose.SERVICE  # default → refreshable
        assert cred.comment == "prod"
        # SDK got a typed AwsIamRole.
        kwargs = service.client.workspace_client.return_value.credentials.create_credential.call_args.kwargs
        assert isinstance(kwargs["aws_iam_role"], AwsIamRole)

    def test_get_list_names_exists(self, service):
        service.create_aws("a", "arn:aws:iam::1:role/A")
        assert service.exists("a") and not service.exists("ghost")
        assert "a" in service.names()
        assert isinstance(service["a"], Credential)

    def test_delete(self, service):
        service.create_aws("tmp", "arn:aws:iam::1:role/T")
        service.delete("tmp")
        assert "tmp" not in service._store


class TestRefreshableAws:
    def test_aws_credentials_mapping(self, service):
        cred = service.get("ro_cred")
        ac = cred.aws_credentials()
        assert ac.access_key_id == "AKIA1" and ac.session_token == "tok" and ac.access_point == "ap"
        assert ac.expiration == "2030-01-01T00:00:00+00:00"
        assert set(ac.to_botocore_metadata()) == {"access_key", "secret_key", "token", "expiry_time"}

    def test_provider_is_singleton_and_refreshes(self, service):
        cred = service.get("ro_cred")
        p1 = cred.aws_provider()
        assert isinstance(p1, DatabricksCredentialAwsProvider)
        assert cred.aws_provider() is p1  # singleton per host|name
        # Each refresh cycle re-mints a fresh token.
        assert p1.get_credentials().access_key_id != p1.get_credentials().access_key_id[:-1] + "0"
        first = p1.get_credentials().access_key_id
        second = p1.get_credentials().access_key_id
        assert first != second  # counter advanced → genuinely re-generated

    def test_aws_client_is_region_bound_and_no_network(self, service):
        from yggdrasil.aws.client import AWSClient

        cred = service.get("ro_cred")
        client = cred.aws_client(region="us-east-1")
        assert isinstance(client, AWSClient)
        assert client.region == "us-east-1"

    def test_unbound_provider_raises(self):
        p = DatabricksCredentialAwsProvider("host|orphan")
        p._credential = None
        with pytest.raises(RuntimeError):
            p.get_credentials()


class TestResource:
    def test_explore_url_and_repr(self, service):
        cred = service["ro_cred"]
        assert str(cred.explore_url) == "https://dbc-x.cloud.databricks.com/explore/credentials/ro_cred"
        assert repr(cred) == f"Credential({cred.explore_url!r})"

    def test_metadata(self, service):
        cred = service.get("ro_cred")
        assert cred.id == "c-0" and cred.read_only is True
        assert cred.aws_iam_role.role_arn == "arn:aws:iam::123:role/RO"


def test_client_credentials_property_cached():
    client = MagicMock(spec=DatabricksClient)
    type(client).credentials = DatabricksClient.credentials
    a = client.credentials
    assert isinstance(a, Credentials)
    assert client.credentials is a
