"""Service: yggdrasil.databricks.credentials.credentials.Credentials."""
from __future__ import annotations

from unittest.mock import MagicMock

from databricks.sdk.service.catalog import AwsIamRole, CredentialPurpose

from yggdrasil.databricks.client import DatabricksClient
from yggdrasil.databricks.credentials import Credential, Credentials


def test_create_aws_easy_path(service):
    cred = service.create_aws("prod_s3", "arn:aws:iam::123:role/Reader", comment="prod")
    assert cred.aws_role_arn == "arn:aws:iam::123:role/Reader"
    assert cred.purpose == CredentialPurpose.SERVICE  # default → refreshable
    assert cred.comment == "prod"
    kwargs = service.client.workspace_client.return_value.credentials.create_credential.call_args.kwargs
    assert isinstance(kwargs["aws_iam_role"], AwsIamRole)


def test_get_list_names_exists(service):
    service.create_aws("a", "arn:aws:iam::1:role/A")
    assert service.exists("a") and not service.exists("ghost")
    assert "a" in service.names()
    assert isinstance(service["a"], Credential)


def test_delete(service):
    service.create_aws("tmp", "arn:aws:iam::1:role/T")
    service.delete("tmp")
    assert "tmp" not in service._store


def test_client_credentials_property_cached():
    client = MagicMock(spec=DatabricksClient)
    type(client).credentials = DatabricksClient.credentials
    a = client.credentials
    assert isinstance(a, Credentials)
    assert client.credentials is a
