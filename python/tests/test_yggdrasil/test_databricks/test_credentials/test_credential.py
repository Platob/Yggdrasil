"""Resource: yggdrasil.databricks.credentials.credential.Credential —
metadata + refreshable AWS credentials."""
from __future__ import annotations

from yggdrasil.aws.client import AWSClient
from yggdrasil.aws.config import AwsCredentials


def test_metadata(service):
    cred = service.get("ro_cred")
    assert cred.id == "c-0" and cred.read_only is True
    assert cred.aws_iam_role.role_arn == "arn:aws:iam::123:role/RO"


def test_explore_url_and_repr(service):
    cred = service["ro_cred"]
    assert str(cred.explore_url) == "https://dbc-x.cloud.databricks.com/explore/credentials/ro_cred"
    assert repr(cred) == f"Credential({cred.explore_url!r})"


def test_aws_credentials_mapping(service):
    ac = service.get("ro_cred").aws_credentials()
    assert ac.access_key_id == "AKIA1" and ac.session_token == "tok" and ac.access_point == "ap"
    assert ac.expiration == "2030-01-01T00:00:00+00:00"
    assert set(ac.to_botocore_metadata()) == {"access_key", "secret_key", "token", "expiry_time"}
    assert isinstance(ac, AwsCredentials)


def test_aws_client_is_region_bound(service):
    client = service.get("ro_cred").aws_client(region="us-east-1")
    assert isinstance(client, AWSClient)
    assert client.region == "us-east-1"
