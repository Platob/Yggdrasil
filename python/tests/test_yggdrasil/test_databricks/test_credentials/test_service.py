"""Service: yggdrasil.databricks.credentials.service.Credentials
(incl. its MutableMapping surface + flexible finder)."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
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


# --- flexible finder -------------------------------------------------------
def test_finder_lazy_by_name(service):
    c = service.credential("ro_cred")  # lazy handle
    assert isinstance(c, Credential) and c.name == "ro_cred"
    service.client.workspace_client.return_value.credentials.get_credential.assert_not_called()


def test_finder_resource_passthrough(service):
    c = service.credential("ro_cred")
    assert service.resolve(c) is c


def test_finder_by_explicit_id(service):
    # explicit credential_id → list + match on info.id (the seeded id is "c-0")
    assert service.resolve(credential_id="c-0").name == "ro_cred"


def test_finder_uuid_string_dispatches_to_id(service):
    # a UUID-shaped string is treated as an id (not a name)
    with pytest.raises(KeyError):
        service.resolve("00000000-0000-0000-0000-000000000000")  # no credential has it


def test_finder_bad_input_raises(service):
    with pytest.raises(ValueError):
        service.resolve()
    with pytest.raises(TypeError):
        service.resolve(123)


# --- MutableMapping surface ------------------------------------------------
def test_getitem_fetches_and_missing_raises(service):
    assert service["ro_cred"].aws_role_arn == "arn:aws:iam::123:role/RO"
    with pytest.raises(KeyError):
        service["ghost"]


def test_contains_len_iter(service):
    service.create_aws("a", "arn:aws:iam::1:role/A")
    assert "ro_cred" in service and "ghost" not in service
    assert "a" in list(service)
    assert len(service) == len(service.names())


def test_setitem_creates_from_role_arn_then_updates(service):
    service["nc"] = "arn:aws:iam::1:role/NC"           # str shorthand → create
    assert service._store["nc"].aws_iam_role.role_arn == "arn:aws:iam::1:role/NC"
    service["nc"] = {"comment": "patched"}              # exists → update
    assert service._store["nc"].comment == "patched"


def test_setitem_create_requires_role_arn(service):
    with pytest.raises(ValueError):
        service["bad"] = {"comment": "no role"}


def test_delitem_and_pop(service):
    service.create_aws("d", "arn:aws:iam::1:role/D")
    del service["d"]
    assert "d" not in service._store
    with pytest.raises(KeyError):
        del service["ghost"]
    service.create_aws("p", "arn:aws:iam::1:role/P")
    assert service.pop("p").name == "p" and "p" not in service._store


def test_clear_is_refused(service):
    with pytest.raises(NotImplementedError):
        service.clear()
