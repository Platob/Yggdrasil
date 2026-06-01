"""Provider: yggdrasil.databricks.credentials.provider
.DatabricksCredentialAwsProvider — singleton + genuine re-mint each cycle."""
from __future__ import annotations

import pytest

from yggdrasil.databricks.credentials import DatabricksCredentialAwsProvider


def test_provider_is_singleton_and_refreshes(service):
    cred = service.get("ro_cred")
    p1 = cred.aws_provider()
    assert isinstance(p1, DatabricksCredentialAwsProvider)
    assert cred.aws_provider() is p1  # singleton per host|name
    # Each refresh cycle re-mints a fresh token (counter advances).
    assert p1.get_credentials().access_key_id != p1.get_credentials().access_key_id


def test_unbound_provider_raises():
    p = DatabricksCredentialAwsProvider("host|orphan")
    p._credential = None
    with pytest.raises(RuntimeError):
        p.get_credentials()
