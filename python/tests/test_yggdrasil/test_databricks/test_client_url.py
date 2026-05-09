"""Behaviors for :class:`DatabricksClient` as a :class:`URLBased`.

The client round-trips through a single ``dbks://...`` URL:

* :meth:`DatabricksClient.to_url` packs the workspace host, the
  active credential (PAT or OAuth), and every other init field
  into URL host / userinfo / query.
* :meth:`DatabricksClient.from_url` unpacks the same URL back into
  the right ``__init__`` kwargs, with caller-supplied ``**kwargs``
  taking precedence.
* :meth:`URLBased.dispatch` routes ``dbks://...`` URLs straight to
  :class:`DatabricksClient` via the lazy registry.
"""
from __future__ import annotations

from yggdrasil.data.enums import Scheme
from yggdrasil.databricks import DatabricksClient
from yggdrasil.io.url import URL, URLBased


class TestSchemeRegistration:

    def test_class_scheme(self) -> None:
        assert DatabricksClient.scheme is Scheme.DATABRICKS

    def test_for_scheme_resolves_to_client(self) -> None:
        assert URLBased.for_scheme(Scheme.DATABRICKS) is DatabricksClient

    def test_for_scheme_lazy_alias(self) -> None:
        # ``"databricks"`` is an alias of ``"dbks"`` — both resolve.
        assert URLBased.for_scheme("dbks") is DatabricksClient
        assert URLBased.for_scheme("databricks") is DatabricksClient


class TestToUrl:

    def test_pat_credential_in_userinfo(self) -> None:
        client = DatabricksClient(
            host="https://ws.example.com",
            token="dapi-secret",
        )
        url = client.to_url()
        # Token rides as the password; user is empty.
        assert url.password == "dapi-secret"
        assert not url.user
        assert url.host == "ws.example.com"
        assert url.scheme == "dbks"

    def test_oauth_credential_in_userinfo(self) -> None:
        client = DatabricksClient(
            host="https://ws.example.com",
            client_id="abc",
            client_secret="xyz",
        )
        url = client.to_url()
        assert url.user == "abc"
        assert url.password == "xyz"

    def test_query_carries_non_secret_fields(self) -> None:
        client = DatabricksClient(
            host="https://ws.example.com",
            token="t",
            profile="dev",
            auth_type="pat",
            account_id="acct-1",
        )
        url = client.to_url()
        items = dict(url.query_items())
        assert items["profile"] == "dev"
        assert items["auth_type"] == "pat"
        assert items["account_id"] == "acct-1"

    def test_query_omits_secrets(self) -> None:
        client = DatabricksClient(
            host="https://ws.example.com",
            token="dapi-secret",
            client_id="abc",
            client_secret="xyz",
        )
        items = dict(client.to_url().query_items())
        assert "host" not in items
        assert "token" not in items
        assert "client_id" not in items
        assert "client_secret" not in items

    def test_scheme_override(self) -> None:
        url = DatabricksClient(host="https://ws.example.com").to_url(scheme="https")
        assert url.scheme == "https"


class TestFromUrl:

    def test_pat_round_trip(self) -> None:
        original = DatabricksClient(
            host="https://ws.example.com",
            token="dapi-1",
            profile="dev",
        )
        rebuilt = DatabricksClient.from_url(original.to_url())
        assert rebuilt.host == "https://ws.example.com"
        assert rebuilt.token == "dapi-1"
        assert rebuilt.profile == "dev"

    def test_oauth_round_trip(self) -> None:
        original = DatabricksClient(
            host="https://ws.example.com",
            client_id="abc",
            client_secret="xyz",
        )
        rebuilt = DatabricksClient.from_url(original.to_url())
        assert rebuilt.client_id == "abc"
        assert rebuilt.client_secret == "xyz"
        # PAT token is empty when the URL carries a client_id/secret pair.
        assert rebuilt.token in (None, "")

    def test_kwargs_override_url(self) -> None:
        url = DatabricksClient(
            host="https://ws.example.com", profile="dev",
        ).to_url()
        rebuilt = DatabricksClient.from_url(url, profile="prod")
        assert rebuilt.profile == "prod"

    def test_host_required(self) -> None:
        # No URL host and no host= query → reject.
        try:
            DatabricksClient.from_url("dbks:///?profile=dev")
        except ValueError as exc:
            assert "Host is required" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("expected ValueError")

    def test_host_via_query_param(self) -> None:
        rebuilt = DatabricksClient.from_url(
            "dbks:///?host=https://ws.example.com&profile=dev",
        )
        assert rebuilt.host == "https://ws.example.com"
        assert rebuilt.profile == "dev"

    def test_password_only_userinfo_is_token(self) -> None:
        rebuilt = DatabricksClient.from_url("dbks://:dapi-2@ws.example.com")
        assert rebuilt.token == "dapi-2"
        assert rebuilt.client_id is None

    def test_dispatch_via_urlbased(self) -> None:
        client = URLBased.dispatch(
            "dbks://abc:xyz@ws.example.com?profile=prod",
        )
        assert isinstance(client, DatabricksClient)
        assert client.client_id == "abc"
        assert client.client_secret == "xyz"
        assert client.profile == "prod"

    def test_legacy_from_parsed_url_alias(self) -> None:
        """Older code paths still call :meth:`from_parsed_url` — keep
        the alias wired so the audit pass can be deferred."""
        url = URL.from_("dbks://:t@ws.example.com")
        rebuilt = DatabricksClient.from_parsed_url(url)
        assert rebuilt.token == "t"
