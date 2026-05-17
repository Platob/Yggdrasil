"""Unit tests for :class:`yggdrasil.io.authorization.msal.MSALAuth`.

Covers config resolution (constructor / env fallback), singleton-by-config
caching through the :class:`ExpiringDict` cache, idempotent ``__init__``,
hash + equality, scope normalization, token lifecycle (mocked MSAL app),
``Authorization`` header formatting, and pickle round-trips both
in-process and across a simulated process boundary.

MSAL itself is mocked end-to-end: no real AAD round-trip is ever issued.
The tests focus on the wrapper's contract, not MSAL's.
"""
from __future__ import annotations

import os
import pickle
import threading
import time
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from yggdrasil.io.authorization import Authorization, MSALAuth
from yggdrasil.io.authorization import msal as msal_module
from yggdrasil.dataclasses.expiring import ExpiringDict


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_msal_singleton_cache():
    """Tear singletons between tests — they're class-level state."""
    MSALAuth._INSTANCES.clear()
    yield
    MSALAuth._INSTANCES.clear()


@pytest.fixture(autouse=True)
def _clear_azure_env(monkeypatch):
    """Wipe AZURE_* env vars so explicit-arg tests aren't shadowed by
    whatever the host machine exports. Individual tests opt back in via
    ``monkeypatch.setenv``.
    """
    for var in (
        "AZURE_TENANT_ID",
        "AZURE_CLIENT_ID",
        "AZURE_CLIENT_SECRET",
        "AZURE_AUTHORITY",
        "AZURE_SCOPES",
    ):
        monkeypatch.delenv(var, raising=False)


def _make_token_result(token: str = "tok-1", expires_in: int = 3600) -> dict:
    return {"access_token": token, "expires_in": expires_in, "token_type": "Bearer"}


def _build_msal_auth(
    *,
    tenant_id: Optional[str] = "tenant-1",
    client_id: Optional[str] = "client-1",
    client_secret: Optional[str] = "secret-1",
    scopes: object = "api://x/.default",
    expiry_skew_seconds: int = 30,
) -> MSALAuth:
    return MSALAuth(
        tenant_id=tenant_id,
        client_id=client_id,
        client_secret=client_secret,
        scopes=scopes,
        expiry_skew_seconds=expiry_skew_seconds,
    )


# ---------------------------------------------------------------------------
# _parse_scopes
# ---------------------------------------------------------------------------


class TestParseScopes:

    def test_none_returns_none(self) -> None:
        assert msal_module._parse_scopes(None) is None

    def test_list_of_strings_stripped_and_filtered(self) -> None:
        assert msal_module._parse_scopes([" a ", "b", "", "  "]) == ["a", "b"]

    def test_empty_list_becomes_none(self) -> None:
        assert msal_module._parse_scopes([]) is None
        assert msal_module._parse_scopes(["", "   "]) is None

    def test_space_separated_string(self) -> None:
        assert msal_module._parse_scopes("a b c") == ["a", "b", "c"]

    def test_comma_separated_string_takes_precedence(self) -> None:
        # Mixed: the presence of a comma flips the splitter to ',' so
        # space-in-scope tokens survive (rare but legal).
        assert msal_module._parse_scopes("a, b ,c") == ["a", "b", "c"]

    def test_empty_string_becomes_none(self) -> None:
        assert msal_module._parse_scopes("") is None
        assert msal_module._parse_scopes("   ") is None

    def test_invalid_type_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="scopes must be"):
            msal_module._parse_scopes(42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _resolve_config / constructor resolution
# ---------------------------------------------------------------------------


class TestResolveConfig:

    def test_explicit_args_win(self) -> None:
        key = msal_module._resolve_config(
            tenant_id="t", client_id="c", client_secret="s",
            authority="https://example.com/t", scopes=["a", "b"],
        )
        assert key == ("t", "c", "s", "https://example.com/t", ("a", "b"))

    def test_ellipsis_falls_back_to_env(self, monkeypatch) -> None:
        monkeypatch.setenv("AZURE_TENANT_ID", "env-tenant")
        monkeypatch.setenv("AZURE_CLIENT_ID", "env-client")
        monkeypatch.setenv("AZURE_CLIENT_SECRET", "env-secret")
        monkeypatch.setenv("AZURE_SCOPES", "scope1 scope2")
        key = msal_module._resolve_config(..., ..., ..., ..., ...)
        assert key[:3] == ("env-tenant", "env-client", "env-secret")
        assert key[3] == "https://login.microsoftonline.com/env-tenant"
        assert key[4] == ("scope1", "scope2")

    def test_explicit_none_stays_none(self) -> None:
        # None != ... — an explicit None means "the caller deliberately
        # cleared this field" and the env fallback should NOT kick in.
        key = msal_module._resolve_config(
            tenant_id="t", client_id=None, client_secret=None,
            authority="https://example.com/t", scopes=None,
        )
        assert key == ("t", None, None, "https://example.com/t", ())

    def test_authority_derived_from_tenant_when_unset(self) -> None:
        key = msal_module._resolve_config(
            tenant_id="abc-123", client_id="c", client_secret="s",
            authority=None, scopes="api/.default",
        )
        assert key[3] == "https://login.microsoftonline.com/abc-123"

    def test_authority_explicit_overrides_tenant_derivation(self) -> None:
        key = msal_module._resolve_config(
            tenant_id="t", client_id="c", client_secret="s",
            authority="https://custom.example.com/t", scopes="api/.default",
        )
        assert key[3] == "https://custom.example.com/t"

    def test_missing_tenant_and_authority_raises(self) -> None:
        with pytest.raises(ValueError, match="tenant_id is required"):
            msal_module._resolve_config(
                tenant_id=None, client_id="c", client_secret="s",
                authority=None, scopes="api/.default",
            )

    def test_whitespace_strings_normalized(self) -> None:
        key = msal_module._resolve_config(
            tenant_id="  t  ", client_id="  c  ", client_secret="  s  ",
            authority="  https://x/t  ", scopes="  a  ",
        )
        assert key == ("t", "c", "s", "https://x/t", ("a",))


# ---------------------------------------------------------------------------
# Singleton behavior
# ---------------------------------------------------------------------------


class TestSingleton:

    def test_same_config_returns_same_instance(self) -> None:
        a = _build_msal_auth()
        b = _build_msal_auth()
        assert a is b

    def test_different_scopes_returns_different_instance(self) -> None:
        a = _build_msal_auth(scopes="scope-a")
        b = _build_msal_auth(scopes="scope-b")
        assert a is not b

    def test_different_tenant_returns_different_instance(self) -> None:
        a = _build_msal_auth(tenant_id="t1")
        b = _build_msal_auth(tenant_id="t2")
        assert a is not b

    def test_authority_alias_collapses_to_one_singleton(self) -> None:
        # Passing tenant_id alone derives the same authority as passing
        # the URL explicitly — both ways must reach the same instance.
        a = _build_msal_auth(tenant_id="t-alias")
        b = MSALAuth(
            tenant_id="t-alias",
            client_id="client-1",
            client_secret="secret-1",
            authority="https://login.microsoftonline.com/t-alias",
            scopes="api://x/.default",
        )
        assert a is b

    def test_subclass_does_not_collide_with_base(self) -> None:
        class Sub(MSALAuth):
            pass

        base = _build_msal_auth()
        sub = Sub(
            tenant_id="tenant-1", client_id="client-1",
            client_secret="secret-1", scopes="api://x/.default",
        )
        assert base is not sub
        assert type(base) is MSALAuth
        assert type(sub) is Sub

    def test_instances_cache_is_expiring_dict(self) -> None:
        # The cache type is part of the public contract for tests /
        # operators that need to clear or introspect it.
        assert isinstance(MSALAuth._INSTANCES, ExpiringDict)

    def test_concurrent_construction_yields_one_instance(self) -> None:
        # Two threads racing on the same config must collapse to a
        # single shared singleton — that's the whole point of the
        # ExpiringDict.get_or_set seam.
        results: list[MSALAuth] = []
        barrier = threading.Barrier(8)

        def make() -> None:
            barrier.wait()
            results.append(_build_msal_auth(tenant_id="race-t"))

        threads = [threading.Thread(target=make) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 8
        assert all(r is results[0] for r in results)


# ---------------------------------------------------------------------------
# __init__ idempotence
# ---------------------------------------------------------------------------


class TestInitIdempotence:

    def test_reinit_does_not_overwrite_runtime_state(self) -> None:
        a = _build_msal_auth()
        a._access_token = "live-token"  # type: ignore[assignment]
        a._expires_at = time.time() + 1000
        # Re-entering the constructor must hit the _initialized guard
        # — otherwise we'd nuke the cached token on every caller.
        b = _build_msal_auth()
        assert b is a
        assert a._access_token == "live-token"
        assert a._expires_at is not None

    def test_initialized_flag_set_after_construction(self) -> None:
        a = _build_msal_auth()
        assert getattr(a, "_initialized", False) is True


# ---------------------------------------------------------------------------
# Hash + equality
# ---------------------------------------------------------------------------


class TestHashEquality:

    def test_equal_configs_are_equal_and_hash_alike(self) -> None:
        a = _build_msal_auth()
        b = _build_msal_auth()
        assert a == b
        assert hash(a) == hash(b)

    def test_different_configs_not_equal(self) -> None:
        a = _build_msal_auth(tenant_id="t1")
        b = _build_msal_auth(tenant_id="t2")
        assert a != b

    def test_different_type_returns_notimplemented(self) -> None:
        a = _build_msal_auth()
        # __eq__ returns NotImplemented for non-MSALAuth, which falls
        # back to default identity comparison — so `==` resolves to False.
        assert a.__eq__("not an MSALAuth") is NotImplemented
        assert (a == "not an MSALAuth") is False

    def test_self_equality_short_circuits(self) -> None:
        a = _build_msal_auth()
        assert a == a

    def test_usable_as_dict_key(self) -> None:
        a = _build_msal_auth(tenant_id="t1")
        b = _build_msal_auth(tenant_id="t2")
        d = {a: "one", b: "two"}
        # Same-config rebuild lands on the same bucket.
        assert d[_build_msal_auth(tenant_id="t1")] == "one"
        assert d[_build_msal_auth(tenant_id="t2")] == "two"


# ---------------------------------------------------------------------------
# Scope normalization
# ---------------------------------------------------------------------------


class TestScopes:

    def test_scopes_normalized_to_list(self) -> None:
        a = _build_msal_auth(scopes="a b")
        assert a.scopes == ["a", "b"]

    def test_scope_string_property_joins_with_space(self) -> None:
        a = _build_msal_auth(scopes=["a", "b"])
        assert a.scope == "a b"

    def test_scope_property_none_when_unset(self) -> None:
        a = MSALAuth(
            tenant_id="t", client_id="c", client_secret="s", scopes=None,
        )
        assert a.scope is None

    def test_scope_setter_resets_cached_token(self) -> None:
        a = _build_msal_auth()
        a._access_token = "old"  # type: ignore[assignment]
        a._expires_at = time.time() + 1000
        a.scope = ["new", "scope"]
        assert a.scopes == ["new", "scope"]
        assert a._access_token is None
        assert a._expires_at is None

    def test_scope_setter_accepts_tuple_and_set(self) -> None:
        a = _build_msal_auth()
        a.scope = ("x", "y")
        assert a.scopes == ["x", "y"]
        # Sets aren't order-preserving — just check membership.
        a.scope = {"p", "q"}
        assert set(a.scopes or []) == {"p", "q"}


# ---------------------------------------------------------------------------
# Token lifecycle (mocked MSAL)
# ---------------------------------------------------------------------------


class TestTokenLifecycle:

    def test_is_expired_true_when_never_acquired(self) -> None:
        a = _build_msal_auth()
        assert a.is_expired is True
        assert a.seconds_to_expiry == float("-inf")

    def test_is_expired_respects_skew(self) -> None:
        a = _build_msal_auth(expiry_skew_seconds=30)
        a._expires_at = time.time() + 20  # within skew window
        assert a.is_expired is True
        a._expires_at = time.time() + 120  # safely past skew
        assert a.is_expired is False

    def test_seconds_to_expiry_signed(self) -> None:
        a = _build_msal_auth()
        a._expires_at = time.time() + 500
        assert a.seconds_to_expiry > 0
        a._expires_at = time.time() - 5
        assert a.seconds_to_expiry < 0

    def test_ensure_confidential_flow_ready_errors(self) -> None:
        a = MSALAuth(
            tenant_id="t", client_id=None, client_secret="s",
            scopes="api/.default",
        )
        with pytest.raises(ValueError, match="client_id is required"):
            a._ensure_confidential_flow_ready()

        b = _build_msal_auth(scopes=None)
        with pytest.raises(ValueError, match="scopes are required"):
            b._ensure_confidential_flow_ready()

    def test_auth_app_uses_confidential_client_when_secret_present(self) -> None:
        a = _build_msal_auth(client_secret="secret-1")
        fake_cls = MagicMock(return_value=MagicMock(name="ConfidentialApp"))
        with patch.object(msal_module, "ConfidentialClientApplication", fake_cls):
            app = a.auth_app
        fake_cls.assert_called_once_with(
            client_id="client-1",
            client_credential="secret-1",
            authority="https://login.microsoftonline.com/tenant-1",
        )
        assert app is fake_cls.return_value
        # Cached on the instance — second access doesn't re-construct.
        with patch.object(msal_module, "ConfidentialClientApplication", fake_cls):
            assert a.auth_app is app
        assert fake_cls.call_count == 1

    def test_auth_app_uses_public_client_when_no_secret(self) -> None:
        a = MSALAuth(
            tenant_id="t", client_id="c", client_secret=None,
            scopes="api/.default",
        )
        fake_cls = MagicMock(return_value=MagicMock(name="PublicApp"))
        with patch.object(msal_module, "PublicClientApplication", fake_cls):
            app = a.auth_app
        fake_cls.assert_called_once_with(
            client_id="c",
            authority="https://login.microsoftonline.com/t",
        )
        assert app is fake_cls.return_value

    def test_refresh_acquires_token_for_client(self) -> None:
        a = _build_msal_auth()
        fake_app = MagicMock()
        fake_app.acquire_token_for_client.return_value = _make_token_result("AAA", 3600)
        a._auth_app = fake_app

        a.refresh()

        fake_app.acquire_token_for_client.assert_called_once_with(scopes=["api://x/.default"])
        assert a._access_token == "AAA"
        assert a._expires_at is not None
        assert a._expires_at > time.time()

    def test_refresh_skipped_when_token_live(self) -> None:
        a = _build_msal_auth()
        fake_app = MagicMock()
        fake_app.acquire_token_for_client.return_value = _make_token_result("AAA", 3600)
        a._auth_app = fake_app

        a.refresh()
        a.refresh()  # second call should hit the live-token fast path

        assert fake_app.acquire_token_for_client.call_count == 1

    def test_refresh_force_overrides_cache(self) -> None:
        a = _build_msal_auth()
        fake_app = MagicMock()
        fake_app.acquire_token_for_client.side_effect = [
            _make_token_result("AAA", 3600),
            _make_token_result("BBB", 3600),
        ]
        a._auth_app = fake_app

        a.refresh()
        assert a._access_token == "AAA"
        a.refresh(force=True)
        assert a._access_token == "BBB"
        assert fake_app.acquire_token_for_client.call_count == 2

    def test_refresh_raises_when_msal_returns_no_token(self) -> None:
        a = _build_msal_auth()
        fake_app = MagicMock()
        fake_app.acquire_token_for_client.return_value = {
            "error_description": "AADSTS70011: bad scope",
        }
        a._auth_app = fake_app

        with pytest.raises(RuntimeError, match="Failed to acquire token"):
            a.refresh()
        assert a._access_token is None

    def test_refresh_uses_public_flow_when_no_secret(self) -> None:
        a = MSALAuth(
            tenant_id="t", client_id="c", client_secret=None,
            scopes="api/.default",
        )
        fake_app = MagicMock()
        fake_app.acquire_token_interactive.return_value = _make_token_result("PUB", 3600)
        a._auth_app = fake_app

        a.refresh()

        fake_app.acquire_token_interactive.assert_called_once_with(scopes=["api/.default"])
        assert a._access_token == "PUB"

    def test_access_token_property_triggers_refresh(self) -> None:
        a = _build_msal_auth()
        fake_app = MagicMock()
        fake_app.acquire_token_for_client.return_value = _make_token_result("AAA", 3600)
        a._auth_app = fake_app

        token = a.access_token

        assert token == "AAA"
        fake_app.acquire_token_for_client.assert_called_once()

    def test_authorization_returns_bearer_header(self) -> None:
        a = _build_msal_auth()
        fake_app = MagicMock()
        fake_app.acquire_token_for_client.return_value = _make_token_result("XYZ", 3600)
        a._auth_app = fake_app

        assert a.authorization == "Bearer XYZ"
        assert str(a) == "Bearer XYZ"

    def test_authorization_subclasses_base(self) -> None:
        assert issubclass(MSALAuth, Authorization)
        a = _build_msal_auth()
        fake_app = MagicMock()
        fake_app.acquire_token_for_client.return_value = _make_token_result("T", 3600)
        a._auth_app = fake_app
        # Duck-type the Authorization contract.
        assert isinstance(a.authorization, str)


# ---------------------------------------------------------------------------
# Mapping sugar + repr
# ---------------------------------------------------------------------------


class TestMappingAndRepr:

    def test_getitem_setitem_delegate_to_attributes(self) -> None:
        a = _build_msal_auth()
        a["client_id"] = "rotated-client"
        assert a.client_id == "rotated-client"
        assert a["client_id"] == "rotated-client"

    def test_repr_hides_secret(self) -> None:
        a = _build_msal_auth()
        text = repr(a)
        assert "tenant-1" in text
        assert "client-1" in text
        # The secret is never written to repr (avoid leaking into logs).
        assert "secret-1" not in text


# ---------------------------------------------------------------------------
# Pickle round-trip
# ---------------------------------------------------------------------------


class TestPickle:

    def test_in_process_pickle_collapses_to_singleton(self) -> None:
        a = _build_msal_auth()
        clone = pickle.loads(pickle.dumps(a))
        # __getnewargs__ routes us back through the cached __new__.
        assert clone is a

    def test_getstate_drops_transient_attrs(self) -> None:
        a = _build_msal_auth()
        a._auth_app = object()  # type: ignore[assignment]
        state = a.__getstate__()
        for transient in MSALAuth.TRANSIENT_STATE_ATTRS:
            assert transient not in state, (
                f"transient attr {transient!r} leaked into pickle state"
            )
        # Non-transient attrs survive.
        assert state["client_id"] == "client-1"
        assert state["tenant_id"] == "tenant-1"

    def test_cross_process_pickle_rebuilds_transients(self) -> None:
        # Simulate the receiver side of a cross-process pickle by
        # clearing the singleton cache before unpickling — that's the
        # exact path a Spark worker / multiprocessing child takes.
        a = _build_msal_auth()
        a._access_token = "cached-tok"  # type: ignore[assignment]
        a._expires_at = time.time() + 1000
        payload = pickle.dumps(a)

        MSALAuth._INSTANCES.clear()

        clone = pickle.loads(payload)

        assert clone is not a
        assert clone.client_id == "client-1"
        assert clone.tenant_id == "tenant-1"
        assert clone.scopes == ["api://x/.default"]
        # Token state survives the pickle so the receiver doesn't pay
        # another AAD round-trip.
        assert clone._access_token == "cached-tok"
        # Transient slots got reset to fresh defaults.
        assert clone._auth_app is None
        assert isinstance(clone._refresh_lock, type(threading.Lock()))

    def test_setstate_short_circuits_on_initialized_singleton(self) -> None:
        # If __new__ returns the live singleton, __setstate__ must not
        # clobber its current state — that's the whole point of the
        # ``_initialized`` guard.
        a = _build_msal_auth()
        a._access_token = "live-token"  # type: ignore[assignment]
        clone = pickle.loads(pickle.dumps(a))
        assert clone is a
        assert a._access_token == "live-token"
