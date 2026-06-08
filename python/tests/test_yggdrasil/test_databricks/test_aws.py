"""Tests for the Databricks-backed AWS credentials providers.

Behavior contract:

* :class:`AWSDatabricksVolumeCredentials` and
  :class:`AWSDatabricksTableCredentials` are process-wide singletons
  keyed by the resource id (``volume_id`` / ``table_id``) alone;
* one provider serves both read and write modes —
  :meth:`get_credentials(mode=...)` resolves the requested
  :class:`Mode` into the right UC operation and re-runs the SDK call;
* :meth:`aws_client(mode, region)` caches one :class:`AWSClient` per
  ``(mode, region)`` and tags the config with a deterministic
  ``refresher_key`` so the AWSClient singleton cache mints distinct
  boto sessions for read vs write;
* the bound :class:`DatabricksClient` is mutable — re-constructing
  with a new client rebinds without breaking the singleton;
* pickling collapses to the live singleton in-process.
"""
from __future__ import annotations

import pickle
from unittest.mock import MagicMock

import pytest

from yggdrasil.aws import AWSClient
from yggdrasil.aws.config import AwsCredentials
from yggdrasil.aws.provider import AwsCredentialsProvider
from yggdrasil.databricks.aws import (
    AWSDatabricksTableCredentials,
    AWSDatabricksVolumeCredentials,
)


@pytest.fixture(autouse=True)
def _clear_singletons():
    AwsCredentialsProvider._INSTANCES.clear()
    AWSClient._INSTANCES.clear()
    yield
    AwsCredentialsProvider._INSTANCES.clear()
    AWSClient._INSTANCES.clear()


def _op_token(op) -> str:
    return getattr(op, "value", None) or getattr(op, "name", None) or str(op)


def _aws_creds_response(
    *,
    access_key_id: str = "AKIA-test",
    secret_access_key: str = "secret-test",
    session_token: str = "session-test",
):
    import datetime as _dt
    return MagicMock(
        aws_temp_credentials=MagicMock(
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            session_token=session_token,
            access_point=None,
        ),
        expiration_time=_dt.datetime(2099, 1, 1, tzinfo=_dt.timezone.utc),
    )


def _volume_client(creds=None):
    """A ``DatabricksClient``-shaped mock wired so
    ``client.workspace_client().temporary_volume_credentials
    .generate_temporary_volume_credentials(...)`` returns *creds*."""
    client = MagicMock()
    ws = client.workspace_client.return_value
    gen = ws.temporary_volume_credentials.generate_temporary_volume_credentials
    gen.return_value = creds or _aws_creds_response()
    return client, gen


def _table_client(creds=None):
    client = MagicMock()
    ws = client.workspace_client.return_value
    gen = ws.temporary_table_credentials.generate_temporary_table_credentials
    gen.return_value = creds or _aws_creds_response()
    return client, gen


# ===========================================================================
# Volume provider
# ===========================================================================


class TestVolumeCredentialsSingleton:

    def test_same_volume_id_collapses_to_one_instance(self) -> None:
        client, _ = _volume_client()
        a = AWSDatabricksVolumeCredentials("vid-A", client=client)
        b = AWSDatabricksVolumeCredentials("vid-A", client=client)
        assert a is b

    def test_different_volume_ids_yield_different_instances(self) -> None:
        client, _ = _volume_client()
        a = AWSDatabricksVolumeCredentials("vid-A", client=client)
        b = AWSDatabricksVolumeCredentials("vid-B", client=client)
        assert a is not b

    def test_client_rebound_on_repeat_construction(self) -> None:
        client_a, _ = _volume_client()
        client_b, _ = _volume_client()
        a = AWSDatabricksVolumeCredentials("vid-A", client=client_a)
        assert a.client is client_a
        b = AWSDatabricksVolumeCredentials("vid-A", client=client_b)
        assert b is a
        # The latest client wins so STS refreshes pick up the freshest
        # workspace auth context.
        assert b.client is client_b

    def test_passing_no_client_preserves_existing_binding(self) -> None:
        client, _ = _volume_client()
        a = AWSDatabricksVolumeCredentials("vid-A", client=client)
        again = AWSDatabricksVolumeCredentials("vid-A")
        assert again is a
        # ``client=None`` (the default) must NOT clobber the live ref.
        assert again.client is client


class TestVolumeGetCredentials:

    def test_read_mode_hits_read_volume_operation(self) -> None:
        client, gen = _volume_client()
        p = AWSDatabricksVolumeCredentials("vid-A", client=client)
        out = p.get_credentials(mode="read")
        assert isinstance(out, AwsCredentials)
        assert _op_token(gen.call_args.kwargs["operation"]) == "READ_VOLUME"
        assert gen.call_args.kwargs["volume_id"] == "vid-A"

    def test_write_mode_hits_write_volume_operation(self) -> None:
        client, gen = _volume_client()
        p = AWSDatabricksVolumeCredentials("vid-A", client=client)
        p.get_credentials(mode="overwrite")
        assert _op_token(gen.call_args.kwargs["operation"]) == "WRITE_VOLUME"

    def test_default_mode_is_read_only(self) -> None:
        client, gen = _volume_client()
        p = AWSDatabricksVolumeCredentials("vid-A", client=client)
        p.get_credentials()  # no mode → DEFAULT_MODE
        assert _op_token(gen.call_args.kwargs["operation"]) == "READ_VOLUME"

    def test_call_dunder_delegates_to_get_credentials(self) -> None:
        client, _ = _volume_client()
        p = AWSDatabricksVolumeCredentials("vid-A", client=client)
        out = p()
        assert isinstance(out, AwsCredentials)
        assert out.access_key_id == "AKIA-test"

    def test_response_without_aws_creds_raises(self) -> None:
        # Azure/GCP-backed volumes don't return ``aws_temp_credentials``
        # — the provider must surface a helpful error rather than
        # AttributeError-ing inside the SDK glue.
        resp = MagicMock(aws_temp_credentials=None, expiration_time=None)
        client, _ = _volume_client(creds=resp)
        p = AWSDatabricksVolumeCredentials("vid-A", client=client)
        with pytest.raises(RuntimeError, match="aws_temp_credentials"):
            p.get_credentials(mode="read")


class TestVolumeAwsClient:

    def test_same_mode_and_region_returns_same_client(self) -> None:
        client, _ = _volume_client()
        p = AWSDatabricksVolumeCredentials("vid-A", client=client)
        c1 = p.aws_client(mode="read", region="us-east-1")
        c2 = p.aws_client(mode="read", region="us-east-1")
        assert c1 is c2

    def test_different_modes_yield_distinct_clients(self) -> None:
        # Read and write vend different STS creds — each needs its own
        # boto session / ``RefreshableCredentials`` cycle.
        client, _ = _volume_client()
        p = AWSDatabricksVolumeCredentials("vid-A", client=client)
        c_read = p.aws_client(mode="read", region="us-east-1")
        c_write = p.aws_client(mode="overwrite", region="us-east-1")
        assert c_read is not c_write
        assert c_read.config.refresher_key != c_write.config.refresher_key

    def test_different_regions_yield_distinct_clients(self) -> None:
        client, _ = _volume_client()
        p = AWSDatabricksVolumeCredentials("vid-A", client=client)
        us = p.aws_client(mode="read", region="us-east-1")
        eu = p.aws_client(mode="read", region="eu-central-1")
        assert us is not eu

    def test_refresher_is_picklable_mode_bound_adapter(self) -> None:
        # The refresher handed to AWSConfig is a frozen dataclass that
        # binds (provider, mode), not a closure — so it survives a
        # cross-process pickle without cloudpickle.
        client, _ = _volume_client()
        p = AWSDatabricksVolumeCredentials("vid-A", client=client)
        c = p.aws_client(mode="read", region="us-east-1")
        refresher = c.config.refresher
        loaded = pickle.loads(pickle.dumps(refresher))
        assert loaded == refresher


class TestVolumePickling:

    def test_in_process_pickle_round_trip_returns_singleton(self) -> None:
        client, _ = _volume_client()
        p = AWSDatabricksVolumeCredentials("vid-A", client=client)
        p.aws_client(mode="read", region="us-east-1")
        loaded = pickle.loads(pickle.dumps(p))
        assert loaded is p

    def test_cross_process_pickle_rebuilds_transients(self) -> None:
        client, _ = _volume_client()
        p = AWSDatabricksVolumeCredentials("vid-A", client=client)
        p.aws_client(mode="read", region="us-east-1")
        payload = pickle.dumps(p)
        # Simulate a different process.
        AwsCredentialsProvider._INSTANCES.clear()
        AWSClient._INSTANCES.clear()
        loaded = pickle.loads(payload)
        assert loaded.volume_id == "vid-A"
        assert loaded._client_cache == {}
        # Bound client is transient — receiver-side resolves it lazily.
        assert loaded._client is None


# ===========================================================================
# Table provider
# ===========================================================================


class TestTableCredentialsSingleton:

    def test_same_table_id_collapses_to_one_instance(self) -> None:
        client, _ = _table_client()
        a = AWSDatabricksTableCredentials("tid-A", client=client)
        b = AWSDatabricksTableCredentials("tid-A", client=client)
        assert a is b

    def test_different_table_ids_yield_different_instances(self) -> None:
        client, _ = _table_client()
        a = AWSDatabricksTableCredentials("tid-A", client=client)
        b = AWSDatabricksTableCredentials("tid-B", client=client)
        assert a is not b

    def test_volume_and_table_keyspaces_are_disjoint(self) -> None:
        # The base provider keys by ``(cls, key)`` — same string id on
        # different subclasses must NOT collide.
        client_v, _ = _volume_client()
        client_t, _ = _table_client()
        v = AWSDatabricksVolumeCredentials("shared-id", client=client_v)
        t = AWSDatabricksTableCredentials("shared-id", client=client_t)
        assert v is not t


class TestTableGetCredentials:

    def test_read_mode_hits_read_table_operation(self) -> None:
        client, gen = _table_client()
        p = AWSDatabricksTableCredentials("tid-A", client=client)
        p.get_credentials(mode="read")
        assert _op_token(gen.call_args.kwargs["operation"]) == "READ"
        assert gen.call_args.kwargs["table_id"] == "tid-A"

    def test_write_mode_hits_read_write_table_operation(self) -> None:
        client, gen = _table_client()
        p = AWSDatabricksTableCredentials("tid-A", client=client)
        p.get_credentials(mode="overwrite")
        assert _op_token(gen.call_args.kwargs["operation"]) == "READ_WRITE"

    def test_default_mode_is_read_only(self) -> None:
        client, gen = _table_client()
        p = AWSDatabricksTableCredentials("tid-A", client=client)
        p.get_credentials()
        assert _op_token(gen.call_args.kwargs["operation"]) == "READ"


class TestTableAwsClient:

    def test_different_modes_yield_distinct_clients(self) -> None:
        client, _ = _table_client()
        p = AWSDatabricksTableCredentials("tid-A", client=client)
        c_read = p.aws_client(mode="read", region="us-east-1")
        c_write = p.aws_client(mode="overwrite", region="us-east-1")
        assert c_read is not c_write

    def test_same_mode_and_region_returns_same_client(self) -> None:
        client, _ = _table_client()
        p = AWSDatabricksTableCredentials("tid-A", client=client)
        a = p.aws_client(mode="read", region="us-east-1")
        b = p.aws_client(mode="read", region="us-east-1")
        assert a is b


# ===========================================================================
# Refresher key — cross-resource distinctness
# ===========================================================================


class TestRefresherKeyIsolation:

    def test_volume_and_table_with_same_id_mint_distinct_clients(self) -> None:
        # Same string id, different provider classes → different
        # ``refresher_key`` stamps → different AWSClient singletons.
        client_v, _ = _volume_client()
        client_t, _ = _table_client()
        v = AWSDatabricksVolumeCredentials("shared-id", client=client_v)
        t = AWSDatabricksTableCredentials("shared-id", client=client_t)
        cv = v.aws_client(mode="read", region="us-east-1")
        ct = t.aws_client(mode="read", region="us-east-1")
        assert cv is not ct

    def test_refresher_key_encodes_cls_resource_and_mode(self) -> None:
        client, _ = _volume_client()
        p = AWSDatabricksVolumeCredentials("vid-A", client=client)
        c_read = p.aws_client(mode="read", region="us-east-1")
        c_write = p.aws_client(mode="overwrite", region="us-east-1")
        key_r = c_read.config.refresher_key
        key_w = c_write.config.refresher_key
        # Stable structure callers can rely on for debugging /
        # cache-key construction.
        assert "vid-A" in key_r and "vid-A" in key_w
        assert "READ_ONLY" in key_r
        assert "OVERWRITE" in key_w
        assert AWSDatabricksVolumeCredentials.__name__ in key_r


class TestRefresherKeyPersistsSingleton:
    """A refresher-backed client must collapse to one singleton even though
    the seed credentials rotate on every vend — the ``refresher_key`` is the
    identity, not the ephemeral temporary creds."""

    @staticmethod
    def _rotating_refresher():
        import datetime as dt
        from yggdrasil.aws.config import AwsCredentials

        def refresher():
            refresher.calls += 1
            i = refresher.calls
            return AwsCredentials(
                access_key_id=f"AKIA-{i}",
                secret_access_key=f"secret-{i}",
                session_token=f"token-{i}",
                expiration=dt.datetime(2099, 1, 1, tzinfo=dt.timezone.utc).isoformat(),
            )
        refresher.calls = 0
        return refresher

    def test_rotating_seed_creds_collapse_to_one_client(self) -> None:
        refresher = self._rotating_refresher()
        key = "AWSDatabricksTableCredentials:tid-A:READ_ONLY"
        c1 = AWSClient.from_refresher(refresher, region="us-east-1", refresher_key=key)
        c2 = AWSClient.from_refresher(refresher, region="us-east-1", refresher_key=key)
        # Different seed creds (AKIA-1 vs AKIA-2) must NOT fragment the cache.
        assert c1 is c2

    def test_distinct_refresher_keys_stay_distinct(self) -> None:
        refresher = self._rotating_refresher()
        c_read = AWSClient.from_refresher(
            refresher, region="us-east-1", refresher_key="X:tid-A:READ_ONLY")
        c_write = AWSClient.from_refresher(
            refresher, region="us-east-1", refresher_key="X:tid-A:OVERWRITE")
        assert c_read is not c_write

    def test_no_refresher_key_keys_on_refresher_identity(self) -> None:
        # Without a refresher_key the singleton keys on the refresher's own
        # identity (the creds are vended lazily, so there are none to key on
        # at build time). The same refresher object therefore collapses to
        # one client — and is never invoked just to construct it.
        refresher = self._rotating_refresher()
        c1 = AWSClient.from_refresher(refresher, region="us-east-1")
        c2 = AWSClient.from_refresher(refresher, region="us-east-1")
        assert c1 is c2
        assert refresher.calls == 0  # lazy: not fetched at construction

    def test_distinct_refresher_objects_stay_distinct(self) -> None:
        # Two different ad-hoc refreshers (no key) must not collapse onto
        # each other — object identity discriminates them.
        c1 = AWSClient.from_refresher(self._rotating_refresher(), region="us-east-1")
        c2 = AWSClient.from_refresher(self._rotating_refresher(), region="us-east-1")
        assert c1 is not c2


# ===========================================================================
# Auto self-grant EXTERNAL_USE_SCHEMA on credential mint failure
# ===========================================================================


class _PermissionDenied(Exception):
    """Locally-typed stand-in for ``databricks.sdk.errors.platform.PermissionDenied``.

    The recovery hook keys off ``type(exc).__name__ == 'PermissionDenied'``
    so an in-test exception with that name is sufficient — we don't need
    to import the real SDK error class.
    """


PermissionDenied = type("PermissionDenied", (_PermissionDenied,), {})


_EXTERNAL_USE_MSG = (
    "User does not have EXTERNAL USE SCHEMA on Schema "
    "'trading_tgp_dev.unittest'. Config: host=...."
)


class TestExternalUseSchemaSelfGrant:

    @staticmethod
    def _wire_self_grant(client) -> MagicMock:
        """Hook ``current_user`` and ``schemas.schema(...).grant(...)``
        on *client* so the recovery path has something to call."""
        # ``name=`` is a reserved MagicMock kwarg — set attributes
        # explicitly so ``getattr(user, "name", None)`` works as
        # expected (and email/username, which aren't reserved, can ride
        # along on the constructor).
        user = MagicMock(email="alice@example.com", username="alice")
        user.name = "alice"
        client.iam.users.current_user = user
        schema = MagicMock()
        client.schemas.schema.return_value = schema
        return schema

    def test_self_grants_and_retries_on_external_use_schema_denied(self) -> None:
        client, gen = _volume_client()
        # First call → PermissionDenied; second call → fresh creds.
        gen.side_effect = [
            PermissionDenied(_EXTERNAL_USE_MSG),
            _aws_creds_response(),
        ]
        schema = self._wire_self_grant(client)

        p = AWSDatabricksVolumeCredentials("vid-A", client=client)
        out = p.get_credentials(mode="overwrite")
        assert isinstance(out, AwsCredentials)
        # The mint was tried twice (fail → recover → succeed).
        assert gen.call_count == 2
        # Schema was looked up by the two-part name from the error
        # message; grant landed on the current user.
        client.schemas.schema.assert_called_once_with(
            catalog_name="trading_tgp_dev", schema_name="unittest",
        )
        schema.grant.assert_called_once_with(
            "alice@example.com", "EXTERNAL_USE_SCHEMA",
        )

    def test_non_external_use_permission_denied_propagates(self) -> None:
        client, gen = _volume_client()
        gen.side_effect = PermissionDenied(
            "User does not have SELECT on Table 'cat.sch.tbl'."
        )
        schema = self._wire_self_grant(client)

        p = AWSDatabricksVolumeCredentials("vid-A", client=client)
        with pytest.raises(PermissionDenied):
            p.get_credentials()
        # No grant attempted — message didn't match.
        schema.grant.assert_not_called()
        assert gen.call_count == 1

    def test_grant_failure_propagates_original_error(self) -> None:
        # When the self-grant itself fails (caller isn't owner), the
        # original PermissionDenied wins — that's the actionable error.
        client, gen = _volume_client()
        original = PermissionDenied(_EXTERNAL_USE_MSG)
        gen.side_effect = [original, _aws_creds_response()]
        schema = self._wire_self_grant(client)
        schema.grant.side_effect = RuntimeError("not owner")

        p = AWSDatabricksVolumeCredentials("vid-A", client=client)
        with pytest.raises(PermissionDenied) as info:
            p.get_credentials()
        assert info.value is original
        # No retry of the mint happened.
        assert gen.call_count == 1

    def test_no_principal_skips_grant(self) -> None:
        client, gen = _volume_client()
        gen.side_effect = PermissionDenied(_EXTERNAL_USE_MSG)
        user = MagicMock(email=None, username=None)
        user.name = None
        client.iam.users.current_user = user

        p = AWSDatabricksVolumeCredentials("vid-A", client=client)
        with pytest.raises(PermissionDenied):
            p.get_credentials()
        client.schemas.schema.assert_not_called()


# ===========================================================================
# Path provider — UC ``generate_temporary_path_credentials`` scoped to a URL
# ===========================================================================


def _path_client(creds=None):
    client = MagicMock()
    ws = client.workspace_client.return_value
    gen = ws.temporary_path_credentials.generate_temporary_path_credentials
    gen.return_value = creds or _aws_creds_response()
    return client, gen


class TestPathCredentials:
    def test_url_normalized_to_trailing_slash(self) -> None:
        from yggdrasil.databricks.aws import AWSDatabricksPathCredentials
        client, _ = _path_client()
        p = AWSDatabricksPathCredentials("s3://b/p", client=client)
        assert p.url == "s3://b/p/"
        assert p.key == "s3://b/p/"

    def test_same_url_collapses_to_one_instance(self) -> None:
        from yggdrasil.databricks.aws import AWSDatabricksPathCredentials
        client, _ = _path_client()
        a = AWSDatabricksPathCredentials("s3://b/p", client=client)
        b = AWSDatabricksPathCredentials("s3://b/p/", client=client)
        assert a is b  # trailing slash normalises to one key

    def test_read_mode_hits_path_read(self) -> None:
        from yggdrasil.databricks.aws import AWSDatabricksPathCredentials
        from yggdrasil.enums import Mode
        client, gen = _path_client()
        p = AWSDatabricksPathCredentials("s3://b/p/", client=client)
        p.get_credentials(Mode.READ_ONLY)
        _, kwargs = gen.call_args
        assert _op_token(kwargs["operation"]) == "PATH_READ"
        assert kwargs["url"] == "s3://b/p/"

    def test_write_mode_hits_path_read_write(self) -> None:
        from yggdrasil.databricks.aws import AWSDatabricksPathCredentials
        from yggdrasil.enums import Mode
        client, gen = _path_client()
        p = AWSDatabricksPathCredentials("s3://b/p/", client=client)
        p.get_credentials(Mode.OVERWRITE)
        _, kwargs = gen.call_args
        assert _op_token(kwargs["operation"]) == "PATH_READ_WRITE"

    def test_pickle_round_trip_returns_singleton(self) -> None:
        from yggdrasil.databricks.aws import AWSDatabricksPathCredentials
        client, _ = _path_client()
        p = AWSDatabricksPathCredentials("s3://b/p/", client=client)
        assert pickle.loads(pickle.dumps(p)) is p


# ===========================================================================
# Secrets-backed persistence — reuse vended creds across calls / processes
# ===========================================================================


def _iso(dt_):
    return dt_.isoformat()


def _future_iso(hours: int = 24) -> str:
    import datetime as _dt
    return _iso(_dt.datetime.now(_dt.timezone.utc) + _dt.timedelta(hours=hours))


def _persisted_payload(expiration: str) -> dict:
    return {
        "access_key_id": "AKIA-persisted",
        "access_point": None,
        "secret_access_key": "secret-persisted",
        "session_token": "session-persisted",
        "expiration": expiration,
    }


def _wire_secret_object(client, obj):
    """Make ``client.secrets.secret(...).refresh(raise_error=False).object``
    return *obj* — the per-resource ``{mode: creds}`` map (or None)."""
    client.secrets.secret.return_value.refresh.return_value.object = obj


class TestSecretsBackedPersistence:
    """Persistence is **opt-in** (``secret_cache=True``); off by default."""

    def test_disabled_by_default(self) -> None:
        # No ``secret_cache`` → no secret read, no write, plain vend.
        client, gen = _volume_client()
        _wire_secret_object(client, None)
        p = AWSDatabricksVolumeCredentials(
            "vid-default-off", client=client, resource_url="cat.sch.vol",
        )

        out = p.get_credentials(mode="read")

        assert out.access_key_id == "AKIA-test"
        gen.assert_called_once()
        client.secrets.secret.assert_not_called()         # no read
        client.secrets.create_secret.assert_not_called()  # no write

    def test_reuse_ignored_when_secret_cache_off(self) -> None:
        # A valid cached credential is ignored unless the caller opts in.
        client, gen = _volume_client()
        _wire_secret_object(client, {"READ_ONLY": _persisted_payload(_future_iso())})
        p = AWSDatabricksVolumeCredentials("vid-default-off-2", client=client)

        out = p.get_credentials(mode="read")

        gen.assert_called_once()                          # vended, not reused
        assert out.access_key_id == "AKIA-test"
        client.secrets.secret.assert_not_called()

    def test_vend_persists_to_per_resource_scope_under_credentials_key(self) -> None:
        client, gen = _volume_client()
        _wire_secret_object(client, None)  # nothing cached yet → must vend
        p = AWSDatabricksVolumeCredentials(
            "vid-persist-1", client=client, resource_url="cat.sch.vol",
            secret_cache=True,
        )

        out = p.get_credentials(mode="read")

        assert isinstance(out, AwsCredentials)
        gen.assert_called_once()
        client.secrets.create_secret.assert_called_once()
        kw = client.secrets.create_secret.call_args.kwargs
        # Scope is per resource, aws-prefixed, named from the resource URL;
        # the credential lives under the single ``credentials`` key as a
        # per-mode map.
        assert kw["scope"] == "aws.volume.cat.sch.vol"
        assert kw["key"] == "credentials"
        assert kw["value"]["READ_ONLY"]["access_key_id"] == "AKIA-test"
        assert kw["value"]["READ_ONLY"]["session_token"] == "session-test"

    def test_secret_cache_opt_in_is_sticky_across_singleton(self) -> None:
        # The provider is a singleton; once any construction opts in, the
        # shared instance keeps persisting.
        client, _ = _volume_client()
        _wire_secret_object(client, None)
        AWSDatabricksVolumeCredentials(
            "vid-sticky", client=client, secret_cache=True,
        )
        # A later plain construction returns the same (now-enabled) singleton.
        p = AWSDatabricksVolumeCredentials("vid-sticky", client=client)

        p.get_credentials(mode="read")

        client.secrets.create_secret.assert_called_once()

    def test_scope_falls_back_to_id_without_resource_url(self) -> None:
        client, _ = _volume_client()
        _wire_secret_object(client, None)
        p = AWSDatabricksVolumeCredentials(
            "vid-noerl", client=client, secret_cache=True,
        )

        p.get_credentials(mode="read")

        assert (
            client.secrets.create_secret.call_args.kwargs["scope"]
            == "aws.volume.vid-noerl"
        )

    def test_reuse_from_secret_skips_vend(self) -> None:
        client, gen = _volume_client()
        _wire_secret_object(client, {"READ_ONLY": _persisted_payload(_future_iso())})
        p = AWSDatabricksVolumeCredentials(
            "vid-persist-2", client=client, secret_cache=True,
        )

        out = p.get_credentials(mode="read")

        gen.assert_not_called()                       # no UC vend
        client.secrets.create_secret.assert_not_called()
        assert out.access_key_id == "AKIA-persisted"
        assert out.session_token == "session-persisted"

    def test_other_mode_entry_is_a_miss(self) -> None:
        # The map only holds a write entry → a read still vends.
        client, gen = _volume_client()
        _wire_secret_object(client, {"OVERWRITE": _persisted_payload(_future_iso())})
        p = AWSDatabricksVolumeCredentials(
            "vid-persist-2b", client=client, secret_cache=True,
        )

        p.get_credentials(mode="read")
        gen.assert_called_once()

    def test_near_expiry_secret_revends(self) -> None:
        client, gen = _volume_client()
        # 1 minute of life left — inside the 10-minute margin → treat as miss.
        import datetime as _dt
        soon = _iso(_dt.datetime.now(_dt.timezone.utc) + _dt.timedelta(minutes=1))
        _wire_secret_object(client, {"READ_ONLY": _persisted_payload(soon)})
        p = AWSDatabricksVolumeCredentials(
            "vid-persist-3", client=client, secret_cache=True,
        )

        out = p.get_credentials(mode="read")

        gen.assert_called_once()                      # stale → fresh vend
        assert out.access_key_id == "AKIA-test"

    def test_in_process_memo_skips_second_secret_read_and_vend(self) -> None:
        client, gen = _volume_client()
        _wire_secret_object(client, None)
        p = AWSDatabricksVolumeCredentials(
            "vid-persist-4", client=client, secret_cache=True,
        )

        p.get_credentials(mode="read")                # vends + memoises
        p.get_credentials(mode="read")                # served from the memo

        gen.assert_called_once()
        # The secret was only read once (the first call's miss); the second
        # call short-circuited on the in-process memo. Persist builds the map
        # from the memo, so it adds no extra read.
        client.secrets.secret.assert_called_once()

    def test_default_path_memoises_without_secret_cache(self) -> None:
        # The common path (no secret_cache) still memoises the vended credential
        # in-process, so repeated resolutions within validity vend from Unity
        # Catalog exactly once and never touch the Secrets API.
        client, gen = _volume_client()
        p = AWSDatabricksVolumeCredentials("vid-memo-default", client=client)
        p.get_credentials(mode="read")
        p.get_credentials(mode="read")
        p.get_credentials(mode="read")
        gen.assert_called_once()
        client.secrets.secret.assert_not_called()
        client.secrets.create_secret.assert_not_called()

    def test_disabled_when_prefix_empty(self, monkeypatch) -> None:
        monkeypatch.setenv("YGG_DATABRICKS_CREDS_SECRET_PREFIX", "")
        client, gen = _volume_client()
        p = AWSDatabricksVolumeCredentials(
            "vid-persist-5", client=client, secret_cache=True,
        )

        p.get_credentials(mode="read")

        gen.assert_called_once()
        client.secrets.secret.assert_not_called()     # no read
        client.secrets.create_secret.assert_not_called()  # no write

    def test_prefix_override_via_env(self, monkeypatch) -> None:
        monkeypatch.setenv("YGG_DATABRICKS_CREDS_SECRET_PREFIX", "myteam")
        client, gen = _volume_client()
        _wire_secret_object(client, None)
        p = AWSDatabricksVolumeCredentials(
            "vid-persist-6", client=client, resource_url="cat.sch.vol",
            secret_cache=True,
        )

        p.get_credentials(mode="read")

        assert (
            client.secrets.create_secret.call_args.kwargs["scope"]
            == "myteam.volume.cat.sch.vol"
        )

    def test_secret_read_failure_falls_back_to_vend(self) -> None:
        client, gen = _volume_client()
        client.secrets.secret.side_effect = RuntimeError("no secrets access")
        p = AWSDatabricksVolumeCredentials(
            "vid-persist-7", client=client, secret_cache=True,
        )

        out = p.get_credentials(mode="read")  # must not raise

        gen.assert_called_once()
        assert out.access_key_id == "AKIA-test"

    def test_persist_failure_does_not_break_vend(self) -> None:
        client, gen = _volume_client()
        _wire_secret_object(client, None)
        client.secrets.create_secret.side_effect = RuntimeError("no write access")
        p = AWSDatabricksVolumeCredentials(
            "vid-persist-8", client=client, secret_cache=True,
        )

        out = p.get_credentials(mode="read")  # best-effort persist swallows the error

        gen.assert_called_once()
        assert out.access_key_id == "AKIA-test"

    def test_read_and_write_modes_share_credentials_key(self) -> None:
        client, _ = _volume_client()
        _wire_secret_object(client, None)
        p = AWSDatabricksVolumeCredentials(
            "vid-persist-9", client=client, secret_cache=True,
        )

        p.get_credentials(mode="read")
        p.get_credentials(mode="overwrite")

        # Both modes write the same ``credentials`` key; the last write carries
        # both entries (built from the in-process memo).
        keys = {c.kwargs["key"] for c in client.secrets.create_secret.call_args_list}
        assert keys == {"credentials"}
        last_value = client.secrets.create_secret.call_args.kwargs["value"]
        assert set(last_value) == {"READ_ONLY", "OVERWRITE"}

    def test_table_provider_persists_to_table_scope(self) -> None:
        client, gen = _table_client()
        _wire_secret_object(client, None)
        p = AWSDatabricksTableCredentials(
            "tid-persist-1", client=client, resource_url="cat.sch.tbl",
            secret_cache=True,
        )

        p.get_credentials(mode="read")

        client.secrets.create_secret.assert_called_once()
        kw = client.secrets.create_secret.call_args.kwargs
        assert kw["scope"] == "aws.table.cat.sch.tbl"
        assert kw["key"] == "credentials"
