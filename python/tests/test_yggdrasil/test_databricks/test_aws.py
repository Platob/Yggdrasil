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
