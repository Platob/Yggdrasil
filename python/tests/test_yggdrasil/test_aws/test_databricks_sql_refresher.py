"""Tests for :class:`DatabricksSQLCredentialsRefresher` and the
:meth:`AWSClient.from_databricks_sql` constructor.

Behavior contract:

* the refresher resolves credentials by re-running a SQL query and
  reading the first row via the standard
  :meth:`StatementResult.read_pylist` path;
* canonical lowercase column names win first, AWS STS JSON casing
  (``AccessKeyId`` / ``SecretAccessKey`` / …) is the fallback, and a
  per-call ``columns=`` mapping overrides both;
* missing required columns and empty result sets raise informative
  errors that name the column that wasn't found and what to try next;
* :class:`datetime` expirations get coerced to ISO strings so
  botocore's ``RefreshableCredentials`` can parse them back;
* the refresher is picklable when no live client is held (the
  ``DatabricksClient.current()`` fallback is resolved lazily inside
  ``__call__``), which is what lets it cross Spark / multiprocessing
  boundaries without dragging the workspace SDK along.
"""
from __future__ import annotations

import datetime
import pickle
from unittest.mock import MagicMock

import pytest

from yggdrasil.aws import AWSClient
from yggdrasil.aws.config import (
    AwsCredentials,
    DatabricksSQLCredentialsRefresher,
)


@pytest.fixture(autouse=True)
def _clear_client_singleton_cache():
    AWSClient._INSTANCES.clear()
    yield
    AWSClient._INSTANCES.clear()


def _fake_db_client(rows: list[dict]) -> MagicMock:
    """Mock that quacks like ``DatabricksClient`` for the refresher's
    one-line ``client.sql.execute(query).read_pylist()`` access."""
    client = MagicMock()
    client.sql.execute.return_value.read_pylist.return_value = rows
    return client


# ---------------------------------------------------------------------------
# Class-level static defaults on AWSConfig
# ---------------------------------------------------------------------------


class TestStaticDefaults:

    def test_canonical_column_aliases_present(self) -> None:
        # All four AwsCredentials fields have a default alias tuple,
        # canonical-lowercase-first then AWS-API-JSON-casing.
        aliases = AWSClient.DATABRICKS_SQL_CREDENTIAL_COLUMNS
        assert aliases["access_key_id"] == ("access_key_id", "AccessKeyId")
        assert aliases["secret_access_key"] == (
            "secret_access_key", "SecretAccessKey",
        )
        assert aliases["session_token"] == ("session_token", "SessionToken")
        assert aliases["expiration"] == ("expiration", "Expiration")

    def test_aliases_are_immutable(self) -> None:
        # MappingProxyType makes the class-var read-only; subclasses
        # have to replace the whole mapping rather than mutating it.
        with pytest.raises(TypeError):
            AWSClient.DATABRICKS_SQL_CREDENTIAL_COLUMNS["foo"] = ("bar",)  # type: ignore[index]


# ---------------------------------------------------------------------------
# DatabricksSQLCredentialsRefresher — column resolution
# ---------------------------------------------------------------------------


class TestColumnResolution:

    def test_canonical_lowercase_columns(self) -> None:
        client = _fake_db_client([{
            "access_key_id": "AKIAEXAMPLE",
            "secret_access_key": "secret",
            "session_token": "tok",
            "expiration": "2026-12-31T23:59:59Z",
        }])
        r = DatabricksSQLCredentialsRefresher(
            query="SELECT * FROM ops.aws_creds",
            client=client,
            column_aliases=dict(AWSClient.DATABRICKS_SQL_CREDENTIAL_COLUMNS),
        )
        creds = r()
        assert isinstance(creds, AwsCredentials)
        assert creds.access_key_id == "AKIAEXAMPLE"
        assert creds.secret_access_key == "secret"
        assert creds.session_token == "tok"
        assert creds.expiration == "2026-12-31T23:59:59Z"

    def test_aws_api_json_casing_fallback(self) -> None:
        client = _fake_db_client([{
            "AccessKeyId": "AKIA",
            "SecretAccessKey": "shh",
            "SessionToken": "t",
            "Expiration": "2026-12-31T23:59:59Z",
        }])
        r = DatabricksSQLCredentialsRefresher(
            query="Q",
            client=client,
            column_aliases=dict(AWSClient.DATABRICKS_SQL_CREDENTIAL_COLUMNS),
        )
        creds = r()
        assert creds.access_key_id == "AKIA"
        assert creds.secret_access_key == "shh"

    def test_per_call_columns_override_wins(self) -> None:
        client = _fake_db_client([{"my_key": "AK", "my_secret": "shh"}])
        r = DatabricksSQLCredentialsRefresher(
            query="Q",
            client=client,
            columns={
                "access_key_id": "my_key",
                "secret_access_key": "my_secret",
            },
            column_aliases=dict(AWSClient.DATABRICKS_SQL_CREDENTIAL_COLUMNS),
        )
        creds = r()
        assert creds.access_key_id == "AK"
        assert creds.secret_access_key == "shh"
        # session_token / expiration weren't in the row → optional, None.
        assert creds.session_token is None
        assert creds.expiration is None

    def test_optional_columns_missing_returns_none(self) -> None:
        # session_token and expiration are optional — missing them is
        # fine for long-lived static keys.
        client = _fake_db_client([{
            "access_key_id": "AK",
            "secret_access_key": "shh",
        }])
        r = DatabricksSQLCredentialsRefresher(
            query="Q",
            client=client,
            column_aliases=dict(AWSClient.DATABRICKS_SQL_CREDENTIAL_COLUMNS),
        )
        creds = r()
        assert creds.session_token is None
        assert creds.expiration is None

    def test_datetime_expiration_stringified(self) -> None:
        # Spark TIMESTAMP / Arrow timestamp columns surface as Python
        # datetimes via to_pylist(); botocore needs an ISO string.
        client = _fake_db_client([{
            "access_key_id": "AK",
            "secret_access_key": "shh",
            "expiration": datetime.datetime(2026, 12, 31, 23, 59, 59),
        }])
        r = DatabricksSQLCredentialsRefresher(
            query="Q",
            client=client,
            column_aliases=dict(AWSClient.DATABRICKS_SQL_CREDENTIAL_COLUMNS),
        )
        creds = r()
        assert creds.expiration == "2026-12-31T23:59:59"


# ---------------------------------------------------------------------------
# DatabricksSQLCredentialsRefresher — error paths
# ---------------------------------------------------------------------------


class TestErrorPaths:

    def test_empty_rows_raises_runtime_error(self) -> None:
        client = _fake_db_client([])
        r = DatabricksSQLCredentialsRefresher(
            query="SELECT * FROM ops.aws_creds WHERE role = 'reader'",
            client=client,
            column_aliases=dict(AWSClient.DATABRICKS_SQL_CREDENTIAL_COLUMNS),
        )
        with pytest.raises(RuntimeError, match="returned no rows"):
            r()

    def test_missing_required_column_raises_with_hint(self) -> None:
        client = _fake_db_client([{"foo": "bar"}])
        r = DatabricksSQLCredentialsRefresher(
            query="Q",
            client=client,
            column_aliases=dict(AWSClient.DATABRICKS_SQL_CREDENTIAL_COLUMNS),
        )
        with pytest.raises(KeyError) as excinfo:
            r()
        msg = str(excinfo.value)
        # Names the canonical field, the aliases tried, the available
        # columns, and the columns= override path.
        assert "access_key_id" in msg
        assert "AccessKeyId" in msg
        assert "foo" in msg
        assert "AWSClient.from_databricks_sql" in msg

    def test_explicit_override_missing_column_raises(self) -> None:
        # When the caller explicitly maps to a column that doesn't
        # exist in the result, the error names the override target.
        client = _fake_db_client([{"actual_column": "x"}])
        r = DatabricksSQLCredentialsRefresher(
            query="Q",
            client=client,
            columns={"access_key_id": "wrong_column"},
            column_aliases=dict(AWSClient.DATABRICKS_SQL_CREDENTIAL_COLUMNS),
        )
        with pytest.raises(KeyError, match="wrong_column"):
            r()


# ---------------------------------------------------------------------------
# Picklability
# ---------------------------------------------------------------------------


class TestPicklable:

    def test_refresher_picklable_without_client(self) -> None:
        # The cross-process shape: client=None, resolved lazily on
        # first refresh. Must round-trip cleanly through stdlib pickle
        # so Spark / multiprocessing can ship it without cloudpickle.
        r = DatabricksSQLCredentialsRefresher(
            query="SELECT * FROM ops.aws_creds",
            columns={"access_key_id": "k"},
            column_aliases=dict(AWSClient.DATABRICKS_SQL_CREDENTIAL_COLUMNS),
        )
        restored = pickle.loads(pickle.dumps(r))
        assert restored.query == r.query
        assert restored.columns == r.columns
        assert restored.column_aliases == r.column_aliases
        assert restored.client is None

    def test_no_closures_in_dataclass(self) -> None:
        # Plain dataclass slots, no inner functions. If anyone adds a
        # closure later this introspection will trip the next reader.
        r = DatabricksSQLCredentialsRefresher(query="Q")
        assert all(
            not callable(v) for v in r.__dict__.values() if v is not None
        ), f"non-None callable in refresher state: {r.__dict__!r}"


# ---------------------------------------------------------------------------
# AWSClient.from_databricks_sql wires up the refresher correctly
# ---------------------------------------------------------------------------


class TestFromDatabricksSql:

    def test_seed_and_refresher_installed(self) -> None:
        client = _fake_db_client([{
            "access_key_id": "AK_SEED",
            "secret_access_key": "secret_seed",
            "session_token": "tok_seed",
            "expiration": "2026-01-01T00:00:00Z",
        }])
        config = AWSClient.from_databricks_sql(
            "SELECT * FROM ops.aws_creds",
            client=client,
            region="us-east-1",
            endpoint_url="https://my-endpoint",
        )
        # The seed snapshot was taken from the first row.
        assert config.access_key_id == "AK_SEED"
        assert config.secret_access_key == "secret_seed"
        assert config.session_token == "tok_seed"
        # And the refresher is wired up so botocore can re-fetch later.
        assert config.has_refresher()
        assert isinstance(config.refresher, DatabricksSQLCredentialsRefresher)
        assert config.region == "us-east-1"
        assert config.endpoint_url == "https://my-endpoint"

    def test_columns_override_threads_through(self) -> None:
        client = _fake_db_client([{"k": "AK", "s": "shh"}])
        config = AWSClient.from_databricks_sql(
            "Q",
            client=client,
            columns={"access_key_id": "k", "secret_access_key": "s"},
        )
        assert config.access_key_id == "AK"
        assert config.secret_access_key == "shh"
        # The refresher carries the override forward for subsequent
        # botocore refreshes.
        assert isinstance(config.refresher, DatabricksSQLCredentialsRefresher)
        assert config.refresher.columns == {
            "access_key_id": "k", "secret_access_key": "s",
        }

    def test_refresh_metadata_round_trip(self) -> None:
        # AWSConfig.refresh_metadata() drives the end-to-end shape
        # botocore reads on a refresh cycle.
        client = _fake_db_client([{
            "access_key_id": "AK", "secret_access_key": "shh",
            "session_token": "t", "expiration": "2026-12-31T23:59:59Z",
        }])
        config = AWSClient.from_databricks_sql("Q", client=client)
        meta = config.refresh_metadata()
        assert meta == {
            "access_key": "AK",
            "secret_key": "shh",
            "token": "t",
            "expiry_time": "2026-12-31T23:59:59Z",
        }

    def test_config_aliases_can_be_subclassed(self) -> None:
        # The class-var is the extension point — a subclass that
        # overrides DATABRICKS_SQL_CREDENTIAL_COLUMNS gets its aliases
        # snapshotted into every refresher it builds.
        class MyAWSClient(AWSClient):
            DATABRICKS_SQL_CREDENTIAL_COLUMNS = {
                "access_key_id":     ("AK",),
                "secret_access_key": ("SK",),
                "session_token":     ("ST",),
                "expiration":        ("EXP",),
            }

        client = _fake_db_client([{
            "AK": "ak", "SK": "sk", "ST": "st", "EXP": "2027-01-01T00:00:00Z",
        }])
        config = MyAWSClient.from_databricks_sql("Q", client=client)
        assert config.access_key_id == "ak"
        assert config.secret_access_key == "sk"
        assert config.session_token == "st"
