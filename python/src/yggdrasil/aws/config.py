"""AWS wire-format credential records.

The :class:`AWSClient` in :mod:`.client` owns both the configuration
*and* the behavior — there is no separate ``AWSConfig`` class. This
module holds the passive wire records that flow into the client and
the refresher callable type.

- :class:`AwsCredentials` mirrors AWS STS's ``Credentials`` shape
  and is what services (Databricks Storage Credentials, STS, etc.)
  hand back. Convert into an :class:`AWSClient` via
  :meth:`AWSClient.from_credentials`.

- :class:`DatabricksSQLCredentialsRefresher` is a picklable refresher
  that re-runs a Databricks SQL query for fresh creds; wired up by
  :meth:`AWSClient.from_databricks_sql`.

- :data:`CredentialsRefresher` is the callable type for vended-creds
  flows — anything returning :class:`AwsCredentials` or a botocore
  metadata mapping.
"""

from __future__ import annotations

import dataclasses
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from yggdrasil.databricks.client import DatabricksClient


__all__ = [
    "AwsCredentials",
    "CredentialsRefresher",
    "DatabricksSQLCredentialsRefresher",
    "DATABRICKS_SQL_CREDENTIAL_COLUMNS",
]


# Refresher callable: returns a fresh :class:`AwsCredentials` (or the
# botocore metadata dict — both shapes are accepted for ergonomics).
CredentialsRefresher = Callable[[], Union["AwsCredentials", Mapping[str, Any]]]


# Default column-name aliases for credential rows pulled from a
# Databricks SQL table. Each canonical AwsCredentials field maps
# to the ordered tuple of column names the refresher will try in
# turn — first match wins. Lowercase Python casing comes first so
# the common "ops.aws_creds" shape works without a per-call
# ``columns=`` override; the AWS API JSON casing is the fallback
# for tables that mirror the STS response shape directly.
DATABRICKS_SQL_CREDENTIAL_COLUMNS: Mapping[str, tuple[str, ...]] = MappingProxyType({
    "access_key_id":     ("access_key_id", "AccessKeyId"),
    "secret_access_key": ("secret_access_key", "SecretAccessKey"),
    "session_token":     ("session_token", "SessionToken"),
    "expiration":        ("expiration", "Expiration"),
})


# ---------------------------------------------------------------------------
# AwsCredentials — passive wire record
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class AwsCredentials:
    """AWS temporary credentials for API authentication.

    Mirrors AWS STS's ``Credentials`` shape and the equivalent
    Databricks-issued temporary credentials.
    See https://docs.aws.amazon.com/STS/latest/APIReference/API_Credentials.html.
    """

    access_key_id: Optional[str] = None
    """The access key ID that identifies the temporary credentials."""

    access_point: Optional[str] = None
    """The Amazon Resource Name (ARN) of the S3 access point for
    temporary credentials related to the external location."""

    secret_access_key: Optional[str] = None
    """The secret access key that can be used to sign AWS API
    requests."""

    session_token: Optional[str] = None
    """The token that users must pass to AWS API to use the
    temporary credentials."""

    expiration: Optional[str] = None
    """ISO-8601 timestamp at which the credentials expire. Optional
    on construction; STS-issued creds always have one, long-lived
    keys don't."""

    def is_complete(self) -> bool:
        """True iff at least access_key_id + secret_access_key are set."""
        return bool(self.access_key_id and self.secret_access_key)

    def to_botocore_metadata(self) -> Mapping[str, Optional[str]]:
        """Render as the metadata dict botocore's RefreshableCredentials
        expects.

        Used by :class:`AWSClient` when seeding refreshable creds
        from an initial static :class:`AwsCredentials` snapshot.
        """
        return {
            "access_key": self.access_key_id,
            "secret_key": self.secret_access_key,
            "token": self.session_token,
            "expiry_time": self.expiration,
        }


# ---------------------------------------------------------------------------
# Refresher adapters
# ---------------------------------------------------------------------------


def _coerce_refresher_output(
    obj: Union[AwsCredentials, Mapping[str, Any]],
) -> Union[AwsCredentials, Mapping[str, Any]]:
    """Normalize a refresher's return value into one of the two
    accepted shapes; raise :class:`TypeError` otherwise."""
    if isinstance(obj, AwsCredentials):
        return obj
    if isinstance(obj, Mapping):
        return obj
    raise TypeError(
        "AWSClient refresher must return an AwsCredentials or a "
        "Mapping with access_key/secret_key/token/expiry_time keys; "
        f"got {type(obj).__name__}."
    )


def _refresher_to_metadata(
    refresher: CredentialsRefresher,
) -> Mapping[str, Any]:
    """Adapter: invoke *refresher* and return a botocore metadata dict.

    Accepts both the canonical :class:`AwsCredentials` shape and the
    raw mapping shape so callers don't have to wrap themselves.
    """
    raw = _coerce_refresher_output(refresher())
    if isinstance(raw, AwsCredentials):
        return dict(raw.to_botocore_metadata())
    return dict(raw)


# ---------------------------------------------------------------------------
# DatabricksSQLCredentialsRefresher — picklable refresher for SQL-vended creds
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class DatabricksSQLCredentialsRefresher:
    """Refresher that re-runs a Databricks SQL query for fresh AWS credentials.

    Picklable by design — closure-free so botocore can hand the
    refresher to a Spark worker, a multiprocessing pool, or a
    cross-process job runner without cloudpickle. Serialization
    follows the project rule: the dataclass body holds plain
    picklable fields, the :class:`DatabricksClient` (when explicitly
    set) goes through its own URL-based round-trip, and the lazy
    ``DatabricksClient.current()`` fallback is resolved inside
    :meth:`__call__` so the refresher stays constructible at
    module-import time without a live workspace.

    Mapped to :class:`AwsCredentials` via *column_aliases* — each
    canonical field tries the user-provided ``columns`` override
    first, then walks the alias tuple in order. Per-call ``columns``
    overrides win over the default
    :data:`DATABRICKS_SQL_CREDENTIAL_COLUMNS`.
    """

    query: str
    """SQL query that returns at least one row of credentials."""

    client: Optional["DatabricksClient"] = None
    """Workspace to query. ``None`` resolves to
    :meth:`DatabricksClient.current` lazily on first call."""

    columns: Optional[dict[str, str]] = None
    """Per-call column-name overrides keyed by canonical field
    (``access_key_id`` / ``secret_access_key`` / ``session_token`` /
    ``expiration``). Wins over :attr:`column_aliases`."""

    column_aliases: dict[str, tuple[str, ...]] = dataclasses.field(
        default_factory=dict,
    )
    """Fallback alias tuples per canonical field. Populated from
    :data:`DATABRICKS_SQL_CREDENTIAL_COLUMNS` by
    :meth:`AWSClient.from_databricks_sql`; pre-snapshotted so the
    refresher survives a cross-process pickle without depending on
    the receiver's class-var defaults."""

    def __call__(self) -> AwsCredentials:
        # Lazy import: DatabricksClient pulls in the databricks-sdk
        # optional dep, plus the SQL stack. Keep the import inside
        # __call__ — and skip it entirely when the caller supplied a
        # client — so a pure-AWS install can construct (and even
        # pickle) the refresher without the dep installed.
        client = self.client
        if client is None:
            from yggdrasil.databricks.client import DatabricksClient
            client = DatabricksClient.current()
        rows = client.sql.execute(self.query).read_pylist()
        if not rows:
            raise RuntimeError(
                "DatabricksSQLCredentialsRefresher returned no rows. "
                f"Query: {self.query!r}. Confirm the table is populated "
                "and the WHERE clause matches an existing row."
            )

        row = rows[0]
        return AwsCredentials(
            access_key_id=self._lookup(row, "access_key_id", required=True),
            secret_access_key=self._lookup(row, "secret_access_key", required=True),
            session_token=self._lookup(row, "session_token"),
            expiration=self._stringify(self._lookup(row, "expiration")),
        )

    def _lookup(
        self,
        row: Mapping[str, Any],
        canonical: str,
        *,
        required: bool = False,
    ) -> Any:
        """Resolve *canonical* against the row using overrides → aliases."""
        if self.columns and canonical in self.columns:
            name = self.columns[canonical]
            if name in row:
                return row[name]
            if required:
                raise KeyError(
                    f"DatabricksSQLCredentialsRefresher: column "
                    f"{name!r} (override for {canonical!r}) is missing "
                    f"from the SQL row. Available: {list(row)!r}."
                )
            return None

        for name in self.column_aliases.get(canonical, (canonical,)):
            if name in row:
                return row[name]

        if required:
            tried = self.column_aliases.get(canonical, (canonical,))
            raise KeyError(
                f"DatabricksSQLCredentialsRefresher: no column for "
                f"{canonical!r} in SQL row. Tried {list(tried)!r}; "
                f"available columns: {list(row)!r}. Pass "
                f"columns={{{canonical!r}: '<your-column>'}} to "
                f"AWSClient.from_databricks_sql to override."
            )
        return None

    @staticmethod
    def _stringify(value: Any) -> Optional[str]:
        """Coerce an expiration value to ISO-8601 string."""
        if value is None:
            return None
        if isinstance(value, str):
            return value
        isoformat = getattr(value, "isoformat", None)
        if callable(isoformat):
            return isoformat()
        return str(value)
