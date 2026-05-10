"""AWS configuration and credential dataclasses.

Pure data: this module holds no behavior beyond construction and
field access. Session-building, boto client minting, and STS
assume-role refresh all live on :class:`AWSClient` in :mod:`.client`.

The split mirrors yggdrasil's broader pattern (Databricks puts
fields on ``Config``, behavior on ``DatabricksClient``):

- :class:`AwsCredentials` is the wire-format AWS API record. Used
  whenever a service hands you a creds block (Databricks Storage
  Credentials, STS, etc.); flows through :meth:`AWSConfig.from_credentials`
  to construct an :class:`AWSConfig`.

- :class:`AWSConfig` is the union of every knob :class:`AWSClient`
  needs to build a boto3 :class:`Session`: static creds, region,
  profile, optional assume-role parameters, optional endpoint
  override (for MinIO / R2 / Ceph), and S3 addressing style.

Defaults are env-driven so a blank ``AWSConfig()`` matches what
boto3 would do walking the default credential chain.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any, Callable, Mapping, Optional, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from .client import AWSClient


__all__ = [
    "AwsCredentials",
    "AWSConfig",
    "CredentialsRefresher",
]


# Refresher callable: returns a fresh :class:`AwsCredentials` (or the
# botocore metadata dict — both shapes are accepted for ergonomics).
CredentialsRefresher = Callable[[], Union["AwsCredentials", Mapping[str, Any]]]


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
# AWSConfig — pure-data session config
# ---------------------------------------------------------------------------


def _env(name: str) -> Optional[str]:
    """Return env var if set and non-empty, else ``None``."""
    value = os.environ.get(name)
    return value if value else None


def _env_factory(name: str):
    """Default-factory for env-backed dataclass fields."""
    def factory() -> Optional[str]:
        return _env(name)
    return factory


@dataclasses.dataclass(unsafe_hash=True)
class AWSConfig:
    """Pure-data AWS session configuration.

    Holds every knob :class:`AWSClient` needs to mint a boto3
    :class:`Session`. Pickle-safe; no cached state. Equality and
    hashing follow the field set — :attr:`refresher` is excluded
    from both (callables aren't comparable), so two configs that
    differ only in their refresher callback collapse to the same
    :class:`AWSClient` singleton.

    Construction shapes:

    - **Static credentials**: pass ``access_key_id`` /
      ``secret_access_key`` / optional ``session_token``.
    - **Profile**: pass ``profile`` (matches ``AWS_PROFILE``); the
      session resolves through ``~/.aws/credentials``.
    - **Assume-role**: pass ``role_arn``, optionally with
      ``role_session_name`` / ``external_id`` / ``duration_seconds``.
      :class:`AWSClient` builds a refreshable credential provider
      that calls STS AssumeRole on demand.
    - **Default chain**: pass nothing. boto3 walks env →
      profile → instance metadata → SSO.
    """

    # --- Base creds ---------------------------------------------------------

    access_key_id: Optional[str] = dataclasses.field(
        default_factory=_env_factory("AWS_ACCESS_KEY_ID"),
    )
    secret_access_key: Optional[str] = dataclasses.field(
        default_factory=_env_factory("AWS_SECRET_ACCESS_KEY"),
        repr=False,
    )
    session_token: Optional[str] = dataclasses.field(
        default_factory=_env_factory("AWS_SESSION_TOKEN"),
        repr=False,
    )

    # --- Region / profile ---------------------------------------------------

    region: Optional[str] = dataclasses.field(
        default_factory=lambda: (
            _env("AWS_REGION") or _env("AWS_DEFAULT_REGION")
        ),
    )
    profile: Optional[str] = dataclasses.field(
        default_factory=_env_factory("AWS_PROFILE"),
    )

    # --- Assume-role --------------------------------------------------------

    role_arn: Optional[str] = None
    role_session_name: Optional[str] = None
    external_id: Optional[str] = None
    duration_seconds: int = 3600
    """STS AssumeRole token lifetime. Botocore refreshes ~5 min before
    this expires, so 1h gives ~55min of usable token per cycle."""

    # --- Endpoint / behaviour overrides ------------------------------------

    endpoint_url: Optional[str] = dataclasses.field(
        default_factory=_env_factory("AWS_ENDPOINT_URL"),
    )
    """For S3-compatible stores (MinIO, R2, Ceph). ``None`` lets boto
    talk to the real AWS endpoint."""

    s3_addressing_style: Optional[str] = None
    """``"path"`` or ``"virtual"``. ``None`` lets boto pick (virtual
    by default for AWS, path for many S3-compatibles)."""

    # --- External credential refresher --------------------------------------

    refresher: Optional[CredentialsRefresher] = dataclasses.field(
        default=None, repr=False, compare=False, hash=False,
    )
    """Callable that returns fresh credentials before they expire.

    Set this when the credentials come from a vending service that
    keeps minting them — Databricks ``temporary_path_credentials`` /
    ``temporary_table_credentials`` / external STS broker — so an
    :class:`AWSClient` built from this config can drive a
    botocore :class:`RefreshableCredentials`-backed session that
    survives token rotation.

    Two accepted return shapes:

    - :class:`AwsCredentials` — the canonical Yggdrasil record.
    - ``Mapping`` matching botocore's metadata
      (``{"access_key", "secret_key", "token", "expiry_time"}``).

    The callable is invoked from inside botocore's refresh hook
    (~5 min before token expiry). It must be idempotent and
    thread-safe; botocore serializes calls but a runaway refresh
    that hangs blocks every signing thread.

    Excluded from equality / hashing: callables aren't comparable,
    and two configs vending creds via different refreshers are still
    "the same" config from a downstream identity standpoint.
    """

    # ------------------------------------------------------------------
    # Coercion entry points
    # ------------------------------------------------------------------

    @classmethod
    def from_credentials(
        cls,
        creds: AwsCredentials,
        *,
        region: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        refresher: Optional[CredentialsRefresher] = None,
        **kwargs,
    ) -> "AWSConfig":
        """Build an AWSConfig wrapping a static :class:`AwsCredentials`.

        Pass ``refresher`` for self-renewing temporary credentials.
        See :attr:`AWSConfig.refresher`.
        """
        return cls(
            access_key_id=creds.access_key_id,
            secret_access_key=creds.secret_access_key,
            session_token=creds.session_token,
            region=region,
            endpoint_url=endpoint_url,
            refresher=refresher,
            **kwargs,
        )

    @classmethod
    def from_refresher(
        cls,
        refresher: CredentialsRefresher,
        *,
        region: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        **kwargs,
    ) -> "AWSConfig":
        """Build a self-refreshing AWSConfig from a credentials callback.

        The first :meth:`to_client` reads a seed snapshot by invoking
        *refresher* once; subsequent botocore refresh cycles re-invoke
        it. Useful when credentials come from a vending service::

            def vend() -> AwsCredentials:
                resp = volume.temporary_credentials(operation=op)
                aws = resp.aws_temp_credentials
                return AwsCredentials(
                    access_key_id=aws.access_key_id,
                    secret_access_key=aws.secret_access_key,
                    session_token=aws.session_token,
                    expiration=resp.expiration_time.isoformat(),
                )

            client = AWSConfig.from_refresher(vend, region="eu-central-1").to_client()
            client.s3_client().list_buckets()  # creds auto-refreshed
        """
        seed = _coerce_refresher_output(refresher())
        if isinstance(seed, AwsCredentials):
            access_key_id = seed.access_key_id
            secret_access_key = seed.secret_access_key
            session_token = seed.session_token
        else:
            access_key_id = seed.get("access_key")
            secret_access_key = seed.get("secret_key")
            session_token = seed.get("token")

        return cls(
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            session_token=session_token,
            region=region,
            endpoint_url=endpoint_url,
            refresher=refresher,
            **kwargs,
        )

    def to_credentials(self) -> AwsCredentials:
        """Snapshot the static credentials into an :class:`AwsCredentials`.

        Returns the configured static fields; does NOT materialize
        assumed-role tokens. For an assume-role config, exporting
        the live STS token would defeat the auto-refresh that's the
        whole point of using a role.
        """
        return AwsCredentials(
            access_key_id=self.access_key_id,
            secret_access_key=self.secret_access_key,
            session_token=self.session_token,
        )

    def to_client(self) -> "AWSClient":
        from .client import AWSClient

        return AWSClient(config=self)

    # ------------------------------------------------------------------
    # Inspection helpers used by AWSClient
    # ------------------------------------------------------------------

    def has_assume_role(self) -> bool:
        return bool(self.role_arn)

    def has_static_credentials(self) -> bool:
        return bool(self.access_key_id and self.secret_access_key)

    def has_refresher(self) -> bool:
        """True iff a :attr:`refresher` callback is wired up.

        Drives :meth:`AWSClient._build_session` to mint a
        :class:`RefreshableCredentials`-backed session instead of a
        static one.
        """
        return self.refresher is not None

    def refresh_metadata(self) -> Mapping[str, Any]:
        """Invoke :attr:`refresher` and return botocore-shaped metadata.

        Raises :class:`RuntimeError` when no refresher is set.
        """
        if self.refresher is None:
            raise RuntimeError(
                "AWSConfig.refresh_metadata() requires a refresher; "
                "none is set. Build the config via "
                "AWSConfig.from_refresher(...) or assign config.refresher."
            )
        return _refresher_to_metadata(self.refresher)


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
        "AWSConfig refresher must return an AwsCredentials or a "
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