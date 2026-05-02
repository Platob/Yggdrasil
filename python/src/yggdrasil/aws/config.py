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
from typing import Mapping, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .client import AWSClient


__all__ = [
    "AwsCredentials",
    "AWSConfig",
]


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


@dataclasses.dataclass
class AWSConfig:
    """Pure-data AWS session configuration.

    Holds every knob :class:`AWSClient` needs to mint a boto3
    :class:`Session`. Pickle-safe; no cached state. Equality and
    hashing follow the field set.

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
        **kwargs,
    ) -> "AWSConfig":
        """Build an AWSConfig wrapping a static :class:`AwsCredentials`."""
        return cls(
            access_key_id=creds.access_key_id,
            secret_access_key=creds.secret_access_key,
            session_token=creds.session_token,
            region=region,
            endpoint_url=endpoint_url,
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