"""yggdrasil AWS integration.

Mirror of the Databricks-side organization:

- :class:`AWSClient` — top-level client. Owns both the static
  configuration (credentials, region, role, endpoint, SSO) and the
  lazy boto :class:`Session`. Singleton-cached per identity-bearing
  init kwarg.
- :class:`AWSService` — abstract base for service objects (one
  per AWS service: S3, DynamoDB, ...).
- :class:`AWSResource` — abstract base for individual entities
  (an S3 object, a DynamoDB row).
- :class:`AwsCredentials` — wire-format credentials record (STS /
  Databricks Storage Credentials shape).

Filesystem
----------

- :class:`S3Service` (in :mod:`yggdrasil.aws.fs.service`) — thin
  S3 service object reachable as ``client.s3``.
- :class:`S3Path` (in :mod:`yggdrasil.aws.fs.path`) — :class:`Path`
  subclass over S3, registered for the ``s3://`` / ``s3a://`` /
  ``s3n://`` URL schemes.

Quick start
-----------

    >>> from yggdrasil.aws import AWSClient
    >>> from yggdrasil.aws.fs.path import S3Path
    >>>
    >>> # Default chain (env / profile / instance metadata):
    >>> p = S3Path("s3://my-bucket/data.parquet")
    >>>
    >>> # Explicit role:
    >>> client = AWSClient(
    ...     role_arn="arn:aws:iam::1234:role/Reader",
    ...     region="us-east-1",
    ... )
    >>> p = client.s3.path("s3://my-bucket/data.parquet")
    >>>
    >>> # IAM Identity Center (SSO) with external browser:
    >>> client = AWSClient(
    ...     sso_start_url="https://example.awsapps.com/start",
    ...     sso_region="us-east-1",
    ...     sso_account_id="123456789012",
    ...     sso_role_name="DataReader",
    ... )
"""

from __future__ import annotations

from .account import AWSAccount, AccountService
from .batch import AWSBatchEnvironment
from .client import AWSClient, AWSResource, AWSService
from .config import AwsCredentials, DatabricksSQLCredentialsRefresher
from .provider import AwsCredentialsProvider


__all__ = [
    "AWSClient",
    "AWSService",
    "AWSResource",
    "AWSAccount",
    "AccountService",
    "AWSBatchEnvironment",
    "AwsCredentials",
    "AwsCredentialsProvider",
    "DatabricksSQLCredentialsRefresher",
]
