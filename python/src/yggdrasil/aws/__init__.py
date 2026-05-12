"""yggdrasil AWS integration.

Mirror of the Databricks-side organization:

- :class:`AWSClient` — top-level client, owns boto session +
  credential refresh.
- :class:`AWSService` — abstract base for service objects (one
  per AWS service: S3, DynamoDB, ...).
- :class:`AWSResource` — abstract base for individual entities
  (an S3 object, a DynamoDB row).
- :class:`AWSConfig` / :class:`AwsCredentials` — pure-data
  configuration shapes.

Filesystem
----------

- :class:`S3Service` (in :mod:`yggdrasil.aws.fs.service`) — thin
  S3 service object reachable as ``client.s3``.
- :class:`S3Path` (in :mod:`yggdrasil.aws.fs.path`) — :class:`Path`
  subclass over S3, registered for the ``s3://`` / ``s3a://`` /
  ``s3n://`` URL schemes.

Quick start
-----------

    >>> from yggdrasil.aws import AWSClient, AWSConfig
    >>> from yggdrasil.aws.fs.path import S3Path
    >>>
    >>> # Default chain (env / profile / instance metadata):
    >>> p = S3Path("s3://my-bucket/data.parquet")
    >>>
    >>> # Explicit role:
    >>> client = AWSClient(AWSConfig(
    ...     role_arn="arn:aws:iam::1234:role/Reader",
    ...     region="us-east-1",
    ... ))
    >>> p = client.s3.path("s3://my-bucket/data.parquet")
"""

from __future__ import annotations

from .client import AWSClient, AWSResource, AWSService
from .config import AWSConfig, AwsCredentials, DatabricksSQLCredentialsRefresher
from .provider import AwsCredentialsProvider


__all__ = [
    "AWSClient",
    "AWSService",
    "AWSResource",
    "AWSConfig",
    "AwsCredentials",
    "AwsCredentialsProvider",
    "DatabricksSQLCredentialsRefresher",
]