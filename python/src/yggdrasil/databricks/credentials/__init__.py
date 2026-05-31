"""Unity Catalog credentials — resource + service (folder module).

* :class:`Credential` (``resource.py``) — one credential; turns an AWS IAM
  role into a refreshable :class:`~yggdrasil.aws.client.AWSClient`.
* :class:`Credentials` (``service.py``) — the workspace collection service,
  reachable as ``client.credentials`` (with :meth:`Credentials.create_aws`).
* :class:`DatabricksCredentialAwsProvider` (``provider.py``) — the refresh
  cycle backing the AWS client.

API: https://docs.databricks.com/api/workspace/credentials
"""
from __future__ import annotations

from .resource import Credential
from .service import Credentials
from .provider import DatabricksCredentialAwsProvider

__all__ = ["Credential", "Credentials", "DatabricksCredentialAwsProvider"]
