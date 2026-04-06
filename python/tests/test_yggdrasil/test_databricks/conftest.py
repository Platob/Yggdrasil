"""
Pytest configuration for Databricks integration tests.

Integration tests are gated on the ``DATABRICKS_HOST`` environment variable.
When the variable is absent the entire integration suite is skipped automatically.

Usage
-----
Set ``DATABRICKS_HOST`` (and usually ``DATABRICKS_TOKEN``) before running::

    export DATABRICKS_HOST=https://dbc-82edd6f4-1e97.cloud.databricks.com/
    export DATABRICKS_TOKEN=dapiXXXXXXXXXXXXXXXXXX
    pytest tests/test_yggdrasil/test_databricks/

To run *only* the offline unit tests (no live workspace required)::

    pytest tests/test_yggdrasil/test_databricks/ -m "not integration"

Apply ``pytestmark = requires_databricks`` (or ``@pytest.mark.integration``) to
every module / class whose tests touch a real Databricks workspace.
"""

from __future__ import annotations

import os

import pytest

from ._base import DatabricksCase

__all__ = ["DatabricksCase", "DATABRICKS_HOST", "requires_databricks"]

# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------

#: The Databricks workspace URL read from the environment.
#: Example value used by this project: https://dbc-82edd6f4-1e97.cloud.databricks.com/
DATABRICKS_HOST: str = os.environ.get("DATABRICKS_HOST", "")

_SKIP_REASON = (
    "Integration tests require DATABRICKS_HOST to be set in the environment. "
    "Example: DATABRICKS_HOST=https://dbc-82edd6f4-1e97.cloud.databricks.com/"
)

# ---------------------------------------------------------------------------
# Shared marker
# ---------------------------------------------------------------------------

#: Apply to any module / class whose tests require a live Databricks workspace::
#:
#:     pytestmark = requires_databricks
#:
#: When DATABRICKS_HOST is absent every test in that module is skipped with a
#: clear, actionable message.
requires_databricks = pytest.mark.skipif(
    not DATABRICKS_HOST,
    reason=_SKIP_REASON,
)


# ---------------------------------------------------------------------------
# Marker registration hook
# ---------------------------------------------------------------------------

def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: marks tests that require a live Databricks workspace "
        "(skipped when DATABRICKS_HOST is not set)",
    )


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def databricks_host() -> str:
    """Return the Databricks host URL, or skip the session if not configured."""
    if not DATABRICKS_HOST:
        pytest.skip(_SKIP_REASON)
    return DATABRICKS_HOST


@pytest.fixture(scope="session")
def databricks_client(databricks_host: str):
    """
    Return a connected :class:`~yggdrasil.databricks.client.DatabricksClient`.

    Skips automatically if the workspace is unreachable (e.g. missing token).
    """
    from yggdrasil.databricks.client import DatabricksClient

    client = DatabricksClient(host=databricks_host)
    try:
        client.workspace_client().current_user.me()
    except Exception as exc:
        pytest.skip(f"Databricks workspace not reachable: {exc}")
    return client


@pytest.fixture(scope="session")
def databricks_workspace(databricks_host: str):
    """
    Return a connected :class:`~yggdrasil.databricks.workspaces.Workspace`.

    Skips automatically if the workspace is unreachable.
    """
    from yggdrasil.databricks.workspaces.workspace import Workspace

    workspace = Workspace(host=databricks_host)
    try:
        workspace.workspace_client().current_user.me()
    except Exception as exc:
        pytest.skip(f"Databricks workspace not reachable: {exc}")
    return workspace

