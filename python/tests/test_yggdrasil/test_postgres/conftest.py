"""
Pytest configuration for Postgres integration tests.

Integration tests are gated on the ``POSTGRES_URI`` environment variable.
When the variable is absent the entire integration suite is skipped automatically.

Usage
-----
Set ``POSTGRES_URI`` before running::

    export POSTGRES_URI=postgresql://postgres:postgres@localhost:5432/postgres
    pytest tests/test_yggdrasil/test_postgres/

To run *only* the offline unit tests (no live Postgres required)::

    pytest tests/test_yggdrasil/test_postgres/ -m "not postgres_integration"

Apply ``pytestmark = pytest.mark.postgres_integration`` to every module whose
tests touch a real Postgres instance.  :class:`PostgresTestCase` already raises
``SkipTest`` when the URI is missing or the optional drivers are absent — the
marker is layered on top so callers can use ``-m`` filters without invoking
the per-class setup.
"""

from __future__ import annotations

import os
from typing import Optional

import pytest

from yggdrasil.io.url import URL

__all__ = ["POSTGRES_URI", "requires_postgres"]


def _load_postgres_uri() -> Optional[URL]:
    """Read ``POSTGRES_URI`` and parse it through :class:`URL`.

    Centralising URI parsing on the project's :class:`URL` lets the
    fixtures inspect ``host`` / ``port`` / ``scheme`` consistently
    without re-implementing :mod:`urllib`-style parsing per test.
    Returns ``None`` when the env var is unset so the skip layer can
    branch on identity rather than on a magic empty string.
    """
    raw = os.environ.get("POSTGRES_URI", "")
    if not raw:
        return None
    return URL.from_(raw)


#: The parsed Postgres URI, or ``None`` when ``POSTGRES_URI`` is unset.
#: Normalised via :class:`URL` so callers see a canonical
#: ``postgresql://`` form regardless of the input scheme.
POSTGRES_URI: Optional[URL] = _load_postgres_uri()

_SKIP_REASON = (
    "Postgres integration tests require POSTGRES_URI to be set in the "
    "environment.  Example: "
    "POSTGRES_URI=postgresql://postgres:postgres@localhost:5432/postgres"
)

#: Apply to any module / class whose tests require a live Postgres instance::
#:
#:     pytestmark = requires_postgres
#:
#: When ``POSTGRES_URI`` is absent every test in that module is skipped with a
#: clear, actionable message.
requires_postgres = pytest.mark.skipif(
    POSTGRES_URI is None,
    reason=_SKIP_REASON,
)


@pytest.fixture(scope="session")
def postgres_uri() -> URL:
    """Return the parsed :class:`URL`, or skip when ``POSTGRES_URI`` is unset."""
    if POSTGRES_URI is None:
        pytest.skip(_SKIP_REASON)
    return POSTGRES_URI


@pytest.fixture(scope="session")
def postgres_engine(postgres_uri: URL):
    """A session-scoped :class:`PostgresEngine` connected to ``POSTGRES_URI``.

    Skips automatically when the optional drivers are missing or the
    server is unreachable.  Closed at session teardown.
    """
    from yggdrasil.postgres.lib import has_psycopg

    if not has_psycopg():
        pytest.skip(
            "psycopg (psycopg 3) is not installed; install with "
            "`pip install ygg[postgres]`."
        )

    from yggdrasil.postgres import PostgresEngine

    engine = PostgresEngine(str(postgres_uri))
    try:
        # Lightweight reachability probe — bad URI / unreachable server / wrong
        # credentials all fail here with the actual driver error.
        cursor = engine.connection.psycopg_cursor()
        try:
            cursor.execute("SELECT 1")
            cursor.fetchone()
        finally:
            cursor.close()
    except Exception as exc:
        engine.connection.close()
        pytest.skip(f"Postgres instance not reachable: {exc}")
    yield engine
    engine.connection.close()
