"""Connection-shape tests that don't open a real Postgres handle."""

from __future__ import annotations

import pytest

from yggdrasil.postgres.connection import (
    PostgresConnection,
    normalize_postgres_uri,
)


def test_normalize_postgres_uri_promotes_short_scheme() -> None:
    assert (
        normalize_postgres_uri("postgres://u:p@h:5432/db")
        == "postgresql://u:p@h:5432/db"
    )


def test_normalize_postgres_uri_passthrough() -> None:
    assert (
        normalize_postgres_uri("postgresql://u@h/db")
        == "postgresql://u@h/db"
    )


def test_normalize_postgres_uri_rejects_empty() -> None:
    with pytest.raises(ValueError):
        normalize_postgres_uri("")


def test_connection_repr_strips_password() -> None:
    conn = PostgresConnection("postgresql://user:secret@h/db")
    assert "secret" not in repr(conn)
    assert "***" in repr(conn)


def test_from_passes_through_existing_instance() -> None:
    conn = PostgresConnection("postgresql://h/db")
    assert PostgresConnection.from_(conn) is conn


def test_from_str_normalizes_scheme() -> None:
    conn = PostgresConnection.from_("postgres://h/db")
    assert conn.uri == "postgresql://h/db"


def test_from_mapping_forwards_kwargs() -> None:
    conn = PostgresConnection.from_({"uri": "postgresql://h/db", "autocommit": False})
    assert conn.autocommit is False


def test_missing_uri_raises_without_env(monkeypatch) -> None:
    monkeypatch.delenv("POSTGRES_URI", raising=False)
    with pytest.raises(ValueError):
        PostgresConnection()


def test_uses_env_when_unspecified(monkeypatch) -> None:
    monkeypatch.setenv("POSTGRES_URI", "postgres://h/from_env")
    conn = PostgresConnection()
    assert conn.uri == "postgresql://h/from_env"
