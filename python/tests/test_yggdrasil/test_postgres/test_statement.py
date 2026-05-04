"""Unit tests for the Postgres statement / batch coercion."""

from __future__ import annotations

from yggdrasil.postgres.statement import (
    POSTGRES_STATEMENT_MIME,
    POSTGRES_TABLE_MIME,
    PostgresPreparedStatement,
    PostgresStatementBatch,
)


class TestPreparedStatement:
    def test_from_str(self) -> None:
        stmt = PostgresPreparedStatement.from_("SELECT 1")
        assert stmt.text == "SELECT 1"
        assert stmt.parameters is None
        assert stmt.prefer_arrow is True

    def test_from_self_is_passthrough(self) -> None:
        stmt = PostgresPreparedStatement("SELECT 1")
        assert PostgresPreparedStatement.from_(stmt) is stmt

    def test_prepare_attaches_parameters(self) -> None:
        stmt = PostgresPreparedStatement.prepare(
            "SELECT * FROM t WHERE id = %s",
            parameters=(42,),
        )
        assert stmt.parameters == (42,)
        assert stmt.text == "SELECT * FROM t WHERE id = %s"

    def test_prepare_routes_scope(self) -> None:
        stmt = PostgresPreparedStatement.prepare(
            "SELECT 1",
            schema_name="s",
            catalog_name="db",
        )
        assert stmt.schema_name == "s"
        assert stmt.catalog_name == "db"

    def test_prepare_disables_arrow(self) -> None:
        stmt = PostgresPreparedStatement.prepare("DROP TABLE t", prefer_arrow=False)
        assert stmt.prefer_arrow is False


class TestStatementBatch:
    def test_coerces_strings(self) -> None:
        batch = PostgresStatementBatch(executor=_Stub())
        batch.add("SELECT 1")
        # ``statements`` is a deque of typed instances after coerce.
        stmt = batch.statements[0]
        assert isinstance(stmt, PostgresPreparedStatement)
        assert stmt.text == "SELECT 1"

    def test_keeps_typed_instances(self) -> None:
        batch = PostgresStatementBatch(executor=_Stub())
        existing = PostgresPreparedStatement("SELECT 2")
        batch.add(existing)
        assert batch.statements[0] is existing


class TestMimeTypes:
    def test_statement_mime_unique(self) -> None:
        assert POSTGRES_STATEMENT_MIME != POSTGRES_TABLE_MIME

    def test_statement_mime_identifies_postgres(self) -> None:
        assert "postgres" in POSTGRES_STATEMENT_MIME.value
        assert "postgres" in POSTGRES_TABLE_MIME.value


class _Stub:
    """Minimal executor stand-in for batch-level coercion tests."""

    def execute(self, *args, **kwargs):  # pragma: no cover — never called
        raise NotImplementedError
