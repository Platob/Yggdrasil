"""Pure-unit tests for the Postgres SQL string helpers."""

from __future__ import annotations

import datetime as dt
import decimal as dec

from yggdrasil.postgres.sql_utils import (
    escape_sql_string,
    parse_dotted_name,
    quote_ident,
    quote_qualified_ident,
    split_qualified_ident,
    sql_literal,
)


class TestQuoting:
    def test_quote_ident_basic(self) -> None:
        assert quote_ident("users") == '"users"'

    def test_quote_ident_escapes_embedded_quote(self) -> None:
        assert quote_ident('weird"name') == '"weird""name"'

    def test_quote_qualified_ident_dotted_string(self) -> None:
        assert quote_qualified_ident("public.users") == '"public"."users"'

    def test_quote_qualified_ident_three_parts(self) -> None:
        assert (
            quote_qualified_ident("mydb.public.users")
            == '"mydb"."public"."users"'
        )

    def test_quote_qualified_ident_list(self) -> None:
        assert (
            quote_qualified_ident(["public", "weird.name"])
            == '"public"."weird.name"'
        )

    def test_split_qualified_ident_handles_quoted_dot(self) -> None:
        # Quoted segment preserves embedded dots.
        parts = split_qualified_ident('"my.db".public.t')
        assert parts == ["my.db", "public", "t"]


class TestParseDottedName:
    def test_three_part(self) -> None:
        c, s, t = parse_dotted_name("db.public.users")
        assert (c, s, t) == ("db", "public", "users")

    def test_two_part(self) -> None:
        c, s, t = parse_dotted_name("public.users")
        assert (c, s, t) == (None, "public", "users")

    def test_single_part(self) -> None:
        c, s, t = parse_dotted_name("users")
        assert (c, s, t) == (None, None, "users")

    def test_overrides_win(self) -> None:
        c, s, t = parse_dotted_name(
            "users",
            schema_name="explicit",
            catalog_name="explicit_db",
        )
        assert (c, s, t) == ("explicit_db", "explicit", "users")

    def test_too_many_parts(self) -> None:
        import pytest
        with pytest.raises(ValueError):
            parse_dotted_name("a.b.c.d")


class TestSqlLiteral:
    def test_none(self) -> None:
        assert sql_literal(None) == "NULL"

    def test_bool(self) -> None:
        assert sql_literal(True) == "TRUE"
        assert sql_literal(False) == "FALSE"

    def test_int(self) -> None:
        assert sql_literal(42) == "42"

    def test_decimal(self) -> None:
        assert sql_literal(dec.Decimal("1.50")) == "1.50"

    def test_date(self) -> None:
        assert sql_literal(dt.date(2024, 1, 2)) == "DATE '2024-01-02'"

    def test_datetime(self) -> None:
        out = sql_literal(dt.datetime(2024, 1, 2, 3, 4, 5))
        assert out == "TIMESTAMP '2024-01-02 03:04:05'"

    def test_string_escapes_quote(self) -> None:
        assert sql_literal("O'Brien") == "'O''Brien'"

    def test_bytes(self) -> None:
        assert sql_literal(b"\x00\xff") == "'\\x00ff'"

    def test_escape_sql_string(self) -> None:
        assert escape_sql_string("O'Brien") == "O''Brien"
