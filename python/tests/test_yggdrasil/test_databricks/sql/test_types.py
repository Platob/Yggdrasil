"""Unit tests for :mod:`yggdrasil.databricks.sql.types`."""

from __future__ import annotations

import pytest

from yggdrasil.data.schema import schema as make_schema
from yggdrasil.data.data_field import Field
from yggdrasil.data.types.primitive import IntegerType, StringType
from yggdrasil.databricks.sql.sql_utils import (
    _build_fk_constraint_sql,
    _parse_fk_ref,
    _qualify_fk_ref,
)
from yggdrasil.databricks.sql.types import ForeignKeySpec, PrimaryKeySpec


def _int_field(
    name: str,
    *,
    tags: dict[str, str] | None = None,
) -> Field:
    return Field(
        name=name,
        dtype=IntegerType(byte_size=8, signed=True),
        nullable=True,
        tags=tags,
    )


def _str_field(name: str) -> Field:
    return Field(name=name, dtype=StringType(), nullable=True)


def test_foreign_key_spec_from_schema_reads_field_tags():
    s = make_schema(
        [
            _int_field("id", tags={"primary_key": "true"}),
            _int_field("parent_id", tags={"foreign_key": "dim_parent.id"}),
            _int_field("org_id", tags={"foreign_key": "main.ref.orgs.id"}),
            _str_field("name"),
        ]
    )

    specs = ForeignKeySpec.from_schema(s)

    assert [(f.column, f.ref) for f in specs] == [
        ("parent_id", "dim_parent.id"),
        ("org_id", "main.ref.orgs.id"),
    ]


def test_foreign_key_spec_from_any_falls_back_to_schema():
    s = make_schema(
        [_int_field("parent_id", tags={"foreign_key": "dim_parent.id"})]
    )

    assert ForeignKeySpec.from_any(None, schema=s) == [
        ForeignKeySpec(column="parent_id", ref="dim_parent.id")
    ]


def test_foreign_key_spec_from_any_accepts_dict_and_string_forms():
    from_dict = ForeignKeySpec.from_any({"parent_id": "dim_parent.id"})
    from_str = ForeignKeySpec.from_any("parent_id->dim_parent.id")
    from_tuple = ForeignKeySpec.from_any(("parent_id", "dim_parent.id"))

    assert from_dict == from_str == from_tuple == [
        ForeignKeySpec(column="parent_id", ref="dim_parent.id")
    ]


def test_primary_key_spec_from_any_list_columns():
    pk = PrimaryKeySpec.from_any(["id"])

    assert pk is not None
    assert pk.columns == ["id"]


def test_parse_fk_ref_completes_partial_refs_with_defaults():
    assert _parse_fk_ref(
        "dim_parent.id",
        default_catalog="main",
        default_schema="ref",
    ) == ("main.ref.dim_parent", ["id"])

    assert _parse_fk_ref(
        "ref.dim_parent.id",
        default_catalog="main",
    ) == ("main.ref.dim_parent", ["id"])

    assert _parse_fk_ref("main.ref.dim_parent.id") == (
        "main.ref.dim_parent",
        ["id"],
    )


def test_parse_fk_ref_rejects_short_refs_without_defaults():
    with pytest.raises(ValueError):
        _parse_fk_ref("dim_parent.id")


def test_qualify_fk_ref_serializes_back_to_dotted_form():
    assert _qualify_fk_ref(
        "dim_parent.id",
        default_catalog="main",
        default_schema="ref",
    ) == "main.ref.dim_parent.id"


def test_build_fk_constraint_sql_uses_defaults_for_partial_ref():
    fk = ForeignKeySpec(column="parent_id", ref="dim_parent.id")

    sql = _build_fk_constraint_sql(
        "child",
        fk,
        default_catalog="main",
        default_schema="ref",
    )

    assert "FOREIGN KEY (`parent_id`)" in sql
    assert "REFERENCES `main`.`ref`.`dim_parent` (`id`)" in sql
    assert "NOT ENFORCED" in sql


def test_foreign_key_spec_from_any_returns_empty_when_schema_has_no_fks():
    s = make_schema([_int_field("id"), _str_field("name")])

    assert ForeignKeySpec.from_any(None, schema=s) == []
