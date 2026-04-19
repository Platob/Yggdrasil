from __future__ import annotations

from types import SimpleNamespace

import pyarrow as pa
import pytest
from databricks.sdk.service.catalog import (
    ColumnInfo as CatalogColumnInfo,
    ForeignKeyConstraint,
    PrimaryKeyConstraint,
    TableConstraint,
)
from databricks.sdk.service.sql import ColumnInfo as SQLColumnInfo
from yggdrasil.data import Field

from yggdrasil.databricks.sql.column import Column
from yggdrasil.databricks.sql.sql_utils import _safe_constraint_name


@pytest.fixture
def executed() -> list[str]:
    return []


@pytest.fixture
def sql(executed: list[str]):
    return SimpleNamespace(execute=executed.append)


class _TableConstraintsAPIStub:
    """Captures ``table_constraints.create`` / ``.delete`` calls for assertions."""

    def __init__(self) -> None:
        self.created: list[tuple[str, TableConstraint]] = []
        self.deleted: list[tuple[str, str, bool]] = []

    def create(self, *, full_name_arg, constraint):
        self.created.append((full_name_arg, constraint))
        return constraint

    def delete(self, *, full_name, constraint_name, cascade):
        self.deleted.append((full_name, constraint_name, cascade))


@pytest.fixture
def table_constraints_api() -> _TableConstraintsAPIStub:
    return _TableConstraintsAPIStub()


@pytest.fixture
def client(table_constraints_api):
    workspace = SimpleNamespace(table_constraints=table_constraints_api)
    return SimpleNamespace(
        tables=SimpleNamespace(
            find_table=lambda **kwargs: None,
        ),
        workspace_client=lambda: workspace,
    )


def _make_table_stub(client, sql, *, catalog, schema, name, constraints=()):
    table = SimpleNamespace(
        client=client,
        sql=sql,
        catalog_name=catalog,
        schema_name=schema,
        table_name=name,
        name=name,
        full_name=lambda safe=False, q_schema=schema, q_catalog=catalog, q_name=name: (
            f"`{q_catalog}`.`{q_schema}`.`{q_name}`" if safe
            else f"{q_catalog}.{q_schema}.{q_name}"
        ),
        infos=SimpleNamespace(table_constraints=list(constraints)),
        _reset_cache=lambda invalidate_cache=False: None,
    )

    def _set_pk(columns, *, constraint_name=None, rely=False, timeseries=None):
        from yggdrasil.databricks.sql.constraints_api import apply_primary_key
        from yggdrasil.databricks.sql.types import PrimaryKeySpec

        apply_primary_key(
            table,
            PrimaryKeySpec(
                columns=list(columns) if not isinstance(columns, str) else [columns],
                constraint_name=constraint_name,
                rely=rely,
                timeseries=timeseries,
            ),
        )
        return table

    def _drop_pk(*, if_exists=True, cascade=False):
        from yggdrasil.databricks.sql.constraints_api import delete_constraint

        pk_name = None
        for c in table.infos.table_constraints:
            pk = getattr(c, "primary_key_constraint", None)
            if pk and getattr(pk, "name", None):
                pk_name = pk.name
                break
        if pk_name is None:
            if not if_exists:
                raise ValueError("no PK")
            return table
        delete_constraint(table, pk_name, cascade=cascade, if_exists=if_exists)
        return table

    def _fk_name(column: str):
        for c in table.infos.table_constraints:
            fk = getattr(c, "foreign_key_constraint", None)
            if fk is None:
                continue
            if column in (fk.child_columns or ()):
                return fk.name
        return None

    table.set_primary_key = _set_pk
    table.drop_primary_key = _drop_pk
    table._foreign_key_constraint_name = _fk_name
    return table


@pytest.fixture
def table(client, sql):
    return _make_table_stub(
        client, sql, catalog="main", schema="analytics", name="trades",
    )


@pytest.fixture
def ref_table(client, sql):
    return _make_table_stub(
        client, sql, catalog="main", schema="refined", name="books",
    )


@pytest.fixture
def field():
    return Field(
        name="trade_id",
        dtype="string",
        nullable=True,
        metadata={b"source": b"test"},
    )


@pytest.fixture
def column(table, field):
    return Column(
        table=table,
        name="trade_id",
        field=field,
    )


@pytest.fixture
def ref_column(ref_table):
    return Column(
        table=ref_table,
        name="book_id",
        field=Field(name="book_id", dtype="string", nullable=True),
    )


def test_engine_property(column, sql):
    assert column.engine is sql


def test_metadata_property(column):
    assert column.metadata == {b"source": b"test"}


def test_metadata_property_empty(table):
    col = Column(
        table=table,
        name="trade_id",
        field=Field(
            name="trade_id",
            dtype="string",
            nullable=True,
            metadata=None,
        ),
    )
    assert col.metadata == {}


def test_qcol(column):
    assert column._qcol() == "`trade_id`"


def test_set_tags_ddl(column):
    query = column.set_tags_ddl({"owner": "nika", "domain": "power"})
    assert query == (
        "ALTER TABLE `main`.`analytics`.`trades` "
        "ALTER COLUMN `trade_id` SET TAGS ('owner' = 'nika', 'domain' = 'power')"
    )


def test_set_tags_ddl_skips_empty_pairs(column):
    query = column.set_tags_ddl({"owner": "nika", "": "x", "domain": ""})
    assert query == (
        "ALTER TABLE `main`.`analytics`.`trades` "
        "ALTER COLUMN `trade_id` SET TAGS ('owner' = 'nika')"
    )


def test_set_tags_ddl_returns_none_when_empty(column):
    assert column.set_tags_ddl({"": "", "x": ""}) is None


def test_rename_executes_alter_table_rename_column(column, executed):
    column.rename("price")
    assert executed == [
        "ALTER TABLE `main`.`analytics`.`trades` RENAME COLUMN `trade_id` TO `price`"
    ]
    assert column.name == "price"


def test_rename_noop_on_same_name(column, executed):
    column.rename("trade_id")
    assert executed == []


def test_rename_empty_raises(column):
    with pytest.raises(ValueError):
        column.rename("")


def test_rename_strips_backticks(column, executed):
    column.rename("`price`")
    assert column.name == "price"
    assert "RENAME COLUMN `trade_id` TO `price`" in executed[0]


def test_set_tags_executes(column, executed):
    result = column.set_tags({"owner": "nika"})
    assert result is column
    assert executed == [
        "ALTER TABLE `main`.`analytics`.`trades` "
        "ALTER COLUMN `trade_id` SET TAGS ('owner' = 'nika')"
    ]


def test_set_tags_noop_when_none(column, executed):
    result = column.set_tags(None)
    assert result is column
    assert executed == []


def test_add_primary_key_ddl_default(column):
    cname = _safe_constraint_name("trades", "trade_id", "pk")
    query = column.add_primary_key_ddl()
    assert query == (
        "ALTER TABLE `main`.`analytics`.`trades` "
        f"ADD CONSTRAINT `{cname}` "
        "PRIMARY KEY (`trade_id`)"
    )


def test_add_primary_key_ddl_with_options(column):
    query = column.add_primary_key_ddl(
        constraint_name="pk trade id",
        rely=True,
        timeseries=True,
    )
    assert query == (
        "ALTER TABLE `main`.`analytics`.`trades` "
        f"ADD CONSTRAINT `{_safe_constraint_name('pk trade id')}` "
        "PRIMARY KEY (`trade_id` TIMESERIES) RELY"
    )


def test_set_primary_key_executes(column, table_constraints_api):
    cname = _safe_constraint_name("trades_trade_id_pk")
    result = column.set_primary_key(rely=True)
    assert result is column
    assert len(table_constraints_api.created) == 1
    full_name, constraint = table_constraints_api.created[0]
    assert full_name == "main.analytics.trades"
    assert constraint.primary_key_constraint is not None
    assert constraint.primary_key_constraint.name == cname
    assert constraint.primary_key_constraint.child_columns == ["trade_id"]
    assert constraint.primary_key_constraint.rely is True


def test_drop_primary_key_ddl_default(column):
    query = column.drop_primary_key_ddl()
    assert query == (
        "ALTER TABLE `main`.`analytics`.`trades` " "DROP PRIMARY KEY IF EXISTS"
    )


def test_drop_primary_key_ddl_variation(column):
    query = column.drop_primary_key_ddl(if_exists=False, cascade=True)
    assert query == (
        "ALTER TABLE `main`.`analytics`.`trades` " "DROP PRIMARY KEY CASCADE"
    )


def test_unset_primary_key_executes(column, table_constraints_api):
    column.table.infos.table_constraints = [
        TableConstraint(
            primary_key_constraint=PrimaryKeyConstraint(
                name="trades_trade_id_pk",
                child_columns=["trade_id"],
            ),
        ),
    ]
    result = column.unset_primary_key(if_exists=False, cascade=True)
    assert result is column
    assert table_constraints_api.deleted == [
        ("main.analytics.trades", "trades_trade_id_pk", True),
    ]


def test_add_foreign_key_ddl(column, ref_table):
    cname = _safe_constraint_name("trades", "trade_id", "books", "book_id", "fk")
    query = column.add_foreign_key_ddl(
        ref_table=ref_table,
        ref_column="book_id",
    )
    assert query == (
        "ALTER TABLE `main`.`analytics`.`trades` "
        f"ADD CONSTRAINT `{cname}` "
        "FOREIGN KEY (`trade_id`) "
        "REFERENCES `main`.`refined`.`books` (`book_id`)"
    )


def test_add_foreign_key_ddl_with_options(column, ref_table, ref_column):
    query = column.add_foreign_key_ddl(
        ref_table=ref_table,
        ref_column=ref_column,
        constraint_name="fk trades book",
        rely=True,
        match_full=True,
        on_update_no_action=True,
        on_delete_no_action=True,
    )
    assert query == (
        "ALTER TABLE `main`.`analytics`.`trades` "
        f"ADD CONSTRAINT `{_safe_constraint_name('fk trades book')}` "
        "FOREIGN KEY (`trade_id`) "
        "REFERENCES `main`.`refined`.`books` (`book_id`) "
        "RELY MATCH FULL ON UPDATE NO ACTION ON DELETE NO ACTION"
    )


def test_set_foreign_key_executes(column, ref_table, table_constraints_api):
    cname = _safe_constraint_name(
        "trades", "trade_id", "main.refined.books", "book_id", "fk",
    )
    result = column.set_foreign_key(
        ref_table=ref_table,
        ref_column="book_id",
        rely=True,
    )
    assert result is column
    assert len(table_constraints_api.created) == 1
    full_name, constraint = table_constraints_api.created[0]
    assert full_name == "main.analytics.trades"
    fk = constraint.foreign_key_constraint
    assert fk is not None
    assert fk.name == cname
    assert fk.child_columns == ["trade_id"]
    assert fk.parent_table == "main.refined.books"
    assert fk.parent_columns == ["book_id"]
    assert fk.rely is True


def test_drop_foreign_key_ddl_default(column):
    query = column.drop_foreign_key_ddl()
    assert query == (
        "ALTER TABLE `main`.`analytics`.`trades` "
        "DROP FOREIGN KEY IF EXISTS (`trade_id`)"
    )


def test_drop_foreign_key_ddl_variation(column):
    query = column.drop_foreign_key_ddl(if_exists=False)
    assert query == (
        "ALTER TABLE `main`.`analytics`.`trades` " "DROP FOREIGN KEY (`trade_id`)"
    )


def test_unset_foreign_key_executes(column, table_constraints_api):
    column.table.infos.table_constraints = [
        TableConstraint(
            foreign_key_constraint=ForeignKeyConstraint(
                name="trades_trade_id_books_book_id_fk",
                child_columns=["trade_id"],
                parent_table="main.refined.books",
                parent_columns=["book_id"],
            ),
        ),
    ]
    result = column.unset_foreign_key(if_exists=False)
    assert result is column
    assert table_constraints_api.deleted == [
        ("main.analytics.trades", "trades_trade_id_books_book_id_fk", False),
    ]


def test_resolve_ref_args_with_column_object(column, ref_column, ref_table):
    resolved_table, resolved_column = column._resolve_ref_args(ref_column, None, None)

    assert resolved_table is ref_table
    assert resolved_column is ref_column


def test_resolve_ref_args_with_ref_table_string(column):
    resolved_ref_table = SimpleNamespace(name="books")
    column.table.client.tables.find_table = lambda **kwargs: resolved_ref_table

    resolved_table, resolved_column = column._resolve_ref_args(
        None,
        "main.refined.books",
        "book_id",
    )

    assert resolved_table is resolved_ref_table
    assert resolved_column == "book_id"


def test_resolve_ref_args_defaults_to_self_table(column):
    resolved_table, resolved_column = column._resolve_ref_args(None, None, "trade_id")
    assert resolved_table is column.table
    assert resolved_column == "trade_id"


def test_resolve_ref_args_with_string_column(column, ref_table):
    column.table.client.tables.find_table = lambda **kwargs: ref_table

    resolved_table, resolved_column = column._resolve_ref_args(
        "main.refined.books.book_id",
        None,
        None,
    )

    assert resolved_table is ref_table
    assert resolved_column == "book_id"


def test_from_api_sql_column_info(table):
    info = SQLColumnInfo(name="trade_id")

    col = Column.from_api(table, info)

    assert col.name == "trade_id"
    assert col.table is table
    assert col.field.metadata[b"engine"] == b"databricks"
    assert col.field.metadata[b"catalog_name"] == b"main"
    assert col.field.metadata[b"schema_name"] == b"analytics"
    assert col.field.metadata[b"table_name"] == b"trades"


def test_from_api_catalog_column_info(table):
    info = CatalogColumnInfo(name="trade_id")

    col = Column.from_api(table, info)

    assert col.name == "trade_id"
    assert col.table is table
    assert col.field.metadata[b"engine"] == b"databricks"
    assert col.field.metadata[b"catalog_name"] == b"main"
    assert col.field.metadata[b"schema_name"] == b"analytics"
    assert col.field.metadata[b"table_name"] == b"trades"


@pytest.mark.parametrize(
    ("info", "expected_name", "expected_nullable", "expected_arrow"),
    [
        (
            SQLColumnInfo(
                name="trade_id",
                type_text="STRING",
                type_name="STRING",
                type_precision=0,
                type_scale=0,
                position=0,
            ),
            "trade_id",
            True,
            pa.string(),
        ),
        (
            SQLColumnInfo(
                name="quantity",
                type_text="DECIMAL(18,4)",
                type_name="DECIMAL",
                type_precision=18,
                type_scale=4,
                position=1,
            ),
            "quantity",
            False,
            pa.decimal128(18, 4),
        ),
        (
            SQLColumnInfo(
                name="tags",
                type_text="ARRAY<STRING>",
                type_name="ARRAY",
                position=2,
            ),
            "tags",
            True,
            pa.list_(pa.string()),
        ),
        (
            SQLColumnInfo(
                name="attributes",
                type_text="MAP<STRING, STRING>",
                type_name="MAP",
                position=3,
            ),
            "attributes",
            True,
            pa.map_(pa.string(), pa.string()),
        ),
        (
            SQLColumnInfo(
                name="book",
                type_text="STRUCT<book_id: STRING, version: INT>",
                type_name="STRUCT",
                position=4,
            ),
            "book",
            False,
            pa.struct(
                [
                    pa.field("book_id", pa.string(), nullable=True),
                    pa.field("version", pa.int32(), nullable=True),
                ]
            ),
        ),
    ],
)
def test_from_api_sql_column_info_parses_types(
    table,
    info,
    expected_name,
    expected_nullable,
    expected_arrow,
):
    col = Column.from_api(table, info)

    assert col.name == expected_name
    assert col.field.name == expected_name
    assert col.field.nullable is True
    assert col.field.to_arrow_type() == expected_arrow
    assert col.field.metadata[b"engine"] == b"databricks"
    assert col.field.metadata[b"catalog_name"] == b"main"
    assert col.field.metadata[b"schema_name"] == b"analytics"
    assert col.field.metadata[b"table_name"] == b"trades"


@pytest.mark.parametrize(
    ("info", "expected_name", "expected_nullable", "expected_arrow"),
    [
        (
            CatalogColumnInfo(
                name="book_id",
                type_text="STRING",
                type_name="STRING",
                type_precision=0,
                type_scale=0,
                position=0,
                nullable=True,
            ),
            "book_id",
            True,
            pa.string(),
        ),
        (
            CatalogColumnInfo(
                name="book_version",
                type_text="INT",
                type_name="INT",
                type_precision=0,
                type_scale=0,
                position=1,
                nullable=False,
            ),
            "book_version",
            False,
            pa.int32(),
        ),
        (
            CatalogColumnInfo(
                name="tags",
                type_text="ARRAY<STRING>",
                type_name="ARRAY",
                position=2,
                nullable=True,
            ),
            "tags",
            True,
            pa.list_(pa.string()),
        ),
        (
            CatalogColumnInfo(
                name="attributes",
                type_text="MAP<STRING, STRING>",
                type_name="MAP",
                position=3,
                nullable=True,
            ),
            "attributes",
            True,
            pa.map_(pa.string(), pa.string()),
        ),
        (
            CatalogColumnInfo(
                name="book",
                type_text="STRUCT<book_id: STRING, version: INT>",
                type_name="STRUCT",
                position=4,
                nullable=False,
            ),
            "book",
            False,
            pa.struct(
                [
                    pa.field("book_id", pa.string(), nullable=True),
                    pa.field("version", pa.int32(), nullable=True),
                ]
            ),
        ),
    ],
)
def test_from_api_catalog_column_info_parses_types(
    table,
    info,
    expected_name,
    expected_nullable,
    expected_arrow,
):
    col = Column.from_api(table, info)

    assert col.name == expected_name
    assert col.field.name == expected_name
    assert col.field.nullable is expected_nullable
    assert col.field.to_arrow_type() == expected_arrow
    assert col.field.metadata[b"engine"] == b"databricks"
    assert col.field.metadata[b"catalog_name"] == b"main"
    assert col.field.metadata[b"schema_name"] == b"analytics"
    assert col.field.metadata[b"table_name"] == b"trades"


def test_from_api_sql_column_info_parses_deeply_nested_struct(table):
    info = SQLColumnInfo(
        name="payload",
        type_text=(
            "STRUCT<"
            "book: STRUCT<book_id: STRING, version: INT>, "
            "tags: ARRAY<STRING>, "
            "attrs: MAP<STRING, STRING>"
            ">"
        ),
        type_name="STRUCT",
        position=5,
    )

    col = Column.from_api(table, info)

    assert col.name == "payload"
    assert col.field.name == "payload"
    assert col.field.nullable is True
    assert col.field.to_arrow_type() == pa.struct(
        [
            pa.field(
                "book",
                pa.struct(
                    [
                        pa.field("book_id", pa.string(), nullable=True),
                        pa.field("version", pa.int32(), nullable=True),
                    ]
                ),
                nullable=True,
            ),
            pa.field("tags", pa.list_(pa.string()), nullable=True),
            pa.field("attrs", pa.map_(pa.string(), pa.string()), nullable=True),
        ]
    )
    assert col.field.metadata[b"engine"] == b"databricks"
    assert col.field.metadata[b"catalog_name"] == b"main"
    assert col.field.metadata[b"schema_name"] == b"analytics"
    assert col.field.metadata[b"table_name"] == b"trades"


def test_from_api_catalog_column_info_parses_deeply_nested_struct(table):
    info = CatalogColumnInfo(
        name="payload",
        type_text=(
            "STRUCT<"
            "book: STRUCT<book_id: STRING, version: INT>, "
            "tags: ARRAY<STRING>, "
            "attrs: MAP<STRING, STRING>"
            ">"
        ),
        type_name="STRUCT",
        position=5,
        nullable=True,
    )

    col = Column.from_api(table, info)

    assert col.name == "payload"
    assert col.field.name == "payload"
    assert col.field.nullable is True
    assert col.field.to_arrow_type() == pa.struct(
        [
            pa.field(
                "book",
                pa.struct(
                    [
                        pa.field("book_id", pa.string(), nullable=True),
                        pa.field("version", pa.int32(), nullable=True),
                    ]
                ),
                nullable=True,
            ),
            pa.field("tags", pa.list_(pa.string()), nullable=True),
            pa.field("attrs", pa.map_(pa.string(), pa.string()), nullable=True),
        ]
    )
    assert col.field.metadata[b"engine"] == b"databricks"
    assert col.field.metadata[b"catalog_name"] == b"main"
    assert col.field.metadata[b"schema_name"] == b"analytics"
    assert col.field.metadata[b"table_name"] == b"trades"


# ---------------------------------------------------------------------------
# GEOGRAPHY type — Databricks native geography DDL parsing
# ---------------------------------------------------------------------------


def test_geography_type_text_parses_to_geography_type(table):
    """Databricks type_text ``GEOGRAPHY(4326)`` should parse to GeographyType."""
    from yggdrasil.data.types.extensions.geography import GeographyType

    info = SQLColumnInfo(
        name="location",
        type_text="GEOGRAPHY(4326)",
        type_name="GEOGRAPHY",
        position=0,
    )

    col = Column.from_api(table, info)

    assert col.name == "location"
    assert col.field.name == "location"
    assert isinstance(col.field.dtype, GeographyType)
    assert col.field.dtype.srid == 4326


def test_geography_type_text_any_srid(table):
    """Databricks ``GEOGRAPHY(ANY)`` parses with srid='ANY'."""
    from yggdrasil.data.types.extensions.geography import GeographyType

    info = SQLColumnInfo(
        name="geo_col",
        type_text="GEOGRAPHY(ANY)",
        type_name="GEOGRAPHY",
        position=0,
    )

    col = Column.from_api(table, info)

    assert isinstance(col.field.dtype, GeographyType)
    assert col.field.dtype.srid == "ANY"


def test_geography_type_text_bare(table):
    """Bare ``GEOGRAPHY`` without SRID defaults to 4326."""
    from yggdrasil.data.types.extensions.geography import GeographyType

    info = SQLColumnInfo(
        name="geo_col",
        type_text="GEOGRAPHY",
        type_name="GEOGRAPHY",
        position=0,
    )

    col = Column.from_api(table, info)

    assert isinstance(col.field.dtype, GeographyType)
    assert col.field.dtype.srid == 4326


def test_geography_type_round_trip_ddl():
    """GeographyType.to_databricks_ddl() matches what Databricks expects."""
    from yggdrasil.data.types.extensions.geography import GeographyType

    assert GeographyType().to_databricks_ddl() == "GEOGRAPHY(4326)"
    assert GeographyType(srid=3857).to_databricks_ddl() == "GEOGRAPHY(3857)"
    assert GeographyType(srid="ANY").to_databricks_ddl() == "GEOGRAPHY(ANY)"


def test_geography_catalog_column_info(table):
    """CatalogColumnInfo with GEOGRAPHY type_text also parses correctly."""
    from yggdrasil.data.types.extensions.geography import GeographyType

    info = CatalogColumnInfo(
        name="delivery_point",
        type_text="GEOGRAPHY(4326)",
        type_name="GEOGRAPHY",
        position=0,
        nullable=True,
    )

    col = Column.from_api(table, info)

    assert col.name == "delivery_point"
    assert isinstance(col.field.dtype, GeographyType)
    assert col.field.dtype.srid == 4326
    assert col.field.nullable is True
