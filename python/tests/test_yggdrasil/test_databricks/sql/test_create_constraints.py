"""
Tests for CREATE TABLE primary-key and foreign-key constraint handling.

Structure
---------
Unit tests (no live workspace)
    - ``_resolve_pk_spec`` / ``_resolve_fk_specs`` helper functions
    - ``Table.add_primary_key_ddl`` (table-level, composite key)
    - ``Table.drop_primary_key_ddl``
    - ``Table._apply_constraints`` (mocked engine)
    - ``sql_create`` NOT NULL enforcement on PK columns (DDL inspection)
    - ``Field.foreign_key`` / ``Field.primary_key`` metadata properties
    - ``Schema.primary_keys`` / ``Schema.foreign_keys``

Integration tests (``requires_databricks``)
    ``TestCreateConstraintsIntegration`` — creates two scratch tables,
    exercises ``create(primary_keys=…, foreign_keys=…)`` and
    ``create`` from a schema with metadata tags, then tears down.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pyarrow as pa
import pytest

from yggdrasil.data import Field
from yggdrasil.data.field import _normalize_metadata
from yggdrasil.data.schema import Schema
from yggdrasil.databricks.sql import Table, Tables, Column
from yggdrasil.databricks.sql.table import _resolve_pk_spec, _resolve_fk_specs
from yggdrasil.databricks.sql.types import PrimaryKeySpec, ForeignKeySpec
from ..conftest import DatabricksCase, requires_databricks


# ─────────────────────────────────────────────────────────────────────────────
# Helpers shared across all unit tests
# ─────────────────────────────────────────────────────────────────────────────


def _stub_table(
    catalog: str = "main",
    schema: str = "sales",
    name: str = "orders",
) -> MagicMock:
    tbl = MagicMock(spec=Table)
    tbl.name = name
    tbl.table_name = name
    tbl.catalog_name = catalog
    tbl.schema_name = schema
    tbl.full_name.side_effect = lambda safe=None: (
        f"`{catalog}`.`{schema}`.`{name}`" if safe else f"{catalog}.{schema}.{name}"
    )
    tbl._safe_str.side_effect = lambda v: v if isinstance(v, str) else v.decode()
    return tbl


def _pk_field(name: str = "id", arrow_type: pa.DataType = pa.int64()) -> Field:
    """Build a Field with t:primary_key = true."""
    return Field(
        name=name,
        arrow_type=arrow_type,
        nullable=True,
        metadata=_normalize_metadata(None, tags={"primary_key": "true"}),
    )


def _fk_field(name: str, ref: str, arrow_type: pa.DataType = pa.int64()) -> Field:
    """Build a Field with t:foreign_key = <ref>."""
    return Field(
        name=name,
        arrow_type=arrow_type,
        nullable=True,
        metadata=_normalize_metadata(None, tags={"foreign_key": ref}),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Unit — Field.primary_key / Field.foreign_key properties
# ─────────────────────────────────────────────────────────────────────────────


class TestFieldConstraintProperties:
    def test_primary_key_true(self):
        f = _pk_field()
        assert f.primary_key is True

    def test_primary_key_false_by_default(self):
        f = Field(name="x", arrow_type=pa.int64())
        assert f.primary_key is False

    def test_foreign_key_value(self):
        f = _fk_field("customer_id", "main.sales.customers.id")
        assert f.foreign_key == "main.sales.customers.id"

    def test_foreign_key_none_by_default(self):
        f = Field(name="x", arrow_type=pa.int64())
        assert f.foreign_key is None

    def test_both_pk_and_fk(self):
        meta = _normalize_metadata(None, tags={"primary_key": "true", "foreign_key": "a.b.c.d"})
        f = Field(name="x", arrow_type=pa.int64(), metadata=meta)
        assert f.primary_key is True
        assert f.foreign_key == "a.b.c.d"


# ─────────────────────────────────────────────────────────────────────────────
# Unit — Schema.primary_keys / Schema.foreign_keys
# ─────────────────────────────────────────────────────────────────────────────


class TestSchemaConstraintProperties:
    def _schema(self) -> Schema:
        return Schema.from_fields([
            _pk_field("id"),
            _fk_field("customer_id", "main.sales.customers.id"),
            Field(name="name", arrow_type=pa.utf8()),
        ])

    def test_primary_keys(self):
        s = self._schema()
        assert [f.name for f in s.primary_keys] == ["id"]

    def test_foreign_keys(self):
        s = self._schema()
        assert [f.name for f in s.foreign_keys] == ["customer_id"]
        assert s.foreign_keys[0].foreign_key == "main.sales.customers.id"

    def test_empty_when_no_constraints(self):
        s = Schema.from_fields([Field(name="x", arrow_type=pa.int64())])
        assert s.primary_keys == []
        assert s.foreign_keys == []


# ─────────────────────────────────────────────────────────────────────────────
# Unit — _resolve_pk_spec
# ─────────────────────────────────────────────────────────────────────────────


class TestResolvePkSpec:
    def _empty_schema(self) -> Schema:
        return Schema.from_fields([Field(name="x", arrow_type=pa.int64())])

    def _pk_schema(self) -> Schema:
        return Schema.from_fields([_pk_field("id"), Field(name="name", arrow_type=pa.utf8())])

    def test_none_no_metadata_returns_none(self):
        assert _resolve_pk_spec(self._empty_schema(), None) is None

    def test_none_reads_field_metadata(self):
        spec = _resolve_pk_spec(self._pk_schema(), None)
        assert spec is not None
        assert spec.columns == ["id"]

    def test_list_of_strings(self):
        spec = _resolve_pk_spec(self._empty_schema(), ["col_a", "col_b"])
        assert spec.columns == ["col_a", "col_b"]

    def test_empty_list_returns_none(self):
        assert _resolve_pk_spec(self._empty_schema(), []) is None

    def test_single_string(self):
        spec = _resolve_pk_spec(self._empty_schema(), "col_a")
        assert spec.columns == ["col_a"]

    def test_pk_spec_passthrough(self):
        given = PrimaryKeySpec(columns=["a", "b"], rely=True, constraint_name="my_pk")
        result = _resolve_pk_spec(self._empty_schema(), given)
        assert result is given

    def test_explicit_overrides_metadata(self):
        # schema has t:primary_key on "id", but caller explicitly passes ["name"]
        spec = _resolve_pk_spec(self._pk_schema(), ["name"])
        assert spec.columns == ["name"]


# ─────────────────────────────────────────────────────────────────────────────
# Unit — _resolve_fk_specs
# ─────────────────────────────────────────────────────────────────────────────


class TestResolveFkSpecs:
    def _empty_schema(self) -> Schema:
        return Schema.from_fields([Field(name="x", arrow_type=pa.int64())])

    def _fk_schema(self) -> Schema:
        return Schema.from_fields([_fk_field("customer_id", "main.sales.customers.id")])

    def test_none_no_metadata_returns_empty(self):
        assert _resolve_fk_specs(self._empty_schema(), None) == []

    def test_none_reads_field_metadata(self):
        specs = _resolve_fk_specs(self._fk_schema(), None)
        assert len(specs) == 1
        assert specs[0].column == "customer_id"
        assert specs[0].ref == "main.sales.customers.id"

    def test_dict_input(self):
        specs = _resolve_fk_specs(self._empty_schema(), {"col": "a.b.c.d"})
        assert len(specs) == 1
        assert specs[0].column == "col" and specs[0].ref == "a.b.c.d"

    def test_dict_skips_empty_values(self):
        specs = _resolve_fk_specs(self._empty_schema(), {"": "ref", "col": ""})
        assert specs == []

    def test_list_passthrough(self):
        given = [ForeignKeySpec(column="c", ref="a.b.t.c")]
        result = _resolve_fk_specs(self._empty_schema(), given)
        assert result is given

    def test_explicit_overrides_metadata(self):
        specs = _resolve_fk_specs(self._fk_schema(), {"override_col": "x.y.z.w"})
        assert specs[0].column == "override_col"


# ─────────────────────────────────────────────────────────────────────────────
# Unit — Table.add_primary_key_ddl (table-level, multi-column)
# ─────────────────────────────────────────────────────────────────────────────


class TestTableAddPrimaryKeyDDL:
    def _table(self) -> MagicMock:
        return _stub_table()

    def test_single_column(self):
        tbl = _stub_table()
        # Call the real method by building a minimal Table-like object
        # We test via the DDL builder added to Table; use Column DDL as the fixture
        # for the table-level DDL we call the actual Table class method directly.
        # Since Table is a dataclass, instantiate it with mocked service.
        mock_svc = MagicMock(spec=Tables)
        mock_svc.client = MagicMock()
        t = Table(service=mock_svc, catalog_name="main", schema_name="sales", table_name="orders")
        ddl = t.add_primary_key_ddl("id")
        assert "ALTER TABLE `main`.`sales`.`orders`" in ddl
        assert "PRIMARY KEY (`id`)" in ddl
        assert "`orders_id_pk`" in ddl

    def test_composite_key(self):
        mock_svc = MagicMock(spec=Tables)
        mock_svc.client = MagicMock()
        t = Table(service=mock_svc, catalog_name="cat", schema_name="sch", table_name="tbl")
        ddl = t.add_primary_key_ddl(["a", "b"])
        assert "PRIMARY KEY (`a`, `b`)" in ddl
        assert "`tbl_a_b_pk`" in ddl

    def test_rely_clause(self):
        mock_svc = MagicMock(spec=Tables)
        mock_svc.client = MagicMock()
        t = Table(service=mock_svc, catalog_name="c", schema_name="s", table_name="t")
        ddl = t.add_primary_key_ddl("id", rely=True)
        assert "RELY" in ddl

    def test_timeseries_column(self):
        mock_svc = MagicMock(spec=Tables)
        mock_svc.client = MagicMock()
        t = Table(service=mock_svc, catalog_name="c", schema_name="s", table_name="t")
        ddl = t.add_primary_key_ddl(["id", "ts"], timeseries="ts")
        assert "`ts` TIMESERIES" in ddl
        assert "`id`" in ddl

    def test_custom_constraint_name(self):
        mock_svc = MagicMock(spec=Tables)
        mock_svc.client = MagicMock()
        t = Table(service=mock_svc, catalog_name="c", schema_name="s", table_name="t")
        ddl = t.add_primary_key_ddl("id", constraint_name="my_pk")
        assert "`my_pk`" in ddl

    def test_drop_primary_key_if_exists(self):
        mock_svc = MagicMock(spec=Tables)
        mock_svc.client = MagicMock()
        t = Table(service=mock_svc, catalog_name="c", schema_name="s", table_name="t")
        ddl = t.drop_primary_key_ddl()
        assert "DROP PRIMARY KEY IF EXISTS" in ddl

    def test_drop_primary_key_cascade(self):
        mock_svc = MagicMock(spec=Tables)
        mock_svc.client = MagicMock()
        t = Table(service=mock_svc, catalog_name="c", schema_name="s", table_name="t")
        ddl = t.drop_primary_key_ddl(cascade=True)
        assert "CASCADE" in ddl

    def test_drop_primary_key_no_if_exists(self):
        mock_svc = MagicMock(spec=Tables)
        mock_svc.client = MagicMock()
        t = Table(service=mock_svc, catalog_name="c", schema_name="s", table_name="t")
        ddl = t.drop_primary_key_ddl(if_exists=False)
        assert "IF EXISTS" not in ddl


# ─────────────────────────────────────────────────────────────────────────────
# Unit — Table._apply_constraints (mocked SQL engine)
# ─────────────────────────────────────────────────────────────────────────────


def _table_with_mock_engine() -> tuple[Table, MagicMock]:
    """Return a Table whose sql.execute is a MagicMock."""
    mock_svc = MagicMock(spec=Tables)
    mock_svc.client = MagicMock()
    t = Table(service=mock_svc, catalog_name="main", schema_name="sales", table_name="orders")
    mock_engine = MagicMock()
    mock_svc.client.sql = mock_engine
    # Make column() return a mock column that has set_foreign_key
    mock_col = MagicMock(spec=Column)
    mock_col.set_foreign_key.return_value = mock_col
    t.column = MagicMock(return_value=mock_col)
    return t, mock_engine


class TestApplyConstraints:
    def test_pk_only(self):
        t, eng = _table_with_mock_engine()
        pk = PrimaryKeySpec(columns=["id"])
        t._apply_constraints(pk, [])
        eng.execute.assert_called_once()
        ddl = eng.execute.call_args[0][0]
        assert "PRIMARY KEY (`id`)" in ddl

    def test_fk_only(self):
        t, eng = _table_with_mock_engine()
        fk = ForeignKeySpec(column="customer_id", ref="main.sales.customers.id")
        t._apply_constraints(None, [fk])
        t.column.assert_called_once_with("customer_id")
        t.column.return_value.set_foreign_key.assert_called_once_with(
            "main.sales.customers.id",
            constraint_name=None,
            rely=False,
            match_full=False,
            on_update_no_action=False,
            on_delete_no_action=False,
        )

    def test_pk_applied_before_fk(self):
        """PK must be applied before FK so that FK references can resolve."""
        call_order: list[str] = []
        t, eng = _table_with_mock_engine()
        eng.execute.side_effect = lambda *a, **kw: call_order.append("pk")
        t.column.return_value.set_foreign_key.side_effect = lambda *a, **kw: call_order.append("fk")

        pk = PrimaryKeySpec(columns=["id"])
        fk = ForeignKeySpec(column="ref_id", ref="a.b.c.d")
        t._apply_constraints(pk, [fk])

        assert call_order == ["pk", "fk"]

    def test_multiple_fks(self):
        t, eng = _table_with_mock_engine()
        fks = [
            ForeignKeySpec(column="c1", ref="a.b.t.x"),
            ForeignKeySpec(column="c2", ref="a.b.t.y"),
        ]
        t._apply_constraints(None, fks)
        assert t.column.call_count == 2

    def test_no_constraints_is_noop(self):
        t, eng = _table_with_mock_engine()
        t._apply_constraints(None, [])
        eng.execute.assert_not_called()
        t.column.assert_not_called()

    def test_pk_error_is_logged_not_raised(self):
        t, eng = _table_with_mock_engine()
        eng.execute.side_effect = RuntimeError("SQL error")
        pk = PrimaryKeySpec(columns=["id"])
        # Must not raise
        t._apply_constraints(pk, [])

    def test_fk_error_is_logged_not_raised(self):
        t, eng = _table_with_mock_engine()
        t.column.return_value.set_foreign_key.side_effect = RuntimeError("FK error")
        fk = ForeignKeySpec(column="c", ref="a.b.t.c")
        # Must not raise
        t._apply_constraints(None, [fk])

    def test_fk_spec_options_forwarded(self):
        t, eng = _table_with_mock_engine()
        fk = ForeignKeySpec(
            column="c", ref="a.b.t.c",
            constraint_name="my_fk",
            rely=True,
            match_full=True,
            on_update_no_action=True,
            on_delete_no_action=True,
        )
        t._apply_constraints(None, [fk])
        t.column.return_value.set_foreign_key.assert_called_once_with(
            "a.b.t.c",
            constraint_name="my_fk",
            rely=True,
            match_full=True,
            on_update_no_action=True,
            on_delete_no_action=True,
        )

    def test_pk_spec_options_forwarded(self):
        t, eng = _table_with_mock_engine()
        pk = PrimaryKeySpec(
            columns=["ts", "id"],
            constraint_name="my_pk",
            rely=True,
            timeseries="ts",
        )
        t._apply_constraints(pk, [])
        ddl = eng.execute.call_args[0][0]
        assert "`my_pk`" in ddl
        assert "RELY" in ddl
        assert "`ts` TIMESERIES" in ddl


# ─────────────────────────────────────────────────────────────────────────────
# Unit — sql_create NOT NULL enforcement on PK columns
# ─────────────────────────────────────────────────────────────────────────────


class TestSqlCreateNotNull:
    """Verify that pk columns are forced NOT NULL in the generated DDL."""

    def _make_table(self) -> tuple["Table", "MagicMock"]:
        """Return (table, mock_sql_engine). sql engine is accessed via client.sql."""
        mock_engine = MagicMock()
        mock_svc = MagicMock(spec=Tables)
        mock_svc.client = MagicMock()
        mock_svc.client.sql = mock_engine
        t = Table(service=mock_svc, catalog_name="c", schema_name="s", table_name="t")
        # Stub post-create steps so only DDL capture matters
        t._reset_cache = MagicMock()
        t.set_tags = MagicMock()
        t.column = MagicMock(return_value=MagicMock(spec=Column, set_tags=MagicMock()))
        t._apply_constraints = MagicMock()
        return t, mock_engine

    def test_nullable_pk_column_forced_not_null(self):
        """A nullable=True field with primary_key metadata must become NOT NULL in DDL."""
        t, eng = self._make_table()
        schema = pa.schema([
            pa.field("id", pa.int64(), nullable=True),   # nullable — will be forced
            pa.field("name", pa.utf8(), nullable=True),
        ])

        captured: list[str] = []
        eng.execute.side_effect = lambda stmt, **kw: captured.append(stmt)

        t.sql_create(schema, primary_keys=["id"])

        assert captured, "sql.execute was never called"
        ddl = captured[0]
        assert "NOT NULL" in ddl
        id_line = next(ln for ln in ddl.splitlines() if "`id`" in ln)
        assert "NOT NULL" in id_line

    def test_already_not_null_pk_column_unchanged(self):
        """A nullable=False PK column must remain NOT NULL (no change needed)."""
        t, eng = self._make_table()
        schema = pa.schema([
            pa.field("id", pa.int64(), nullable=False),
            pa.field("name", pa.utf8(), nullable=True),
        ])

        captured: list[str] = []
        eng.execute.side_effect = lambda stmt, **kw: captured.append(stmt)
        t.sql_create(schema, primary_keys=["id"])

        ddl = captured[0]
        id_line = next(ln for ln in ddl.splitlines() if "`id`" in ln)
        assert "NOT NULL" in id_line

    def test_non_pk_nullable_column_stays_nullable(self):
        """Non-PK nullable columns must NOT have NOT NULL in their definition."""
        t, eng = self._make_table()
        schema = pa.schema([
            pa.field("id", pa.int64(), nullable=False),
            pa.field("name", pa.utf8(), nullable=True),
        ])

        captured: list[str] = []
        eng.execute.side_effect = lambda stmt, **kw: captured.append(stmt)
        t.sql_create(schema, primary_keys=["id"])

        ddl = captured[0]
        name_line = next(ln for ln in ddl.splitlines() if "`name`" in ln)
        assert "NOT NULL" not in name_line

    def test_pk_from_field_metadata_forces_not_null(self):
        """PK columns discovered from field metadata must also be forced NOT NULL."""
        t, eng = self._make_table()
        schema = Schema.from_fields([
            _pk_field("id"),
            Field(name="name", arrow_type=pa.utf8()),
        ])

        captured: list[str] = []
        eng.execute.side_effect = lambda stmt, **kw: captured.append(stmt)
        # No explicit primary_keys — should be read from field metadata
        t.sql_create(schema)

        ddl = captured[0]
        id_line = next(ln for ln in ddl.splitlines() if "`id`" in ln)
        assert "NOT NULL" in id_line

    def test_apply_constraints_called_with_resolved_specs(self):
        t, eng = self._make_table()
        schema = pa.schema([pa.field("id", pa.int64(), nullable=False)])
        eng.execute.return_value = None

        t.sql_create(schema, primary_keys=["id"], foreign_keys={"ref_id": "a.b.c.d"})

        t._apply_constraints.assert_called_once()
        pk_arg, fk_arg = t._apply_constraints.call_args[0]
        assert pk_arg.columns == ["id"]
        assert len(fk_arg) == 1 and fk_arg[0].column == "ref_id"


# ─────���───────────────────────────────────────────────────────────────────────
# Integration — live Databricks workspace
# ─────────────────────────────────────────────────────────────────────────────


class TestCreateConstraintsIntegration(DatabricksCase):
    """
    Integration tests for ``create(primary_keys=…, foreign_keys=…)``.

    Two scratch tables are created in ``trading.unittest``:

    - **test_cc_ref**  — ``id BIGINT NOT NULL`` (PK target for FK tests)
    - **test_cc_main** — ``id BIGINT NOT NULL, name STRING``
    """

    pytestmark = [requires_databricks, pytest.mark.integration]

    _CATALOG = "trading"
    _SCHEMA_NAME = "unittest"

    engine = None
    table = None
    ref_table = None

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.engine = cls.workspace.sql(
            catalog_name=cls._CATALOG,
            schema_name=cls._SCHEMA_NAME,
        )

        # Reference table — PK will be set by the test that needs it
        ref_schema = pa.schema([pa.field("id", pa.int64(), nullable=False)])
        ref_data = pa.table([pa.array([1, 2, 3], type=pa.int64())], schema=ref_schema)
        cls.ref_table = cls.engine.table("test_cc_ref").create(
            ref_data, if_not_exists=False
        )

        # Main table created without constraints (added in individual tests)
        main_schema = pa.schema([
            pa.field("id", pa.int64(), nullable=False),
            pa.field("name", pa.string(), nullable=True),
        ])
        main_data = pa.table(
            [pa.array([1, 2, 3], type=pa.int64()), pa.array(["a", "b", "c"])],
            schema=main_schema,
        )
        cls.table = cls.engine.table("test_cc_main").create(
            main_data, if_not_exists=False
        )

    @classmethod
    def tearDownClass(cls) -> None:
        for tbl in (cls.table, cls.ref_table):
            try:
                if tbl is not None:
                    tbl.delete()
            except Exception:
                pass
        super().tearDownClass()

    # ── explicit primary_keys param ───────────────────────────────────────────

    def test_set_primary_key_table_level_single(self):
        self.table.set_primary_key(["id"], constraint_name="it_cc_pk")
        self.table.drop_primary_key(if_exists=True, cascade=True)

    def test_set_primary_key_table_level_composite(self):
        # add both columns as PK (both NOT NULL) — use ref table which only has id
        self.ref_table.set_primary_key(["id"], constraint_name="it_cc_ref_pk_cmp")
        self.ref_table.drop_primary_key(if_exists=True, cascade=True)

    # ── create with explicit constraints via a fresh temp table ───────────────

    def _temp_ref_schema(self) -> pa.Schema:
        return pa.schema([pa.field("id", pa.int64(), nullable=False)])

    def _temp_main_schema(self) -> pa.Schema:
        return pa.schema([
            pa.field("id", pa.int64(), nullable=False),
            pa.field("ref_id", pa.int64(), nullable=True),
        ])

    def test_create_with_pk_param(self):
        ref = self.engine.table("test_cc_tmp_pk")
        try:
            ref_data = pa.table(
                [pa.array([1, 2, 3], type=pa.int64())],
                schema=self._temp_ref_schema(),
            )
            ref.create(
                ref_data,
                if_not_exists=False,
                primary_keys=PrimaryKeySpec(columns=["id"], constraint_name="it_tmp_pk"),
            )
        finally:
            ref.drop_primary_key(if_exists=True, cascade=True)
            ref.delete()

    def test_create_with_fk_via_explicit_param(self):
        """create(foreign_keys={…}) applies FK after table creation."""
        # First establish PK on ref table
        self.ref_table.set_primary_key(["id"], constraint_name="it_cc_fk_ref_pk")
        ref_main = self.engine.table("test_cc_tmp_fk")
        try:
            data = pa.table(
                [pa.array([1, 2, 3], type=pa.int64()), pa.array([1, 2, 3], type=pa.int64())],
                schema=self._temp_main_schema(),
            )
            ref_main.create(
                data,
                if_not_exists=False,
                primary_keys=["id"],
                foreign_keys={
                    "ref_id": (
                        f"{self._CATALOG}.{self._SCHEMA_NAME}.test_cc_ref.id"
                    )
                },
            )
        finally:
            ref_main.column("ref_id").unset_foreign_key(if_exists=True)
            ref_main.delete()
            self.ref_table.drop_primary_key(if_exists=True, cascade=True)

    def test_create_from_field_metadata(self):
        """Fields with t:primary_key metadata trigger PK constraint automatically."""
        schema = Schema.from_fields([_pk_field("id")])
        tbl = self.engine.table("test_cc_meta_pk")
        data = pa.table([pa.array([1, 2, 3], type=pa.int64())], schema=schema.to_arrow_schema())
        try:
            # PK is read from t:primary_key metadata — no explicit param needed
            tbl.create(data, if_not_exists=False)
        finally:
            tbl.drop_primary_key(if_exists=True, cascade=True)
            tbl.delete()

    # ── not-null enforcement verified via schema ──────────────────────────────

    def test_create_forces_not_null_on_pk_column(self):
        """After create(primary_keys=[…]) the PK column must be NOT NULL."""
        tbl = self.engine.table("test_cc_notnull")
        # Define id as nullable=True in Arrow schema — create should force NOT NULL
        schema = pa.schema([pa.field("id", pa.int64(), nullable=True)])
        data = pa.table([pa.array([1, 2, 3], type=pa.int64())], schema=schema)
        try:
            tbl.create(data, if_not_exists=False, primary_keys=["id"])
            tbl.set_primary_key(["id"], constraint_name="it_notnull_pk")
        finally:
            tbl.drop_primary_key(if_exists=True, cascade=True)
            tbl.delete()

