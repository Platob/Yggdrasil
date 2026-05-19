"""Tests for the unity statement layer and :class:`UnityExecutionPlan`."""
from __future__ import annotations

from yggdrasil.arrow.tests import ArrowTestCase
from yggdrasil.data.data_field import Field
from yggdrasil.data.enums import Mode
from yggdrasil.data.enums.state import State
from yggdrasil.data.executor import StatementExecutor
from yggdrasil.data.schema import Schema
from yggdrasil.data.statement import PreparedStatement
from yggdrasil.data.types.primitive import Int64Type, StringType
from yggdrasil.io.path import LocalPath
from yggdrasil.io.session import Session
from yggdrasil.unity import (
    CreateCatalog,
    CreateSchema,
    CreateTable,
    CreateView,
    DropTable,
    Insert,
    Select,
    ShowCatalogs,
    ShowSchemas,
    ShowTables,
    ShowViews,
    UnityCatalog,
    UnityExecutionPlan,
    UnityStatement,
    UnityStatementResult,
)
from yggdrasil.unity.fs import FSEngine, FSTable


def _sales_schema() -> Schema:
    return Schema([
        Field(name="id", dtype=Int64Type()),
        Field(name="name", dtype=StringType()),
    ])


class TestEngineInheritsSession(ArrowTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.engine = FSEngine(base=LocalPath(str(self.tmp_path / "wh")))

    def test_engine_is_statement_executor(self) -> None:
        self.assertIsInstance(self.engine, StatementExecutor)

    def test_engine_is_session(self) -> None:
        self.assertIsInstance(self.engine, Session)

    def test_engine_singleton_same_base(self) -> None:
        twin = FSEngine(base=LocalPath(str(self.tmp_path / "wh")))
        self.assertIs(twin, self.engine)

    def test_engine_singleton_different_base(self) -> None:
        other = FSEngine(base=LocalPath(str(self.tmp_path / "other")))
        self.assertIsNot(other, self.engine)

    def test_engine_missing_base_raises(self) -> None:
        with self.assertRaises(TypeError):
            FSEngine()


class TestUnityStatementBasics(ArrowTestCase):
    def test_statement_extends_prepared_statement(self) -> None:
        self.assertTrue(issubclass(CreateCatalog, UnityStatement))
        self.assertTrue(issubclass(UnityStatement, PreparedStatement))

    def test_statement_render_text_create_catalog(self) -> None:
        stmt = CreateCatalog("main")
        self.assertIn("CREATE CATALOG", stmt.text)
        self.assertIn("main", stmt.text)

    def test_statement_render_text_create_table(self) -> None:
        stmt = CreateTable(
            "main", "default", "sales", schema=_sales_schema(),
            partition_by=("name",),
        )
        self.assertIn("CREATE TABLE", stmt.text)
        self.assertIn("main.default.sales", stmt.text)
        self.assertIn("PARTITIONED BY (name)", stmt.text)

    def test_statement_render_text_insert(self) -> None:
        stmt = Insert("main", "default", "sales", data=None, mode=Mode.APPEND)
        self.assertIn("INSERT INTO main.default.sales", stmt.text)


class TestExecuteIndividualStatements(ArrowTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.engine = FSEngine(base=LocalPath(str(self.tmp_path / "wh")))

    def test_execute_create_catalog(self) -> None:
        result = self.engine.execute(CreateCatalog("main"))
        self.assertIsInstance(result, UnityStatementResult)
        self.assertTrue(result.done)
        self.assertEqual(result.state, State.SUCCEEDED)
        self.assertIsInstance(result.output, UnityCatalog)
        self.assertEqual(result.output.full_name, "main")
        self.assertTrue(self.engine["main"].exists)

    def test_execute_create_chain(self) -> None:
        self.engine.execute(CreateCatalog("main"))
        self.engine.execute(CreateSchema("main", "default"))
        result = self.engine.execute(CreateTable(
            "main", "default", "sales", schema=_sales_schema(),
        ))
        self.assertTrue(result.done)
        self.assertIsInstance(result.output, FSTable)

    def test_execute_insert_returns_row_count(self) -> None:
        self.engine.execute(CreateCatalog("main"))
        self.engine.execute(CreateSchema("main", "default"))
        self.engine.execute(CreateTable(
            "main", "default", "sales", schema=_sales_schema(),
        ))
        data = self.pa.table({"id": [1, 2, 3], "name": ["a", "b", "c"]})
        result = self.engine.execute(Insert(
            "main", "default", "sales", data=data,
        ))
        self.assertEqual(result.output, 3)

    def test_execute_select_returns_tabular_forwarding(self) -> None:
        self.engine.execute(CreateCatalog("main"))
        self.engine.execute(CreateSchema("main", "default"))
        self.engine.execute(CreateTable(
            "main", "default", "sales", schema=_sales_schema(),
        ))
        self.engine.execute(Insert(
            "main", "default", "sales",
            data=self.pa.table({"id": [1, 2], "name": ["a", "b"]}),
        ))
        result = self.engine.execute(Select("main", "default", "sales"))
        # The result is a Tabular — read_arrow_table forwards to the
        # resolved table via _read_arrow_batches.
        arrow = result.read_arrow_table()
        self.assertEqual(arrow.num_rows, 2)
        self.assertEqual(arrow.column_names, ["id", "name"])

    def test_execute_show_catalogs(self) -> None:
        self.engine.execute(CreateCatalog("main"))
        self.engine.execute(CreateCatalog("other"))
        result = self.engine.execute(ShowCatalogs())
        self.assertEqual(result.output, ["main", "other"])

    def test_execute_show_schemas_tables_views(self) -> None:
        self.engine.execute(CreateCatalog("main"))
        self.engine.execute(CreateSchema("main", "default"))
        self.engine.execute(CreateTable(
            "main", "default", "t", schema=_sales_schema(),
        ))
        self.engine.execute(CreateView(
            "main", "default", "v", source="main.default.t",
        ))
        self.assertEqual(
            self.engine.execute(ShowSchemas("main")).output, ["default"],
        )
        self.assertEqual(
            self.engine.execute(ShowTables("main", "default")).output, ["t"],
        )
        self.assertEqual(
            self.engine.execute(ShowViews("main", "default")).output, ["v"],
        )

    def test_execute_drop_table(self) -> None:
        self.engine.execute(CreateCatalog("main"))
        self.engine.execute(CreateSchema("main", "default"))
        self.engine.execute(CreateTable(
            "main", "default", "t", schema=_sales_schema(),
        ))
        self.engine.execute(DropTable("main", "default", "t"))
        self.assertFalse(self.engine["main"]["default"].table("t").exists)

    def test_execute_missing_table_raises(self) -> None:
        self.engine.execute(CreateCatalog("main"))
        self.engine.execute(CreateSchema("main", "default"))
        with self.assertRaises(FileNotFoundError):
            self.engine.execute(Insert(
                "main", "default", "nope",
                data=self.pa.table({"id": [1], "name": ["a"]}),
            ))

    def test_send_without_start_idle(self) -> None:
        idle = self.engine.send(CreateCatalog("main"), start=False)
        self.assertEqual(idle.state, State.IDLE)
        self.assertFalse(self.engine.catalog("main").exists)
        idle.start()
        self.assertTrue(idle.done)
        self.assertTrue(self.engine["main"].exists)


class TestUnityExecutionPlan(ArrowTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.engine = FSEngine(base=LocalPath(str(self.tmp_path / "wh")))

    def test_plan_is_immutable(self) -> None:
        base = UnityExecutionPlan()
        next_plan = base.then_create_catalog("main")
        self.assertEqual(len(base), 0)
        self.assertEqual(len(next_plan), 1)

    def test_plan_iter_and_len(self) -> None:
        plan = (
            UnityExecutionPlan()
            .then_create_catalog("main")
            .then_create_schema("main", "default")
        )
        self.assertEqual(len(plan), 2)
        kinds = [type(s).__name__ for s in plan]
        self.assertEqual(kinds, ["CreateCatalog", "CreateSchema"])

    def test_plan_then_rejects_non_unity_statement(self) -> None:
        with self.assertRaises(TypeError):
            UnityExecutionPlan().then(PreparedStatement(text="SELECT 1"))

    def test_plan_concatenation(self) -> None:
        a = UnityExecutionPlan().then_create_catalog("main")
        b = UnityExecutionPlan().then_create_schema("main", "default")
        combined = a + b
        self.assertEqual(len(combined), 2)

    def test_plan_execute_via_engine_method(self) -> None:
        plan = (
            UnityExecutionPlan()
            .then_create_catalog("main")
            .then_create_schema("main", "default")
            .then_create_table(
                "main", "default", "sales", schema=_sales_schema(),
            )
            .then_insert(
                "main", "default", "sales",
                self.pa.table({"id": [1, 2, 3], "name": ["a", "b", "c"]}),
            )
        )
        results = self.engine.execute_plan(plan)
        self.assertEqual(len(results), 4)
        self.assertTrue(all(r.done for r in results))
        self.assertEqual(results[-1].output, 3)
        self.assertEqual(
            self.engine["main"]["default"]["sales"].read_arrow_table().num_rows,
            3,
        )

    def test_plan_execute_via_plan_method(self) -> None:
        plan = UnityExecutionPlan().then_create_catalog("main")
        results = plan.execute(self.engine)
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].done)

    def test_plan_repr_includes_text(self) -> None:
        plan = (
            UnityExecutionPlan()
            .then_create_catalog("main")
            .then_create_schema("main", "default")
        )
        rep = repr(plan)
        self.assertIn("CREATE CATALOG", rep)
        self.assertIn("CREATE SCHEMA", rep)

    def test_plan_drop_cascade(self) -> None:
        # Build up, then a teardown plan reuses the same surface.
        (
            UnityExecutionPlan()
            .then_create_catalog("main")
            .then_create_schema("main", "default")
            .then_create_table("main", "default", "t", schema=_sales_schema())
            .execute(self.engine)
        )
        teardown = UnityExecutionPlan().then_drop_catalog(
            "main", recursive=True,
        )
        teardown.execute(self.engine)
        self.assertFalse(self.engine.catalog("main").exists)

    def test_plan_select_then_read(self) -> None:
        (
            UnityExecutionPlan()
            .then_create_catalog("main")
            .then_create_schema("main", "default")
            .then_create_table("main", "default", "sales", schema=_sales_schema())
            .then_insert(
                "main", "default", "sales",
                self.pa.table({"id": [7], "name": ["g"]}),
            )
            .execute(self.engine)
        )
        results = (
            UnityExecutionPlan()
            .then_select("main", "default", "sales")
            .execute(self.engine)
        )
        self.assertEqual(results[0].read_arrow_table().num_rows, 1)

    def test_plan_raise_error_false_continues(self) -> None:
        plan = (
            UnityExecutionPlan()
            .then_create_catalog("main")
            .then_insert(  # Fails — table doesn't exist
                "main", "default", "missing",
                self.pa.table({"id": [1], "name": ["a"]}),
            )
            .then_show_catalogs()  # Should still run
        )
        results = plan.execute(self.engine, raise_error=False)
        self.assertEqual(len(results), 3)
        self.assertTrue(results[0].done and not results[0].failed)
        self.assertTrue(results[1].failed)
        self.assertEqual(results[2].output, ["main"])
