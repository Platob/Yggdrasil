"""Tests for the :class:`PlanTypeId` classification + specialised sub-plans."""
from __future__ import annotations

from yggdrasil.arrow.tests import ArrowTestCase
from yggdrasil.data.data_field import Field
from yggdrasil.data.enums import Mode
from yggdrasil.data.schema import Schema
from yggdrasil.data.types.primitive import Int64Type, StringType
from yggdrasil.io.path import LocalPath
from yggdrasil.unity import (
    CreateCatalog,
    CreateTable,
    ExecutionPlan,
    Insert,
    InsertExecutionPlan,
    MutationExecutionPlan,
    PlanCategory,
    PlanTypeId,
    Select,
    SelectExecutionPlan,
    ShowCatalogs,
    ShowExecutionPlan,
)
from yggdrasil.unity.fs import FSEngine


def _sales_schema() -> Schema:
    return Schema([
        Field(name="id", dtype=Int64Type()),
        Field(name="name", dtype=StringType()),
    ])


# ============================================================
# PlanTypeId / PlanCategory
# ============================================================


class TestPlanTypeIdClassification(ArrowTestCase):
    def test_category_derivation(self) -> None:
        self.assertIs(PlanTypeId.CREATE_CATALOG.category, PlanCategory.DDL)
        self.assertIs(PlanTypeId.DROP_TABLE.category, PlanCategory.DDL)
        self.assertIs(PlanTypeId.INSERT.category, PlanCategory.DML)
        self.assertIs(PlanTypeId.SELECT.category, PlanCategory.DQL)
        self.assertIs(PlanTypeId.SHOW_TABLES.category, PlanCategory.META)

    def test_is_mutation(self) -> None:
        self.assertTrue(PlanTypeId.CREATE_CATALOG.is_mutation)
        self.assertTrue(PlanTypeId.INSERT.is_mutation)
        self.assertFalse(PlanTypeId.SELECT.is_mutation)
        self.assertFalse(PlanTypeId.SHOW_TABLES.is_mutation)

    def test_is_query(self) -> None:
        self.assertTrue(PlanTypeId.SELECT.is_query)
        self.assertTrue(PlanTypeId.SHOW_CATALOGS.is_query)
        self.assertFalse(PlanTypeId.CREATE_CATALOG.is_query)

    def test_is_create_and_is_drop(self) -> None:
        self.assertTrue(PlanTypeId.CREATE_CATALOG.is_create)
        self.assertFalse(PlanTypeId.CREATE_CATALOG.is_drop)
        self.assertTrue(PlanTypeId.DROP_TABLE.is_drop)
        self.assertFalse(PlanTypeId.DROP_TABLE.is_create)

    def test_statement_carries_plan_type_id(self) -> None:
        self.assertIs(CreateCatalog("x").plan_type_id, PlanTypeId.CREATE_CATALOG)
        self.assertIs(
            CreateTable("a", "b", "c", schema=_sales_schema()).plan_type_id,
            PlanTypeId.CREATE_TABLE,
        )
        self.assertIs(
            Insert("a", "b", "c", data=None).plan_type_id, PlanTypeId.INSERT,
        )
        self.assertIs(Select("a", "b", "c").plan_type_id, PlanTypeId.SELECT)
        self.assertIs(ShowCatalogs().plan_type_id, PlanTypeId.SHOW_CATALOGS)

    def test_statement_derived_flags(self) -> None:
        self.assertTrue(CreateCatalog("x").is_mutation)
        self.assertFalse(CreateCatalog("x").is_query)
        self.assertTrue(Select("a", "b", "c").is_query)
        self.assertFalse(Select("a", "b", "c").is_mutation)


# ============================================================
# ExecutionPlan categorisation queries
# ============================================================


class TestExecutionPlanCategorisation(ArrowTestCase):
    def test_plan_type_ids_listing(self) -> None:
        plan = (
            ExecutionPlan()
            .then_create_catalog("main")
            .then_create_schema("main", "default")
            .then_show_catalogs()
        )
        self.assertEqual(
            plan.plan_type_ids,
            (PlanTypeId.CREATE_CATALOG, PlanTypeId.CREATE_SCHEMA, PlanTypeId.SHOW_CATALOGS),
        )

    def test_categories_set(self) -> None:
        plan = (
            ExecutionPlan()
            .then_create_catalog("main")
            .then_show_catalogs()
        )
        self.assertEqual(plan.categories, frozenset({PlanCategory.DDL, PlanCategory.META}))

    def test_is_homogeneous(self) -> None:
        self.assertTrue(ExecutionPlan().is_homogeneous)
        self.assertTrue(
            ExecutionPlan().then_create_catalog("a").is_homogeneous
        )
        self.assertTrue(
            ExecutionPlan()
            .then_create_catalog("a")
            .then_create_catalog("b")
            .is_homogeneous
        )
        self.assertFalse(
            ExecutionPlan()
            .then_create_catalog("a")
            .then_show_catalogs()
            .is_homogeneous
        )

    def test_is_mutation_aggregate(self) -> None:
        self.assertFalse(ExecutionPlan().is_mutation)  # empty plan
        ddl = ExecutionPlan().then_create_catalog("a")
        self.assertTrue(ddl.is_mutation)
        self.assertFalse(ddl.is_query)
        mixed = ddl.then_show_catalogs()
        self.assertFalse(mixed.is_mutation)
        self.assertFalse(mixed.is_query)


# ============================================================
# Specialiser coercion
# ============================================================


class TestPlanSpecialisers(ArrowTestCase):
    def test_as_insert_plan_succeeds(self) -> None:
        plan = ExecutionPlan().then_insert(
            "main", "default", "sales",
            self.pa.table({"id": [1], "name": ["a"]}),
        )
        coerced = plan.as_insert_plan()
        self.assertIsInstance(coerced, InsertExecutionPlan)
        self.assertEqual(coerced.target_full_name, "main.default.sales")

    def test_as_insert_plan_rejects_non_insert(self) -> None:
        plan = ExecutionPlan().then_create_catalog("main")
        with self.assertRaises(TypeError):
            plan.as_insert_plan()

    def test_as_select_plan_rejects_zero_or_many(self) -> None:
        with self.assertRaises(ValueError):
            ExecutionPlan().as_select_plan()  # empty
        with self.assertRaises(ValueError):
            (
                ExecutionPlan()
                .then_select("a", "b", "c")
                .then_select("a", "b", "d")
                .as_select_plan()
            )

    def test_as_mutation_plan(self) -> None:
        plan = (
            ExecutionPlan()
            .then_create_catalog("main")
            .then_create_schema("main", "default")
            .then_insert(
                "main", "default", "t",
                self.pa.table({"x": [1]}),
            )
        )
        coerced = plan.as_mutation_plan()
        self.assertIsInstance(coerced, MutationExecutionPlan)

    def test_as_mutation_plan_rejects_query(self) -> None:
        plan = ExecutionPlan().then_show_catalogs()
        with self.assertRaises(TypeError):
            plan.as_mutation_plan()

    def test_as_show_plan(self) -> None:
        plan = (
            ExecutionPlan()
            .then_show_catalogs()
            .then_show_schemas("main")
        )
        coerced = plan.as_show_plan()
        self.assertIsInstance(coerced, ShowExecutionPlan)


# ============================================================
# InsertExecutionPlan
# ============================================================


class TestInsertExecutionPlan(ArrowTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.engine = FSEngine(base=LocalPath(str(self.tmp_path / "wh")))
        (
            ExecutionPlan()
            .then_create_catalog("main")
            .then_create_schema("main", "default")
            .then_create_table("main", "default", "sales", schema=_sales_schema())
            .execute(self.engine)
        )

    def test_for_table_starts_empty(self) -> None:
        plan = InsertExecutionPlan.for_table("main", "default", "sales")
        self.assertEqual(len(plan), 0)
        self.assertEqual(plan.target_full_name, "main.default.sales")

    def test_then_data_appends_insert(self) -> None:
        plan = (
            InsertExecutionPlan.for_table("main", "default", "sales")
            .then_data(self.pa.table({"id": [1], "name": ["a"]}))
            .then_data(self.pa.table({"id": [2], "name": ["b"]}))
        )
        self.assertEqual(len(plan), 2)
        self.assertTrue(plan.is_homogeneous)
        self.assertEqual(plan.plan_type_ids, (PlanTypeId.INSERT, PlanTypeId.INSERT))

    def test_with_mode_restamps_every_insert(self) -> None:
        plan = (
            InsertExecutionPlan.for_table("main", "default", "sales")
            .then_data(self.pa.table({"id": [1], "name": ["a"]}))
            .then_data(self.pa.table({"id": [2], "name": ["b"]}))
            .with_mode(Mode.OVERWRITE)
        )
        for stmt in plan:
            self.assertIs(stmt.mode, Mode.OVERWRITE)

    def test_total_rows_aggregates_results(self) -> None:
        plan = (
            InsertExecutionPlan.for_table("main", "default", "sales")
            .then_data(self.pa.table({"id": [1, 2], "name": ["a", "b"]}))
            .then_data(self.pa.table({"id": [3], "name": ["c"]}))
        )
        results = plan.execute(self.engine)
        self.assertEqual(InsertExecutionPlan.total_rows(results), 3)

    def test_constructor_rejects_cross_target(self) -> None:
        a = Insert("main", "default", "sales", data=None)
        b = Insert("main", "default", "other", data=None)
        with self.assertRaises(ValueError):
            InsertExecutionPlan(statements=(a, b))

    def test_constructor_rejects_non_insert(self) -> None:
        with self.assertRaises(TypeError):
            InsertExecutionPlan(statements=(CreateCatalog("x"),))

    def test_then_insert_forbidden(self) -> None:
        plan = InsertExecutionPlan.for_table("main", "default", "sales")
        with self.assertRaises(TypeError):
            plan.then_insert(
                "main", "default", "other",
                self.pa.table({"id": [1], "name": ["a"]}),
            )

    def test_then_create_catalog_rejected_by_validation(self) -> None:
        # Sub-plan inherits parent's then_create_catalog; validation rejects.
        plan = InsertExecutionPlan.for_table("main", "default", "sales")
        with self.assertRaises(TypeError):
            plan.then_create_catalog("other")


# ============================================================
# SelectExecutionPlan
# ============================================================


class TestSelectExecutionPlan(ArrowTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.engine = FSEngine(base=LocalPath(str(self.tmp_path / "wh")))
        (
            ExecutionPlan()
            .then_create_catalog("main")
            .then_create_schema("main", "default")
            .then_create_table("main", "default", "sales", schema=_sales_schema())
            .then_insert(
                "main", "default", "sales",
                self.pa.table({"id": [1, 2, 3], "name": ["a", "b", "c"]}),
            )
            .execute(self.engine)
        )

    def test_for_target(self) -> None:
        plan = SelectExecutionPlan.for_target("main", "default", "sales")
        self.assertEqual(len(plan), 1)
        self.assertEqual(plan.target_full_name, "main.default.sales")

    def test_must_have_exactly_one_select(self) -> None:
        with self.assertRaises(ValueError):
            SelectExecutionPlan(())
        with self.assertRaises(ValueError):
            SelectExecutionPlan((
                Select("a", "b", "c"),
                Select("a", "b", "d"),
            ))

    def test_constructor_rejects_non_select(self) -> None:
        with self.assertRaises(TypeError):
            SelectExecutionPlan((CreateCatalog("x"),))

    def test_columns_projects_subset(self) -> None:
        plan = (
            SelectExecutionPlan.for_target("main", "default", "sales")
            .columns("id")
        )
        self.assertEqual(plan.column_projection, ("id",))
        arrow = plan.read_arrow_table(self.engine)
        self.assertEqual(arrow.column_names, ["id"])

    def test_columns_missing_raises_keyerror(self) -> None:
        plan = (
            SelectExecutionPlan.for_target("main", "default", "sales")
            .columns("nonexistent")
        )
        with self.assertRaises(KeyError):
            plan.read_arrow_table(self.engine)

    def test_where_attaches_predicate(self) -> None:
        # Pred shape is opaque here — we just verify it lands on options.
        plan = SelectExecutionPlan.for_target("main", "default", "sales").where("dummy_pred")
        self.assertIsNotNone(plan.options)
        self.assertEqual(plan.options.predicate, "dummy_pred")

    def test_limit_sets_row_limit(self) -> None:
        plan = SelectExecutionPlan.for_target("main", "default", "sales").limit(2)
        self.assertIsNotNone(plan.options)
        self.assertEqual(plan.options.row_limit, 2)

    def test_limit_negative_raises(self) -> None:
        with self.assertRaises(ValueError):
            SelectExecutionPlan.for_target("main", "default", "sales").limit(-1)

    def test_refinements_are_immutable(self) -> None:
        base = SelectExecutionPlan.for_target("main", "default", "sales")
        a = base.columns("id")
        b = base.columns("name")
        self.assertIsNone(base.column_projection)
        self.assertEqual(a.column_projection, ("id",))
        self.assertEqual(b.column_projection, ("name",))

    def test_read_arrow_table(self) -> None:
        plan = SelectExecutionPlan.for_target("main", "default", "sales")
        arrow = plan.read_arrow_table(self.engine)
        self.assertEqual(arrow.num_rows, 3)
        self.assertEqual(arrow.column_names, ["id", "name"])


# ============================================================
# MutationExecutionPlan
# ============================================================


class TestMutationExecutionPlan(ArrowTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.engine = FSEngine(base=LocalPath(str(self.tmp_path / "wh")))

    def test_accepts_ddl_and_dml(self) -> None:
        plan = (
            MutationExecutionPlan()
            .then_create_catalog("main")
            .then_create_schema("main", "default")
            .then_create_table("main", "default", "t", schema=_sales_schema())
            .then_insert(
                "main", "default", "t",
                self.pa.table({"id": [1], "name": ["a"]}),
            )
        )
        self.assertTrue(plan.is_mutation)
        self.assertEqual(
            plan.categories,
            frozenset({PlanCategory.DDL, PlanCategory.DML}),
        )

    def test_rejects_select_at_construction(self) -> None:
        with self.assertRaises(TypeError):
            MutationExecutionPlan((Select("a", "b", "c"),))

    def test_rejects_show_at_construction(self) -> None:
        with self.assertRaises(TypeError):
            MutationExecutionPlan((ShowCatalogs(),))

    def test_summary_counts_creates_and_rows(self) -> None:
        plan = (
            MutationExecutionPlan()
            .then_create_catalog("main")
            .then_create_schema("main", "default")
            .then_create_table("main", "default", "t", schema=_sales_schema())
            .then_insert(
                "main", "default", "t",
                self.pa.table({"id": [1, 2, 3], "name": ["a", "b", "c"]}),
            )
        )
        results = plan.execute(self.engine)
        summary = MutationExecutionPlan.summary(results)
        self.assertEqual(summary[PlanTypeId.CREATE_CATALOG], 1)
        self.assertEqual(summary[PlanTypeId.CREATE_SCHEMA], 1)
        self.assertEqual(summary[PlanTypeId.CREATE_TABLE], 1)
        self.assertEqual(summary[PlanTypeId.INSERT], 3)


# ============================================================
# ShowExecutionPlan
# ============================================================


class TestShowExecutionPlan(ArrowTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.engine = FSEngine(base=LocalPath(str(self.tmp_path / "wh")))
        (
            ExecutionPlan()
            .then_create_catalog("main")
            .then_create_schema("main", "default")
            .then_create_table("main", "default", "t", schema=_sales_schema())
            .execute(self.engine)
        )

    def test_only_accepts_show_statements(self) -> None:
        with self.assertRaises(TypeError):
            ShowExecutionPlan((CreateCatalog("x"),))
        with self.assertRaises(TypeError):
            ShowExecutionPlan((Insert("a", "b", "c", data=None),))

    def test_then_show_chain(self) -> None:
        plan = (
            ShowExecutionPlan()
            .then_show_catalogs()
            .then_show_schemas("main")
            .then_show_tables("main", "default")
        )
        self.assertEqual(len(plan), 3)
        self.assertTrue(plan.is_query)

    def test_results_folds_output(self) -> None:
        plan = (
            ShowExecutionPlan()
            .then_show_catalogs()
            .then_show_schemas("main")
            .then_show_tables("main", "default")
        )
        results = plan.execute(self.engine)
        folded = ShowExecutionPlan.results(results)
        self.assertEqual(folded[PlanTypeId.SHOW_CATALOGS], ["main"])
        self.assertEqual(folded[PlanTypeId.SHOW_SCHEMAS], ["default"])
        self.assertEqual(folded[PlanTypeId.SHOW_TABLES], ["t"])

    def test_results_merges_same_type(self) -> None:
        (
            ExecutionPlan()
            .then_create_catalog("other")
            .then_create_schema("other", "default")
            .execute(self.engine)
        )
        plan = (
            ShowExecutionPlan()
            .then_show_schemas("main")
            .then_show_schemas("other")
        )
        results = plan.execute(self.engine)
        folded = ShowExecutionPlan.results(results)
        self.assertEqual(folded[PlanTypeId.SHOW_SCHEMAS], ["default"])
