"""Field engine-projection caching — lazy build, parent-cascade invalidation.

``Field.to_arrow_field`` / ``to_arrow_schema`` / ``to_polars_field`` /
``to_polars_schema`` / ``to_pyspark_field`` / ``to_spark_schema`` are
lazy-cached on the Field. Mutating the field (or one of its nested
children) drops the cache for the field itself and every ancestor
reachable via :attr:`Field.parent`, so the next access rebuilds the
engine projection from the current state.
"""
from __future__ import annotations

from yggdrasil.arrow.tests import ArrowTestCase
from yggdrasil.data.data_field import Field
from yggdrasil.data.types.nested import StructType
from yggdrasil.data.types.primitive import IntegerType, StringType


def _build_struct() -> Field:
    return Field(
        name="root",
        dtype=StructType(fields=(
            Field(name="a", dtype=IntegerType(byte_size=8, signed=True)),
            Field(name="b", dtype=StringType()),
        )),
    )


class TestArrowCache(ArrowTestCase):

    def test_to_arrow_field_returns_cached_instance(self) -> None:
        f = Field(name="x", dtype=IntegerType(byte_size=8, signed=True))
        first = f.to_arrow_field()
        second = f.to_arrow_field()
        self.assertIs(first, second)

    def test_to_arrow_schema_returns_cached_instance(self) -> None:
        f = _build_struct()
        first = f.to_arrow_schema()
        second = f.to_arrow_schema()
        self.assertIs(first, second)

    def test_with_name_invalidates_cache(self) -> None:
        f = Field(name="x", dtype=IntegerType(byte_size=8, signed=True))
        cached = f.to_arrow_field()
        f.with_name("y", inplace=True)
        rebuilt = f.to_arrow_field()
        self.assertIsNot(cached, rebuilt)
        self.assertEqual(rebuilt.name, "y")

    def test_child_mutation_cascades_to_parent_arrow_schema(self) -> None:
        parent = _build_struct()
        cached_schema = parent.to_arrow_schema()
        # Mutate a child in place; its ``parent`` is the root field
        # (stamped by ``_adopt_children`` during root.__init__), so the
        # cascading invalidation should clear ``cached_schema``.
        child = parent.children[0]
        child.with_name("renamed", inplace=True)
        rebuilt = parent.to_arrow_schema()
        self.assertIsNot(cached_schema, rebuilt)
        self.assertEqual(rebuilt.field(0).name, "renamed")

    def test_public_invalidate_cache_cascades(self) -> None:
        parent = _build_struct()
        cached_parent = parent.to_arrow_schema()
        child = parent.children[0]
        cached_child = child.to_arrow_field()
        child.invalidate_cache()
        # Both child and parent rebuild fresh instances.
        self.assertIsNot(cached_child, child.to_arrow_field())
        self.assertIsNot(cached_parent, parent.to_arrow_schema())

    def test_invalidate_cache_without_cascade_keeps_parent_cached(self) -> None:
        parent = _build_struct()
        cached_parent = parent.to_arrow_schema()
        child = parent.children[0]
        cached_child = child.to_arrow_field()
        child.invalidate_cache(cascade=False)
        # Child rebuilds but parent keeps its cached schema.
        self.assertIsNot(cached_child, child.to_arrow_field())
        self.assertIs(cached_parent, parent.to_arrow_schema())


class TestPolarsCache(ArrowTestCase):

    def test_to_polars_field_returns_cached_instance(self) -> None:
        try:
            import polars  # noqa: F401
        except ImportError:
            self.skipTest("polars not installed")
        f = Field(name="x", dtype=IntegerType(byte_size=8, signed=True))
        first = f.to_polars_field()
        second = f.to_polars_field()
        self.assertIs(first, second)

    def test_child_mutation_invalidates_parent_polars_schema(self) -> None:
        try:
            import polars  # noqa: F401
        except ImportError:
            self.skipTest("polars not installed")
        parent = _build_struct()
        cached = parent.to_polars_schema()
        parent.children[1].with_name("b_renamed", inplace=True)
        rebuilt = parent.to_polars_schema()
        self.assertIsNot(cached, rebuilt)
        self.assertIn("b_renamed", rebuilt)


class TestSparkCache(ArrowTestCase):

    def test_to_pyspark_field_returns_cached_instance(self) -> None:
        try:
            import pyspark.sql  # noqa: F401
        except ImportError:
            self.skipTest("pyspark not installed")
        f = Field(name="x", dtype=IntegerType(byte_size=8, signed=True))
        first = f.to_pyspark_field()
        second = f.to_pyspark_field()
        self.assertIs(first, second)

    def test_child_mutation_invalidates_parent_spark_schema(self) -> None:
        try:
            import pyspark.sql  # noqa: F401
        except ImportError:
            self.skipTest("pyspark not installed")
        parent = _build_struct()
        cached = parent.to_spark_schema()
        parent.children[0].with_nullable(False, inplace=True)
        rebuilt = parent.to_spark_schema()
        self.assertIsNot(cached, rebuilt)
        # Spark StructField carries the nullable flag — the rebuilt
        # schema reflects the new value.
        self.assertFalse(rebuilt.fields[0].nullable)
