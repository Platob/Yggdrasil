""":class:`DataType` is the single source of truth for Python typing.

Every typing-resolution call site in the codebase — ``safe_function``,
``cast.registry``, ``data_field``, ``arrow.python_defaults`` — routes
through these classmethods rather than forking its own copy of the
``get_origin`` / ``get_args`` / ``Annotated`` / ``NewType`` / Optional
unwrap / alias-prefix expansion logic. The tests below lock the
canonical surface.
"""
from __future__ import annotations

import unittest
from typing import Annotated, NewType, Optional, Union

from yggdrasil.arrow.tests import ArrowTestCase
from yggdrasil.data.types.base import DataType


# A NewType chain to exercise unwrap_newtype.
UserId = NewType("UserId", int)
WrappedUserId = NewType("WrappedUserId", UserId)


class TestExpandAlias(ArrowTestCase):
    """``DataType.PYHINT_ALIASES`` — the central alias table."""

    def test_expand_known_prefixes(self):
        self.assertEqual(DataType.expand_alias("pa.Table"), "pyarrow.Table")
        self.assertEqual(DataType.expand_alias("pl.DataFrame"), "polars.DataFrame")
        self.assertEqual(DataType.expand_alias("pd.DataFrame"), "pandas.DataFrame")
        self.assertEqual(DataType.expand_alias("np.ndarray"), "numpy.ndarray")
        self.assertEqual(DataType.expand_alias("ps.DataFrame"), "pyspark.DataFrame")

    def test_unknown_prefix_passes_through(self):
        self.assertEqual(DataType.expand_alias("MyOwn.Type"), "MyOwn.Type")
        self.assertEqual(DataType.expand_alias("int"), "int")

    def test_table_is_editable(self):
        # Third-party can register its own prefix.
        DataType.PYHINT_ALIASES["myco."] = "myco.lib."
        try:
            self.assertEqual(DataType.expand_alias("myco.Frame"), "myco.lib.Frame")
        finally:
            DataType.PYHINT_ALIASES.pop("myco.", None)


class TestStripAnnotated(ArrowTestCase):

    def test_strips_single_annotated(self):
        self.assertIs(DataType.strip_annotated(Annotated[int, "tag"]), int)

    def test_strips_nested_annotated(self):
        self.assertIs(
            DataType.strip_annotated(Annotated[Annotated[int, "a"], "b"]),
            int,
        )

    def test_passes_through_non_annotated(self):
        self.assertIs(DataType.strip_annotated(int), int)
        self.assertIs(DataType.strip_annotated(str), str)


class TestUnwrapNewtype(ArrowTestCase):

    def test_unwraps_single(self):
        self.assertIs(DataType.unwrap_newtype(UserId), int)

    def test_unwraps_chain(self):
        self.assertIs(DataType.unwrap_newtype(WrappedUserId), int)

    def test_passes_through_non_newtype(self):
        self.assertIs(DataType.unwrap_newtype(int), int)


class TestNormalizeHint(ArrowTestCase):

    def test_strips_annotated_then_unwraps_newtype(self):
        self.assertIs(
            DataType.normalize_hint(Annotated[UserId, "tag"]),
            int,
        )

    def test_passes_through_plain_type(self):
        self.assertIs(DataType.normalize_hint(int), int)


class TestUnwrapOptional(ArrowTestCase):

    def test_optional_int_unwraps(self):
        is_opt, inner = DataType.unwrap_optional(Optional[int])
        self.assertTrue(is_opt)
        self.assertIs(inner, int)

    def test_pep604_unwraps(self):
        is_opt, inner = DataType.unwrap_optional(int | None)
        self.assertTrue(is_opt)
        self.assertIs(inner, int)

    def test_non_union_stays(self):
        is_opt, inner = DataType.unwrap_optional(int)
        self.assertFalse(is_opt)
        self.assertIs(inner, int)

    def test_multi_union_stays(self):
        # Union[int, str, None] has TWO non-None arms — not Optional[T].
        hint = Union[int, str, None]
        is_opt, inner = DataType.unwrap_optional(hint)
        self.assertFalse(is_opt)
        self.assertEqual(inner, hint)


class TestUnwrapNullableHint(ArrowTestCase):
    """Field-flavoured Optional unwrap: ``(inner, has_null)``."""

    def test_optional_unwraps(self):
        inner, has_null = DataType.unwrap_nullable_hint(int | None)
        self.assertIs(inner, int)
        self.assertTrue(has_null)

    def test_non_optional(self):
        inner, has_null = DataType.unwrap_nullable_hint(int)
        self.assertIs(inner, int)
        self.assertFalse(has_null)

    def test_multi_union_keeps_shape(self):
        # Multi-arm Union keeps the union; nullability flag tracks None.
        hint = Union[int, str, None]
        inner, has_null = DataType.unwrap_nullable_hint(hint)
        self.assertEqual(inner, Union[int, str, None])
        self.assertTrue(has_null)

    def test_bare_none_hint(self):
        # ``None`` as a hint stays ``None`` (not a Union), so it flows
        # through the non-Union return. ``type(None)`` flows the same
        # way — Python's typing system treats ``None`` and ``type(None)``
        # interchangeably as annotations.
        inner, has_null = DataType.unwrap_nullable_hint(type(None))
        self.assertIs(inner, type(None))
        self.assertFalse(has_null)


class TestIsRuntimeValue(ArrowTestCase):

    def test_value_is_runtime(self):
        self.assertTrue(DataType.is_runtime_value(42))
        self.assertTrue(DataType.is_runtime_value("x"))
        self.assertTrue(DataType.is_runtime_value([]))

    def test_class_is_not_runtime(self):
        self.assertFalse(DataType.is_runtime_value(int))
        self.assertFalse(DataType.is_runtime_value(str))

    def test_generic_alias_is_not_runtime(self):
        self.assertFalse(DataType.is_runtime_value(list[int]))


class TestResolveStrAnnotation(ArrowTestCase):

    def test_resolves_builtin(self):
        self.assertIs(DataType.resolve_str_annotation("int"), int)
        self.assertIs(DataType.resolve_str_annotation("str"), str)

    def test_resolves_typing_form(self):
        self.assertEqual(
            DataType.resolve_str_annotation("Optional[int]"),
            Optional[int],
        )

    def test_resolves_alias_prefix(self):
        result = DataType.resolve_str_annotation("pa.Table")
        import pyarrow as pa
        self.assertIs(result, pa.Table)

    def test_unresolvable_returns_string(self):
        result = DataType.resolve_str_annotation("does.not.exist.Anywhere")
        self.assertEqual(result, "does.not.exist.Anywhere")


class TestResolveFunctionAnnotations(ArrowTestCase):

    def test_resolves_function(self):
        def f(x: int, y: str) -> bool: ...
        annotations = DataType.resolve_function_annotations(f)
        self.assertEqual(annotations, {"x": int, "y": str, "return": bool})

    def test_resolves_stringified_aliases(self):
        # PEP-563-style stringified annotations using short aliases —
        # ``inspect.get_annotations(eval_str=True)`` would leave these as
        # strings; the per-annotation fallback through
        # ``resolve_str_annotation`` finishes the job.
        def f(t: "pa.Table") -> "pa.RecordBatch": ...
        annotations = DataType.resolve_function_annotations(f)
        import pyarrow as pa
        self.assertIs(annotations["t"], pa.Table)
        self.assertIs(annotations["return"], pa.RecordBatch)

    def test_empty_signature(self):
        def f(): ...
        self.assertEqual(DataType.resolve_function_annotations(f), {})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
