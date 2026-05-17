"""Tests for :class:`SecretRef` and the :func:`secret` factory.

The interesting bits live in :meth:`SecretRef.__repr__` (which the
staged-script renderer reads literally) and the dual-form
:func:`secret` constructor.
"""
from __future__ import annotations

import unittest

from yggdrasil.databricks.workflow import SecretRef, secret


class TestSecretFactory(unittest.TestCase):
    """:func:`secret` accepts two-arg and ``"scope/key"`` forms."""

    def test_two_arg_form(self) -> None:
        ref = secret("vendor", "api-key")
        self.assertIsInstance(ref, SecretRef)
        self.assertEqual(ref.scope, "vendor")
        self.assertEqual(ref.key, "api-key")

    def test_single_arg_slash_form(self) -> None:
        ref = secret("vendor/api-key")
        self.assertEqual(ref.scope, "vendor")
        self.assertEqual(ref.key, "api-key")

    def test_single_arg_colon_form(self) -> None:
        ref = secret("vendor:api-key")
        self.assertEqual(ref.scope, "vendor")
        self.assertEqual(ref.key, "api-key")

    def test_single_arg_without_separator_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "no separator"):
            secret("just-a-scope")

    def test_empty_scope_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be non-empty"):
            secret("", "key")


class TestSecretRefRepr(unittest.TestCase):
    """``repr(SecretRef)`` must produce a valid runtime call expression.

    The staged-script renderer in :mod:`yggdrasil.databricks.jobs.task`
    reads this directly into the rendered invocation. Any change here
    is an ABI change for the staged code — guard it.
    """

    def test_repr_calls_runtime_secret(self) -> None:
        self.assertEqual(
            repr(SecretRef(scope="vendor", key="api-key")),
            "ygg.secret('vendor', 'api-key')",
        )

    def test_repr_escapes_special_characters(self) -> None:
        # Keys with quotes / unicode would otherwise produce syntactically
        # invalid Python on the runner side. ``repr`` is the safe form.
        ref = SecretRef(scope="vendor's-prod", key="API/Key — value")
        rendered = repr(ref)
        # Round-trip: the rendered expression must parse as Python.
        import ast
        ast.parse(rendered, mode="eval")

    def test_secret_ref_is_hashable(self) -> None:
        # Stage-time helpers stash SecretRefs in sets / dict keys.
        a = SecretRef(scope="vendor", key="key")
        b = SecretRef(scope="vendor", key="key")
        self.assertEqual({a, b}, {b})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
