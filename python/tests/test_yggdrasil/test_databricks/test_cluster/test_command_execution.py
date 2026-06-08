"""Tests for :class:`CommandExecution` after the ``@dataclass`` removal.

Covers the surface the rest of the code paths depend on: keyword-only
construction past ``context`` / ``command_id``, the language default
(Python for pyfunc, Python for no-language SQL contexts, the bound
context's language otherwise), the environ coercion (iterables of
pairs → dict), and the ``(context_id, command_id)`` equality/hash
contract that two handles to the same in-flight command rely on.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from databricks.sdk.service.compute import Language

from yggdrasil.databricks.compute.command_execution import CommandExecution


def _ctx(context_id: str = "ctx-1", language: Language | None = None) -> MagicMock:
    """Build a minimal :class:`ExecutionContext` stand-in.

    ``CommandExecution.__init__`` reads only ``context.context_id``
    (for hash/equality) and ``context.language`` (for the language
    default), so a MagicMock with those two attributes is enough.
    """
    ctx = MagicMock(name="ExecutionContext")
    ctx.context_id = context_id
    ctx.language = language
    return ctx


class TestInitialization:

    def test_minimal_construction(self):
        cmd = CommandExecution(_ctx(), command_id="cmd-1")
        assert cmd.command_id == "cmd-1"
        assert cmd.command is None
        assert cmd.pyfunc is None
        assert cmd.environ is None
        # Default language: ``Language.PYTHON`` when neither a pyfunc
        # is set nor the context carries a language.
        assert cmd.language == Language.PYTHON

    def test_language_inherits_from_context(self):
        cmd = CommandExecution(_ctx(language=Language.SQL))
        assert cmd.language == Language.SQL

    def test_explicit_language_wins_over_context(self):
        cmd = CommandExecution(
            _ctx(language=Language.SQL), language=Language.SCALA,
        )
        assert cmd.language == Language.SCALA

    def test_pyfunc_forces_python_when_language_unset(self):
        cmd = CommandExecution(_ctx(language=Language.SQL), pyfunc=lambda: 1)
        assert cmd.language == Language.PYTHON

    def test_non_init_attrs_default_to_none(self):
        cmd = CommandExecution(_ctx())
        # _details / _remote_payload_path are populated by the
        # SDK lifecycle later; they must start unset.
        assert cmd._details is None
        assert cmd._remote_payload_path is None
        # The shutdown-registration flag is the bare-bool replacement
        # for the dataclass-side ``field(init=False)`` slot.
        assert cmd._shutdown_registered is False

    def test_keyword_only_past_context_and_command_id(self):
        # All beyond ``command_id`` are keyword-only — calling
        # positionally is a TypeError. Guards against accidental
        # positional drift if the field order changes.
        with pytest.raises(TypeError):
            CommandExecution(_ctx(), "cmd-1", Language.SQL)  # type: ignore[misc]


class TestEnvironCoercion:

    def test_dict_pass_through(self):
        cmd = CommandExecution(_ctx(), environ={"A": "1"})
        assert cmd.environ == {"A": "1"}

    def test_iterable_of_pairs_coerced(self):
        cmd = CommandExecution(_ctx(), environ=[("A", "1"), ("B", "2")])
        assert cmd.environ == {"A": "1", "B": "2"}

    def test_invalid_environ_raises_value_error(self):
        with pytest.raises(ValueError) as exc:
            CommandExecution(_ctx(), environ=42)  # type: ignore[arg-type]
        assert "environ" in str(exc.value)


class TestEqualityAndHash:
    """Two handles to the *same* command (same context + same
    command_id) must compare equal and hash to the same bucket — the
    legacy dataclass-generated ``__eq__`` compared no fields (all
    ``compare=False``) and collapsed every instance to equal, which
    silently broke ``set`` / dict lookups across distinct commands.
    """

    def test_same_context_same_command_id_are_equal(self):
        ctx = _ctx()
        a = CommandExecution(ctx, command_id="cmd-1")
        b = CommandExecution(ctx, command_id="cmd-1")
        assert a == b
        assert hash(a) == hash(b)

    def test_different_command_id_unequal(self):
        ctx = _ctx()
        a = CommandExecution(ctx, command_id="cmd-1")
        b = CommandExecution(ctx, command_id="cmd-2")
        assert a != b
        assert hash(a) != hash(b)

    def test_different_context_unequal(self):
        a = CommandExecution(_ctx("ctx-1"), command_id="cmd-1")
        b = CommandExecution(_ctx("ctx-2"), command_id="cmd-1")
        assert a != b

    def test_equality_returns_notimplemented_for_other_types(self):
        cmd = CommandExecution(_ctx(), command_id="cmd-1")
        # ``a == 42`` must not blow up — falls through to identity.
        assert (cmd == 42) is False

    def test_usable_as_dict_key_and_set_member(self):
        ctx = _ctx()
        a = CommandExecution(ctx, command_id="cmd-1")
        b = CommandExecution(ctx, command_id="cmd-1")
        c = CommandExecution(ctx, command_id="cmd-2")

        # ``a`` and ``b`` collapse to one entry (same identity).
        assert {a, b, c} == {a, c}
        bucket = {a: "first"}
        bucket[b] = "second"  # overwrites — same key
        bucket[c] = "third"
        assert len(bucket) == 2
        assert bucket[a] == "second"
        assert bucket[c] == "third"
