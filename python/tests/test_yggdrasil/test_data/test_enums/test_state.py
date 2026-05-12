"""Behavior tests for :class:`yggdrasil.data.enums.state.State`.

``State`` is the cross-backend execution-state enum: every async surface
(Databricks warehouse / Spark / Mongo / Postgres statement results,
plus any future job-style API) maps its backend-specific status into
this fixed vocabulary so ``done`` / ``failed`` / ``started`` can be
derived from a single source.

The contract under test:

* The five canonical members exist and have stable integer codes.
* The ``is_*`` predicates match the membership-set semantics that the
  legacy warehouse ``DONE_STATES`` / ``FAILED_STATES`` encoded:
  CANCELED counts as failed, CLOSED maps to SUCCEEDED, RUNNING /
  PENDING are non-terminal.
* :meth:`State.from_` is forgiving — alias strings, SDK-style enum
  objects (anything with a ``.name``), integer codes, mixed-case
  tokens, and ``None`` all resolve. Unknown / illegal inputs raise.
"""
from __future__ import annotations

import pytest

from yggdrasil.data.enums.state import State


class TestCanonicalMembers:

    def test_member_count(self) -> None:
        assert len(State) == 5

    def test_members_round_trip_via_int(self) -> None:
        for m in State:
            assert State(int(m)) is m

    def test_codes_are_stable(self) -> None:
        # Stable codes — these are what get serialized into pickled
        # caches and event payloads. Bumping them is a breaking change.
        assert int(State.PENDING) == 0
        assert int(State.RUNNING) == 1
        assert int(State.SUCCEEDED) == 2
        assert int(State.FAILED) == 3
        assert int(State.CANCELED) == 4


class TestPredicates:
    """``is_done`` / ``is_failed`` / ``is_started`` membership semantics."""

    def test_done_states(self) -> None:
        for m in (State.SUCCEEDED, State.FAILED, State.CANCELED):
            assert m.is_done, f"{m.name} should be terminal"
        for m in (State.PENDING, State.RUNNING):
            assert not m.is_done

    def test_failed_states(self) -> None:
        # CANCELED is failed — the warehouse path raises on cancel.
        for m in (State.FAILED, State.CANCELED):
            assert m.is_failed
        for m in (State.PENDING, State.RUNNING, State.SUCCEEDED):
            assert not m.is_failed

    def test_started_states(self) -> None:
        # Anything past PENDING counts as started.
        for m in (State.RUNNING, State.SUCCEEDED, State.FAILED, State.CANCELED):
            assert m.is_started
        assert not State.PENDING.is_started

    def test_pending_only_pending(self) -> None:
        assert State.PENDING.is_pending
        for m in (State.RUNNING, State.SUCCEEDED, State.FAILED, State.CANCELED):
            assert not m.is_pending

    def test_running_only_running(self) -> None:
        assert State.RUNNING.is_running
        for m in (State.PENDING, State.SUCCEEDED, State.FAILED, State.CANCELED):
            assert not m.is_running

    def test_succeeded_only_succeeded(self) -> None:
        assert State.SUCCEEDED.is_succeeded
        for m in (State.PENDING, State.RUNNING, State.FAILED, State.CANCELED):
            assert not m.is_succeeded

    def test_canceled_only_canceled(self) -> None:
        assert State.CANCELED.is_canceled
        for m in (State.PENDING, State.RUNNING, State.SUCCEEDED, State.FAILED):
            assert not m.is_canceled


class TestFromIdentity:

    def test_state_passes_through(self) -> None:
        for m in State:
            assert State.from_(m) is m

    def test_none_returns_pending(self) -> None:
        assert State.from_(None) is State.PENDING

    def test_none_returns_default_when_supplied(self) -> None:
        assert State.from_(None, default=State.RUNNING) is State.RUNNING

    def test_integer_codes(self) -> None:
        for m in State:
            assert State.from_(int(m)) is m

    def test_invalid_integer_raises(self) -> None:
        with pytest.raises(ValueError):
            State.from_(99)

    def test_invalid_integer_returns_default(self) -> None:
        assert State.from_(99, default=State.PENDING) is State.PENDING


class TestFromStringAliases:
    """Common backend tokens normalize to canonical members."""

    @pytest.mark.parametrize(
        "token, expected",
        [
            # Pending family.
            ("pending", State.PENDING),
            ("PENDING", State.PENDING),
            ("queued", State.PENDING),
            ("waiting", State.PENDING),
            ("submitted", State.PENDING),
            ("not-started", State.PENDING),
            ("not started", State.PENDING),
            ("", State.PENDING),
            # Running family.
            ("running", State.RUNNING),
            ("RUNNING", State.RUNNING),
            ("in_progress", State.RUNNING),
            ("in-progress", State.RUNNING),
            ("in progress", State.RUNNING),
            ("active", State.RUNNING),
            ("executing", State.RUNNING),
            # Succeeded family — including the Databricks ``CLOSED`` token
            # which maps to SUCCEEDED (terminal, not an error).
            ("succeeded", State.SUCCEEDED),
            ("SUCCESS", State.SUCCEEDED),
            ("completed", State.SUCCEEDED),
            ("ok", State.SUCCEEDED),
            ("finished", State.SUCCEEDED),
            ("CLOSED", State.SUCCEEDED),
            # Failed family.
            ("failed", State.FAILED),
            ("FAIL", State.FAILED),
            ("error", State.FAILED),
            ("errored", State.FAILED),
            # Canceled family — British spelling + abort synonyms.
            ("canceled", State.CANCELED),
            ("cancelled", State.CANCELED),
            ("aborted", State.CANCELED),
            ("killed", State.CANCELED),
        ],
    )
    def test_alias_resolves(self, token: str, expected: State) -> None:
        assert State.from_(token) is expected

    def test_unknown_string_raises(self) -> None:
        with pytest.raises(ValueError):
            State.from_("definitely_not_a_state")

    def test_unknown_string_returns_default(self) -> None:
        assert (
            State.from_("definitely_not_a_state", default=State.PENDING)
            is State.PENDING
        )


class TestFromSdkLikeObject:
    """SDK enums (Databricks ``StatementState`` etc.) expose ``.name`` /
    ``.value`` — ``from_`` should normalize via either."""

    def test_object_with_name(self) -> None:
        class _Fake:
            name = "RUNNING"

        assert State.from_(_Fake()) is State.RUNNING

    def test_object_with_string_value(self) -> None:
        class _Fake:
            name = None  # not a string — fall through to .value
            value = "SUCCEEDED"

        assert State.from_(_Fake()) is State.SUCCEEDED

    def test_databricks_statement_state(self) -> None:
        # Don't import at module level — the SDK is an optional dep and
        # we want this file importable in a base install. Also tolerate
        # SDK imports that blow up on unrelated env issues (cryptography
        # / cffi binding mismatches surface as PanicException here).
        try:
            from databricks.sdk.service.sql import StatementState
        except BaseException as exc:
            pytest.skip(f"databricks SDK not importable: {exc!r}")

        assert State.from_(StatementState.SUCCEEDED) is State.SUCCEEDED
        assert State.from_(StatementState.PENDING) is State.PENDING
        assert State.from_(StatementState.RUNNING) is State.RUNNING
        assert State.from_(StatementState.FAILED) is State.FAILED
        assert State.from_(StatementState.CANCELED) is State.CANCELED
        # CLOSED is terminal-but-not-an-error — should bucket with
        # SUCCEEDED so the unified ``failed`` predicate matches the
        # legacy warehouse semantics.
        assert State.from_(StatementState.CLOSED) is State.SUCCEEDED


class TestIsValid:

    def test_valid_inputs(self) -> None:
        for value in (State.PENDING, 0, "pending", "queued", None, "closed"):
            assert State.is_valid(value)

    def test_invalid_inputs(self) -> None:
        for value in ("not_a_state", 99):
            assert not State.is_valid(value)


class TestStr:

    def test_str_is_lowercase_name(self) -> None:
        assert str(State.PENDING) == "pending"
        assert str(State.SUCCEEDED) == "succeeded"
