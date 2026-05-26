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

from yggdrasil.enums.state import State


class TestCanonicalMembers:

    def test_member_count(self) -> None:
        assert len(State) == 11

    def test_members_round_trip_via_int(self) -> None:
        for m in State:
            assert State(int(m)) is m

    def test_codes_are_stable(self) -> None:
        # Stable codes — these are what get serialized into pickled
        # caches and event payloads. Bumping them is a breaking change.
        assert int(State.IDLE) == 0
        assert int(State.QUEUED) == 1
        assert int(State.PENDING) == 2
        assert int(State.ACCEPTED) == 3
        assert int(State.RUNNING) == 4
        assert int(State.PARTIAL) == 5
        assert int(State.SUCCEEDED) == 6
        assert int(State.REJECTED) == 7
        assert int(State.FAILED) == 8
        assert int(State.CANCELED) == 9
        assert int(State.EXPIRED) == 10


class TestPredicates:
    """``is_done`` / ``is_failed`` / ``is_started`` membership semantics."""

    _NON_TERMINAL = (State.IDLE, State.QUEUED, State.PENDING,
                     State.ACCEPTED, State.RUNNING, State.PARTIAL)
    _TERMINAL_FAIL = (State.REJECTED, State.FAILED, State.CANCELED, State.EXPIRED)
    _TERMINAL = (State.SUCCEEDED,) + _TERMINAL_FAIL

    def test_done_states(self) -> None:
        for m in self._TERMINAL:
            assert m.is_done, f"{m.name} should be terminal"
        for m in self._NON_TERMINAL:
            assert not m.is_done, f"{m.name} should not be terminal"

    def test_failed_states(self) -> None:
        # CANCELED / REJECTED / EXPIRED all count as failed — every
        # non-success terminal leaves the caller without a result.
        for m in self._TERMINAL_FAIL:
            assert m.is_failed, f"{m.name} should be failed-terminal"
        for m in (State.IDLE, State.QUEUED, State.PENDING,
                  State.ACCEPTED, State.RUNNING, State.PARTIAL,
                  State.SUCCEEDED):
            assert not m.is_failed, f"{m.name} should not be failed"

    def test_started_states(self) -> None:
        # Anything from RUNNING onward counts as started. ACCEPTED is
        # NOT started — the venue holds the order but execution hasn't
        # begun.
        for m in (State.RUNNING, State.PARTIAL, State.SUCCEEDED,
                  State.REJECTED, State.FAILED, State.CANCELED, State.EXPIRED):
            assert m.is_started, f"{m.name} should be started"
        for m in (State.IDLE, State.QUEUED, State.PENDING, State.ACCEPTED):
            assert not m.is_started, f"{m.name} should not be started"

    def test_active_states(self) -> None:
        # Non-terminal states with backend awareness — what a caller
        # can meaningfully wait on. IDLE is excluded (no submission yet);
        # terminal states are excluded (no more transitions).
        for m in (State.QUEUED, State.PENDING, State.ACCEPTED,
                  State.RUNNING, State.PARTIAL):
            assert m.is_active, f"{m.name} should be active"
        for m in (State.IDLE,) + self._TERMINAL:
            assert not m.is_active, f"{m.name} should not be active"

    @pytest.mark.parametrize(
        "member, predicate",
        [
            (State.IDLE, "is_idle"),
            (State.QUEUED, "is_queued"),
            (State.PENDING, "is_pending"),
            (State.ACCEPTED, "is_accepted"),
            (State.RUNNING, "is_running"),
            (State.PARTIAL, "is_partial"),
            (State.SUCCEEDED, "is_succeeded"),
            (State.REJECTED, "is_rejected"),
            (State.CANCELED, "is_canceled"),
            (State.EXPIRED, "is_expired"),
        ],
    )
    def test_single_member_predicate_isolates(
        self, member: State, predicate: str,
    ) -> None:
        # Each ``is_<name>`` predicate fires only for its named member.
        assert getattr(member, predicate)
        for other in State:
            if other is member:
                continue
            assert not getattr(other, predicate), (
                f"{predicate} should be False for {other.name}"
            )


class TestFromIdentity:

    def test_state_passes_through(self) -> None:
        for m in State:
            assert State.from_(m) is m

    def test_none_returns_idle(self) -> None:
        assert State.from_(None) is State.IDLE

    def test_none_returns_default_when_supplied(self) -> None:
        assert State.from_(None, default=State.RUNNING) is State.RUNNING

    def test_integer_codes(self) -> None:
        for m in State:
            assert State.from_(int(m)) is m

    def test_invalid_integer_raises(self) -> None:
        with pytest.raises(ValueError):
            State.from_(99)

    def test_invalid_integer_returns_default(self) -> None:
        assert State.from_(99, default=State.IDLE) is State.IDLE


class TestFromStringAliases:
    """Common backend tokens normalize to canonical members."""

    @pytest.mark.parametrize(
        "token, expected",
        [
            # Idle — built locally, not yet submitted.
            ("idle", State.IDLE),
            ("IDLE", State.IDLE),
            ("new", State.IDLE),
            ("draft", State.IDLE),
            ("not-started", State.IDLE),
            ("not started", State.IDLE),
            ("", State.IDLE),
            # Queued — sitting in a submission queue.
            ("queued", State.QUEUED),
            ("waiting", State.QUEUED),
            ("scheduled", State.QUEUED),
            # Pending — submitted to backend, awaiting ack.
            ("pending", State.PENDING),
            ("PENDING", State.PENDING),
            ("submitted", State.PENDING),
            ("sent", State.PENDING),
            ("pending_new", State.PENDING),
            # Accepted — acknowledged, parked.
            ("accepted", State.ACCEPTED),
            ("ack", State.ACCEPTED),
            ("working", State.ACCEPTED),
            ("open", State.ACCEPTED),
            ("ready", State.ACCEPTED),
            # Partial — partially filled / streaming.
            ("partial", State.PARTIAL),
            ("partially_filled", State.PARTIAL),
            ("partial_fill", State.PARTIAL),
            ("streaming", State.PARTIAL),
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
            ("filled", State.SUCCEEDED),
            ("settled", State.SUCCEEDED),
            # Rejected — refused at submission / accept.
            ("rejected", State.REJECTED),
            ("refused", State.REJECTED),
            ("denied", State.REJECTED),
            # Failed family — generic mid-run errors.
            ("failed", State.FAILED),
            ("FAIL", State.FAILED),
            ("error", State.FAILED),
            ("errored", State.FAILED),
            # Canceled family — British spelling + abort synonyms.
            ("canceled", State.CANCELED),
            ("cancelled", State.CANCELED),
            ("aborted", State.CANCELED),
            ("killed", State.CANCELED),
            ("stopped", State.CANCELED),
            # Expired — TTL / day-rollover.
            ("expired", State.EXPIRED),
            ("timed_out", State.EXPIRED),
            ("timeout", State.EXPIRED),
            ("done_for_day", State.EXPIRED),
        ],
    )
    def test_alias_resolves(self, token: str, expected: State) -> None:
        assert State.from_(token) is expected

    def test_unknown_string_raises(self) -> None:
        with pytest.raises(ValueError):
            State.from_("definitely_not_a_state")

    def test_unknown_string_returns_default(self) -> None:
        assert (
            State.from_("definitely_not_a_state", default=State.IDLE)
            is State.IDLE
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
        for value in (State.IDLE, 0, "idle", "pending", "queued", None, "closed"):
            assert State.is_valid(value)

    def test_invalid_inputs(self) -> None:
        for value in ("not_a_state", 99):
            assert not State.is_valid(value)


class TestStr:

    def test_str_is_lowercase_name(self) -> None:
        assert str(State.PENDING) == "pending"
        assert str(State.SUCCEEDED) == "succeeded"
