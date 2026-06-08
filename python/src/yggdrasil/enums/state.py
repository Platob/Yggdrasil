"""Backend-agnostic execution + order state enum.

PARITY: ported to JS/TS at packages/yggdrasil/enums/state.ts — keep in sync.


Every async-execution surface in yggdrasil — Databricks SQL warehouse
statements, Spark jobs, Mongo / Postgres commands, FastAPI long-running
tasks — *and* every order-style lifecycle (FIX-protocol order flows,
trading-venue acknowledgements, transaction state machines) reports
its progress through one shared vocabulary. Centralizing it on
:class:`State` lets :class:`StatementResult` and any order resource
derive ``done`` / ``failed`` / ``started`` from a single typed enum
and keeps the per-backend status mapping in one place.

Lifecycle ordering (low → high integer codes):

- :attr:`IDLE` — built locally, no submission attempt yet.
- :attr:`QUEUED` — prepared, sitting in a submission queue.
- :attr:`PENDING` — handed to a backend / venue, awaiting acknowledgement.
- :attr:`ACCEPTED` — acknowledged / accepted but not yet executing.
- :attr:`RUNNING` — actively executing (FIX: working / new).
- :attr:`PARTIAL` — partially filled / streaming results, still active.
- :attr:`SUCCEEDED` — terminal, fully filled / completed.
- :attr:`REJECTED` — terminal, rejected at submission or accept time.
- :attr:`FAILED` — terminal, errored mid-execution.
- :attr:`CANCELED` — terminal, user or system abort.
- :attr:`EXPIRED` — terminal, TTL / done-for-day elapsed.

Parsing accepts:

* :class:`State` (returned as-is);
* a string alias — ``"queued"``, ``"pending"``, ``"running"``,
  ``"succeeded"``, ``"completed"``, ``"failed"``, ``"canceled"``,
  ``"closed"``, … (full alias table below). Mixed case, hyphens,
  underscores, and spaces all normalize;
* an integer code matching a member's value (round-trips with ``int``);
* an SDK enum that exposes ``.name`` / ``.value`` whose token matches
  an alias (Databricks ``StatementState.SUCCEEDED`` → ``State.SUCCEEDED``);
* ``None`` — returns *default* if supplied, else :data:`State.IDLE`.
"""
from __future__ import annotations

from enum import IntEnum
from typing import Any, Union

__all__ = ["State", "StateLike"]


StateLike = Union["State", str, int, None]


# Alias table — every spelling we normalize to a canonical member.
# Includes Databricks ``StatementState`` tokens (CLOSED is treated as
# terminal-but-not-an-error, i.e. SUCCEEDED), Spark / Mongo / Postgres
# synonyms (``completed``, ``finished``), and the human-readable
# verbs callers reach for (``running``, ``in_progress``, ``ok``).
_STATE_ALIASES: dict[str, str] = {
    # Idle — built locally, no submission attempt yet.
    "": "IDLE",
    "idle": "IDLE",
    "new": "IDLE",
    "not_started": "IDLE",
    "draft": "IDLE",
    "created": "IDLE",

    # Queued — prepared, sitting in a submission queue.
    "queued": "QUEUED",
    "waiting": "QUEUED",
    "scheduled": "QUEUED",

    # Pending — handed to a backend / venue, awaiting acknowledgement.
    "pending": "PENDING",
    "submitted": "PENDING",
    "pending_new": "PENDING",
    "sent": "PENDING",

    # Accepted — acknowledged, parked, not yet executing. FIX "working"
    # / "new" lifecycle, exchange order-book entry, AMQP "ready".
    "accepted": "ACCEPTED",
    "ack": "ACCEPTED",
    "acknowledged": "ACCEPTED",
    "working": "ACCEPTED",
    "open": "ACCEPTED",
    "ready": "ACCEPTED",

    # Running — actively executing.
    "running": "RUNNING",
    "in_progress": "RUNNING",
    "started": "RUNNING",
    "active": "RUNNING",
    "executing": "RUNNING",

    # Partial — partially filled / streaming, still active.
    "partial": "PARTIAL",
    "partially_filled": "PARTIAL",
    "partial_fill": "PARTIAL",
    "partially_complete": "PARTIAL",
    "streaming": "PARTIAL",

    # Succeeded — terminal, fully completed / filled.
    "succeeded": "SUCCEEDED",
    "success": "SUCCEEDED",
    "completed": "SUCCEEDED",
    "complete": "SUCCEEDED",
    "done": "SUCCEEDED",
    "ok": "SUCCEEDED",
    "finished": "SUCCEEDED",
    "filled": "SUCCEEDED",
    "settled": "SUCCEEDED",
    # Databricks' ``CLOSED`` is "result already fetched / TTL elapsed";
    # terminal and not an error — bucket with SUCCEEDED.
    "closed": "SUCCEEDED",

    # Rejected — terminal, refused at submission / accept.
    "rejected": "REJECTED",
    "reject": "REJECTED",
    "refused": "REJECTED",
    "denied": "REJECTED",

    # Failed — terminal, errored mid-execution.
    "failed": "FAILED",
    "fail": "FAILED",
    "error": "FAILED",
    "errored": "FAILED",
    "broken": "FAILED",

    # Canceled — terminal, user / system abort. Sits on the failed
    # side of ``is_failed`` to match the warehouse semantics
    # (``raise_for_status`` raises on cancel).
    "canceled": "CANCELED",
    "cancelled": "CANCELED",
    "aborted": "CANCELED",
    "abort": "CANCELED",
    "killed": "CANCELED",
    "stopped": "CANCELED",

    # Expired — terminal, TTL / day-rollover elapsed before completion.
    "expired": "EXPIRED",
    "expire": "EXPIRED",
    "timed_out": "EXPIRED",
    "timeout": "EXPIRED",
    "ttl_elapsed": "EXPIRED",
    "done_for_day": "EXPIRED",
}


class State(IntEnum):
    """Unified execution state for async statement / job results.

    Use :meth:`from_` to normalize backend-specific tokens, and the
    ``is_*`` predicates to derive ``done`` / ``failed`` / ``started``
    without re-implementing the membership sets per backend.
    """

    IDLE = 0
    QUEUED = 1
    PENDING = 2
    ACCEPTED = 3
    RUNNING = 4
    PARTIAL = 5
    SUCCEEDED = 6
    REJECTED = 7
    FAILED = 8
    CANCELED = 9
    EXPIRED = 10

    # ── Predicates ──────────────────────────────────────────────────────────

    @property
    def is_idle(self) -> bool:
        """``True`` for :attr:`IDLE` — built locally, not submitted yet."""
        return self is State.IDLE

    @property
    def is_queued(self) -> bool:
        """``True`` for :attr:`QUEUED` — prepared, sitting in a submission queue."""
        return self is State.QUEUED

    @property
    def is_pending(self) -> bool:
        """``True`` for :attr:`PENDING` — handed to a backend, awaiting ack."""
        return self is State.PENDING

    @property
    def is_accepted(self) -> bool:
        """``True`` for :attr:`ACCEPTED` — acknowledged but not yet running."""
        return self is State.ACCEPTED

    @property
    def is_running(self) -> bool:
        """``True`` for :attr:`RUNNING` — actively executing."""
        return self is State.RUNNING

    @property
    def is_partial(self) -> bool:
        """``True`` for :attr:`PARTIAL` — partially filled, still active."""
        return self is State.PARTIAL

    @property
    def is_started(self) -> bool:
        """``True`` for anything from :attr:`RUNNING` onward.

        Mirrors :attr:`StatementResult.started`: once the backend has
        actually started executing, ``is_started`` flips and stays
        ``True`` through every terminal state. :attr:`ACCEPTED` is
        *not* started — the venue holds the order, no execution yet.
        """
        return self.value >= self.RUNNING.value

    @property
    def is_active(self) -> bool:
        """``True`` for non-terminal states with backend awareness.

        Covers :attr:`QUEUED` through :attr:`PARTIAL` — anything the
        caller can reasonably wait on. :attr:`IDLE` is excluded
        (nothing has been submitted yet) and every terminal state is
        excluded (no more transitions).
        """
        return self in _ACTIVE_STATES

    @property
    def is_done(self) -> bool:
        """``True`` for terminal states (:attr:`SUCCEEDED`, :attr:`REJECTED`,
        :attr:`FAILED`, :attr:`CANCELED`, :attr:`EXPIRED`) — no more
        transitions expected."""
        return self in _DONE_STATES

    @property
    def is_failed(self) -> bool:
        """``True`` for :attr:`REJECTED` / :attr:`FAILED` / :attr:`CANCELED` /
        :attr:`EXPIRED`.

        Every non-success terminal state counts as failed because each
        one means "the caller asked for a result and didn't get one":
        cancellation, rejection, mid-run error, or TTL expiry all leave
        the operation incomplete from the caller's view, and the
        per-backend ``raise_for_status`` raises on each.
        """
        return self in _FAILED_STATES

    @property
    def is_succeeded(self) -> bool:
        """``True`` for :attr:`SUCCEEDED` — terminal with a full result."""
        return self is State.SUCCEEDED

    @property
    def is_rejected(self) -> bool:
        """``True`` for :attr:`REJECTED` — refused at submission / accept."""
        return self is State.REJECTED

    @property
    def is_canceled(self) -> bool:
        """``True`` for :attr:`CANCELED`."""
        return self is State.CANCELED

    @property
    def is_expired(self) -> bool:
        """``True`` for :attr:`EXPIRED` — TTL / day-rollover terminal."""
        return self is State.EXPIRED

    # ── Coercion ────────────────────────────────────────────────────────────

    @classmethod
    def from_(cls, value: Any, *, default: Any = ...) -> "State":
        """Coerce any Python value into a :class:`State`.

        See module docstring for the accepted shapes. ``default``
        swallows unknown / unparseable input; without it, unknown
        tokens raise :class:`ValueError` and unsupported types raise
        :class:`TypeError`.
        """
        if isinstance(value, cls):
            return value

        if value is None:
            if default is not ...:
                return default
            return cls.IDLE

        # IntEnum members compare equal to ints; allow integer-code
        # lookups so persisted State codes round-trip.
        if isinstance(value, int) and not isinstance(value, bool):
            try:
                return cls(value)
            except ValueError:
                if default is not ...:
                    return default
                raise ValueError(
                    f"Cannot parse {value!r} as a State. Accepted "
                    f"integer codes: {sorted(int(m) for m in cls)}."
                )

        if isinstance(value, str):
            return cls._from_str(value, default=default)

        # SDK enum / dataclass with a .name (or .value that's a string).
        # Databricks ``StatementState.SUCCEEDED`` lands here.
        name = getattr(value, "name", None)
        if isinstance(name, str):
            try:
                return cls._from_str(name, default=default)
            except (TypeError, ValueError):
                pass
        sdk_value = getattr(value, "value", None)
        if isinstance(sdk_value, str):
            try:
                return cls._from_str(sdk_value, default=default)
            except (TypeError, ValueError):
                pass

        if default is not ...:
            return default
        raise TypeError(
            f"Cannot derive State from {type(value).__name__}: {value!r}"
        )

    @classmethod
    def _from_str(cls, value: str, *, default: Any = ...) -> "State":
        # Fast path: most callers pass an already-canonical token
        # (``"succeeded"`` / ``"SUCCEEDED"`` / ``"running"``).
        # A single dict probe resolves them without paying any string
        # normalisation cost.
        hit = _STATE_LOOKUP.get(value)
        if hit is not None:
            return hit

        token = value.strip().lower().replace("-", "_").replace(" ", "_")
        if not token:
            if default is not ...:
                return default
            return cls.IDLE

        hit = _STATE_LOOKUP.get(token)
        if hit is not None:
            return hit

        if default is not ...:
            return default
        raise ValueError(
            f"Cannot parse {value!r} as a State. Accepted values: "
            f"{sorted(m.name for m in cls)} or aliases like "
            f"{sorted(_STATE_ALIASES)}."
        )

    @classmethod
    def is_valid(cls, value: Any) -> bool:
        """Return ``True`` when :meth:`from_` would succeed for *value*."""
        try:
            cls.from_(value)
            return True
        except (TypeError, ValueError):
            return False

    # ── Dunder ──────────────────────────────────────────────────────────────

    def __str__(self) -> str:
        return self.name.lower()


_DONE_STATES: frozenset["State"] = frozenset({
    State.SUCCEEDED,
    State.REJECTED,
    State.FAILED,
    State.CANCELED,
    State.EXPIRED,
})
_FAILED_STATES: frozenset["State"] = frozenset({
    State.REJECTED,
    State.FAILED,
    State.CANCELED,
    State.EXPIRED,
})
# Non-terminal states with backend awareness — anything the caller can
# meaningfully wait on. IDLE is excluded (no submission yet); every
# terminal state is excluded (no more transitions expected).
_ACTIVE_STATES: frozenset["State"] = frozenset({
    State.QUEUED,
    State.PENDING,
    State.ACCEPTED,
    State.RUNNING,
    State.PARTIAL,
})


def _build_state_lookup() -> dict[str, State]:
    """Pre-compute every accepted spelling → :class:`State` member.

    Folds :data:`_STATE_ALIASES` (lower-case keys) with the canonical
    member names (``"QUEUED"`` / ``"PENDING"`` / ``"RUNNING"`` / …),
    their upper-case form, and lower-case so :meth:`State._from_str`
    resolves any common spelling with a single ``dict.get`` and no
    string allocation.
    """
    out: dict[str, State] = {}
    for alias, canonical in _STATE_ALIASES.items():
        member = State[canonical]
        out[alias] = member
        upper = alias.upper()
        if upper != alias:
            out[upper] = member
    for member in State:
        out[member.name] = member
        out[member.name.lower()] = member
    return out


_STATE_LOOKUP: dict[str, State] = _build_state_lookup()
