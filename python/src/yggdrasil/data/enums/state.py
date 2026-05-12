"""Backend-agnostic execution state enum.

Every async-execution surface in yggdrasil — Databricks SQL warehouse
statements, Spark jobs, Mongo / Postgres commands, FastAPI long-running
tasks — reports its progress through a small fixed vocabulary: pending,
running, succeeded, failed, plus the cancel terminator. Centralizing
that vocabulary on :class:`State` lets :class:`StatementResult` derive
``done`` / ``failed`` / ``started`` from a single typed enum and keeps
the per-backend status mapping in one place.

Parsing accepts:

* :class:`State` (returned as-is);
* a string alias — ``"pending"``, ``"queued"``, ``"running"``,
  ``"succeeded"``, ``"completed"``, ``"failed"``, ``"canceled"``,
  ``"closed"``, … (full alias table below). Mixed case, hyphens,
  underscores, and spaces all normalize;
* an integer code matching a member's value (round-trips with ``int``);
* an SDK enum that exposes ``.name`` / ``.value`` whose token matches
  an alias (Databricks ``StatementState.SUCCEEDED`` → ``State.SUCCEEDED``);
* ``None`` — returns *default* if supplied, else :data:`State.PENDING`.
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
    # Pending — submitted, queued, waiting to start.
    "": "PENDING",
    "pending": "PENDING",
    "queued": "PENDING",
    "waiting": "PENDING",
    "submitted": "PENDING",
    "not_started": "PENDING",
    "scheduled": "PENDING",

    # Running — actively executing.
    "running": "RUNNING",
    "in_progress": "RUNNING",
    "started": "RUNNING",
    "active": "RUNNING",
    "executing": "RUNNING",

    # Succeeded — terminal, no error.
    "succeeded": "SUCCEEDED",
    "success": "SUCCEEDED",
    "completed": "SUCCEEDED",
    "complete": "SUCCEEDED",
    "done": "SUCCEEDED",
    "ok": "SUCCEEDED",
    "finished": "SUCCEEDED",
    # Databricks' ``CLOSED`` is "result already fetched / TTL elapsed";
    # terminal and not an error — bucket with SUCCEEDED.
    "closed": "SUCCEEDED",

    # Failed — terminal, error.
    "failed": "FAILED",
    "fail": "FAILED",
    "error": "FAILED",
    "errored": "FAILED",

    # Canceled — terminal, user / system abort. Sits on the failed
    # side of ``is_failed`` to match the warehouse semantics
    # (``raise_for_status`` raises on cancel).
    "canceled": "CANCELED",
    "cancelled": "CANCELED",
    "aborted": "CANCELED",
    "abort": "CANCELED",
    "killed": "CANCELED",
}


class State(IntEnum):
    """Unified execution state for async statement / job results.

    Use :meth:`from_` to normalize backend-specific tokens, and the
    ``is_*`` predicates to derive ``done`` / ``failed`` / ``started``
    without re-implementing the membership sets per backend.
    """

    PENDING = 0
    RUNNING = 1
    SUCCEEDED = 2
    FAILED = 3
    CANCELED = 4

    # ── Predicates ──────────────────────────────────────────────────────────

    @property
    def is_pending(self) -> bool:
        """``True`` for :attr:`PENDING` — submitted but not yet running."""
        return self is State.PENDING

    @property
    def is_running(self) -> bool:
        """``True`` for :attr:`RUNNING` — actively executing."""
        return self is State.RUNNING

    @property
    def is_started(self) -> bool:
        """``True`` for anything past :attr:`PENDING`.

        Mirrors :attr:`StatementResult.started`: once the backend has
        accepted the submission, ``is_started`` flips and stays ``True``
        through every terminal state.
        """
        return self is not State.PENDING

    @property
    def is_done(self) -> bool:
        """``True`` for terminal states (:attr:`SUCCEEDED`, :attr:`FAILED`,
        :attr:`CANCELED`) — no more transitions expected."""
        return self in _DONE_STATES

    @property
    def is_failed(self) -> bool:
        """``True`` for :attr:`FAILED` / :attr:`CANCELED`.

        Cancellation counts as failed because every backend's
        ``raise_for_status`` raises on cancel — the caller asked for a
        result and didn't get one.
        """
        return self in _FAILED_STATES

    @property
    def is_succeeded(self) -> bool:
        """``True`` for :attr:`SUCCEEDED` — terminal with a result."""
        return self is State.SUCCEEDED

    @property
    def is_canceled(self) -> bool:
        """``True`` for :attr:`CANCELED`."""
        return self is State.CANCELED

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
            return cls.PENDING

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
            return cls.PENDING

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


_DONE_STATES: frozenset["State"] = frozenset(
    {State.SUCCEEDED, State.FAILED, State.CANCELED}
)
_FAILED_STATES: frozenset["State"] = frozenset({State.FAILED, State.CANCELED})


def _build_state_lookup() -> dict[str, State]:
    """Pre-compute every accepted spelling → :class:`State` member.

    Folds :data:`_STATE_ALIASES` (lower-case keys) with the canonical
    member names (``"PENDING"`` / ``"RUNNING"`` / …), their upper-case
    form, and lower-case so :meth:`State._from_str` resolves any
    common spelling with a single ``dict.get`` and no string
    allocation.
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
