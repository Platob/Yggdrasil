"""Lightweight operation descriptors for :class:`ExecutionPlan`.

Each dataclass is a value-only record — no behavior, no engine
imports. The execution engine in :mod:`execution_plan` reads these
and dispatches to the appropriate Tabular / Arrow / Spark surface.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from yggdrasil.enums import JoinType, Mode
    from yggdrasil.execution.expr import Predicate
    from yggdrasil.io.tabular import Tabular


@dataclasses.dataclass(slots=True, frozen=True)
class JoinOp:
    """A pending join between the current result and *right*."""
    right: Tabular
    on: list[str]
    how: JoinType
    right_suffix: str = "_right"


@dataclasses.dataclass(slots=True, frozen=True)
class UnionOp:
    """A pending UNION ALL with *other*."""
    other: Tabular
    mode: Mode


@dataclasses.dataclass(slots=True, frozen=True)
class ResampleOp:
    """A pending time-grid resample."""
    time_column: str
    sampling_seconds: int
    partition_by: list[str]
    fill_strategy: str | None = "ffill"
