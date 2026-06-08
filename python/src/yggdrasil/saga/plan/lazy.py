""":class:`LazyTabular` — an alias of :class:`ExecutionResult`.

A lazy tabular is just an :class:`ExecutionResult` over a ``SelectPlan``
bound to a source that hasn't started yet: the transform builders
(``select`` / ``filter`` / ``join`` / …) mutate the held plan while it is
idle, and the first ``read_*`` / ``await`` executes it (as a graph of inner
``ExecutionResult``s). See :mod:`yggdrasil.saga.plan.execution_result`.
"""

from __future__ import annotations

from .execution_result import ExecutionResult as LazyTabular

__all__ = ["LazyTabular"]
