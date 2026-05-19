"""Synchronous :class:`StatementResult` for :class:`ExecutionEngine`.

Unity operations are local (no remote queue, no streaming wire
protocol): :meth:`ExecutionStatementResult.start` runs the statement's
:meth:`ExecutionStatement.apply` and stashes the return value on
:attr:`output`. The result is terminal once :meth:`start` returns —
:attr:`done` flips to ``True`` and :meth:`refresh_status` is a no-op.

When the output is itself a :class:`Tabular` (the common case for
:class:`Select` / :class:`CreateView` / :class:`CreateTable` etc.), the
result forwards :meth:`_read_arrow_batches` to it so callers can chain
``engine.execute(SELECT(...)).read_arrow_table()`` without a separate
unwrap step.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar, Iterable, Iterator, Optional

import pyarrow as pa

from yggdrasil.data.enums.state import State
from yggdrasil.data.options import CastOptions
from yggdrasil.data.statement import StatementResult
from yggdrasil.unity.statement import ExecutionStatement

if TYPE_CHECKING:
    from yggdrasil.io.tabular.base import Tabular
    from yggdrasil.unity.engine import ExecutionEngine


__all__ = ["ExecutionStatementResult"]


logger = logging.getLogger(__name__)


class ExecutionStatementResult(StatementResult[ExecutionStatement]):
    """Local-sync result wrapping the return value of a :class:`ExecutionStatement`."""

    _PREPARED_CLASS: ClassVar[type[ExecutionStatement]] = ExecutionStatement

    def __init__(
        self,
        statement: ExecutionStatement,
        *,
        executor: "Optional[ExecutionEngine]" = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(statement=statement, executor=executor, **kwargs)
        self._started: bool = False
        self._failure: BaseException | None = None
        #: Whatever the statement's :meth:`apply` returned. Backends
        #: stash either a :class:`ExecutionResource`, an ``int`` row count,
        #: a :class:`Tabular`, or a ``list[str]`` here.
        self.output: Any = None

    # ── lifecycle ──────────────────────────────────────────────────────

    def _compute_state(self) -> State:
        if not self._started:
            return State.IDLE
        if self._failure is not None:
            return State.FAILED
        return State.SUCCEEDED

    def refresh_status(self) -> None:
        """No-op — unity execution is synchronous, no remote state to poll."""
        return None

    def _raise_for_status(self) -> None:
        if self._failure is not None:
            raise self._failure

    def start(
        self,
        reset: bool = False,
        *,
        wait: Any = True,
        raise_error: bool = True,
        **kwargs: Any,
    ) -> "ExecutionStatementResult":
        """Run :meth:`ExecutionStatement.apply` against the bound executor."""
        del wait, kwargs  # Unity execution is fully synchronous.
        if self._started and not reset:
            return self
        self._started = True
        self._failure = None
        self.output = None

        engine = self.executor
        if engine is None:
            raise RuntimeError(
                f"Cannot start {self!r}: no executor bound. Construct the "
                "result via ExecutionEngine.send() / ExecutionEngine.execute() so "
                "the executor back-reference is set."
            )

        logger.debug("Applying unity statement %r against %r", self.statement, engine)
        try:
            self.output = self.statement.apply(engine)
        except BaseException as exc:
            self._failure = exc
            logger.debug(
                "Unity statement %r failed: %s", self.statement, exc,
            )
            if raise_error:
                raise
        else:
            logger.debug(
                "Applied unity statement %r → %r",
                self.statement, self.output,
            )
        return self

    def cancel(
        self,
        wait: Any = None,
        raise_error: bool = False,
        **kwargs: Any,
    ) -> "ExecutionStatementResult":
        """No-op — synchronous results either ran already or never started."""
        del wait, raise_error, kwargs
        return self

    # ── Tabular surface — forward to ``output`` when it's a Tabular ────

    def _output_as_tabular(self) -> "Tabular | None":
        from yggdrasil.io.tabular.base import Tabular

        return self.output if isinstance(self.output, Tabular) else None

    def _read_arrow_batches(self, options: CastOptions) -> Iterator[pa.RecordBatch]:
        target = self._output_as_tabular()
        if target is None:
            return
        yield from target._read_arrow_batches(options)

    def _write_arrow_batches(
        self,
        batches: Iterable[pa.RecordBatch],
        options: CastOptions,
    ) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} is read-only. Write into the target "
            "table directly (e.g. by sending an Insert statement)."
        )
