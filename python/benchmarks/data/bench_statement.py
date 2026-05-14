"""Benchmark :class:`PreparedStatement`, :class:`StatementResult`,
:class:`StatementBatch`, and :class:`StatementExecutor`.

Why this exists
---------------

Every backend executor (Databricks SQL warehouse, Spark, Postgres,
MongoDB, in-process SQL) routes through the abstractions in
:mod:`yggdrasil.data.statement` and :mod:`yggdrasil.data.executor`.
Hot paths exercised on real workloads (and therefore benched here):

* ``PreparedStatement.__init__`` / ``from_`` / ``with_text`` /
  ``with_retry`` — every statement coercion and copy.
* ``PreparedStatement.looks_like_query`` — the dispatch heuristic
  on ``sql()`` / Tabular read coercion.
* ``PreparedStatement.apply_external_substitution`` — placeholder
  substitution in the warehouse / spark submit path.
* ``StatementResult`` state machine — ``state`` / ``done`` /
  ``failed`` / ``started`` / ``state_snapshot``.
* ``StatementResult.wait`` (already-terminal fast-exit) and
  ``retry`` (non-retryable fast-exit) — both hit on every
  successful synchronous statement.
* ``StatementBatch`` add / extend / submit / wait / done / iter —
  the per-key bookkeeping paid even when every result is already
  terminal.
* ``StatementExecutor.execute`` / ``execute_many`` /
  ``_resolve_options`` — kwargs merging plus the synchronous
  submit path.

We use a tiny in-process ``_FastResult`` subclass so the lifecycle
hooks degenerate to constant-time branches; the benchmark
measures the abstraction-level overhead, not any backend cost.

Usage::

    PYTHONPATH=src python benchmarks/data/bench_statement.py
    PYTHONPATH=src python benchmarks/data/bench_statement.py --repeat 7
"""
from __future__ import annotations

import argparse
import statistics
import time
from typing import Any, Callable, Optional

from yggdrasil.data.enums import State
from yggdrasil.data.executor import ExecutionOptions, StatementExecutor
from yggdrasil.data.statement import (
    ExternalStatementData,
    PreparedStatement,
    StatementBatch,
    StatementResult,
)
from yggdrasil.dataclasses.waiting import WaitingConfig


# ---------------------------------------------------------------------------
# Fixtures — a minimal synchronous backend so the lifecycle hooks
# degenerate to constant-time branches and the benchmark measures
# the abstraction overhead, not any backend cost.
# ---------------------------------------------------------------------------


class _FastResult(StatementResult[PreparedStatement]):
    """In-process, always-succeeded :class:`StatementResult`.

    Cheapest possible concrete subclass: ``_compute_state`` returns
    a cached enum, every lifecycle hook is a no-op. The hot path
    becomes pure abstraction overhead — exactly what we want to
    measure.
    """

    def __init__(self, statement: Any, *, key: Optional[str] = None,
                 executor: Optional["_FastExecutor"] = None) -> None:
        super().__init__(statement, key=key, executor=executor)
        self._fixed_state = State.SUCCEEDED

    def _compute_state(self) -> State:
        return self._fixed_state

    def refresh_status(self) -> None:
        return None

    def start(self, reset: bool = False, *, wait: Any = True,
              raise_error: bool = True, **kwargs: Any) -> "_FastResult":
        self._fixed_state = State.SUCCEEDED
        return self

    def cancel(self, wait: WaitingConfigArg = None, raise_error: bool = False, **kwargs) -> "_FastResult":
        self._fixed_state = State.FAILED
        return self

    def _raise_for_status(self) -> None:
        return None

    def _read_arrow_batches(self, options):  # pragma: no cover - never hit
        return iter(())

    def _write_arrow_batches(self, batches, options):  # pragma: no cover
        return None


class _FastExecutor(StatementExecutor[PreparedStatement, _FastResult, StatementBatch]):
    _PREPARED_STATEMENT_CLASS = PreparedStatement
    _STATEMENT_RESULT_CLASS = _FastResult
    _STATEMENT_BATCH_CLASS = StatementBatch

    def _submit_statement(self, statement: PreparedStatement, start: bool = True) -> _FastResult:
        return _FastResult(statement, executor=self)


# Hot inputs reused across scenarios so we don't pay allocation
# costs in the timed section unless that's exactly what's measured.
_STMT_TEXT = "SELECT a, b, c FROM main.t WHERE a > 10 LIMIT 100"
_LONG_QUERY = (
    "-- comment\n"
    "/* multi\nline */\n"
    "  WITH cte AS (SELECT * FROM t)\n"
    "SELECT * FROM cte WHERE id > 0\n"
)
_NOT_QUERY = "INSERT INTO t VALUES (1, 2, 3)"

_RETRY = WaitingConfig(timeout=10.0, interval=0.1, backoff=2.0,
                       max_interval=1.0, retries=3)


# ---------------------------------------------------------------------------
# Timing helpers — same shape as the sibling cast benchmarks.
# ---------------------------------------------------------------------------


def _time_one(label: str, fn: Callable[[], None], *,
              repeat: int, inner: int) -> dict:
    for _ in range(min(inner, 200)):
        fn()
    samples: list[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        for _ in range(inner):
            fn()
        samples.append((time.perf_counter() - t0) / inner)
    return {
        "label": label,
        "best": min(samples),
        "median": statistics.median(samples),
        "mean": statistics.fmean(samples),
    }


def _fmt(r: dict) -> str:
    scale = 1e6
    unit = "us"
    if r["best"] < 1e-6:
        scale = 1e9
        unit = "ns"
    return (
        f"{r['label']:<60s}  "
        f"best={r['best']*scale:9.2f} {unit}  "
        f"median={r['median']*scale:9.2f} {unit}  "
        f"mean={r['mean']*scale:9.2f} {unit}"
    )


# ---------------------------------------------------------------------------
# PreparedStatement scenarios
# ---------------------------------------------------------------------------


def _prepared_scenarios(repeat: int) -> list[dict]:
    out: list[dict] = []

    out.append(_time_one(
        "PreparedStatement(text)",
        lambda: PreparedStatement(_STMT_TEXT),
        repeat=repeat, inner=50_000,
    ))
    out.append(_time_one(
        "PreparedStatement(text, key=...)",
        lambda: PreparedStatement(_STMT_TEXT, key="k"),
        repeat=repeat, inner=50_000,
    ))
    out.append(_time_one(
        "PreparedStatement(text, retry=WaitingConfig(...))",
        lambda: PreparedStatement(_STMT_TEXT, retry=_RETRY),
        repeat=repeat, inner=50_000,
    ))

    ps = PreparedStatement(_STMT_TEXT)
    out.append(_time_one(
        "PreparedStatement.from_(PreparedStatement)  pass-through",
        lambda: PreparedStatement.from_(ps),
        repeat=repeat, inner=200_000,
    ))
    out.append(_time_one(
        "PreparedStatement.from_(str)",
        lambda: PreparedStatement.from_(_STMT_TEXT),
        repeat=repeat, inner=50_000,
    ))
    out.append(_time_one(
        "PreparedStatement.with_text(...)  copy",
        lambda: ps.with_text("SELECT 1"),
        repeat=repeat, inner=200_000,
    ))
    out.append(_time_one(
        "PreparedStatement.with_retry(WaitingConfig)  copy",
        lambda: ps.with_retry(_RETRY),
        repeat=repeat, inner=200_000,
    ))

    out.append(_time_one(
        "PreparedStatement.looks_like_query(short SELECT)",
        lambda: PreparedStatement.looks_like_query(_STMT_TEXT),
        repeat=repeat, inner=200_000,
    ))
    out.append(_time_one(
        "PreparedStatement.looks_like_query(comment + CTE)",
        lambda: PreparedStatement.looks_like_query(_LONG_QUERY),
        repeat=repeat, inner=200_000,
    ))
    out.append(_time_one(
        "PreparedStatement.looks_like_query(INSERT)",
        lambda: PreparedStatement.looks_like_query(_NOT_QUERY),
        repeat=repeat, inner=200_000,
    ))

    # External substitution — exercised on every warehouse / spark submit.
    text_with = "SELECT * FROM {src} WHERE x IN (SELECT x FROM {ref})"
    ext = {
        "src": ExternalStatementData("src", text_value="stage.src_v"),
        "ref": ExternalStatementData("ref", text_value="stage.ref_v"),
    }
    out.append(_time_one(
        "PreparedStatement.apply_external_substitution(2 keys)",
        lambda: PreparedStatement.apply_external_substitution(text_with, ext),
        repeat=repeat, inner=100_000,
    ))
    out.append(_time_one(
        "PreparedStatement.apply_external_substitution(empty)",
        lambda: PreparedStatement.apply_external_substitution(_STMT_TEXT, None),
        repeat=repeat, inner=500_000,
    ))

    return out


# ---------------------------------------------------------------------------
# StatementResult scenarios
# ---------------------------------------------------------------------------


def _result_scenarios(repeat: int) -> list[dict]:
    out: list[dict] = []
    exec_ = _FastExecutor()
    ps = PreparedStatement(_STMT_TEXT)

    out.append(_time_one(
        "StatementResult(PreparedStatement)  construction",
        lambda: _FastResult(ps, executor=exec_),
        repeat=repeat, inner=50_000,
    ))

    result = _FastResult(ps, executor=exec_)
    out.append(_time_one(
        "result.state",
        lambda: result.state,
        repeat=repeat, inner=500_000,
    ))
    out.append(_time_one(
        "result.text  (property -> statement.text)",
        lambda: result.text,
        repeat=repeat, inner=500_000,
    ))

    out.append(_time_one(
        "result.wait(False, raise_error=False)  no-op",
        lambda: result.wait(False, raise_error=False),
        repeat=repeat, inner=200_000,
    ))
    out.append(_time_one(
        "result.raise_for_status()  succeeded",
        lambda: result.raise_for_status(),
        repeat=repeat, inner=200_000,
    ))
    out.append(_time_one(
        "result.retryable  (non-retryable)",
        lambda: result.retryable,
        repeat=repeat, inner=500_000,
    ))

    return out


# ---------------------------------------------------------------------------
# StatementBatch scenarios
# ---------------------------------------------------------------------------


def _batch_scenarios(repeat: int) -> list[dict]:
    out: list[dict] = []
    exec_ = _FastExecutor()

    statements_str = [f"SELECT {i}" for i in range(16)]
    statements_ps = [PreparedStatement(s) for s in statements_str]

    def _make_pending_batch() -> StatementBatch:
        b = exec_.batch()
        b.extend(statements_ps)
        return b

    out.append(_time_one(
        "executor.batch().extend(16 PreparedStatement)",
        _make_pending_batch,
        repeat=repeat, inner=10_000,
    ))

    out.append(_time_one(
        "executor.batch().extend(16 str)",
        lambda: exec_.batch().extend(statements_str),
        repeat=repeat, inner=10_000,
    ))

    # Pre-built submitted batch for the read-side ops.
    submitted = exec_.batch()
    submitted.extend(statements_ps)

    out.append(_time_one(
        "batch.done  16 succeeded",
        lambda: submitted.done,
        repeat=repeat, inner=50_000,
    ))
    out.append(_time_one(
        "batch.failed  16 succeeded",
        lambda: submitted.failed,
        repeat=repeat, inner=50_000,
    ))
    out.append(_time_one(
        "batch.wait(True, raise_error=False)  all terminal",
        lambda: submitted.wait(True, raise_error=False),
        repeat=repeat, inner=5_000,
    ))
    first_key = next(iter(submitted))
    out.append(_time_one(
        "batch[key]",
        lambda: submitted[first_key],
        repeat=repeat, inner=500_000,
    ))
    out.append(_time_one(
        "key in batch  (hit)",
        lambda: first_key in submitted,
        repeat=repeat, inner=500_000,
    ))
    out.append(_time_one(
        "key in batch  (miss)",
        lambda: "missing" in submitted,
        repeat=repeat, inner=500_000,
    ))
    out.append(_time_one(
        "len(batch)",
        lambda: len(submitted),
        repeat=repeat, inner=500_000,
    ))
    out.append(_time_one(
        "list(batch.materialized())  16",
        lambda: list(submitted.materialized()),
        repeat=repeat, inner=50_000,
    ))

    return out


# ---------------------------------------------------------------------------
# Executor scenarios
# ---------------------------------------------------------------------------


def _executor_scenarios(repeat: int) -> list[dict]:
    out: list[dict] = []
    exec_ = _FastExecutor()
    opts_default = ExecutionOptions()
    opts_nowait = ExecutionOptions(wait=False, raise_error=False)

    out.append(_time_one(
        "ExecutionOptions.from_(None)",
        lambda: ExecutionOptions.from_(None),
        repeat=repeat, inner=500_000,
    ))
    out.append(_time_one(
        "ExecutionOptions.from_(opts)  pass-through",
        lambda: ExecutionOptions.from_(opts_default),
        repeat=repeat, inner=500_000,
    ))
    out.append(_time_one(
        "_resolve_options(None, wait=True, raise_error=True)",
        lambda: StatementExecutor._resolve_options(None, wait=True, raise_error=True),
        repeat=repeat, inner=500_000,
    ))
    out.append(_time_one(
        "_resolve_options(opts, defaults)",
        lambda: StatementExecutor._resolve_options(opts_default, wait=True, raise_error=True),
        repeat=repeat, inner=500_000,
    ))
    out.append(_time_one(
        "_resolve_options(opts, wait=False)",
        lambda: StatementExecutor._resolve_options(opts_default, wait=False, raise_error=True),
        repeat=repeat, inner=500_000,
    ))

    ps = PreparedStatement(_STMT_TEXT)
    out.append(_time_one(
        "executor.execute(stmt, wait=False, raise_error=False)",
        lambda: exec_.execute(ps, wait=False, raise_error=False),
        repeat=repeat, inner=20_000,
    ))
    out.append(_time_one(
        "executor.execute(str, wait=False, raise_error=False)",
        lambda: exec_.execute(_STMT_TEXT, wait=False, raise_error=False),
        repeat=repeat, inner=20_000,
    ))
    out.append(_time_one(
        "executor.execute(stmt, options=opts_nowait)",
        lambda: exec_.execute(ps, options=opts_nowait),
        repeat=repeat, inner=20_000,
    ))

    statements_ps = [PreparedStatement(f"SELECT {i}") for i in range(16)]
    out.append(_time_one(
        "executor.execute_many(16, wait=False)",
        lambda: exec_.execute_many(statements_ps, wait=False, raise_error=False),
        repeat=repeat, inner=2_000,
    ))

    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def scenarios(repeat: int) -> list[dict]:
    return [
        *_prepared_scenarios(repeat),
        *_result_scenarios(repeat),
        *_batch_scenarios(repeat),
        *_executor_scenarios(repeat),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=5,
                    help="Outer repeat count per scenario (median across).")
    args = ap.parse_args()

    print(f"# repeat={args.repeat}")
    print(f"# {'label':<60s}  {'best':>15s}  {'median':>18s}  {'mean':>15s}")
    for row in scenarios(args.repeat):
        print(_fmt(row))


if __name__ == "__main__":
    main()
