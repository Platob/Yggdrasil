"""Benchmark the :class:`CastOptions` hot path.

Why this exists
---------------

Every :class:`DataIO` public method funnels through
:meth:`CastOptions.check`, so the cost of constructing / merging /
copying a :class:`CastOptions` instance is paid on every read, write,
batch, predicate evaluation, and inter-engine handoff. In a
folder-of-folders persist or a streaming batch pipeline, this can be
hundreds of options objects per logical operation — each one
allocating a frozen dataclass, normalizing a target field, running
``__post_init__`` and resolving the engine dispatch.

This benchmark targets the construction / coercion / copy paths that
fire whether or not any real data is touched, plus the cached
property accessors (``merged``, ``column_names``) that pipelines
walk repeatedly. It does **not** measure the per-engine cast kernels
— see ``bench_engine_type_bypass`` for those.

Usage::

    PYTHONPATH=src python benchmarks/bench_castoptions.py
    PYTHONPATH=src python benchmarks/bench_castoptions.py --repeat 7
"""
from __future__ import annotations

import argparse
import statistics
import time
from typing import Callable

import pyarrow as pa

from yggdrasil.data.options import CastOptions
from yggdrasil.data.data_field import Field
from yggdrasil.data.enums import Mode
from yggdrasil.io.primitive.csv_file import CsvOptions


# ---------------------------------------------------------------------------
# Shared fixtures — kept module-level so the timing loop doesn't pay for them.
# ---------------------------------------------------------------------------


PA_SCHEMA = pa.schema(
    [
        pa.field("id", pa.int64(), nullable=False),
        pa.field("amount", pa.float64()),
        pa.field("qty", pa.int32()),
        pa.field("name", pa.string()),
        pa.field("ts", pa.timestamp("us")),
        pa.field("active", pa.bool_()),
        pa.field("tags", pa.list_(pa.string())),
        pa.field("meta", pa.struct([("k", pa.string()), ("v", pa.int32())])),
    ]
)

PA_FIELD = pa.field("payload", pa.struct([("id", pa.int64()), ("name", pa.string())]))
PA_DTYPE = pa.int64()


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _time_one(label: str, fn: Callable[[], None], *, repeat: int, inner: int) -> dict:
    # Warm-up — first call pays JIT / lazy-import / cache-population costs.
    for _ in range(min(inner, 1000)):
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
        f"{r['label']:<48s}  "
        f"best={r['best']*scale:8.2f} {unit}  "
        f"median={r['median']*scale:8.2f} {unit}  "
        f"mean={r['mean']*scale:8.2f} {unit}"
    )


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def scenarios(repeat: int) -> list[dict]:
    results: list[dict] = []

    # 1. Cheapest possible construction — no target/source bound. This
    # path fires every time a DataIO public method gets called with no
    # explicit options (``read_arrow_table()`` etc.).
    results.append(_time_one(
        "construct: CastOptions() defaults",
        lambda: CastOptions(),
        repeat=repeat, inner=50_000,
    ))

    # 2. Common construction shape — a pa.Schema landing as target.
    # ``__post_init__`` runs ``Field.from_(pa.Schema)`` which builds
    # the full Field tree.
    results.append(_time_one(
        "construct: CastOptions(target=pa.Schema)",
        lambda: CastOptions(target=PA_SCHEMA),
        repeat=repeat, inner=5_000,
    ))

    # 3. Single-field target — common for column-level cast pipelines.
    results.append(_time_one(
        "construct: CastOptions(target=pa.Field)",
        lambda: CastOptions(target=PA_FIELD),
        repeat=repeat, inner=10_000,
    ))

    # 4. Bare dtype — single-column-cast shape (Field/Schema not needed).
    results.append(_time_one(
        "construct: CastOptions(target=pa.DataType)",
        lambda: CastOptions(target=PA_DTYPE),
        repeat=repeat, inner=20_000,
    ))

    # 5. .check(None) — no options, no overrides. Fires once per DataIO
    # public method on entry. Should be cheap.
    results.append(_time_one(
        "check: CastOptions.check(None)",
        lambda: CastOptions.check(None),
        repeat=repeat, inner=50_000,
    ))

    # 6. .check(existing) — passthrough. Fastest possible path.
    existing = CastOptions(target=PA_SCHEMA)
    results.append(_time_one(
        "check: CastOptions.check(existing) passthrough",
        lambda: CastOptions.check(existing),
        repeat=repeat, inner=200_000,
    ))

    # 7. .check(existing, **overrides) — copy on override.
    results.append(_time_one(
        "check: CastOptions.check(existing, safe=True)",
        lambda: CastOptions.check(existing, safe=True),
        repeat=repeat, inner=5_000,
    ))

    # 8. .check(pa.Schema) — schema coerced to target.
    results.append(_time_one(
        "check: CastOptions.check(pa.Schema)",
        lambda: CastOptions.check(PA_SCHEMA),
        repeat=repeat, inner=5_000,
    ))

    # 9. .check(dict) — caller passed kwargs as a mapping.
    cfg = {"target": PA_SCHEMA, "safe": True, "row_size": 1024}
    results.append(_time_one(
        "check: CastOptions.check(dict)",
        lambda: CastOptions.check(cfg),
        repeat=repeat, inner=5_000,
    ))

    # 10. .copy() — pure clone. Hit by every with_source / with_target
    # when the caller didn't opt into in-place edits.
    opts_full = CastOptions(target=PA_SCHEMA, source=PA_SCHEMA)
    results.append(_time_one(
        "copy: opts.copy() no overrides",
        lambda: opts_full.copy(),
        repeat=repeat, inner=10_000,
    ))

    # 11. .copy(override) — common shape: re-target with a tweak.
    results.append(_time_one(
        "copy: opts.copy(safe=True)",
        lambda: opts_full.copy(safe=True),
        repeat=repeat, inner=10_000,
    ))

    # 12. .cast() short-circuit — no target bound. Hot in fall-through
    # pipelines where the cast site is unconditional but options often
    # carry no target.
    no_target = CastOptions()
    table = pa.table({"a": pa.array([1, 2, 3], type=pa.int64())})
    results.append(_time_one(
        "cast: opts.cast(t) target=None short-circuit",
        lambda: no_target.cast(table),
        repeat=repeat, inner=200_000,
    ))

    # 13. merged — cached on the instance; subsequent reads are
    # what the cast pipeline actually pays for once the cache warms up.
    opts_both = CastOptions(source=PA_SCHEMA, target=PA_SCHEMA)
    _ = opts_both.merged  # warm the slot
    results.append(_time_one(
        "prop: opts.merged cached read",
        lambda: opts_both.merged,
        repeat=repeat, inner=500_000,
    ))

    # 14. column_names — goes through merged then walks names.
    results.append(_time_one(
        "prop: opts.column_names",
        lambda: opts_both.column_names,
        repeat=repeat, inner=200_000,
    ))

    # 15. target attribute — direct slot read. Replaces the old
    # cached ``target_schema`` property: since :class:`Schema` is a
    # subclass of :class:`StructField`, the same Field-shaped slot
    # serves both "give me the column-ish thing" and "give me a
    # Schema-shaped view" callers.
    opts_t = CastOptions(target=PA_SCHEMA)
    results.append(_time_one(
        "prop: opts.target",
        lambda: opts_t.target,
        repeat=repeat, inner=200_000,
    ))

    # 16. field_names() — classmethod, lru_cached. Called from _build.
    results.append(_time_one(
        "meta: CastOptions.field_names()",
        lambda: CastOptions.field_names(),
        repeat=repeat, inner=500_000,
    ))

    # 17. with_source(None) — common reset path in DataIO pipelines.
    results.append(_time_one(
        "mutate: opts.with_source(None, copy=True)",
        lambda: opts_full.with_source(None, copy=True),
        repeat=repeat, inner=10_000,
    ))

    # 18. need_cast — called from every cast site on entry.
    results.append(_time_one(
        "decide: opts.need_cast()",
        lambda: opts_both.need_cast(),
        repeat=repeat, inner=100_000,
    ))

    # 19. __repr__ — exercised by tracebacks / logging on the error path.
    results.append(_time_one(
        "repr: repr(opts_full)",
        lambda: repr(opts_full),
        repeat=repeat, inner=20_000,
    ))

    # 20. Mode-only construction — exercises only Mode.from_ in __post_init__.
    results.append(_time_one(
        "construct: CastOptions(mode='overwrite')",
        lambda: CastOptions(mode="overwrite"),
        repeat=repeat, inner=20_000,
    ))

    # 21. Mode-with-enum — already a Mode, no parse work.
    results.append(_time_one(
        "construct: CastOptions(mode=Mode.OVERWRITE)",
        lambda: CastOptions(mode=Mode.OVERWRITE),
        repeat=repeat, inner=50_000,
    ))

    # ----------------------------------------------------------------
    # Focused .check() micro-scenarios — every DataIO public method
    # funnels through .check(), so each shape below maps to a real
    # caller pattern (engine cast wrappers, registry dispatch, table
    # writers, etc.). The numbers here drive any future optimization
    # of the dispatch chain inside .check().
    # ----------------------------------------------------------------

    # 22. .check(None, safe=True) — None options + a single override.
    # Fires when a cast wrapper hands a kwarg through without a base
    # options object (``cast(value, target=..., safe=True)`` on the
    # registry dispatcher).
    results.append(_time_one(
        "check: CastOptions.check(None, safe=True)",
        lambda: CastOptions.check(None, safe=True),
        repeat=repeat, inner=20_000,
    ))

    # 23. .check(existing, target=pa.Schema) — copy with a field
    # override. Hot in cast wrappers that pin a target onto an
    # existing options bundle.
    existing_no_target = CastOptions(safe=True)
    results.append(_time_one(
        "check: CastOptions.check(existing, target=pa.Schema)",
        lambda: CastOptions.check(existing_no_target, target=PA_SCHEMA),
        repeat=repeat, inner=2_000,
    ))

    # 24. .check(pa.DataType) — cheapest schema-promote shape:
    # single-column casts where the caller passes a bare dtype.
    results.append(_time_one(
        "check: CastOptions.check(pa.DataType)",
        lambda: CastOptions.check(PA_DTYPE),
        repeat=repeat, inner=20_000,
    ))

    # 25. .check(pa.Field) — schema-promote with a single field.
    results.append(_time_one(
        "check: CastOptions.check(pa.Field)",
        lambda: CastOptions.check(PA_FIELD),
        repeat=repeat, inner=10_000,
    ))

    # 26. .check(yggdrasil Field) — already-Field input. Tests the
    # isinstance-tuple match path on the schema-promote branch.
    YG_FIELD = Field.from_(PA_SCHEMA)
    results.append(_time_one(
        "check: CastOptions.check(yggdrasil Field)",
        lambda: CastOptions.check(YG_FIELD),
        repeat=repeat, inner=20_000,
    ))

    # 27. .check(dict, safe=True) — Mapping options + an explicit
    # override. Exercises the {**options, **overrides} merge.
    cfg_no_safe = {"target": PA_SCHEMA, "row_size": 1024}
    results.append(_time_one(
        "check: CastOptions.check(dict, safe=True)",
        lambda: CastOptions.check(cfg_no_safe, safe=True),
        repeat=repeat, inner=2_000,
    ))

    # 28. .check({"columns": [...]}) — columns shortcut from a dict.
    # Forces the columns-extraction branch (dict copy + struct-of-
    # ObjectType source build).
    cfg_columns = {"columns": ["id", "name", "amount", "ts"]}
    results.append(_time_one(
        "check: CastOptions.check({'columns': [...]})",
        lambda: CastOptions.check(cfg_columns),
        repeat=repeat, inner=5_000,
    ))

    # 29. .check(None, columns=[...]) — columns shortcut as a kwarg,
    # no options. Fires when callers want a placeholder source.
    cols_only = ["id", "name", "amount", "ts"]
    results.append(_time_one(
        "check: CastOptions.check(None, columns=[...])",
        lambda: CastOptions.check(None, columns=cols_only),
        repeat=repeat, inner=5_000,
    ))

    # 30. .check(subclass_existing) — passthrough on a subclass
    # options instance. Same as scenario 6 but with a CsvOptions
    # input (catches any regression when the cls check moves).
    csv_opts = CsvOptions(target=PA_SCHEMA)
    results.append(_time_one(
        "check: CsvOptions.check(csv_opts) passthrough",
        lambda: CsvOptions.check(csv_opts),
        repeat=repeat, inner=200_000,
    ))

    # 31. .check(base_options) on subclass — re-home from
    # CastOptions to CsvOptions. Walks dataclasses.fields and
    # carries over only the shared base fields.
    base_opts = CastOptions(target=PA_SCHEMA, safe=True)
    results.append(_time_one(
        "check: CsvOptions.check(base_castoptions) re-home",
        lambda: CsvOptions.check(base_opts),
        repeat=repeat, inner=2_000,
    ))

    # 32. .check(existing) on subclass — passthrough fast-path on
    # subclass.
    csv_no_target = CsvOptions()
    results.append(_time_one(
        "check: CsvOptions.check(csv_opts_default) passthrough",
        lambda: CsvOptions.check(csv_no_target),
        repeat=repeat, inner=200_000,
    ))

    # 33. .check(...) — Ellipsis-as-options sentinel: equivalent to
    # None. Some callers thread ``...`` through a default to
    # distinguish "didn't pass" from "passed None".
    results.append(_time_one(
        "check: CastOptions.check(...)",
        lambda: CastOptions.check(...),
        repeat=repeat, inner=50_000,
    ))

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=5,
                    help="Outer repeat count per scenario (median across).")
    args = ap.parse_args()

    print(f"# repeat={args.repeat}")
    print(f"# {'label':<48s}  {'best':>14s}  {'median':>16s}  {'mean':>14s}")
    for row in scenarios(args.repeat):
        print(_fmt(row))


if __name__ == "__main__":
    main()
