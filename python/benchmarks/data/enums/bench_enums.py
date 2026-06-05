"""Benchmark the :mod:`yggdrasil.enums` parser hot paths.

Why this exists
---------------

The enums under :mod:`yggdrasil.enums` (``ByteUnit``, ``Mode``,
``TimeUnit``, ``JoinType``, ``State``, ``Scheme``, ``NodeType``,
``Currency``, ``Timezone``, ``MimeType``, ``Codec``) are the canonical
entry points for normalising fixed-vocabulary tokens at every API
boundary. They get hit on every ``CastOptions(mode=...)`` construct,
every URL parse, every temporal-type ``unit=...`` coercion, every
HTTP / SQL / Spark dispatch — sometimes thousands of times per logical
operation.

This bench focuses on the per-call cost of:

* the **identity / passthrough** path (``X.from_(member)``) — already
  an enum, no work to do;
* the **canonical alias** path (``X.from_("overwrite")`` /
  ``ByteUnit.from_("MIB")``) — fully matches an entry in the alias
  table on the first try;
* the **case / whitespace tolerant** path
  (``X.from_("  OverWrite ")`` / ``ByteUnit.from_("128 MB")``) — pays
  ``strip().lower()`` plus regex parsing where applicable;
* the **integer / structural** path (``X.from_(7)`` for ``IntEnum`` /
  ``ByteUnit.parse_size("1.5 GiB")``).

Usage::

    PYTHONPATH=src python benchmarks/data/enums/bench_enums.py
    PYTHONPATH=src python benchmarks/data/enums/bench_enums.py --repeat 7
"""
from __future__ import annotations

import argparse
import statistics
import time
from typing import Callable

from yggdrasil.enums import (
    ByteUnit,
    JoinType,
    Mode,
    NodeType,
    Scheme,
    State,
    TimeUnit,
)
from yggdrasil.enums.currency.currency import Currency
from yggdrasil.enums.mime_type import MimeType, MimeTypes
from yggdrasil.enums.codec import Codec
from yggdrasil.enums.timezone import Timezone


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _time_one(label: str, fn: Callable[[], None], *, repeat: int, inner: int) -> dict:
    # Warm-up — first call pays import/cache-population costs.
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
        f"{r['label']:<58s}  "
        f"best={r['best']*scale:8.2f} {unit}  "
        f"median={r['median']*scale:8.2f} {unit}  "
        f"mean={r['mean']*scale:8.2f} {unit}"
    )


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def _byteunit_scenarios(repeat: int) -> list[dict]:
    results: list[dict] = []
    mib = ByteUnit.MIB

    results.append(_time_one(
        "ByteUnit.from_(member) passthrough",
        lambda: ByteUnit.from_(mib),
        repeat=repeat, inner=200_000,
    ))
    results.append(_time_one(
        "ByteUnit.from_('MIB') canonical name",
        lambda: ByteUnit.from_("MIB"),
        repeat=repeat, inner=100_000,
    ))
    results.append(_time_one(
        "ByteUnit.from_('mib') lowercase alias",
        lambda: ByteUnit.from_("mib"),
        repeat=repeat, inner=100_000,
    ))
    results.append(_time_one(
        "ByteUnit.from_('MB') colloquial",
        lambda: ByteUnit.from_("MB"),
        repeat=repeat, inner=100_000,
    ))
    results.append(_time_one(
        "ByteUnit.from_('megabytes') long form",
        lambda: ByteUnit.from_("megabytes"),
        repeat=repeat, inner=100_000,
    ))
    results.append(_time_one(
        "ByteUnit.from_(1048576) int value",
        lambda: ByteUnit.from_(1048576),
        repeat=repeat, inner=100_000,
    ))
    results.append(_time_one(
        "ByteUnit.parse_size(int)",
        lambda: ByteUnit.parse_size(1048576),
        repeat=repeat, inner=200_000,
    ))
    results.append(_time_one(
        "ByteUnit.parse_size('128 MB')",
        lambda: ByteUnit.parse_size("128 MB"),
        repeat=repeat, inner=50_000,
    ))
    results.append(_time_one(
        "ByteUnit.parse_size('1.5 GiB')",
        lambda: ByteUnit.parse_size("1.5 GiB"),
        repeat=repeat, inner=50_000,
    ))
    results.append(_time_one(
        "ByteUnit.parse_size('MiB') bare unit",
        lambda: ByteUnit.parse_size("MiB"),
        repeat=repeat, inner=100_000,
    ))
    return results


def _mode_scenarios(repeat: int) -> list[dict]:
    results: list[dict] = []
    ow = Mode.OVERWRITE

    results.append(_time_one(
        "Mode.from_(member) passthrough",
        lambda: Mode.from_(ow),
        repeat=repeat, inner=200_000,
    ))
    results.append(_time_one(
        "Mode.from_('overwrite') alias",
        lambda: Mode.from_("overwrite"),
        repeat=repeat, inner=100_000,
    ))
    results.append(_time_one(
        "Mode.from_('OVERWRITE') member name",
        lambda: Mode.from_("OVERWRITE"),
        repeat=repeat, inner=100_000,
    ))
    results.append(_time_one(
        "Mode.from_('rb') OS short",
        lambda: Mode.from_("rb"),
        repeat=repeat, inner=100_000,
    ))
    results.append(_time_one(
        "Mode.from_('wb+') OS plus",
        lambda: Mode.from_("wb+"),
        repeat=repeat, inner=100_000,
    ))
    results.append(_time_one(
        "Mode.from_('error_if_exists') long alias",
        lambda: Mode.from_("error_if_exists"),
        repeat=repeat, inner=100_000,
    ))
    results.append(_time_one(
        "Mode.from_(2) int",
        lambda: Mode.from_(2),
        repeat=repeat, inner=200_000,
    ))
    results.append(_time_one(
        "Mode.from_(None) default",
        lambda: Mode.from_(None),
        repeat=repeat, inner=200_000,
    ))
    return results


def _timeunit_scenarios(repeat: int) -> list[dict]:
    results: list[dict] = []
    us = TimeUnit.MICROSECOND

    results.append(_time_one(
        "TimeUnit.from_(member) passthrough",
        lambda: TimeUnit.from_(us),
        repeat=repeat, inner=200_000,
    ))
    results.append(_time_one(
        "TimeUnit.from_('us') canonical",
        lambda: TimeUnit.from_("us"),
        repeat=repeat, inner=200_000,
    ))
    results.append(_time_one(
        "TimeUnit.from_('US') case",
        lambda: TimeUnit.from_("US"),
        repeat=repeat, inner=100_000,
    ))
    results.append(_time_one(
        "TimeUnit.from_('microsecond') long form",
        lambda: TimeUnit.from_("microsecond"),
        repeat=repeat, inner=100_000,
    ))
    results.append(_time_one(
        "TimeUnit.from_('microseconds') plural",
        lambda: TimeUnit.from_("microseconds"),
        repeat=repeat, inner=100_000,
    ))
    results.append(_time_one(
        "TimeUnit.from_('µs') unicode",
        lambda: TimeUnit.from_("µs"),
        repeat=repeat, inner=100_000,
    ))
    return results


def _jointype_scenarios(repeat: int) -> list[dict]:
    results: list[dict] = []
    lo = JoinType.LEFT_OUTER

    results.append(_time_one(
        "JoinType.from_(member) passthrough",
        lambda: JoinType.from_(lo),
        repeat=repeat, inner=200_000,
    ))
    results.append(_time_one(
        "JoinType.from_('inner')",
        lambda: JoinType.from_("inner"),
        repeat=repeat, inner=100_000,
    ))
    results.append(_time_one(
        "JoinType.from_('left')",
        lambda: JoinType.from_("left"),
        repeat=repeat, inner=100_000,
    ))
    results.append(_time_one(
        "JoinType.from_('left outer') space",
        lambda: JoinType.from_("left outer"),
        repeat=repeat, inner=100_000,
    ))
    results.append(_time_one(
        "JoinType.from_('LEFT JOIN') SQL",
        lambda: JoinType.from_("LEFT JOIN"),
        repeat=repeat, inner=100_000,
    ))
    results.append(_time_one(
        "JoinType.from_('left_anti') underscore",
        lambda: JoinType.from_("left_anti"),
        repeat=repeat, inner=100_000,
    ))
    results.append(_time_one(
        "JoinType.from_(1) int",
        lambda: JoinType.from_(1),
        repeat=repeat, inner=200_000,
    ))
    return results


def _state_scenarios(repeat: int) -> list[dict]:
    results: list[dict] = []
    succ = State.SUCCEEDED

    results.append(_time_one(
        "State.from_(member) passthrough",
        lambda: State.from_(succ),
        repeat=repeat, inner=200_000,
    ))
    results.append(_time_one(
        "State.from_('succeeded')",
        lambda: State.from_("succeeded"),
        repeat=repeat, inner=100_000,
    ))
    results.append(_time_one(
        "State.from_('SUCCEEDED')",
        lambda: State.from_("SUCCEEDED"),
        repeat=repeat, inner=100_000,
    ))
    results.append(_time_one(
        "State.from_('running')",
        lambda: State.from_("running"),
        repeat=repeat, inner=100_000,
    ))
    results.append(_time_one(
        "State.from_('in_progress') alias",
        lambda: State.from_("in_progress"),
        repeat=repeat, inner=100_000,
    ))
    results.append(_time_one(
        "State.from_(2) int",
        lambda: State.from_(2),
        repeat=repeat, inner=200_000,
    ))
    return results


def _scheme_scenarios(repeat: int) -> list[dict]:
    results: list[dict] = []
    s3 = Scheme.S3

    results.append(_time_one(
        "Scheme.from_(member) passthrough",
        lambda: Scheme.from_(s3),
        repeat=repeat, inner=200_000,
    ))
    results.append(_time_one(
        "Scheme.from_('s3')",
        lambda: Scheme.from_("s3"),
        repeat=repeat, inner=100_000,
    ))
    results.append(_time_one(
        "Scheme.from_('S3')",
        lambda: Scheme.from_("S3"),
        repeat=repeat, inner=100_000,
    ))
    results.append(_time_one(
        "Scheme.from_('s3a') alias",
        lambda: Scheme.from_("s3a"),
        repeat=repeat, inner=100_000,
    ))
    results.append(_time_one(
        "Scheme.from_('https://') trailing",
        lambda: Scheme.from_("https://"),
        repeat=repeat, inner=100_000,
    ))
    return results


def _nodetype_scenarios(repeat: int) -> list[dict]:
    results: list[dict] = []
    fx = NodeType.FLEET_XLARGE

    results.append(_time_one(
        "NodeType.from_(member) passthrough",
        lambda: NodeType.from_(fx),
        repeat=repeat, inner=200_000,
    ))
    results.append(_time_one(
        "NodeType.from_('default')",
        lambda: NodeType.from_("default"),
        repeat=repeat, inner=100_000,
    ))
    results.append(_time_one(
        "NodeType.from_('FLEET_XLARGE')",
        lambda: NodeType.from_("FLEET_XLARGE"),
        repeat=repeat, inner=100_000,
    ))
    results.append(_time_one(
        "NodeType.from_('rd-fleet.xlarge') value",
        lambda: NodeType.from_("rd-fleet.xlarge"),
        repeat=repeat, inner=100_000,
    ))
    results.append(_time_one(
        "NodeType.to_id('m5.large')",
        lambda: NodeType.to_id("m5.large"),
        repeat=repeat, inner=100_000,
    ))
    results.append(_time_one(
        "NodeType.to_id('custom.sku') passthrough",
        lambda: NodeType.to_id("custom.sku"),
        repeat=repeat, inner=100_000,
    ))
    return results


def _currency_scenarios(repeat: int) -> list[dict]:
    results: list[dict] = []
    usd = Currency.USD

    results.append(_time_one(
        "Currency.parse(member) passthrough",
        lambda: Currency.parse(usd),
        repeat=repeat, inner=200_000,
    ))
    results.append(_time_one(
        "Currency.parse('USD')",
        lambda: Currency.parse("USD"),
        repeat=repeat, inner=100_000,
    ))
    results.append(_time_one(
        "Currency.parse('usd')",
        lambda: Currency.parse("usd"),
        repeat=repeat, inner=100_000,
    ))
    results.append(_time_one(
        "Currency.parse('$') alias",
        lambda: Currency.parse("$"),
        repeat=repeat, inner=100_000,
    ))
    results.append(_time_one(
        "Currency.parse('EUR')",
        lambda: Currency.parse("EUR"),
        repeat=repeat, inner=100_000,
    ))
    return results


def _timezone_scenarios(repeat: int) -> list[dict]:
    results: list[dict] = []
    utc = Timezone.UTC

    results.append(_time_one(
        "Timezone.from_(member) passthrough",
        lambda: Timezone.from_(utc),
        repeat=repeat, inner=200_000,
    ))
    results.append(_time_one(
        "Timezone.from_('UTC')",
        lambda: Timezone.from_("UTC"),
        repeat=repeat, inner=100_000,
    ))
    results.append(_time_one(
        "Timezone.from_('Europe/Paris') IANA",
        lambda: Timezone.from_("Europe/Paris"),
        repeat=repeat, inner=100_000,
    ))
    results.append(_time_one(
        "Timezone.from_('CET') alias",
        lambda: Timezone.from_("CET"),
        repeat=repeat, inner=100_000,
    ))
    results.append(_time_one(
        "Timezone.from_('+01:00') offset",
        lambda: Timezone.from_("+01:00"),
        repeat=repeat, inner=50_000,
    ))
    results.append(_time_one(
        "Timezone.from_(None) default",
        lambda: Timezone.from_(None),
        repeat=repeat, inner=200_000,
    ))
    return results


def _mime_scenarios(repeat: int) -> list[dict]:
    results: list[dict] = []
    csv = MimeTypes.CSV

    results.append(_time_one(
        "MimeType.from_(member) passthrough",
        lambda: MimeType.from_(csv),
        repeat=repeat, inner=200_000,
    ))
    results.append(_time_one(
        "MimeType.from_('text/csv') value",
        lambda: MimeType.from_("text/csv"),
        repeat=repeat, inner=100_000,
    ))
    results.append(_time_one(
        "MimeType.from_('csv') extension",
        lambda: MimeType.from_("csv"),
        repeat=repeat, inner=100_000,
    ))
    results.append(_time_one(
        "MimeType.from_('.parquet') dot ext",
        lambda: MimeType.from_(".parquet"),
        repeat=repeat, inner=100_000,
    ))
    results.append(_time_one(
        "MimeType.from_('json')",
        lambda: MimeType.from_("json"),
        repeat=repeat, inner=100_000,
    ))
    return results


def _codec_scenarios(repeat: int) -> list[dict]:
    results: list[dict] = []
    from yggdrasil.enums.codec import GZIP

    results.append(_time_one(
        "Codec.from_(instance) passthrough",
        lambda: Codec.from_(GZIP),
        repeat=repeat, inner=200_000,
    ))
    results.append(_time_one(
        "Codec.from_('gzip')",
        lambda: Codec.from_("gzip"),
        repeat=repeat, inner=100_000,
    ))
    results.append(_time_one(
        "Codec.from_('zstd')",
        lambda: Codec.from_("zstd"),
        repeat=repeat, inner=100_000,
    ))
    results.append(_time_one(
        "Codec.from_('GZIP')",
        lambda: Codec.from_("GZIP"),
        repeat=repeat, inner=100_000,
    ))
    return results


def scenarios(repeat: int) -> list[dict]:
    out: list[dict] = []
    out += _byteunit_scenarios(repeat)
    out += _mode_scenarios(repeat)
    out += _timeunit_scenarios(repeat)
    out += _jointype_scenarios(repeat)
    out += _state_scenarios(repeat)
    out += _scheme_scenarios(repeat)
    out += _nodetype_scenarios(repeat)
    out += _currency_scenarios(repeat)
    out += _timezone_scenarios(repeat)
    out += _mime_scenarios(repeat)
    out += _codec_scenarios(repeat)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=5,
                    help="Outer repeat count per scenario (median across).")
    args = ap.parse_args()

    print(f"# repeat={args.repeat}")
    print(f"# {'label':<58s}  {'best':>14s}  {'median':>16s}  {'mean':>14s}")
    for row in scenarios(args.repeat):
        print(_fmt(row))


if __name__ == "__main__":
    main()
