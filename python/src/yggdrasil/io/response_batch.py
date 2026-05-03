"""Triage views over batched ``Session.send_many`` results.

`send_many` historically returned an `Iterator[Response]` and the Spark
variant returned a single fused DataFrame. Both lose the *origin* of every
row — the caller can't tell whether a response came from the local
on-disk cache, the remote SQL cache, or a fresh network fetch.

This module adds two structured carriers that keep that split visible:

- :class:`ResponseBatch`     — Python-side, three lists of `Response`.
- :class:`SparkResponseBatch` — Spark-side, three optional DataFrames.

Both stay iterable / unionable so existing call sites that only want a
flat result keep working with one extra method call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator, Optional

import pyarrow as pa

from .response import RESPONSE_SCHEMA, Response

if TYPE_CHECKING:
    from pyspark.sql import DataFrame as SparkDataFrame, SparkSession


__all__ = ["ResponseBatch", "SparkResponseBatch", "responses_to_spark_df"]


@dataclass
class ResponseBatch:
    """Origin-tagged view of a batch of responses.

    Three buckets, in pipeline order:

    - :attr:`local_hits`  — served from the on-disk pickle cache.
    - :attr:`remote_hits` — served from the remote SQL cache.
    - :attr:`new_hits`    — fetched from the network this run.

    Iteration walks the buckets in that order (local → remote → new), so a
    caller that just wants the flat stream still gets a stable order and
    the cache-cheap responses come out first.
    """

    local_hits: list[Response] = field(default_factory=list)
    remote_hits: list[Response] = field(default_factory=list)
    new_hits: list[Response] = field(default_factory=list)

    def __iter__(self) -> Iterator[Response]:
        yield from self.local_hits
        yield from self.remote_hits
        yield from self.new_hits

    def __len__(self) -> int:
        return len(self.local_hits) + len(self.remote_hits) + len(self.new_hits)

    def __bool__(self) -> bool:
        return len(self) > 0

    @property
    def responses(self) -> list[Response]:
        """Flat list of every response, local → remote → new."""
        return [*self.local_hits, *self.remote_hits, *self.new_hits]

    @property
    def counts(self) -> dict[str, int]:
        """Per-origin counts, handy for logging or quick assertions."""
        return {
            "local": len(self.local_hits),
            "remote": len(self.remote_hits),
            "new": len(self.new_hits),
        }

    def extend(self, other: "ResponseBatch") -> "ResponseBatch":
        """Merge another batch in place. Returns self for chaining."""
        self.local_hits.extend(other.local_hits)
        self.remote_hits.extend(other.remote_hits)
        self.new_hits.extend(other.new_hits)
        return self

    def to_spark(self, spark: "SparkSession") -> "SparkResponseBatch":
        """Lift this Python-side batch into a Spark-side batch.

        Each non-empty bucket becomes a Spark DataFrame typed as
        ``RESPONSE_SCHEMA``; empty buckets stay as ``None`` so the caller
        can tell "no rows" apart from "no fetch happened".
        """
        return SparkResponseBatch(
            local_hits=responses_to_spark_df(self.local_hits, spark),
            remote_hits=responses_to_spark_df(self.remote_hits, spark),
            new_hits=responses_to_spark_df(self.new_hits, spark),
            spark=spark,
        )


@dataclass
class SparkResponseBatch:
    """Spark-flavored sibling of :class:`ResponseBatch`.

    Each bucket holds either a Spark DataFrame typed as
    ``RESPONSE_SCHEMA`` or ``None`` when that origin produced no rows.
    Use :meth:`to_dataframe` when you just want the fused result.

    The optional :attr:`spark` reference lets :meth:`to_dataframe` synthesize
    a typed empty DataFrame when every bucket is empty — without it the
    caller would have to special-case "all three None" themselves.
    """

    local_hits: Optional["SparkDataFrame"] = None
    remote_hits: Optional["SparkDataFrame"] = None
    new_hits: Optional["SparkDataFrame"] = None
    spark: Optional["SparkSession"] = None

    def parts(self) -> list["SparkDataFrame"]:
        """Non-empty buckets in pipeline order."""
        return [
            df for df in (self.local_hits, self.remote_hits, self.new_hits)
            if df is not None
        ]

    @property
    def counts(self) -> dict[str, Optional[int]]:
        """Per-origin row counts. ``None`` means that bucket was skipped.

        Spark counts trigger a job — fine for debugging, not for hot paths.
        """
        return {
            "local": self.local_hits.count() if self.local_hits is not None else None,
            "remote": self.remote_hits.count() if self.remote_hits is not None else None,
            "new": self.new_hits.count() if self.new_hits is not None else None,
        }

    def __iter__(self) -> Iterator["SparkDataFrame"]:
        yield from self.parts()

    def __bool__(self) -> bool:
        return any(df is not None for df in (self.local_hits, self.remote_hits, self.new_hits))

    def to_dataframe(self) -> "SparkDataFrame":
        """Union all non-empty buckets into one DataFrame.

        Buckets are unioned with ``allowMissingColumns=False`` because every
        bucket is built against the same ``RESPONSE_SCHEMA`` — a missing
        column would mean a real schema drift, and silent column-fill is
        worse than a loud failure.
        """
        parts = self.parts()
        if not parts:
            if self.spark is None:
                raise RuntimeError(
                    "SparkResponseBatch has no parts and no SparkSession bound; "
                    "cannot synthesize an empty DataFrame. Pass `spark=` when "
                    "constructing the batch, or check `.parts()` before calling "
                    "to_dataframe()."
                )
            return self.spark.createDataFrame(
                [], schema=RESPONSE_SCHEMA.to_spark_schema(),
            )
        result = parts[0]
        for part in parts[1:]:
            result = result.unionByName(part, allowMissingColumns=False)
        return result


def responses_to_spark_df(
    responses: list[Response],
    spark: "SparkSession",
) -> Optional["SparkDataFrame"]:
    """Build a Spark DataFrame from a list of :class:`Response`.

    Returns ``None`` for an empty list so callers can keep the
    "skipped vs empty" distinction. PySpark 3.4+ accepts a
    ``pyarrow.Table`` directly and preserves the full schema (maps,
    nullable bytes); older releases need the schema-typed pandas
    fallback because the bare ``to_pandas()`` path drops type info and
    Spark fails to infer nested map / binary columns.
    """
    if not responses:
        return None
    batches = [r.to_arrow_batch(parse=False) for r in responses]
    table = pa.Table.from_batches(batches).combine_chunks()
    try:
        return spark.createDataFrame(table)
    except (TypeError, AttributeError):
        return spark.createDataFrame(
            table.to_pandas(),
            schema=RESPONSE_SCHEMA.to_spark_schema(),
        )
