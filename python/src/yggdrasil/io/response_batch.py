"""Origin-tagged carrier for batched ``Session.send_many`` results.

`send_many` historically returned an `Iterator[Response]` and the Spark
variant returned a single fused DataFrame. Both lose the *origin* of every
row — the caller can't tell whether a response came from the local
on-disk cache, the remote SQL cache, or a fresh network fetch.

:class:`ResponseBatch` keeps that split visible. The three buckets are
all stored as the same type — :class:`TabularIO` — so Python
(:class:`MemoryArrowIO`) and Spark (:class:`MemorySparkIO`) share one
contract. Empty buckets keep their :data:`RESPONSE_SCHEMA` so a batch
with no responses still answers schema questions correctly. Iteration
walks each holder's Arrow batches and rebuilds :class:`Response` objects
via :meth:`Response.from_arrow_tabular`, so the batch never has to
special-case which engine produced the rows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator, Optional, Union

from .buffer.base import TabularIO
from .buffer.memory import MemoryArrowIO, MemorySparkIO
from .response import RESPONSE_ARROW_SCHEMA, RESPONSE_SCHEMA, Response

if TYPE_CHECKING:
    from pyspark.sql import DataFrame as SparkDataFrame, SparkSession


__all__ = [
    "ResponseBatch",
    "BucketInput",
    "responses_to_tabular",
    "spark_to_tabular",
    "empty_arrow_holder",
    "empty_spark_holder",
]


# What the public constructor accepts per bucket. ``None`` is coerced to
# a schema-bearing empty holder so every bucket carries
# :data:`RESPONSE_SCHEMA` — Spark when the batch was built with a
# session, Arrow otherwise. ``TabularIO`` is taken as-is; lists and
# Spark DataFrames are funnelled through :func:`responses_to_tabular`
# / :func:`spark_to_tabular`.
BucketInput = Union[
    list[Response],
    "TabularIO",
    "SparkDataFrame",
    None,
]


# ---------------------------------------------------------------------------
# Schema-bearing empty holder constructors
# ---------------------------------------------------------------------------


def empty_arrow_holder() -> MemoryArrowIO:
    """Empty :class:`MemoryArrowIO` keyed to :data:`RESPONSE_ARROW_SCHEMA`.

    Use as the default bucket in Python mode so consumers can read
    ``holder.schema`` even when no responses landed in this origin.
    """
    return MemoryArrowIO(schema=RESPONSE_ARROW_SCHEMA)


def empty_spark_holder(spark: "SparkSession") -> MemorySparkIO:
    """Empty :class:`MemorySparkIO` keyed to the Spark response schema.

    The held frame is built with ``spark.createDataFrame([], schema=...)``
    so the holder advertises :data:`RESPONSE_SCHEMA` (Spark form) without
    materialising any rows.
    """
    df = spark.createDataFrame([], schema=RESPONSE_SCHEMA.to_spark_schema())
    return MemorySparkIO(df, spark=spark)


# ---------------------------------------------------------------------------
# Coercion helpers — one shape in, one TabularIO out
# ---------------------------------------------------------------------------

def responses_to_tabular(responses: list[Response]) -> MemoryArrowIO:
    """Wrap a list of :class:`Response` in a :class:`MemoryArrowIO`.

    Always returns a schema-bearing holder — an empty list yields an
    empty :class:`MemoryArrowIO` whose ``.schema`` is
    :data:`RESPONSE_ARROW_SCHEMA`, so a batch with zero rows still
    answers schema questions correctly. Each response is serialized via
    ``to_arrow_batch(parse=False)`` (one row per batch) and held in
    memory — no IPC bytes, no spill. Reads come back through
    :meth:`Response.from_arrow_tabular`.
    """
    if not responses:
        return empty_arrow_holder()
    return MemoryArrowIO(
        (r.to_arrow_batch(parse=False) for r in responses),
        schema=RESPONSE_ARROW_SCHEMA,
    )


def spark_to_tabular(df: "SparkDataFrame") -> MemorySparkIO:
    """Wrap a Spark DataFrame in a :class:`MemorySparkIO` (no collect).

    The DataFrame lives on the holder's mutable ``frame`` slot, so
    :meth:`TabularIO.read_spark_frame` returns it untouched. A
    subsequent :meth:`TabularIO.read_arrow_batches` would force
    ``df.toArrow()`` — fine for small frames, but Spark-mode iteration
    on the :class:`ResponseBatch` is disallowed precisely so callers
    don't trip over that collect by accident.
    """
    return MemorySparkIO(df)


def _coerce_bucket(
    value: BucketInput,
    *,
    spark: Optional["SparkSession"] = None,
) -> TabularIO:
    """Funnel any accepted bucket input down to a schema-bearing ``TabularIO``.

    Strict on meaning: an unknown shape raises rather than silently
    being treated as Spark. ``None`` is coerced to an empty
    schema-bearing holder — Spark when ``spark`` is passed (so the
    bucket aligns with the rest of a Spark batch), Arrow otherwise.
    Lists land in :func:`responses_to_tabular`, Spark DataFrames in
    :func:`spark_to_tabular`, ``TabularIO`` passes through.
    """
    if value is None:
        if spark is not None:
            return empty_spark_holder(spark)
        return empty_arrow_holder()
    if isinstance(value, TabularIO):
        return value
    if isinstance(value, list):
        return responses_to_tabular(value)
    # Heuristic: anything that exposes a Spark-DataFrame-shaped surface
    # (an `unionByName` method) is treated as a Spark DataFrame. We
    # avoid `import pyspark` here so the optional dependency stays
    # optional.
    if hasattr(value, "unionByName") and hasattr(value, "toArrow"):
        return spark_to_tabular(value)
    raise TypeError(
        f"Unsupported bucket input: {type(value).__name__}. "
        "Expected list[Response], TabularIO, Spark DataFrame, or None."
    )


# ---------------------------------------------------------------------------
# ResponseBatch
# ---------------------------------------------------------------------------


class ResponseBatch:
    """Origin-tagged view of a batch of responses.

    Three buckets, in pipeline order:

    - ``local_hits``  — served from the on-disk pickle cache.
    - ``remote_hits`` — served from the remote SQL cache.
    - ``new_hits``    — fetched from the network this run.

    Each bucket lives on a private :class:`TabularIO` holder
    (``_local_response`` / ``_remote_response`` / ``_new_response``);
    the public ``local_hits`` / ``remote_hits`` / ``new_hits``
    properties expose them. Constructor inputs go through a coercion
    step so callers can hand in a ``list[Response]``, a Spark
    DataFrame, an existing ``TabularIO``, or ``None``. ``None`` and
    empty inputs become schema-bearing empty holders so every bucket
    advertises :data:`RESPONSE_SCHEMA` — useful when downstream code
    inspects the schema before any rows arrive.

    Iteration rebuilds :class:`Response` objects from each holder's
    Arrow batches via :meth:`Response.from_arrow_tabular`, so the
    Python and Spark paths share one read contract. Spark-mode
    iteration is rejected up front (it would force a driver-side
    collect); use :meth:`to_dataframe` instead.
    """

    __slots__ = (
        "_local_response",
        "_remote_response",
        "_new_response",
        "spark",
    )

    def __init__(
        self,
        local_hits: BucketInput = None,
        remote_hits: BucketInput = None,
        new_hits: BucketInput = None,
        *,
        spark: Optional["SparkSession"] = None,
    ) -> None:
        self.spark: Optional["SparkSession"] = spark
        self._local_response: TabularIO = _coerce_bucket(local_hits, spark=spark)
        self._remote_response: TabularIO = _coerce_bucket(remote_hits, spark=spark)
        self._new_response: TabularIO = _coerce_bucket(new_hits, spark=spark)

    def __repr__(self) -> str:
        return (
            f"ResponseBatch(local_hits={self._local_response!r}, "
            f"remote_hits={self._remote_response!r}, "
            f"new_hits={self._new_response!r})"
        )

    # ------------------------------------------------------------------
    # Public accessors over the private holders
    # ------------------------------------------------------------------

    @property
    def local_hits(self) -> TabularIO:
        return self._local_response

    @local_hits.setter
    def local_hits(self, value: BucketInput) -> None:
        self._local_response = _coerce_bucket(value, spark=self.spark)

    @property
    def remote_hits(self) -> TabularIO:
        return self._remote_response

    @remote_hits.setter
    def remote_hits(self, value: BucketInput) -> None:
        self._remote_response = _coerce_bucket(value, spark=self.spark)

    @property
    def new_hits(self) -> TabularIO:
        return self._new_response

    @new_hits.setter
    def new_hits(self, value: BucketInput) -> None:
        self._new_response = _coerce_bucket(value, spark=self.spark)

    # ------------------------------------------------------------------
    # Shape helpers
    # ------------------------------------------------------------------

    def _holders(self) -> tuple[TabularIO, TabularIO, TabularIO]:
        return (self._local_response, self._remote_response, self._new_response)

    @staticmethod
    def _is_spark_holder(holder: TabularIO) -> bool:
        return isinstance(holder, MemorySparkIO) and holder.frame is not None

    @property
    def is_spark(self) -> bool:
        """True if any holder carries a Spark DataFrame."""
        return any(self._is_spark_holder(h) for h in self._holders())

    def parts(self) -> list[TabularIO]:
        """All bucket holders in pipeline order.

        Every bucket is schema-bearing, including empty ones, so this
        always returns three holders.
        """
        return list(self._holders())

    @property
    def counts(self) -> dict[str, int]:
        """Per-origin row counts.

        For :class:`MemorySparkIO` this triggers ``df.count()``; for
        :class:`MemoryArrowIO` it sums ``num_rows`` across the
        in-memory batches — fine for debugging or small assertions,
        not for hot paths.
        """
        out: dict[str, int] = {}
        for name, h in zip(("local", "remote", "new"), self._holders()):
            if isinstance(h, MemorySparkIO):
                out[name] = h.frame.count() if h.frame is not None else 0
            elif isinstance(h, MemoryArrowIO):
                out[name] = h.num_rows
            else:
                out[name] = h.read_arrow_table().num_rows
        return out

    # ------------------------------------------------------------------
    # Python-mode iteration — rebuild Response objects from Arrow batches
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[Response]:
        if self.is_spark:
            raise TypeError(
                "ResponseBatch is in Spark mode — iterating Response objects "
                "would force a driver-side collect via df.toArrow(). Call "
                "`.to_dataframe()` for the fused DataFrame, or read individual "
                "`.local_hits` / `.remote_hits` / `.new_hits` holders if you "
                "really need the Spark frames."
            )
        # Route through `read_records` so each Response is built from a
        # `Record` sharing the holder's singleton Schema — one Schema
        # allocation per holder regardless of row count.
        for holder in self._holders():
            yield from Response.from_records(holder.read_records())

    def __len__(self) -> int:
        if self.is_spark:
            raise TypeError(
                "len() is not defined for a Spark ResponseBatch — use "
                "`.counts` (which calls `df.count()` per holder) or check "
                "the individual DataFrames."
            )
        total = 0
        for holder in self._holders():
            total += holder.read_arrow_table().num_rows
        return total

    def __bool__(self) -> bool:
        # Python mode: cheap row-count via in-memory Arrow batches.
        # Spark mode: forcing ``df.count()`` for an implicit truthiness
        # check would surprise callers, so any Spark-backed holder
        # makes the batch truthy regardless of contents — use
        # :attr:`counts` when you need the precise size.
        for holder in self._holders():
            if isinstance(holder, MemorySparkIO):
                if holder.frame is not None:
                    return True
            elif isinstance(holder, MemoryArrowIO):
                if holder.num_rows > 0:
                    return True
            else:
                if holder.read_arrow_table().num_rows > 0:
                    return True
        return False

    @property
    def responses(self) -> list[Response]:
        """Flat list of every response, local → remote → new.

        Python mode only — raises if any holder is Spark-backed.
        """
        return list(self)

    # ------------------------------------------------------------------
    # Mutation / merge
    # ------------------------------------------------------------------

    def extend(self, other: "ResponseBatch") -> "ResponseBatch":
        """Merge another batch in place. Returns self for chaining.

        Per-bucket merges append the other side's rows into the
        matching holder. Both sides must agree on engine — Python
        merges append Arrow batches, Spark merges union Spark frames
        via ``unionByName``. Mixing engines is rejected because it
        would force a hidden Spark conversion at merge time.
        """
        if self.is_spark != other.is_spark:
            raise TypeError(
                "extend() requires both batches in the same engine. "
                "Lift one side with `to_dataframe()` (or rebuild Python-side) "
                "before merging across modes."
            )

        from .enums import Mode

        for name in ("_local_response", "_remote_response", "_new_response"):
            mine = getattr(self, name)
            theirs = getattr(other, name)
            if isinstance(theirs, MemorySparkIO) and theirs.frame is not None:
                if isinstance(mine, MemorySparkIO) and mine.frame is not None:
                    mine.frame = mine.frame.unionByName(
                        theirs.frame, allowMissingColumns=False,
                    )
                else:
                    setattr(self, name, theirs)
                continue
            mine.write_arrow_batches(theirs.read_arrow_batches(), mode=Mode.APPEND)
        return self

    # ------------------------------------------------------------------
    # Spark interop
    # ------------------------------------------------------------------

    def to_dataframe(
        self,
        spark: Optional["SparkSession"] = None,
    ) -> "SparkDataFrame":
        """Union all buckets into one Spark DataFrame.

        Works in both modes: Spark-backed holders return their cached
        frame directly (no collect), Arrow-IPC holders go through
        :meth:`TabularIO.read_spark_frame` which materializes on the
        driver and lifts to Spark. Buckets are unioned with
        ``allowMissingColumns=False`` because every bucket carries the
        same :data:`RESPONSE_SCHEMA` — a missing column would mean a
        real schema drift, and silent column-fill is worse than a loud
        failure.
        """
        target_spark = spark or self.spark
        if target_spark is None:
            raise RuntimeError(
                "to_dataframe() needs a SparkSession. Pass `spark=` or build "
                "the batch with `spark=...` so we can synthesize an empty "
                "DataFrame and lift any Arrow-IPC holders."
            )

        frames: list["SparkDataFrame"] = []
        for holder in self._holders():
            if isinstance(holder, MemorySparkIO) and holder.frame is not None:
                frames.append(holder.frame)
            else:
                frames.append(holder.read_spark_frame())

        if not frames:
            return target_spark.createDataFrame(
                [], schema=RESPONSE_SCHEMA.to_spark_schema(),
            )
        result = frames[0]
        for part in frames[1:]:
            result = result.unionByName(part, allowMissingColumns=False)
        return result
