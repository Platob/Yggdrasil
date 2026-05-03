"""Origin-tagged carrier for batched ``Session.send_many`` results.

`send_many` historically returned an `Iterator[Response]` and the Spark
variant returned a single fused DataFrame. Both lose the *origin* of every
row — the caller can't tell whether a response came from the local
on-disk cache, the remote SQL cache, or a fresh network fetch.

:class:`ResponseBatch` keeps that split visible. The three buckets are
all stored as the same type — :class:`TabularIO` — so Python
(:class:`MemoryArrowIO`) and Spark (:class:`MemorySparkIO`) share one
contract. Iteration walks each holder's Arrow batches and rebuilds
:class:`Response` objects via :meth:`Response.from_arrow_tabular`, so
the batch never has to special-case which engine produced the rows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator, Optional, Union

from .buffer.base import TabularIO
from .buffer.memory import MemoryArrowIO, MemorySparkIO
from .response import RESPONSE_SCHEMA, Response

if TYPE_CHECKING:
    from pyspark.sql import DataFrame as SparkDataFrame, SparkSession


__all__ = [
    "ResponseBatch",
    "BucketInput",
    "responses_to_tabular",
    "spark_to_tabular",
]


# What the public constructor accepts per bucket. ``None`` means "stage
# skipped"; a :class:`TabularIO` is taken as-is; everything else is
# coerced through :func:`responses_to_tabular` /
# :func:`spark_to_tabular` so callers don't have to know which helper to
# pick.
BucketInput = Union[
    list[Response],
    "TabularIO",
    "SparkDataFrame",
    None,
]


# ---------------------------------------------------------------------------
# Coercion helpers — one shape in, one TabularIO out
# ---------------------------------------------------------------------------

def responses_to_tabular(responses: list[Response]) -> Optional[MemoryArrowIO]:
    """Wrap a list of :class:`Response` in a :class:`MemoryArrowIO`.

    Returns ``None`` for an empty list so callers can keep the
    "skipped vs empty" distinction. Each response is serialized via
    ``to_arrow_batch(parse=False)`` (one row per batch) and held in
    memory — no IPC bytes, no spill. Reads come back through
    :meth:`Response.from_arrow_tabular`.
    """
    if not responses:
        return None
    return MemoryArrowIO(r.to_arrow_batch(parse=False) for r in responses)


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


def _coerce_bucket(value: BucketInput) -> Optional[TabularIO]:
    """Funnel any accepted bucket input down to ``Optional[TabularIO]``.

    Strict on meaning: an unknown shape raises rather than silently
    being treated as Spark. Lists land in :func:`responses_to_tabular`,
    Spark DataFrames in :func:`spark_to_tabular`, ``TabularIO`` passes
    through, ``None`` means "stage skipped".
    """
    if value is None:
        return None
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
    DataFrame, an existing ``TabularIO``, or ``None`` for "stage
    skipped".

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
        self._local_response: Optional[TabularIO] = _coerce_bucket(local_hits)
        self._remote_response: Optional[TabularIO] = _coerce_bucket(remote_hits)
        self._new_response: Optional[TabularIO] = _coerce_bucket(new_hits)
        self.spark: Optional["SparkSession"] = spark

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
    def local_hits(self) -> Optional[TabularIO]:
        return self._local_response

    @local_hits.setter
    def local_hits(self, value: BucketInput) -> None:
        self._local_response = _coerce_bucket(value)

    @property
    def remote_hits(self) -> Optional[TabularIO]:
        return self._remote_response

    @remote_hits.setter
    def remote_hits(self, value: BucketInput) -> None:
        self._remote_response = _coerce_bucket(value)

    @property
    def new_hits(self) -> Optional[TabularIO]:
        return self._new_response

    @new_hits.setter
    def new_hits(self, value: BucketInput) -> None:
        self._new_response = _coerce_bucket(value)

    # ------------------------------------------------------------------
    # Shape helpers
    # ------------------------------------------------------------------

    def _holders(self) -> tuple[Optional[TabularIO], Optional[TabularIO], Optional[TabularIO]]:
        return (self._local_response, self._remote_response, self._new_response)

    @staticmethod
    def _is_spark_holder(holder: Optional[TabularIO]) -> bool:
        return isinstance(holder, MemorySparkIO) and holder.frame is not None

    @property
    def is_spark(self) -> bool:
        """True if any holder carries a Spark DataFrame."""
        return any(self._is_spark_holder(h) for h in self._holders())

    def parts(self) -> list[TabularIO]:
        """Non-None holders in pipeline order."""
        return [h for h in self._holders() if h is not None]

    @property
    def counts(self) -> dict[str, int]:
        """Per-origin row counts.

        Skipped holders count as ``0``. For :class:`MemorySparkIO`
        this triggers ``df.count()``; for :class:`MemoryArrowIO` it
        sums ``num_rows`` across the in-memory batches — fine for
        debugging or small assertions, not for hot paths.
        """
        out: dict[str, int] = {}
        for name, h in zip(("local", "remote", "new"), self._holders()):
            if h is None:
                out[name] = 0
            elif isinstance(h, MemorySparkIO):
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
            if holder is None:
                continue
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
            if holder is None:
                continue
            total += holder.read_arrow_table().num_rows
        return total

    def __bool__(self) -> bool:
        return bool(self.parts())

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

        Python-mode only — mixing Arrow-backed and Spark-backed
        holders would force a Spark conversion at merge time, which is
        too sneaky to do silently. Convert one side first if you need
        that.

        Per-bucket merges append the other side's batches into our
        :class:`MemoryArrowIO` holder — no IPC rewrite, just a list
        extend on the held batches.
        """
        if self.is_spark or other.is_spark:
            raise TypeError(
                "extend() only supports Python-mode batches. Lift one side "
                "with `to_dataframe()` and union the DataFrames yourself if "
                "you need to merge across modes."
            )

        from .enums import Mode

        for name in ("_local_response", "_remote_response", "_new_response"):
            mine = getattr(self, name)
            theirs = getattr(other, name)
            if theirs is None:
                continue
            if mine is None:
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
        """Union all non-empty buckets into one Spark DataFrame.

        Works in both modes: Spark-backed holders return their cached
        ``_spark_frame`` directly (no collect), Arrow-IPC holders go
        through :meth:`TabularIO.read_spark_frame` which materializes
        on the driver and lifts to Spark. Buckets are unioned with
        ``allowMissingColumns=False`` because every bucket carries the
        same ``RESPONSE_SCHEMA`` — a missing column would mean a real
        schema drift, and silent column-fill is worse than a loud
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
            if holder is None:
                continue
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


