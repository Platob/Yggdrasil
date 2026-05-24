"""Origin-tagged carrier for batched ``HTTPSession.send_many`` results.

`send_many` historically returned an `Iterator[Response]` and the Spark
variant returned a single fused DataFrame. Both lose the *origin* of every
row — the caller can't tell whether a response came from the local
on-disk cache, the remote SQL cache, or a fresh network fetch.

:class:`HTTPResponseBatch` keeps that split visible. Three buckets, each
an ``Optional[Tabular]``: ``local_hits`` (on-disk cache), ``remote_hits``
(remote SQL cache), ``new_hits`` (network). ``None`` means "nothing
landed in this bucket"; a non-None holder is the same :class:`Tabular`
contract for Python (:class:`ArrowTabular`) and Spark
(:class:`Dataset`) modes, so iteration and counts share one read path.
Iteration rebuilds :class:`Response` objects via
:meth:`Response.from_records`; Spark-mode iteration is rejected up
front (it would force a driver-side collect — use
:meth:`to_dataframe` instead).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator, Optional, Union

from yggdrasil.io.tabular import ArrowTabular, Dataset
from yggdrasil.io.tabular.base import Tabular
from yggdrasil.io.response import RESPONSE_ARROW_SCHEMA, RESPONSE_SCHEMA, Response

if TYPE_CHECKING:
    from pyspark.sql import DataFrame as SparkDataFrame, SparkSession


__all__ = [
    "HTTPResponseBatch",
    "BucketInput",
    "responses_to_tabular",
    "spark_to_tabular",
]


# What the public constructor accepts per bucket. ``None`` / ``[]`` mean
# "no rows landed here" and are stored as ``None``; lists go through
# :func:`responses_to_tabular`, Spark DataFrames through
# :func:`spark_to_tabular`, and any ``Tabular`` passes through.
BucketInput = Union[
    list[Response],
    "Tabular",
    "SparkDataFrame",
    None,
]


# ---------------------------------------------------------------------------
# Coercion helpers — one shape in, one Optional[Tabular] out
# ---------------------------------------------------------------------------

def _union_holders(
    mine: Optional[Tabular],
    theirs: Optional[Tabular],
) -> Optional[Tabular]:
    if theirs is None:
        return mine
    if mine is None:
        return theirs
    return mine.union(theirs)


def responses_to_tabular(responses: list[Response]) -> ArrowTabular:
    """Wrap a non-empty list of :class:`Response` in an :class:`ArrowTabular`.

    The whole list lands in one
    :meth:`Response.values_to_arrow_batch` C++ pass instead of one
    per-row build — at 64 rows that's ~30x cheaper. Reads come back
    through :meth:`Response.from_arrow_tabular`.
    """
    return ArrowTabular(
        [Response.values_to_arrow_batch(responses)],
        schema=RESPONSE_ARROW_SCHEMA,
    )


def spark_to_tabular(df: "SparkDataFrame") -> Dataset:
    """Wrap a Spark DataFrame in a :class:`Dataset` (no collect).

    The DataFrame lives on the holder's mutable ``frame`` slot, so
    :meth:`Tabular.read_spark_frame` returns it untouched. A
    subsequent :meth:`Tabular.read_arrow_batches` would force
    ``df.toArrow()`` — fine for small frames, but Spark-mode iteration
    on the :class:`HTTPResponseBatch` is disallowed precisely so callers
    don't trip over that collect by accident.
    """
    return Dataset(df)


def _coerce_bucket(value: BucketInput) -> Optional[Tabular]:
    """Funnel an accepted bucket input down to ``Optional[Tabular]``.

    Strict on meaning: an unknown shape raises rather than being
    silently treated as Spark. Empty inputs (``None`` / ``[]``)
    collapse to ``None`` — there's no schema-bearing placeholder, the
    bucket is simply absent.
    """
    if value is None:
        return None
    if isinstance(value, Tabular):
        return value
    if isinstance(value, list):
        if not value:
            return None
        return responses_to_tabular(value)
    # Heuristic: anything that exposes a Spark-DataFrame-shaped surface
    # (an `unionByName` method) is treated as a Spark DataFrame. We
    # avoid `import pyspark` here so the optional dependency stays
    # optional.
    if hasattr(value, "unionByName") and hasattr(value, "toArrow"):
        return spark_to_tabular(value)
    raise TypeError(
        f"Unsupported bucket input: {type(value).__name__}. "
        "Expected list[Response], Tabular, Spark DataFrame, or None."
    )


# ---------------------------------------------------------------------------
# HTTPResponseBatch
# ---------------------------------------------------------------------------


class HTTPResponseBatch:
    """Origin-tagged view of a batch of responses.

    Three optional buckets, in pipeline order:

    - ``local_hits``  — served from the on-disk pickle cache.
    - ``remote_hits`` — served from the remote SQL cache.
    - ``new_hits``    — fetched from the network this run.

    Each is an ``Optional[Tabular]`` — ``None`` when nothing landed
    in that bucket, otherwise a holder that speaks the same contract
    in Python (:class:`ArrowTabular`) and Spark (:class:`Dataset`)
    modes. Constructor inputs go through coercion so callers can hand
    in a ``list[Response]``, a Spark DataFrame, an existing
    :class:`Tabular`, or ``None``.

    Iteration rebuilds :class:`Response` objects from each non-empty
    holder's records, so the Python and Spark paths share one read
    contract. Spark-mode iteration is rejected up front (it would
    force a driver-side collect); use :meth:`to_dataframe` instead.
    """

    __slots__ = (
        "_local",
        "_remote",
        "_new",
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
        self._local: Optional[Tabular] = _coerce_bucket(local_hits)
        self._remote: Optional[Tabular] = _coerce_bucket(remote_hits)
        self._new: Optional[Tabular] = _coerce_bucket(new_hits)

    def __repr__(self) -> str:
        return (
            f"HTTPResponseBatch(local_hits={self._local!r}, "
            f"remote_hits={self._remote!r}, "
            f"new_hits={self._new!r})"
        )

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def local_hits(self) -> Optional[Tabular]:
        return self._local

    @local_hits.setter
    def local_hits(self, value: BucketInput) -> None:
        self._local = _coerce_bucket(value)

    @property
    def remote_hits(self) -> Optional[Tabular]:
        return self._remote

    @remote_hits.setter
    def remote_hits(self, value: BucketInput) -> None:
        self._remote = _coerce_bucket(value)

    @property
    def new_hits(self) -> Optional[Tabular]:
        return self._new

    @new_hits.setter
    def new_hits(self, value: BucketInput) -> None:
        self._new = _coerce_bucket(value)

    def local_responses(self) -> Iterator[Response]:
        """Yield every local response, in stored order."""
        if self._local is None:
            return
        yield from Response.from_records(self._local.read_records())

    def remote_responses(self) -> Iterator[Response]:
        """Yield every remote response, in stored order."""
        if self._remote is None:
            return
        yield from Response.from_records(self._remote.read_records())

    def new_responses(self) -> Iterator[Response]:
        """Yield every new response, in stored order."""
        if self._new is None:
            return
        yield from Response.from_records(self._new.read_records())

    # ------------------------------------------------------------------
    # Shape helpers
    # ------------------------------------------------------------------

    def _holders(self) -> list[Tabular]:
        return [h for h in (self._local, self._remote, self._new) if h is not None]

    @staticmethod
    def _is_spark_holder(holder: Tabular) -> bool:
        return isinstance(holder, Dataset) and holder.frame is not None

    @staticmethod
    def _holder_count(holder: Optional[Tabular]) -> int:
        if holder is None:
            return 0
        if isinstance(holder, Dataset):
            return holder.frame.count() if holder.frame is not None else 0
        if isinstance(holder, ArrowTabular):
            return holder.num_rows
        return holder.read_arrow_table().num_rows

    @property
    def is_spark(self) -> bool:
        """True if any holder carries a Spark DataFrame."""
        return any(self._is_spark_holder(h) for h in self._holders())

    def parts(self) -> list[Tabular]:
        """Non-empty bucket holders in pipeline order: local, remote, new."""
        return self._holders()

    @property
    def counts(self) -> dict[str, int]:
        """Per-origin row counts.

        For :class:`Dataset` this triggers ``df.count()``; for
        :class:`ArrowTabular` it sums ``num_rows`` across the
        in-memory batches — fine for debugging or small assertions,
        not for hot paths. Missing buckets count as ``0``.
        """
        return {
            "local": self._holder_count(self._local),
            "remote": self._holder_count(self._remote),
            "new": self._holder_count(self._new),
        }

    # ------------------------------------------------------------------
    # Python-mode iteration — rebuild Response objects from Arrow batches
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[Response]:
        return self.iter_responses()

    def __len__(self) -> int:
        if self.is_spark:
            raise TypeError(
                "len() is not defined for a Spark HTTPResponseBatch — use "
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
            if isinstance(holder, Dataset):
                if holder.frame is not None:
                    return True
            elif isinstance(holder, ArrowTabular):
                if holder.num_rows > 0:
                    return True
            else:
                if holder.read_arrow_table().num_rows > 0:
                    return True
        return False

    def responses(self) -> list[Response]:
        """Flat list of every response, local → remote → new.

        Python mode only — raises if any holder is Spark-backed.
        """
        return list(self.iter_responses())

    def iter_responses(self) -> Iterator[Response]:
        for holder in self._holders():
            yield from Response.from_records(holder.read_records())

    # ------------------------------------------------------------------
    # Mutation / merge
    # ------------------------------------------------------------------

    def extend(self, other: "HTTPResponseBatch") -> "HTTPResponseBatch":
        """Merge another batch in place via :meth:`Tabular.union`.

        ``None`` on either side is treated as empty: the other side
        wins. Engine mixing (Arrow + Spark) is handled transparently
        by :meth:`Tabular.union`.
        """
        self._local = _union_holders(self._local, other._local)
        self._remote = _union_holders(self._remote, other._remote)
        self._new = _union_holders(self._new, other._new)
        return self

    # ------------------------------------------------------------------
    # Spark interop
    # ------------------------------------------------------------------

    def to_tabular(
        self,
        spark: Optional["SparkSession"] = None,
    ) -> Tabular:
        """Concat every non-empty bucket into one :class:`Tabular`.

        Engine-agnostic counterpart to :meth:`to_dataframe`: returns a
        :class:`Dataset` wrapping the unioned Spark frame when any
        holder is Spark-backed (or *spark* / ``self.spark`` is set),
        otherwise an :class:`ArrowTabular` carrying every Arrow batch
        across local / remote / new buckets. Empty batch in Python mode
        returns a schema-bearing empty :class:`ArrowTabular`.
        """
        target_spark = spark or self.spark
        if self.is_spark or target_spark is not None:
            return spark_to_tabular(self.to_dataframe(target_spark))
        holders = self._holders()
        if not holders:
            return ArrowTabular(
                RESPONSE_ARROW_SCHEMA.empty_table(),
                schema=RESPONSE_ARROW_SCHEMA,
            )
        result = holders[0]
        for h in holders[1:]:
            result = result.union(h)
        return result

    def to_dataframe(
        self,
        spark: Optional["SparkSession"] = None,
    ) -> "SparkDataFrame":
        """Union all non-empty buckets into one Spark DataFrame.

        Works in both modes: Spark-backed holders return their cached
        frame directly (no collect), Arrow-IPC holders go through
        :meth:`Tabular.read_spark_frame` which materializes on the
        driver and lifts to Spark. Buckets are unioned in pipeline
        order — local, then remote, then new — with
        ``allowMissingColumns=True`` because every bucket carries the
        same :data:`RESPONSE_SCHEMA`. A missing column would mean a
        real schema drift, and silent column-fill is worse than a
        loud failure. An all-empty batch returns an empty DataFrame
        with :data:`RESPONSE_SCHEMA`.
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
            if isinstance(holder, Dataset) and holder.frame is not None:
                frames.append(holder.frame)
            else:
                frames.append(holder.read_spark_frame())

        if not frames:
            return target_spark.createDataFrame(
                [], schema=RESPONSE_SCHEMA.to_spark_schema(),
            )
        result = frames[0]
        for part in frames[1:]:
            if "__rn" in part.columns:
                part = part.drop("__rn")
            result = result.unionByName(part, allowMissingColumns=True)
        return result
