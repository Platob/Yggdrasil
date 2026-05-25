"""Carrier for batched ``HTTPSession.send_many`` results.

:class:`HTTPResponseBatch` holds three optional :class:`Tabular` buckets
(local cache hits, remote cache hits, network fetches) and exposes
iteration, counts, and union across them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator, Optional

from yggdrasil.io.tabular import ArrowTabular, Dataset
from yggdrasil.io.tabular.base import Tabular
from yggdrasil.io.response import RESPONSE_ARROW_SCHEMA, RESPONSE_SCHEMA, Response

if TYPE_CHECKING:
    from pyspark.sql import DataFrame as SparkDataFrame, SparkSession


__all__ = [
    "HTTPResponseBatch",
    "responses_to_tabular",
]


def responses_to_tabular(responses: list[Response]) -> ArrowTabular:
    """Wrap a non-empty list of :class:`Response` in an :class:`ArrowTabular`."""
    return ArrowTabular(
        [Response.values_to_arrow_batch(responses)],
        schema=RESPONSE_ARROW_SCHEMA,
    )


def _union(a: Optional[Tabular], b: Optional[Tabular]) -> Optional[Tabular]:
    if b is None:
        return a
    if a is None:
        return b
    return a.union(b)


class HTTPResponseBatch(Tabular):
    """Origin-tagged view of a batch of responses.

    Three optional :class:`Tabular` buckets:

    - ``local``  — served from the local cache.
    - ``remote`` — served from the remote cache.
    - ``new``    — fetched from the network.

    Also a :class:`Tabular` itself — ``read_arrow_batches`` chains
    all buckets, ``read_spark_frame`` unions Spark frames directly.
    """

    def __init__(
        self,
        local: "Tabular | None" = None,
        remote: "Tabular | None" = None,
        new: "Tabular | list[Response] | SparkDataFrame | None" = None,
        *,
        misses: "list | None" = None,
        failed: "list | None" = None,
    ) -> None:
        super().__init__()
        self.local: Optional[Tabular] = local
        self.remote: Optional[Tabular] = remote
        if isinstance(new, list):
            self.new: Optional[Tabular] = responses_to_tabular(new) if new else None
        elif new is not None and not isinstance(new, Tabular):
            self.new = Dataset(new)
        else:
            self.new = new
        self.misses: list = misses or []
        self.failed: list = failed or []

    def __repr__(self) -> str:
        return (
            f"HTTPResponseBatch(local={self.local!r}, "
            f"remote={self.remote!r}, new={self.new!r})"
        )

    # ------------------------------------------------------------------
    # Holders
    # ------------------------------------------------------------------

    def _holders(self) -> list[Tabular]:
        return [h for h in (self.local, self.remote, self.new) if h is not None]

    @property
    def is_spark(self) -> bool:
        return any(isinstance(h, Dataset) for h in self._holders())

    # ------------------------------------------------------------------
    # Tabular implementation
    # ------------------------------------------------------------------

    def _collect_schema(self, options=None):
        return RESPONSE_SCHEMA

    def _read_arrow_batches(self, options=None):
        for holder in self._holders():
            yield from holder.read_arrow_batches(options=options)

    def _write_arrow_batches(self, batches, options=None):
        raise NotImplementedError("HTTPResponseBatch is read-only")

    def _read_spark_frame(self, options=None):
        from yggdrasil.environ import PyEnv
        spark = getattr(options, "spark_session", None) or PyEnv.spark_session(create=True)
        result = None
        for holder in self._holders():
            if isinstance(holder, Dataset) and holder.frame is not None:
                df = holder.frame
            else:
                df = spark.createDataFrame(holder.read_arrow_table())
            result = df if result is None else result.unionByName(
                df, allowMissingColumns=True,
            )
        if result is None:
            return spark.createDataFrame(
                [], schema=RESPONSE_SCHEMA.to_spark_schema(),
            )
        return result

    # ------------------------------------------------------------------
    # Counts
    # ------------------------------------------------------------------

    @staticmethod
    def _count(holder: Optional[Tabular]) -> int:
        if holder is None:
            return 0
        if isinstance(holder, ArrowTabular):
            return holder.num_rows
        if isinstance(holder, Dataset):
            return holder.frame.count() if holder.frame is not None else 0
        return sum(b.num_rows for b in holder.read_arrow_batches())

    @property
    def counts(self) -> dict[str, int]:
        return {
            "local": self._count(self.local),
            "remote": self._count(self.remote),
            "new": self._count(self.new),
        }

    def __len__(self) -> int:
        return sum(self.counts.values())

    def __bool__(self) -> bool:
        return bool(self._holders())

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[Response]:
        return self.iter_responses()

    def iter_responses(self) -> Iterator[Response]:
        for label, holder in (
            ("local", self.local),
            ("remote", self.remote),
            ("new", self.new),
        ):
            if holder is None:
                continue
            yield from Response.from_records(holder.read_records())

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    def extend(self, other: "HTTPResponseBatch") -> "HTTPResponseBatch":
        self.local = _union(self.local, other.local)
        self.remote = _union(self.remote, other.remote)
        self.new = _union(self.new, other.new)
        self.misses.extend(other.misses)
        self.failed.extend(other.failed)
        return self
