"""Carrier for batched ``HTTPSession.send_many`` results.

:class:`HTTPResponseBatch` holds three optional :class:`Tabular` buckets
(local cache hits, remote cache hits, network fetches) and exposes
iteration, counts, and union across them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator, Optional

from yggdrasil.io.tabular import ArrowTabular, Dataset
from yggdrasil.io.tabular.base import Tabular
from yggdrasil.io.response import RESPONSE_ARROW_SCHEMA, Response

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


class HTTPResponseBatch:
    """Origin-tagged view of a batch of responses.

    Three optional :class:`Tabular` buckets:

    - ``local``  — served from the local cache.
    - ``remote`` — served from the remote cache.
    - ``new``    — fetched from the network.

    Iteration yields :class:`Response` objects with ``local_cached`` /
    ``remote_cached`` stamped per origin.
    """

    __slots__ = ("local", "remote", "new", "misses", "failed")

    def __init__(
        self,
        local: "Tabular | None" = None,
        remote: "Tabular | None" = None,
        new: "Tabular | list[Response] | SparkDataFrame | None" = None,
        *,
        misses: "list | None" = None,
        failed: "list | None" = None,
    ) -> None:
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

    def tabular(self) -> Optional[Tabular]:
        """Union all non-empty buckets into one :class:`Tabular`."""
        holders = self._holders()
        if not holders:
            return None
        result = holders[0]
        for h in holders[1:]:
            result = result.union(h)
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
            for response in Response.from_records(holder.read_records()):
                response.local_cached = (label == "local")
                response.remote_cached = (label == "remote")
                yield response

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
