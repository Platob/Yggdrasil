"""Carrier for batched ``HTTPSession.send_many`` results.

:class:`HTTPResponseBatch` holds the send config and requests, lazily
resolving cache hits (local/remote) and network responses on access.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING, Iterator, Optional

import pyarrow as pa

from yggdrasil.io.tabular import ArrowTabular, Dataset
from yggdrasil.io.tabular.base import Tabular
from yggdrasil.io.response import RESPONSE_ARROW_SCHEMA, RESPONSE_SCHEMA, Response

if TYPE_CHECKING:
    from pyspark.sql import DataFrame as SparkDataFrame, SparkSession
    from yggdrasil.io.request import PreparedRequest
    from yggdrasil.io.send_config import SendConfig


__all__ = [
    "HTTPResponseBatch",
    "responses_to_tabular",
]


def responses_to_tabular(responses: list[Response]) -> ArrowTabular:
    return ArrowTabular(
        [Response.values_to_arrow_batch(responses)],
        schema=RESPONSE_ARROW_SCHEMA,
    )


def _to_tabular(data) -> "Tabular | None":
    if data is None:
        return None
    if isinstance(data, list):
        return responses_to_tabular(data) if data else None
    if isinstance(data, pa.Table):
        return ArrowTabular(data.to_batches(), schema=data.schema) if data.num_rows > 0 else None
    if isinstance(data, pa.RecordBatch):
        return ArrowTabular([data], schema=data.schema) if data.num_rows > 0 else None
    if isinstance(data, Tabular):
        return data
    if hasattr(data, "toArrow") or hasattr(data, "toPandas"):
        return Dataset(data)
    return None


class HTTPResponseBatch(Tabular):
    """Lazy batch of responses with send config context.

    Holds the send config and original requests. Cache hits (local/remote)
    are resolved lazily via ``send_config.split_requests`` on first access.
    Network responses are set eagerly after fetch.
    """

    __slots__ = (
        "_send_config", "_requests", "_session", "_local", "_remote",
        "_new", "_split_done", "_misses", "failed",
    )

    def __init__(
        self,
        send_config: "SendConfig",
        requests: "list[PreparedRequest]",
        new_responses: "list[Response] | None" = None,
        new_responses_tabular: "Tabular | pa.Table | SparkDataFrame | None" = None,
        *,
        misses: "list[PreparedRequest] | None" = None,
        failed: "list[Response] | None" = None,
        session: "Any" = None,
    ) -> None:
        super().__init__()
        self._send_config = send_config
        self._requests = requests
        self._local: Optional[Tabular] = None
        self._remote: Optional[Tabular] = None
        self._split_done = False
        self._session = session

        if new_responses is not None:
            self._new: Optional[Tabular] = responses_to_tabular(new_responses)
        else:
            self._new = _to_tabular(new_responses_tabular)

        self._misses: list = misses if misses is not None else list(requests)
        self.failed: list = failed or []

    def _ensure_split(self) -> None:
        if self._split_done:
            return
        self._split_done = True
        if self._send_config is None or not self._requests:
            return
        local_tab, remote_tab, remaining = self._send_config.split_requests(
            self._requests, session=self._session,
        )
        self._local = local_tab
        self._remote = remote_tab
        self._misses = remaining

    @property
    def misses(self) -> list:
        self._ensure_split()
        return self._misses

    @misses.setter
    def misses(self, value: list) -> None:
        self._misses = value

    @property
    def local(self) -> "Tabular | None":
        self._ensure_split()
        return self._local

    @property
    def remote(self) -> "Tabular | None":
        self._ensure_split()
        return self._remote

    @property
    def new(self) -> "Tabular | None":
        return self._new

    @new.setter
    def new(self, value) -> None:
        self._new = _to_tabular(value)

    @property
    def send_config(self) -> "SendConfig":
        return self._send_config

    def __repr__(self) -> str:
        return (
            f"HTTPResponseBatch(requests={len(self._requests)}, "
            f"split={self._split_done})"
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
        for holder in self._holders():
            if holder is None:
                continue
            yield from Response.from_records(holder.read_records())

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    def extend(self, other: "HTTPResponseBatch") -> "HTTPResponseBatch":
        other._ensure_split()
        self._ensure_split()
        self._local = _union(self._local, other._local)
        self._remote = _union(self._remote, other._remote)
        self._new = _union(self._new, other._new)
        self._misses.extend(other._misses)
        self.failed.extend(other.failed)
        return self


def _union(a: Optional[Tabular], b: Optional[Tabular]) -> Optional[Tabular]:
    if b is None:
        return a
    if a is None:
        return b
    return a.union(b)
