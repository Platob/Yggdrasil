"""Carrier for batched ``HTTPSession.send_many`` results.

:class:`HTTPResponseBatch` holds the send config and requests, lazily
resolving cache hits (local/remote) and network responses on access.
"""

from __future__ import annotations

import datetime as dt
import logging
import pickle
from typing import Any, TYPE_CHECKING, Iterator, Optional

import pyarrow as pa

from yggdrasil.arrow.cast import rechunk_arrow_batches
from yggdrasil.io.tabular import ArrowTabular, Dataset
from yggdrasil.io.tabular.base import Tabular
from yggdrasil.io.response import RESPONSE_ARROW_SCHEMA, RESPONSE_SCHEMA, Response

if TYPE_CHECKING:
    from pyspark.sql import DataFrame as SparkDataFrame, SparkSession
    from yggdrasil.io.request import PreparedRequest
    from yggdrasil.io.send_config import SendConfig

LOGGER = logging.getLogger(__name__)

_SPARK_RESPONSE_BATCH_BYTE_LIMIT: int = 128 * 1024 * 1024


__all__ = [
    "HTTPResponseBatch",
    "responses_to_tabular",
]


def _synthetic_not_found(request: "PreparedRequest") -> Response:
    return Response(
        request=request,
        status_code=404,
        headers={"Content-Type": "application/json"},
        tags={"synthetic": "cache_only_miss"},
        buffer=b'{"error": "not found in cache"}',
        received_at=dt.datetime.now(dt.timezone.utc),
    )


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

        request_map = {r.match_value("public_hash"): r for r in self._requests}

        def _filter_tab(tab, cache_cfg):
            if tab is None or cache_cfg is None:
                return tab, []
            kept: list[Response] = []
            rejected_keys: list = []
            for resp in Response.from_arrow_tabular(tab.read_arrow_batches()):
                req = request_map.get(
                    resp.match_value("public_hash") if hasattr(resp, "match_value") else None
                )
                if req is not None and cache_cfg.filter_response(resp, request=req):
                    kept.append(resp)
                else:
                    rejected_keys.append(resp)
            if not kept:
                return None, rejected_keys
            return responses_to_tabular(kept), rejected_keys

        local_tab, local_rejected = _filter_tab(local_tab, self._send_config.local_cache)
        if local_rejected:
            local_rejected_hashes = {
                r.match_value("public_hash") for r in local_rejected
                if hasattr(r, "match_value")
            }
            remaining = remaining + [
                r for r in self._requests
                if r.match_value("public_hash") in local_rejected_hashes
            ]

        remote_tab, _ = _filter_tab(remote_tab, self._send_config.remote_cache)

        self._local = local_tab
        self._remote = remote_tab
        self._misses = remaining

    def _fetch(
        self,
        *,
        ordered: bool = False,
        max_in_flight: int | None = None,
    ) -> "HTTPResponseBatch":
        """Resolve cache hits, fetch misses, and write back.

        Triggers the lazy split (local → remote → misses), fetches
        remaining misses via the session's thread pool or Spark,
        writes ok responses back to caches, and returns self.
        """
        cfg = self._send_config
        session = self._session
        misses = self.misses

        if self._remote is not None and cfg.local_cache is not None:
            from yggdrasil.data.enums import Mode
            try:
                remote_table = self._remote.read_arrow_table()
                if remote_table.num_rows > 0:
                    cfg.local_cache.write_responses_tabular(
                        remote_table, mode=Mode.OVERWRITE, session=session,
                    )
            except Exception:
                LOGGER.debug("Remote→local backfill failed", exc_info=True)

        if not misses or cfg.cache_only:
            if misses:
                self.new = [_synthetic_not_found(r) for r in misses]
            return self

        spark = cfg.get_spark_session()
        LOGGER.debug(
            "Fetching %d miss(es) via %s",
            len(misses), "spark" if spark else "thread pool",
        )

        write_data = None
        if spark is not None:
            result_df = self._spark_fetch(misses, spark)
            if cfg.raise_error:
                from pyspark.sql import functions as F
                from yggdrasil.spark.cast import spark_dataframe_to_arrow

                ok_df = result_df.where(
                    (F.col("status_code") >= 200) & (F.col("status_code") < 400)
                )
                err_df = result_df.where(
                    (F.col("status_code") < 200) | (F.col("status_code") >= 400)
                )
                err_table = spark_dataframe_to_arrow(err_df)
                if len(err_table) > 0:
                    self.failed = list(Response.from_arrow_tabular(err_table))
                write_data = ok_df
            else:
                write_data = result_df
            self.new = result_df
        else:
            ok_list: list[Response] = []
            all_list: list[Response] = []
            for response in session._fetch_misses(
                misses, ordered=ordered, max_in_flight=max_in_flight,
            ):
                all_list.append(response)
                if response.ok:
                    ok_list.append(response)
                elif cfg.raise_error:
                    self.failed.append(response)
            LOGGER.info(
                "Fetched %d/%d miss(es) (ok=%d, failed=%d)",
                len(ok_list) + len(self.failed), len(misses),
                len(ok_list), len(self.failed),
            )
            if all_list:
                self.new = pa.Table.from_batches(
                    [Response.values_to_arrow_batch(all_list)]
                )
            if ok_list:
                write_data = pa.Table.from_batches(
                    [Response.values_to_arrow_batch(ok_list)]
                )

        if write_data is not None:
            cfg.write_responses_tabular(write_data, session=session)

        return self

    def _spark_fetch(
        self,
        misses: "list[PreparedRequest]",
        spark: "SparkSession",
    ) -> "SparkDataFrame":
        """Scatter misses to Spark workers via mapInArrow."""
        from yggdrasil.io.request import PreparedRequest
        from yggdrasil.io.send_config import SendConfig

        session = self._session
        cfg = self._send_config

        if not misses:
            schema = RESPONSE_SCHEMA.to_spark_schema()
            return spark.createDataFrame([], schema=schema)

        for req in misses:
            session.prepare_request_before_send(req)

        request_table = pa.Table.from_batches(
            [PreparedRequest.values_to_arrow_batch(misses)]
        )

        try:
            default_par = max(spark.sparkContext.defaultParallelism, 1)
        except Exception:
            default_par = 8
        n_parts = max(1, min(len(misses), default_par * 8))
        LOGGER.info(
            "Scattering %d miss(es) across %d Spark partition(s)",
            len(misses), n_parts,
        )
        request_df = spark.createDataFrame(request_table).repartition(n_parts)

        worker_config = cfg.copy(
            remote_cache=None,
            spark_session=False,
            raise_error=False,
        )
        lc = worker_config.local_cache
        if lc is not None:
            now = dt.datetime.now(dt.timezone.utc)
            worker_config = worker_config.copy(
                local_cache=lc.copy(
                    received_to=now,
                    received_from=now - dt.timedelta(minutes=15),
                ),
            )

        session_bytes = pickle.dumps(session)
        response_spark_schema = RESPONSE_SCHEMA.to_spark_schema()

        try:
            bc_session = spark.sparkContext.broadcast(session_bytes)
            bc_config = spark.sparkContext.broadcast(worker_config)
            use_broadcast = True
        except Exception:
            use_broadcast = False

        def _send_partition(
            batches: Iterator[pa.RecordBatch],
        ) -> Iterator[pa.RecordBatch]:
            if use_broadcast:
                sess = pickle.loads(bc_session.value)
                part_config = bc_config.value
            else:
                sess = pickle.loads(session_bytes)
                part_config = worker_config
            for batch in batches:
                reqs = list(PreparedRequest.from_arrow(batch))
                if not reqs:
                    continue

                def _row_batches() -> Iterator[pa.RecordBatch]:
                    for resp in sess.send_many(iter(reqs), part_config):
                        yield resp.to_arrow_batch(parse=False)

                yield from rechunk_arrow_batches(
                    _row_batches(),
                    byte_size=_SPARK_RESPONSE_BATCH_BYTE_LIMIT,
                )

        result_df = request_df.mapInArrow(
            _send_partition, schema=response_spark_schema,
        )
        try:
            result_df = result_df.cache()
        except Exception:
            LOGGER.warning(
                "Failed to cache mapInArrow result", exc_info=True,
            )
        return result_df

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
