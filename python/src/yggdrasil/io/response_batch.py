"""Origin-tagged carrier for batched ``Session.send_many`` results.

`send_many` historically returned an `Iterator[Response]` and the Spark
variant returned a single fused DataFrame. Both lose the *origin* of every
row — the caller can't tell whether a response came from the local
on-disk cache, the remote SQL cache, or a fresh network fetch.

:class:`ResponseBatch` keeps that split visible. The local and remote
buckets are both ``dict[str, TabularIO]`` so the per-config split
survives all the way to the consumer: local hits keyed by local-cache
folder path, remote hits keyed by remote-cache table full name. The
new-hits bucket stays single-holder — network fetches don't carry a
meaningful per-config split before they're persisted. All holders are
the same type — Python (:class:`MemoryArrowIO`) and Spark
(:class:`MemorySparkIO`) share one contract — and empty buckets keep
their :data:`RESPONSE_SCHEMA` so a batch with no responses still answers
schema questions correctly. Iteration walks each holder's Arrow batches
and rebuilds :class:`Response` objects via
:meth:`Response.from_arrow_tabular`, so the batch never has to
special-case which engine produced the rows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator, Mapping, Optional, Union

from .buffer.base import TabularIO
from .buffer.memory import MemoryArrowIO, MemorySparkIO
from .response import RESPONSE_ARROW_SCHEMA, RESPONSE_SCHEMA, Response

if TYPE_CHECKING:
    from pyspark.sql import DataFrame as SparkDataFrame, SparkSession


__all__ = [
    "ResponseBatch",
    "BucketInput",
    "KeyedBucketInput",
    "responses_to_tabular",
    "spark_to_tabular",
    "empty_arrow_holder",
    "empty_spark_holder",
    "DEFAULT_BUCKET_KEY",
    "DEFAULT_REMOTE_TABLE_KEY",
    "DEFAULT_LOCAL_PATH_KEY",
]


# Sentinel key used when a per-config bucket carries a holder that
# wasn't tagged with a specific cache-table name or local-cache folder
# — either because the caller passed a bare ``BucketInput`` instead of
# a per-key mapping, or because the batch is empty and we still want a
# schema-bearing default entry.
DEFAULT_BUCKET_KEY = ""

# Back-compat aliases — both buckets share one default sentinel but the
# named constants document which key to expect when introspecting a
# specific bucket.
DEFAULT_REMOTE_TABLE_KEY = DEFAULT_BUCKET_KEY
DEFAULT_LOCAL_PATH_KEY = DEFAULT_BUCKET_KEY


# What the public constructor accepts per simple bucket. ``None`` is
# coerced to a schema-bearing empty holder so every bucket carries
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


# What the public constructor accepts for the ``local_hits`` and
# ``remote_hits`` buckets. In addition to the simple shapes, callers
# can pass a mapping keyed by cache identity (local-cache folder path
# or remote-cache table full name) so the per-config split survives
# all the way out to the consumer.
KeyedBucketInput = Union[
    BucketInput,
    Mapping[str, BucketInput],
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
    """Funnel any accepted simple bucket input down to a ``TabularIO``.

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


def _coerce_keyed_bucket(
    value: KeyedBucketInput,
    *,
    spark: Optional["SparkSession"] = None,
) -> dict[str, TabularIO]:
    """Funnel any accepted per-key bucket input down to a keyed dict.

    Always returns at least one entry so the bucket can answer schema
    questions even when nothing matched — an empty mapping or ``None``
    becomes ``{DEFAULT_BUCKET_KEY: empty_holder}``. Mapping keys are
    stringified so cache-table full names and ``Path``-derived
    local-cache folders both produce stable, comparable string keys.
    A bare :class:`BucketInput` is treated as a single untagged bucket
    and stored under :data:`DEFAULT_BUCKET_KEY`.
    """
    if isinstance(value, Mapping):
        if not value:
            return {DEFAULT_BUCKET_KEY: _coerce_bucket(None, spark=spark)}
        return {
            str(k): _coerce_bucket(v, spark=spark)
            for k, v in value.items()
        }
    return {DEFAULT_BUCKET_KEY: _coerce_bucket(value, spark=spark)}


# ---------------------------------------------------------------------------
# ResponseBatch
# ---------------------------------------------------------------------------


class ResponseBatch:
    """Origin-tagged view of a batch of responses.

    Three buckets, in pipeline order:

    - ``local_hits``  — served from the on-disk pickle cache, split by
      local-cache folder path (``dict[str, TabularIO]``) so callers can
      see which configured cache root answered which subset.
    - ``remote_hits`` — served from the remote SQL cache, split by
      cache-table full name (``dict[str, TabularIO]``) so callers can
      see which configured table answered which subset of the batch.
    - ``new_hits``    — fetched from the network this run.

    The local and remote buckets live on insertion-ordered ``dict``
    holders (``_local_responses`` / ``_remote_responses``) so the
    per-config split survives. The new bucket stays a single private
    :class:`TabularIO` (``_new_response``) — fresh network fetches
    haven't been bucketed by destination at this stage of the
    pipeline. Constructor inputs go through coercion so callers can
    hand in a ``list[Response]``, a Spark DataFrame, an existing
    ``TabularIO``, ``None``, or — for ``local_hits`` / ``remote_hits``
    — a mapping from key (path or table name) to any of those.
    ``None`` and empty inputs become schema-bearing empty holders so
    every bucket advertises :data:`RESPONSE_SCHEMA`.

    Iteration rebuilds :class:`Response` objects from each holder's
    Arrow batches via :meth:`Response.from_arrow_tabular`, so the
    Python and Spark paths share one read contract. Spark-mode
    iteration is rejected up front (it would force a driver-side
    collect); use :meth:`to_dataframe` instead.
    """

    __slots__ = (
        "_local_responses",
        "_remote_responses",
        "_new_response",
        "spark",
    )

    def __init__(
        self,
        local_hits: KeyedBucketInput = None,
        remote_hits: KeyedBucketInput = None,
        new_hits: BucketInput = None,
        *,
        spark: Optional["SparkSession"] = None,
    ) -> None:
        self.spark: Optional["SparkSession"] = spark
        self._local_responses: dict[str, TabularIO] = _coerce_keyed_bucket(
            local_hits, spark=spark,
        )
        self._remote_responses: dict[str, TabularIO] = _coerce_keyed_bucket(
            remote_hits, spark=spark,
        )
        self._new_response: TabularIO = _coerce_bucket(new_hits, spark=spark)

    def __repr__(self) -> str:
        return (
            f"ResponseBatch(local_hits={self._local_responses!r}, "
            f"remote_hits={self._remote_responses!r}, "
            f"new_hits={self._new_response!r})"
        )

    # ------------------------------------------------------------------
    # Public accessors over the private holders
    # ------------------------------------------------------------------

    @property
    def local_hits(self) -> dict[str, TabularIO]:
        """Per-path local-cache holders, keyed by local-cache folder.

        Always returns at least one entry — :data:`DEFAULT_LOCAL_PATH_KEY`
        with a schema-bearing empty holder when the batch carries no
        local hits — so callers can introspect schema without checking
        for an empty dict.
        """
        return self._local_responses

    @local_hits.setter
    def local_hits(self, value: KeyedBucketInput) -> None:
        self._local_responses = _coerce_keyed_bucket(value, spark=self.spark)

    @property
    def local_paths(self) -> list[str]:
        """Local-cache folders that contributed to this batch.

        Drops the :data:`DEFAULT_LOCAL_PATH_KEY` placeholder — that
        key is bookkeeping for schema-bearing empties, not a real
        cache root.
        """
        return [k for k in self._local_responses if k != DEFAULT_LOCAL_PATH_KEY]

    @property
    def remote_hits(self) -> dict[str, TabularIO]:
        """Per-table remote-cache holders, keyed by cache-table full name.

        Always returns at least one entry — :data:`DEFAULT_REMOTE_TABLE_KEY`
        with a schema-bearing empty holder when the batch carries no
        remote hits — so callers can introspect schema without checking
        for an empty dict.
        """
        return self._remote_responses

    @remote_hits.setter
    def remote_hits(self, value: KeyedBucketInput) -> None:
        self._remote_responses = _coerce_keyed_bucket(value, spark=self.spark)

    @property
    def remote_tables(self) -> list[str]:
        """Names of remote-cache tables that contributed to this batch.

        Drops the :data:`DEFAULT_REMOTE_TABLE_KEY` placeholder — that
        key is bookkeeping for schema-bearing empties, not a real
        cache table.
        """
        return [k for k in self._remote_responses if k != DEFAULT_REMOTE_TABLE_KEY]

    @property
    def new_hits(self) -> TabularIO:
        return self._new_response

    @new_hits.setter
    def new_hits(self, value: BucketInput) -> None:
        self._new_response = _coerce_bucket(value, spark=self.spark)

    # ------------------------------------------------------------------
    # Shape helpers
    # ------------------------------------------------------------------

    def _holders(self) -> list[TabularIO]:
        return [
            *self._local_responses.values(),
            *self._remote_responses.values(),
            self._new_response,
        ]

    @staticmethod
    def _is_spark_holder(holder: TabularIO) -> bool:
        return isinstance(holder, MemorySparkIO) and holder.frame is not None

    @staticmethod
    def _holder_count(holder: TabularIO) -> int:
        if isinstance(holder, MemorySparkIO):
            return holder.frame.count() if holder.frame is not None else 0
        if isinstance(holder, MemoryArrowIO):
            return holder.num_rows
        return holder.read_arrow_table().num_rows

    @property
    def is_spark(self) -> bool:
        """True if any holder carries a Spark DataFrame."""
        return any(self._is_spark_holder(h) for h in self._holders())

    def parts(self) -> list[TabularIO]:
        """All bucket holders in pipeline order: locals…, remotes…, new.

        Every bucket is schema-bearing, including empty ones. Per-key
        holders appear in the order they were registered on this
        batch (or merged in via :meth:`extend`), so consumers that
        iterate ``parts()`` see a stable per-key ordering.
        """
        return self._holders()

    @property
    def counts(self) -> dict[str, int]:
        """Per-origin row counts.

        ``local`` and ``remote`` are totals summed across every
        contributing key — use :attr:`local_counts` and
        :attr:`remote_counts` for the per-key breakdowns. For
        :class:`MemorySparkIO` this triggers ``df.count()``; for
        :class:`MemoryArrowIO` it sums ``num_rows`` across the
        in-memory batches — fine for debugging or small assertions,
        not for hot paths.
        """
        return {
            "local": sum(
                self._holder_count(h) for h in self._local_responses.values()
            ),
            "remote": sum(
                self._holder_count(h) for h in self._remote_responses.values()
            ),
            "new": self._holder_count(self._new_response),
        }

    @property
    def local_counts(self) -> dict[str, int]:
        """Row counts per local-cache folder, in registration order.

        Includes the :data:`DEFAULT_LOCAL_PATH_KEY` entry only when
        it actually holds rows — the placeholder empty default is
        elided so the breakdown reflects real cache roots.
        """
        return self._keyed_counts(self._local_responses, DEFAULT_LOCAL_PATH_KEY)

    @property
    def remote_counts(self) -> dict[str, int]:
        """Row counts per remote-cache table, in registration order.

        Includes the :data:`DEFAULT_REMOTE_TABLE_KEY` entry only when
        it actually holds rows — the placeholder empty default is
        elided so the breakdown reflects real cache tables.
        """
        return self._keyed_counts(self._remote_responses, DEFAULT_REMOTE_TABLE_KEY)

    def _keyed_counts(
        self,
        holders: dict[str, TabularIO],
        default_key: str,
    ) -> dict[str, int]:
        out: dict[str, int] = {}
        for key, holder in holders.items():
            n = self._holder_count(holder)
            if key == default_key and n == 0:
                continue
            out[key] = n
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

        Local and remote merges are keyed by their respective
        identities (cache-folder path / cache-table full name): rows
        sharing a key on both sides are unioned in place; keys
        present only on ``other`` are inserted at the tail of the
        per-key dict so iteration order remains deterministic. The
        placeholder :data:`DEFAULT_BUCKET_KEY` empty default is
        dropped from ``self`` as soon as ``other`` brings in any real
        key, so a merged batch never carries a stale empty bucket
        alongside real per-key data.
        """
        if self.is_spark != other.is_spark:
            raise TypeError(
                "extend() requires both batches in the same engine. "
                "Lift one side with `to_dataframe()` (or rebuild Python-side) "
                "before merging across modes."
            )

        self._local_responses = self._merge_keyed(
            self._local_responses, other._local_responses,
        )
        self._remote_responses = self._merge_keyed(
            self._remote_responses, other._remote_responses,
        )
        self._merge_simple_holder("_new_response", other._new_response)
        return self

    def _merge_simple_holder(self, attr: str, theirs: TabularIO) -> None:
        from .enums import Mode

        mine: TabularIO = getattr(self, attr)
        if isinstance(theirs, MemorySparkIO) and theirs.frame is not None:
            if isinstance(mine, MemorySparkIO) and mine.frame is not None:
                mine.frame = mine.frame.unionByName(
                    theirs.frame, allowMissingColumns=False,
                )
            else:
                setattr(self, attr, theirs)
            return
        mine.write_arrow_batches(theirs.read_arrow_batches(), mode=Mode.APPEND)

    def _merge_keyed(
        self,
        mine: dict[str, TabularIO],
        theirs: dict[str, TabularIO],
    ) -> dict[str, TabularIO]:
        from .enums import Mode

        # Drop the placeholder empty default as soon as the incoming
        # side brings in any real (non-default) key — keeps the
        # merged dict free of bookkeeping entries that would clutter
        # ``*_paths`` / ``*_tables`` and ``parts()``.
        incoming_real = [k for k in theirs if k != DEFAULT_BUCKET_KEY]
        if incoming_real and self._is_placeholder(mine):
            mine = {}

        for key, their_holder in theirs.items():
            existing = mine.get(key)
            if existing is None:
                mine[key] = their_holder
                continue
            if isinstance(their_holder, MemorySparkIO) and their_holder.frame is not None:
                if isinstance(existing, MemorySparkIO) and existing.frame is not None:
                    existing.frame = existing.frame.unionByName(
                        their_holder.frame, allowMissingColumns=False,
                    )
                else:
                    mine[key] = their_holder
                continue
            existing.write_arrow_batches(
                their_holder.read_arrow_batches(), mode=Mode.APPEND,
            )
        return mine

    def _is_placeholder(self, holders: dict[str, TabularIO]) -> bool:
        """True when ``holders`` contains only the empty default entry."""
        if list(holders) != [DEFAULT_BUCKET_KEY]:
            return False
        return self._holder_count(holders[DEFAULT_BUCKET_KEY]) == 0

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
        driver and lifts to Spark. Buckets are unioned in pipeline
        order — every per-path local holder, then every per-table
        remote holder, then new — with ``allowMissingColumns=False``
        because every bucket carries the same :data:`RESPONSE_SCHEMA`.
        A missing column would mean a real schema drift, and silent
        column-fill is worse than a loud failure.
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
