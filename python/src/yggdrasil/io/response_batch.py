"""Origin-tagged carrier for batched ``Session.send_many`` results.

`send_many` historically returned an `Iterator[Response]` and the Spark
variant returned a single fused DataFrame. Both lose the *origin* of every
row — the caller can't tell whether a response came from the local
on-disk cache, the remote SQL cache, or a fresh network fetch.

:class:`ResponseBatch` keeps that split visible. The same class works for
both the Python and Spark paths: each bucket is either a list of
:class:`Response` (Python path) or a Spark DataFrame typed as
``RESPONSE_SCHEMA`` (Spark path), or ``None`` when that origin produced
nothing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterator, Optional, Union

import pyarrow as pa

from .response import RESPONSE_SCHEMA, Response

if TYPE_CHECKING:
    from pyspark.sql import DataFrame as SparkDataFrame, SparkSession


__all__ = ["ResponseBatch", "Bucket", "responses_to_spark_df"]


# A single bucket holds one origin's responses. ``None`` means "this stage
# was skipped"; an empty list means "stage ran, found zero responses". The
# distinction matters for cache-hit metrics, so we keep both shapes.
Bucket = Union[list[Response], "SparkDataFrame", None]


class ResponseBatch:
    """Origin-tagged view of a batch of responses.

    Three buckets, in pipeline order:

    - ``local_hits``  — served from the on-disk pickle cache.
    - ``remote_hits`` — served from the remote SQL cache.
    - ``new_hits``    — fetched from the network this run.

    Each bucket is either a ``list[Response]`` (Python path), a Spark
    ``DataFrame`` (Spark path), or ``None`` (stage skipped). Iteration,
    ``len()``, and the ``responses`` accessor work on Python-mode
    batches; :meth:`to_dataframe` works in both modes and is the right
    call when you want one fused DataFrame.

    Internally the buckets live on private holders
    (``_local_response`` / ``_remote_response`` / ``_new_response``) and
    the public names are read/write properties — that keeps the
    polymorphic shape opaque to callers who only want the typed
    accessors.
    """

    __slots__ = (
        "_local_response",
        "_remote_response",
        "_new_response",
        "spark",
    )

    def __init__(
        self,
        local_hits: "Bucket" = None,
        remote_hits: "Bucket" = None,
        new_hits: "Bucket" = None,
        *,
        spark: Optional["SparkSession"] = None,
    ) -> None:
        self._local_response: "Bucket" = local_hits
        self._remote_response: "Bucket" = remote_hits
        self._new_response: "Bucket" = new_hits
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
    def local_hits(self) -> "Bucket":
        return self._local_response

    @local_hits.setter
    def local_hits(self, value: "Bucket") -> None:
        self._local_response = value

    @property
    def remote_hits(self) -> "Bucket":
        return self._remote_response

    @remote_hits.setter
    def remote_hits(self, value: "Bucket") -> None:
        self._remote_response = value

    @property
    def new_hits(self) -> "Bucket":
        return self._new_response

    @new_hits.setter
    def new_hits(self, value: "Bucket") -> None:
        self._new_response = value

    # ------------------------------------------------------------------
    # Shape helpers
    # ------------------------------------------------------------------

    def _holders(self) -> tuple["Bucket", "Bucket", "Bucket"]:
        return (self._local_response, self._remote_response, self._new_response)

    @property
    def is_spark(self) -> bool:
        """True if any non-None holder is a Spark DataFrame."""
        for h in self._holders():
            if h is None:
                continue
            if not isinstance(h, list):
                return True
        return False

    def parts(self) -> list[Any]:
        """Non-empty buckets in pipeline order.

        Skipped (``None``) and empty (``[]``) buckets are omitted —
        callers iterating ``parts()`` get only the holders worth doing
        work on.
        """
        out: list[Any] = []
        for h in self._holders():
            if h is None:
                continue
            if isinstance(h, list) and not h:
                continue
            out.append(h)
        return out

    @property
    def counts(self) -> dict[str, int]:
        """Per-origin row counts.

        Skipped holders count as ``0``. For Spark holders this triggers
        a job — fine for debugging or small assertions, not for hot paths.
        """
        out: dict[str, int] = {}
        for name, h in zip(("local", "remote", "new"), self._holders()):
            if h is None:
                out[name] = 0
            elif isinstance(h, list):
                out[name] = len(h)
            else:
                out[name] = h.count()
        return out

    # ------------------------------------------------------------------
    # Python-mode iteration
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[Response]:
        if self.is_spark:
            raise TypeError(
                "ResponseBatch is in Spark mode — iterating Response objects "
                "would force a driver-side collect. Call `.to_dataframe()` for "
                "the fused DataFrame, or read individual `.local_hits` / "
                "`.remote_hits` / `.new_hits` holders if you really need the "
                "Spark DataFrames."
            )
        for h in self._holders():
            if h:
                yield from h  # type: ignore[misc]

    def __len__(self) -> int:
        if self.is_spark:
            raise TypeError(
                "len() is not defined for a Spark ResponseBatch — use "
                "`.counts` (which calls `.count()` per holder) or check the "
                "individual DataFrames."
            )
        total = 0
        for h in self._holders():
            if h:
                total += len(h)  # type: ignore[arg-type]
        return total

    def __bool__(self) -> bool:
        return bool(self.parts())

    @property
    def responses(self) -> list[Response]:
        """Flat list of every response, local → remote → new.

        Python mode only — raises if any holder is a Spark DataFrame.
        """
        return list(self)

    # ------------------------------------------------------------------
    # Mutation / merge
    # ------------------------------------------------------------------

    def extend(self, other: "ResponseBatch") -> "ResponseBatch":
        """Merge another batch in place. Returns self for chaining.

        Python-mode only — mixing list and DataFrame holders would force
        a Spark conversion at merge time, which is too sneaky to do
        silently. Convert one side first if you need that.
        """
        if self.is_spark or other.is_spark:
            raise TypeError(
                "extend() only supports Python-mode batches. Lift one side "
                "with `responses_to_spark_df` and union the DataFrames "
                "yourself if you need to merge across modes."
            )

        for name in ("_local_response", "_remote_response", "_new_response"):
            mine = getattr(self, name)
            theirs = getattr(other, name)
            if not theirs:
                continue
            if mine is None:
                setattr(self, name, list(theirs))
            else:
                mine.extend(theirs)
        return self

    # ------------------------------------------------------------------
    # Spark interop
    # ------------------------------------------------------------------

    def to_spark(self, spark: "SparkSession") -> "ResponseBatch":
        """Return a new batch with every Python-list holder lifted to Spark.

        DataFrame holders are passed through. The returned batch carries
        the same ``spark`` reference so :meth:`to_dataframe` can synthesize
        a typed empty DataFrame even when every bucket is empty.
        """
        def _lift(h: "Bucket") -> "Bucket":
            if h is None:
                return None
            if isinstance(h, list):
                return responses_to_spark_df(h, spark)
            return h

        return ResponseBatch(
            local_hits=_lift(self._local_response),
            remote_hits=_lift(self._remote_response),
            new_hits=_lift(self._new_response),
            spark=spark,
        )

    def to_dataframe(
        self,
        spark: Optional["SparkSession"] = None,
    ) -> "SparkDataFrame":
        """Union all non-empty buckets into one Spark DataFrame.

        Works in both modes: list holders are lifted to Spark first
        (using ``spark`` or ``self.spark``), DataFrame holders pass
        through. Buckets are unioned with ``allowMissingColumns=False``
        because every bucket carries the same ``RESPONSE_SCHEMA`` —
        a missing column would mean a real schema drift, and silent
        column-fill is worse than a loud failure.
        """
        target_spark = spark or self.spark
        if target_spark is None:
            raise RuntimeError(
                "to_dataframe() needs a SparkSession. Pass `spark=` or build "
                "the batch with `spark=...` so we can synthesize an empty "
                "DataFrame and lift any Python-list holders."
            )

        lifted = self.to_spark(target_spark)
        parts = lifted.parts()
        if not parts:
            return target_spark.createDataFrame(
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
