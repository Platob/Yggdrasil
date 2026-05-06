"""Apache Kafka :class:`TabularIO` — publish and read records.

:class:`KafkaIO` is the canonical Kafka transport surface in
yggdrasil. A single instance is bound to a topic on a cluster and
satisfies the :class:`TabularIO` contract, so a Kafka topic can be
consumed and produced through the same Arrow / Polars / Pandas /
Spark surface that every other tabular source exposes.

The implementation here is intentionally minimal — broker config is
forwarded as-is to ``confluent-kafka`` via the ``config`` mapping
(or its ``producer_config`` / ``consumer_config`` counterparts when
the two sides need to diverge), values are JSON-encoded by default,
and reads stop on a configurable poll-timeout / message-cap. Richer
behaviour (Schema Registry, Avro / Protobuf codecs, exactly-once
producer semantics, dedicated partition assignment, transactional
writes) can be layered on top by subclassing — the base class
deliberately leaves those policy decisions to the caller.

The Kafka client is loaded through :mod:`yggdrasil.kafka.lib`, so
base installs that never touch this module don't pay for the
optional ``confluent-kafka`` dependency.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, Callable, Iterator

import pyarrow as pa

from yggdrasil.data.options import CastOptions
from yggdrasil.environ import PyEnv
from yggdrasil.io.buffer.base import TabularIO
from yggdrasil.io.enums import MimeType, MimeTypes

if TYPE_CHECKING:
    from confluent_kafka import Consumer, Producer
    from pyspark.sql import DataFrame as SparkDataFrame


__all__ = ["KafkaIO", "default_value_serializer", "default_value_deserializer"]


# ---------------------------------------------------------------------------
# Default codec — JSON over UTF-8
# ---------------------------------------------------------------------------


def default_value_serializer(value: Any) -> bytes:
    """Encode a Python value as a Kafka message payload.

    ``bytes`` / ``bytearray`` pass through untouched, ``None`` becomes
    a null-tombstone (``b""`` rather than ``None`` — Kafka tombstones
    are produced by passing ``value=None`` separately, and the
    serializer never sees them in this base path), everything else is
    JSON-encoded with ``ensure_ascii=False`` so unicode round-trips
    cleanly.
    """
    if value is None:
        return b""
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    return json.dumps(value, ensure_ascii=False, default=str).encode("utf-8")


def default_value_deserializer(payload: "bytes | None") -> Any:
    """Inverse of :func:`default_value_serializer`.

    Best-effort JSON decode — payloads that aren't valid JSON come
    back as raw ``bytes`` so binary topics still work.
    """
    if payload is None:
        return None
    if not payload:
        return None
    try:
        return json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return payload


def _default_key_serializer(key: Any) -> "bytes | None":
    if key is None:
        return None
    if isinstance(key, (bytes, bytearray)):
        return bytes(key)
    if isinstance(key, str):
        return key.encode("utf-8")
    return str(key).encode("utf-8")


# ---------------------------------------------------------------------------
# KafkaIO
# ---------------------------------------------------------------------------


class KafkaIO(TabularIO[CastOptions]):
    """A single Kafka topic, exposed as a :class:`TabularIO`.

    Parameters
    ----------
    topic:
        Topic name to read from / write to.
    bootstrap_servers:
        ``"host:port"`` or ``"host1:port,host2:port"``. Forwarded to
        ``confluent-kafka`` as ``bootstrap.servers``.
    group_id:
        Consumer group id used by :meth:`_read_arrow_batches`. A
        random group id keeps each reader independent; a stable id
        lets multiple processes share the topic load.
    config:
        Extra ``confluent-kafka`` config applied to *both* the
        producer and the consumer. Values here are overridden by
        ``producer_config`` / ``consumer_config`` on the side that
        sets them.
    producer_config / consumer_config:
        Side-specific overrides. Useful when (e.g.) the producer
        needs ``acks=all`` and the consumer needs
        ``enable.auto.commit=false``.
    poll_timeout:
        Seconds to wait for a single message during a read. The
        consumer loop exits once a poll returns nothing, so this
        also caps the "drain to end" behaviour for finite topics.
    max_messages:
        Optional hard cap on the total number of messages a read
        will yield before stopping. ``None`` (default) means "drain
        until the poll timeout".
    value_serializer / value_deserializer:
        Pluggable codecs. Defaults are JSON-over-UTF-8 with raw-bytes
        fallback on decode failure.
    key_serializer:
        Codec for message keys. ``str`` keys are UTF-8 encoded by
        default; ``bytes`` pass through.
    value_column / key_column:
        Column names used when projecting decoded messages into Arrow
        record batches. ``key_column=None`` (default) drops the key
        column entirely on the read path.
    """

    _FINAL_TABULAR_IO = True

    @classmethod
    def default_media_type(cls) -> MimeType:
        return MimeTypes.KAFKA_TOPIC

    @classmethod
    def options_class(cls) -> type[CastOptions]:
        return CastOptions

    def __init__(
        self,
        topic: str,
        *,
        bootstrap_servers: str = "localhost:9092",
        group_id: str | None = None,
        config: Mapping[str, Any] | None = None,
        producer_config: Mapping[str, Any] | None = None,
        consumer_config: Mapping[str, Any] | None = None,
        poll_timeout: float = 1.0,
        max_messages: int | None = None,
        value_serializer: Callable[[Any], bytes] = default_value_serializer,
        value_deserializer: Callable[["bytes | None"], Any] = default_value_deserializer,
        key_serializer: Callable[[Any], "bytes | None"] = _default_key_serializer,
        value_column: str = "value",
        key_column: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not topic:
            raise ValueError("KafkaIO requires a non-empty topic name")
        self.topic = topic
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self.config = dict(config) if config else {}
        self.producer_config = dict(producer_config) if producer_config else {}
        self.consumer_config = dict(consumer_config) if consumer_config else {}
        self.poll_timeout = float(poll_timeout)
        self.max_messages = max_messages
        self.value_serializer = value_serializer
        self.value_deserializer = value_deserializer
        self.key_serializer = key_serializer
        self.value_column = value_column
        self.key_column = key_column

        # Lazily constructed clients — opened on first use, closed
        # by :meth:`_release` (Disposable lifecycle).
        self._producer: "Producer | None" = None
        self._consumer: "Consumer | None" = None

    # ------------------------------------------------------------------
    # Disposable lifecycle
    # ------------------------------------------------------------------

    def _release(self) -> None:
        super()._release()
        consumer = self._consumer
        producer = self._producer
        self._consumer = None
        self._producer = None
        if consumer is not None:
            try:
                consumer.close()
            except Exception:
                pass
        if producer is not None:
            try:
                producer.flush(self.poll_timeout)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Client construction
    # ------------------------------------------------------------------

    def _producer_client(self) -> "Producer":
        if self._producer is not None:
            return self._producer
        from yggdrasil.kafka.lib import confluent_kafka

        cfg: dict[str, Any] = {"bootstrap.servers": self.bootstrap_servers}
        cfg.update(self.config)
        cfg.update(self.producer_config)
        self._producer = confluent_kafka.Producer(cfg)
        return self._producer

    def _consumer_client(self) -> "Consumer":
        if self._consumer is not None:
            return self._consumer
        from yggdrasil.kafka.lib import confluent_kafka

        cfg: dict[str, Any] = {
            "bootstrap.servers": self.bootstrap_servers,
            "group.id": self.group_id or f"ygg-kafka-{id(self):x}",
            "auto.offset.reset": "earliest",
            "enable.auto.commit": True,
        }
        cfg.update(self.config)
        cfg.update(self.consumer_config)
        consumer = confluent_kafka.Consumer(cfg)
        consumer.subscribe([self.topic])
        self._consumer = consumer
        return consumer

    # ------------------------------------------------------------------
    # TabularIO contract
    # ------------------------------------------------------------------

    def stat(self):
        return self._stats

    def _read_arrow_batches(self, options: CastOptions) -> Iterator[pa.RecordBatch]:
        """Drain the topic into Arrow record batches.

        Polls the underlying consumer until a poll times out or
        ``max_messages`` is reached, decodes each payload with
        :attr:`value_deserializer`, and yields one batch per
        ``options.row_size`` (default: a single batch on EOF).
        """
        consumer = self._consumer_client()
        from yggdrasil.kafka.lib import confluent_kafka  # noqa: F401  (clarity)

        row_size = getattr(options, "row_size", None) or 0
        max_total = self.max_messages
        produced = 0
        buffer: list[dict[str, Any]] = []

        while True:
            if max_total is not None and produced >= max_total:
                break
            message = consumer.poll(timeout=self.poll_timeout)
            if message is None:
                break
            err = message.error()
            if err is not None:
                # Partition EOF is expected and not a real error;
                # everything else is surfaced.
                from confluent_kafka import KafkaError

                if err.code() == KafkaError._PARTITION_EOF:
                    break
                raise RuntimeError(f"Kafka consumer error: {err}")

            row: dict[str, Any] = {
                self.value_column: self.value_deserializer(message.value()),
            }
            if self.key_column is not None:
                key = message.key()
                row[self.key_column] = (
                    key.decode("utf-8") if isinstance(key, (bytes, bytearray)) else key
                )
            buffer.append(row)
            produced += 1

            if row_size and len(buffer) >= row_size:
                yield self._records_to_batch(buffer)
                buffer = []

        if buffer:
            yield self._records_to_batch(buffer)

    def _scan_spark_frame(self, options: CastOptions) -> "SparkDataFrame":
        """Native Spark Structured Streaming source for Kafka.

        Skips the parquet-spill fallback in :class:`TabularIO` and
        wires straight into Spark's built-in ``kafka`` source —
        ``spark.readStream.format("kafka")`` — so ``isStreaming`` is
        ``True`` and the consumer keeps tailing the topic for the
        lifetime of the streaming query.

        ``key`` and ``value`` arrive as Spark ``BinaryType`` columns;
        callers typically follow with a ``selectExpr("CAST(value AS
        STRING)")`` and a JSON / Avro decode to recover the row
        shape. We don't apply :attr:`value_deserializer` here — the
        Spark side has its own decoders that fuse into the streaming
        plan.
        """
        spark = PyEnv.spark_session(create=True)

        reader = (
            spark.readStream.format("kafka")
            .option("kafka.bootstrap.servers", self.bootstrap_servers)
            .option("subscribe", self.topic)
        )
        if self.group_id:
            reader = reader.option("kafka.group.id", self.group_id)
        # Forward every passthrough config under the ``kafka.``
        # prefix so SASL/SSL credentials reach the Spark Kafka
        # source unchanged.
        for key, value in {**self.config, **self.consumer_config}.items():
            reader = reader.option(f"kafka.{key}", str(value))
        return reader.load()

    def _records_to_batch(self, records: list[dict[str, Any]]) -> pa.RecordBatch:
        """Build a :class:`pa.RecordBatch` from a list of decoded rows.

        Falls through :meth:`TabularIO._normalize_records` so rows
        with diverging keys are backfilled to a uniform schema before
        Arrow type inference runs.
        """
        normalized = self._normalize_records(records)
        return pa.RecordBatch.from_pylist(normalized)

    # ------------------------------------------------------------------
    # TabularIO write path
    # ------------------------------------------------------------------

    def _write_arrow_batches(
        self,
        batches: Iterable[pa.RecordBatch],
        options: CastOptions,
    ) -> None:
        """Publish each row of every batch as a Kafka message.

        When :attr:`key_column` is set, that column is removed from
        the value payload and used as the message key — useful for
        compacted topics. Otherwise the entire row dict is sent as
        the value, encoded by :attr:`value_serializer`.
        """
        producer = self._producer_client()
        key_column = self.key_column

        for batch in batches:
            for row in batch.to_pylist():
                if key_column is not None and key_column in row:
                    key = self.key_serializer(row.pop(key_column))
                else:
                    key = None
                payload = self.value_serializer(row)
                producer.produce(self.topic, value=payload, key=key)
                # Service delivery callbacks / queue pressure between
                # produces — keeps the local queue from overflowing
                # on large batches.
                producer.poll(0)

        producer.flush()
