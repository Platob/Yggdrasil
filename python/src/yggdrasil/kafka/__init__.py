"""yggdrasil Kafka integration.

Surfaces an Apache Kafka topic as a :class:`Tabular`, so the same
Arrow / Polars / Pandas / Spark conversion stack used everywhere
else in yggdrasil also covers Kafka publish / consume. The runtime
client (``confluent-kafka``) is loaded through
:mod:`yggdrasil.kafka.lib` — base installs that never touch this
module don't need the optional dependency.

Quick start
-----------

    >>> from yggdrasil.kafka import KafkaIO
    >>>
    >>> sink = KafkaIO("events", bootstrap_servers="localhost:9092")
    >>> sink.write_arrow_table(table)            # publish
    >>>
    >>> source = KafkaIO("events", bootstrap_servers="localhost:9092",
    ...                  group_id="reader", max_messages=1000)
    >>> df = source.read_polars_frame()          # consume
"""

from __future__ import annotations

from .base import (
    KafkaIO,
    default_value_deserializer,
    default_value_serializer,
)


__all__ = [
    "KafkaIO",
    "default_value_serializer",
    "default_value_deserializer",
]
