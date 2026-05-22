# yggdrasil.kafka

Apache Kafka topic surfaced as a `Tabular` — publish and consume with the same Arrow / Polars / pandas conversion stack used everywhere else in yggdrasil.

**Optional dependency:** `confluent-kafka`. Install with `pip install "ygg[kafka]"` (or `pip install confluent-kafka`).

## One-liner

```python
from yggdrasil.kafka import KafkaIO
import pyarrow as pa

KafkaIO("events", bootstrap_servers="localhost:9092").write_arrow_table(table)
```

## Publish (write)

```python
from yggdrasil.kafka import KafkaIO
import pyarrow as pa

sink = KafkaIO("events", bootstrap_servers="localhost:9092")

# From Arrow table
table = pa.table({
    "user_id": [1, 2, 3],
    "action":  ["click", "view", "purchase"],
})
sink.write_arrow_table(table)

# From Polars
import polars as pl
sink.write_polars_frame(pl.from_arrow(table))

# From pandas
sink.write_pandas_frame(table.to_pandas())

# From list of dicts
sink.write_pylist([{"user_id": 4, "action": "logout"}])
```

## Consume (read)

```python
from yggdrasil.kafka import KafkaIO

source = KafkaIO(
    "events",
    bootstrap_servers="localhost:9092",
    group_id="my-consumer-group",
    max_messages=1000,       # stop after N messages
    timeout=5.0,             # poll timeout in seconds
)

# Arrow table
arrow_tbl = source.read_arrow_table()

# Polars DataFrame
df_polars = source.read_polars_frame()

# pandas DataFrame
df_pandas = source.read_pandas_frame()

# Python list of dicts
docs = source.read_pylist()
```

## Custom serializers / deserializers

By default values are JSON-serialized bytes. Swap in your own:

```python
from yggdrasil.kafka import KafkaIO, default_value_serializer, default_value_deserializer
import json

def avro_serializer(row: dict) -> bytes:
    # serialize with fastavro or confluent schema registry
    return json.dumps(row).encode()

def avro_deserializer(data: bytes) -> dict:
    return json.loads(data)

sink = KafkaIO(
    "events",
    bootstrap_servers="localhost:9092",
    value_serializer=avro_serializer,
    value_deserializer=avro_deserializer,
)
```

## Producer configuration

Pass any [confluent-kafka producer config](https://docs.confluent.io/platform/current/clients/confluent-kafka-python/html/index.html) as keyword arguments:

```python
from yggdrasil.kafka import KafkaIO

sink = KafkaIO(
    "events",
    bootstrap_servers="broker1:9092,broker2:9092",
    security_protocol="SASL_SSL",
    sasl_mechanism="PLAIN",
    sasl_username="user",
    sasl_password="secret",
    acks="all",
    compression_type="lz4",
)
```

## Full pipeline: HTTP API → Kafka → Arrow

```python
from yggdrasil.kafka import KafkaIO
from yggdrasil.http_ import HTTPSession
import pyarrow as pa

http = HTTPSession()
sink = KafkaIO("raw-events", bootstrap_servers="localhost:9092")

# Fetch from API and publish to Kafka
resp  = http.get("https://api.example.com/events")
rows  = resp.json()
table = pa.table({
    "id":    [r["id"] for r in rows],
    "event": [r["event"] for r in rows],
})
sink.write_arrow_table(table)

# Consume and aggregate
source = KafkaIO("raw-events", bootstrap_servers="localhost:9092",
                 group_id="agg", max_messages=len(rows))
df = source.read_polars_frame()
print(df.group_by("event").agg(pl.count()))
```
