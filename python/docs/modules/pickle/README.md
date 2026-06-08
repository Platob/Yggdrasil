# yggdrasil.pickle

Custom serialization wire format with optional compression (zstandard), pluggable codecs (cloudpickle, dill), and a rich JSON layer powered by orjson.

## One-liner

```python
from yggdrasil.pickle import dumps, loads

blob = dumps({"event": "login", "user_id": 42})
obj  = loads(blob)   # {"event": "login", "user_id": 42}
```

## JSON (`yggdrasil.pickle.json`)

Drop-in replacement for stdlib `json` with datetime / UUID / dataclass / Enum / Path / Decimal / set / Mapping support, 3–10× faster than stdlib.

```python
from yggdrasil.pickle.json import dumps, loads, load, dump

# Serialize rich types without a custom encoder
import datetime, uuid
data = {
    "ts": datetime.datetime(2026, 1, 1, 12, 0, tzinfo=datetime.timezone.utc),
    "id": uuid.uuid4(),
    "score": 9.95,
}
blob = dumps(data)           # bytes (orjson, UTF-8)
text = dumps(data, to_bytes=False)  # str

back = loads(blob)           # ts → str (ISO 8601), id → str
```

Pretty-print (2-space indent):

```python
from yggdrasil.pickle.json import dumps

print(dumps({"a": 1, "b": [2, 3]}, indent=2, to_bytes=False))
```

Sorted keys + ensure-ASCII (falls back to stdlib):

```python
blob = dumps(data, sort_keys=True, ensure_ascii=True, to_bytes=True)
```

Safe `loads` with rich type restoration (datetime / UUID strings → Python objects):

```python
from yggdrasil.pickle.json import loads

restored = loads('{"ts": "2026-01-01T12:00:00+00:00"}', safe=False)
# restored["ts"] is a datetime.datetime object
```

File round-trip:

```python
from yggdrasil.pickle.json import dump, load
from pathlib import Path

with open("/tmp/payload.json", "wb") as fp:
    dump({"k": "v"}, fp)

with open("/tmp/payload.json", "rb") as fp:
    obj = load(fp)
```

## Binary pickle (`yggdrasil.pickle`)

The binary format adds a typed header (magic bytes + codec + metadata) around the standard pickle payload, with optional zstandard compression.

```python
from yggdrasil.pickle import dumps, loads, serialize, dump, load

# Basic round-trip
blob = dumps({"key": [1, 2, 3]})
obj  = loads(blob)

# Inspect the envelope without unpickling
from yggdrasil.pickle import Serialized
s = serialize({"key": "value"})
print(s.codec, s.metadata)

# File round-trip
import io
buf = io.BytesIO()
dump({"records": list(range(10))}, buf)
buf.seek(0)
obj = load(buf)

# Base-64 round-trip (for embedding in JSON/text)
b64_str = dumps({"v": 1}, ).hex()
```

### Codec selection

The codec is chosen automatically based on installed extras.
Priority: `cloudpickle` → `dill` → stdlib `pickle`.

```python
# Force a specific codec
from yggdrasil.pickle.ser.codec import Codec

blob = dumps(obj, codec=Codec.CLOUDPICKLE)
blob = dumps(obj, codec=Codec.DILL)
blob = dumps(obj, codec=Codec.PICKLE)
```

### Metadata tags

```python
from yggdrasil.pickle import Tags, dumps, loads, serialize

# Attach metadata to the wire envelope
s = serialize(
    {"data": [1, 2, 3]},
    metadata={Tags.VERSION: b"1.2.3", Tags.SOURCE: b"etl-job"},
)
print(s.metadata)           # {b"version": b"1.2.3", b"source": b"etl-job"}
```

## Error handling

```python
from yggdrasil.pickle import (
    SerializationError,
    HeaderDecodeError,
    MetadataDecodeError,
    InvalidCodecError,
)
from yggdrasil.exceptions import YGGException

try:
    from yggdrasil.pickle import loads
    loads(b"not-a-valid-payload")
except HeaderDecodeError as exc:
    print("Bad header:", exc)
except SerializationError as exc:           # all pickle errors
    print("Serde error:", exc)
except YGGException:                        # every yggdrasil error
    raise
```

## Advanced: Serialized dataclass

```python
from yggdrasil.pickle import serialize, Serialized
import io

s: Serialized = serialize({"batch": list(range(1000))})

# Inspect
print(s.codec)           # e.g. Codec.CLOUDPICKLE
print(len(s.payload))    # compressed byte length

# Write to any file-like
buf = io.BytesIO()
s.write_to(buf)
buf.seek(0)
```
