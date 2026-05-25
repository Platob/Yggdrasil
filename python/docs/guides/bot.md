# Bot — Remote Execution & Mesh Chat

`yggdrasil.bot` is a lightweight HTTP server that turns any Python process into
a remotely-controllable execution node.  Nodes discover each other, execute
functions, stream Arrow-encoded results, and exchange messages in real-time
channels.

Install the `[bot]` extra:

```bash
pip install "ygg[bot]"
```

---

## Quick start

### Start a node

```bash
ygg bot serve --port 8100 --allow-remote
```

Or from Python:

```python
from yggdrasil.bot import create_app
app = create_app()  # uvicorn-ready FastAPI app
```

### Register a function for remote execution

```python
from yggdrasil.bot import remote

@remote
def predict(features: list[float]) -> dict:
    import numpy as np
    model = ...  # load your model
    return {"score": float(np.mean(features))}
```

The `@remote` decorator:

- Registers the function on the local node's `/api/call` endpoint.
- **Auto-infers third-party imports** via AST analysis (`numpy` is
  detected from `import numpy as np`).  Missing packages are
  auto-installed on the remote node before execution.
- Works normally for local calls.

### Call from a client

```python
from yggdrasil.bot import BotClient

client = BotClient("http://worker-node:8100")
result = client.call(predict, [1.0, 2.0, 3.0])
# {'score': 2.0}
```

Tabular results stream as Arrow IPC automatically:

```python
@remote
def query(table: str, limit: int = 1000) -> pa.Table:
    ...

# Full table in one call
table = client.call(query, "sales", limit=500)

# Batch-by-batch streaming
for batch in client.call_stream(query, "events", limit=1_000_000):
    process(batch)
```

---

## Architecture

```
Node A (:8100)              Node B (:8101)              Node C (:8102)
  |                            |                            |
  +-- /api/hello -----------> discovers B, C               |
  +-- /api/call ------------> execute @remote fn           |
  +-- /api/messenger -------> broadcast to channel ------> receives via poll
  +-- /api/job -------------> define task DAG, trigger runs
  +-- /api/cmd -------------> shell execution
  +-- /api/python ----------> Python execution
  +-- /api/env -------------> environment variables
```

### Peer discovery

Every node exposes `GET /api/hello` (identity) and `POST /api/hello`
(peer registration).  When node A calls `POST /api/hello` on node B,
both nodes memorise each other.  Stale peers (not seen in 5 minutes)
are automatically pruned.

```python
# Discover peers programmatically
client = BotClient("http://node-a:8100")
peers = client.discover(["http://node-b:8101", "http://node-c:8102"])
```

Or from the CLI:

```bash
ygg bot chat --url http://node-a:8100
# then type /peers to see discovered nodes
```

### Transport layer

| Result type | Wire format | Content-Type |
|---|---|---|
| `pa.Table`, `pa.RecordBatch` | Arrow IPC Stream | `application/vnd.apache.arrow.stream` |
| `pl.DataFrame`, `pd.DataFrame` | Arrow IPC Stream (zero-copy bridge) | `application/vnd.apache.arrow.stream` |
| Everything else | `yggdrasil.pickle` (YgD1 + zstd) | `application/x-yggdrasil-pickle` |

Tabular results always stream.  Complex Python objects are serialized
via the yggdrasil wire format with zstd compression.

---

## Messenger

Channels are chat rooms that any connected node can join.  Messages
are delivered in real-time via long-poll (no busy-wait).

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/messenger` | Send message |
| GET | `/api/messenger/channels` | List channels |
| POST | `/api/messenger/channels?name=X` | Create channel |
| GET | `/api/messenger/channels/{name}` | Channel info |
| DELETE | `/api/messenger/channels/{name}` | Delete channel |
| GET | `/api/messenger/channels/{name}/messages` | Fetch messages |
| GET | `/api/messenger/channels/{name}/poll` | Long-poll for new messages |

Channels inactive for more than 24 hours are automatically deleted.
The `general` channel is never deleted.

### Terminal chat

```bash
ygg bot chat --url http://localhost:8100 --user alice
```

Features:

- Real-time message delivery via background long-poll thread
- ANSI-colored usernames (hash-based, 6 colors)
- Typing animation for incoming messages
- Slash commands: `/join`, `/channels`, `/create`, `/users`, `/help`, `/quit`

---

## Jobs

Define multi-task workflows with dependency DAGs:

```python
import requests

resp = requests.post("http://localhost:8100/api/job", json={
    "name": "daily-pipeline",
    "tasks": {
        "fetch":     {"type": "cmd", "command": ["curl", "-o", "/tmp/data.json", "https://..."]},
        "transform": {"type": "python", "code": "...", "depends_on": ["fetch"]},
        "notify":    {"type": "cmd", "command": ["echo", "done"], "depends_on": ["transform"]},
    },
})
job_id = resp.json()["job"]["job_id"]

# Trigger a run
requests.post(f"http://localhost:8100/api/job/{job_id}/run")
```

Tasks execute in topological order.  Failed dependencies skip
downstream tasks.

---

## CLI Reference

```bash
# Start the server
ygg bot serve [--host H] [--port P] [--allow-remote] [--reload]

# Call a remote function
ygg bot run <func_key> [args...] [--url URL] [--kwarg K=V] [--stream]

# Open the chat terminal
ygg bot chat [--url URL] [--user NAME] [--channel CH]
```

---

## All endpoints

| Group | Method | Path | Purpose |
|---|---|---|---|
| **Discovery** | GET | `/api/hello` | Node identity |
| | POST | `/api/hello` | Register peer |
| | GET | `/api/hello/peers` | List known peers |
| | POST | `/api/hello/discover` | Discover peers from URL list |
| **Execution** | POST | `/api/call` | Execute @remote function |
| | POST | `/api/call/stream` | Always-streaming variant |
| | GET | `/api/call/registry` | List registered functions |
| **Messenger** | POST | `/api/messenger` | Send message |
| | GET | `/api/messenger/channels` | List channels |
| | POST | `/api/messenger/channels` | Create channel |
| | GET | `/api/messenger/channels/{n}` | Channel info |
| | DELETE | `/api/messenger/channels/{n}` | Delete channel |
| | GET | `/api/messenger/channels/{n}/messages` | Fetch messages |
| | GET | `/api/messenger/channels/{n}/poll` | Long-poll |
| **Jobs** | POST | `/api/job` | Create job |
| | GET | `/api/job` | List jobs |
| | GET | `/api/job/{id}` | Get job |
| | PUT | `/api/job/{id}` | Update job |
| | DELETE | `/api/job/{id}` | Delete job |
| | POST | `/api/job/{id}/run` | Trigger run |
| | GET | `/api/job/{id}/run` | List runs |
| | GET | `/api/job/{id}/run/{rid}` | Get run |
| | PUT | `/api/job/{id}/run/{rid}` | Re-trigger run |
| | DELETE | `/api/job/{id}/run/{rid}` | Delete run |
| **Shell** | POST | `/api/cmd` | Execute command |
| | GET | `/api/cmd` | List history |
| | GET | `/api/cmd/{id}` | Get result |
| | DELETE | `/api/cmd/{id}` | Delete result |
| **Python** | POST | `/api/python` | Execute code |
| | GET | `/api/python` | List history |
| | GET | `/api/python/{id}` | Get result |
| | DELETE | `/api/python/{id}` | Delete result |
| **Env** | GET | `/api/env` | Read env vars |
| | POST | `/api/env` | Set env vars |

---

## Performance

Benchmarked on a single node (repeat=5):

| Operation | Throughput |
|---|---|
| Message send (direct) | 67,000 msg/s |
| Arrow IPC stream (10K rows) | 2.4 GB/s |
| `/api/call` scalar | 3.3 ms |
| `/api/call` tabular 10K rows | 17 ms |
| Polars DF round-trip | 749 MB/s |
