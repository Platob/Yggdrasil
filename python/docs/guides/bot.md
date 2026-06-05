# Bot — Remote Execution & Mesh Chat

`yggdrasil.bot` turns any Python process into a remotely-controllable
execution node.  Nodes discover each other, execute functions, stream
Arrow-encoded results, and exchange messages in real-time channels.

---

## 1. Install

```bash
pip install "ygg[bot]"
```

This adds `fastapi`, `uvicorn`, and `pydantic` to the base `ygg` install.

---

## 2. Start a bot (simplest)

=== "CLI"

    ```bash
    ygg bot serve
    ```

    The server starts on `http://127.0.0.1:8100` with interactive docs at `/docs`.

=== "Python"

    ```python
    from yggdrasil.bot import create_app

    app = create_app()   # FastAPI app — pass to uvicorn
    ```

=== "Auto (background)"

    Every `ygg` CLI command auto-spawns a background bot if none is running.
    Just use any command and the bot is there:

    ```bash
    ygg bot chat          # bot starts automatically
    ygg bot status        # check what's running
    ```

### Server options

```bash
ygg bot serve --host 0.0.0.0 --port 9000 --allow-remote --reload
```

| Flag | Default | Purpose |
|---|---|---|
| `--host` | `127.0.0.1` | Bind address |
| `--port` | `8100` (auto-scans if busy) | Bind port |
| `--allow-remote` | off | Accept non-localhost clients |
| `--reload` | off | Auto-reload on file changes |

---

## 3. Execute a shell command

=== "curl"

    ```bash
    curl -X POST http://localhost:8100/api/cmd \
      -H "Content-Type: application/json" \
      -d '{"command": ["echo", "hello world"]}'
    ```

=== "Python"

    ```python
    from yggdrasil.bot.client import BotClient

    client = BotClient("http://localhost:8100")
    result = client.cmd(["echo", "hello world"])
    print(result["stdout"])  # "hello world\n"
    ```

---

## 4. Execute Python code

=== "curl"

    ```bash
    curl -X POST http://localhost:8100/api/python \
      -H "Content-Type: application/json" \
      -d '{"code": "import sys; __result__ = sys.version"}'
    ```

=== "Python"

    ```python
    result = client.execute("import sys; __result__ = sys.version")
    print(result["result"])  # "3.11.15 ..."
    ```

Set `__result__` in your code to return a value.  Anything printed goes to
`stdout`.

---

## 5. Remote function execution (`@remote`)

### Register a function

```python
from yggdrasil.bot import remote

@remote
def compute(x: int, y: int) -> int:
    return x + y
```

The `@remote` decorator:

- Registers the function on the local node's `/api/call` endpoint.
- **Auto-infers third-party imports** via AST analysis — `import numpy`
  inside the function body is detected, and `numpy` is auto-installed on
  the remote node if missing.
- Works normally for local calls — `compute(2, 3)` returns `5`.

### Call from another machine

```python
from yggdrasil.bot import BotClient

client = BotClient("http://worker-node:8100")
result = client.call(compute, 10, 20)   # 30
```

### Explicit dependencies

```python
@remote(timeout=60, modules=["numpy", "scipy"])
def predict(features: list[float]) -> dict:
    import numpy as np
    return {"mean": float(np.mean(features))}
```

### Call by name (no local function needed)

```python
result = client.call("mymodule:predict", [1.0, 2.0, 3.0])
```

---

## 6. Tabular results with Arrow IPC streaming

Functions that return tabular data (Arrow Table, Polars DataFrame,
pandas DataFrame) are automatically serialized as Arrow IPC streams —
zero-copy, columnar, and fast.

```python
import pyarrow as pa
from yggdrasil.bot import remote

@remote
def query(table: str, limit: int = 1000) -> pa.Table:
    # ... your query logic
    return pa.table({"id": range(limit), "value": range(limit)})
```

=== "Full result"

    ```python
    table = client.call(query, "sales", limit=500)
    print(table.to_pandas())
    ```

=== "Streaming (batch-by-batch)"

    ```python
    for batch in client.call_stream(query, "events", limit=1_000_000):
        process(batch)  # pa.RecordBatch
    ```

=== "CLI"

    ```bash
    ygg bot run "mymodule:query" sales --kwarg limit=500 --stream
    ```

### Transport format

| Result type | Wire format | Content-Type |
|---|---|---|
| `pa.Table`, `pa.RecordBatch` | Arrow IPC Stream | `application/vnd.apache.arrow.stream` |
| `pl.DataFrame`, `pd.DataFrame` | Arrow IPC Stream (zero-copy bridge) | `application/vnd.apache.arrow.stream` |
| Everything else | yggdrasil pickle (zstd compressed) | `application/x-yggdrasil-pickle` |

---

## 7. Chat messenger

Channels are chat rooms with real-time message delivery via long-poll.

### Terminal chat

```bash
ygg bot chat
```

```
  __   __ ___ ___
  \ \ / // __/ __|
   \ V /| (_ | (_ |
    |_|  \___|\___|
              chat

  * connecting to http://127.0.0.1:8100 as alice
  * channels:
      #general  12 msgs, 3 members <--
  * joined #general — type /help for commands

  17:30 bob: hey alice!
  17:31 alice: hi!

[general] alice>
```

### Chat commands

| Command | Purpose |
|---|---|
| `/join <channel>` | Switch to a channel |
| `/channels` | List all channels |
| `/create <channel>` | Create a new channel |
| `/users` | Show members in current channel |
| `/help` | Show command list |
| `/quit` | Exit |

### From code

=== "Send a message"

    ```bash
    curl -X POST http://localhost:8100/api/messenger \
      -H "Content-Type: application/json" \
      -d '{"text": "hello!", "sender": "bot-script", "channel": "general"}'
    ```

=== "Poll for new messages"

    ```bash
    # Long-poll — blocks up to 30s waiting for new messages
    curl "http://localhost:8100/api/messenger/channels/general/poll?after_id=abc123&timeout=30"
    ```

### Channel auto-deletion

Channels inactive for more than **24 hours** are automatically deleted.
The `general` channel is never deleted.

---

## 8. Peer discovery

Nodes discover each other via a simple hello handshake.

### How it works

```
Node A                          Node B
  |                                |
  |--- POST /api/hello ---------->|  (A introduces itself)
  |<-- {node_id, peers} ----------|  (B responds + memorizes A)
  |                                |
  |  Both nodes now know each other
```

### Discover peers from CLI

```bash
# Check who the local bot knows about
curl http://localhost:8100/api/hello/peers

# Introduce this node to others
curl -X POST http://localhost:8100/api/hello/discover \
  -H "Content-Type: application/json" \
  -d '["http://node-b:8100", "http://node-c:8100"]'
```

### From Python

```python
client = BotClient("http://node-a:8100")

# Node identity
info = client.list_functions()  # also accessible via GET /api/hello

# The hello endpoint returns node_id, version, uptime, channels, functions
```

Stale peers (not seen in 5 minutes) are automatically pruned.

---

## 9. Jobs with task DAGs

Define multi-task workflows where tasks execute in dependency order.

```python
import requests

# Create a job with three tasks
resp = requests.post("http://localhost:8100/api/job", json={
    "name": "daily-pipeline",
    "tasks": {
        "fetch": {
            "type": "cmd",
            "command": ["curl", "-o", "/tmp/data.json", "https://api.example.com/data"],
        },
        "transform": {
            "type": "python",
            "code": "import json; data = json.load(open('/tmp/data.json')); print(len(data))",
            "depends_on": ["fetch"],
        },
        "notify": {
            "type": "cmd",
            "command": ["echo", "pipeline complete"],
            "depends_on": ["transform"],
        },
    },
})
job_id = resp.json()["job"]["job_id"]

# Trigger a run
run = requests.post(f"http://localhost:8100/api/job/{job_id}/run")
print(run.json()["run"]["status"])       # "completed"
print(run.json()["run"]["task_results"]) # per-task stdout/stderr/returncode
```

Tasks execute in topological order. Failed dependencies skip downstream
tasks.

---

## 10. Environment variables

Read and set environment variables on the remote node:

```bash
# Read specific vars
curl "http://localhost:8100/api/env?keys=PATH,HOME"

# Set a variable
curl -X POST http://localhost:8100/api/env \
  -H "Content-Type: application/json" \
  -d '{"variables": {"MY_VAR": "hello", "OLD_VAR": null}}'
```

Setting a value to `null` unsets the variable.

---

## 11. Bot home directory

Every user gets a dedicated directory at `~/.bot/{user_key}/`:

```
~/.bot/d2a0080137c2/
├── data/       # job history, execution history
├── cache/      # local caching for responses
├── spill/      # spill-to-disk buffers for large payloads
├── logs/       # bot-2026-05-25.log (daily rotation)
│   └── bot-2026-05-25.log
├── bot.pid     # PID of the running bot process
└── bot.port    # port the bot is listening on
```

`user_key` is a 12-character hash of `login@hostname`.

### Log management

Log files older than **7 days** are automatically cleaned on every bot
startup.  Configure via `YGG_BOT_LOG_RETENTION_DAYS`.

### Check status

```bash
ygg bot status
```

```
Bot home:  /home/alice/.bot/a1b2c3d4e5f6
Logs:      /home/alice/.bot/a1b2c3d4e5f6/logs
Cache:     /home/alice/.bot/a1b2c3d4e5f6/cache
Status:    running (pid=12345, port=8100)
URL:       http://127.0.0.1:8100
```

---

## 12. Multi-node setup

### Start nodes on different ports

```bash
# Terminal 1
YGG_BOT_NODE_ID=node-a ygg bot serve --port 8100 --allow-remote

# Terminal 2
YGG_BOT_NODE_ID=node-b ygg bot serve --port 8101 --allow-remote
```

### Connect them

```bash
# From node-a, discover node-b
curl -X POST http://localhost:8100/api/hello/discover \
  -H "Content-Type: application/json" \
  -d '["http://localhost:8101"]'
```

### Broadcast a message across nodes

```bash
# Send to node-a's general channel
curl -X POST http://localhost:8100/api/messenger \
  -H "Content-Type: application/json" \
  -d '{"text": "hello from node-a!", "sender": "node-a"}'
```

### Fan out work

```python
from yggdrasil.bot import BotClient, remote

@remote
def heavy_compute(chunk: list[int]) -> int:
    return sum(x * x for x in chunk)

nodes = [
    BotClient("http://node-a:8100"),
    BotClient("http://node-b:8101"),
]

data = list(range(10_000))
mid = len(data) // 2
results = [
    nodes[0].call(heavy_compute, data[:mid]),
    nodes[1].call(heavy_compute, data[mid:]),
]
total = sum(results)
```

---

## CLI reference

```
ygg bot serve   [--host H] [--port P] [--allow-remote] [--reload]
ygg bot run     <func> [args...] [--url URL] [--kwarg K=V] [--stream]
ygg bot chat    [--url URL] [--user NAME] [--channel CH]
ygg bot status
ygg bot stop
```

---

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `YGG_BOT_HOST` | `127.0.0.1` | Bind address |
| `YGG_BOT_PORT` | `8100` | Bind port |
| `YGG_BOT_ALLOW_REMOTE` | `0` | Allow non-local clients |
| `YGG_BOT_NODE_ID` | `{hostname}-{random}` | Node identifier |
| `YGG_BOT_HOME` | `~/.bot/{user_key}` | Bot data directory |
| `YGG_BOT_LOG_RETENTION_DAYS` | `7` | Days to keep log files |
| `YGG_BOT_MAX_CMD_TIMEOUT` | `300` | Max shell command timeout (s) |
| `YGG_BOT_MAX_PYTHON_TIMEOUT` | `600` | Max Python execution timeout (s) |

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
| Message burst send | 67,000 msg/s |
| Arrow IPC stream write (10K rows) | 2.4 GB/s |
| Arrow IPC stream read (10K rows) | 21 GB/s |
| `/api/call` scalar return | 3.3 ms |
| `/api/call` tabular 10K rows | 17 ms |
| `/api/messenger` send | 3.1 ms |
| Polars DataFrame round-trip | 749 MB/s |
