# Node Server

The node server is the core of Yggdrasil — a FastAPI application that provides
distributed function execution, environment management, and peer discovery.

## Quick Start

```bash
pip install "yggdrasil[node]"
ygg node serve          # starts node + frontend
ygg node serve --no-front  # node API only
```

## Architecture

```
node/
├── routers/         Thin HTTP handlers
├── services/        Business logic + state (thread-safe)
├── schemas/         Pydantic StrictModel contracts
├── fn.py            @function decorator framework
├── path.py          NodePath — pathlib-like remote/local files
├── geo.py           IP geolocation
├── ids.py           Int64 ID generation (xxhash)
└── middleware.py    Response caching
```

## API Endpoints

### Discovery
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/hello` | Node info with geolocation |
| POST | `/api/hello` | Register peer, get peers list |
| GET | `/api/hello/peers` | All known peers |

### Functions
| Method | Path | Description |
|--------|------|-------------|
| GET/POST | `/api/function` | List / create-or-update |
| GET/PUT/DELETE | `/api/function/{id}` | CRUD |
| POST | `/api/function/{id}/run` | Trigger execution |
| GET | `/api/function/{id}/run` | List runs |
| POST | `/api/function/{id}/clone` | Clone function |

### Environments
| Method | Path | Description |
|--------|------|-------------|
| GET/POST | `/api/environment` | List / create-or-update |
| GET/PUT/DELETE | `/api/environment/{id}` | CRUD |
| POST | `/api/environment/{id}/install` | Install packages |
| POST | `/api/environment/{id}/clone` | Clone environment |

### Runs
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/run` | List all runs |
| GET | `/api/run/{id}` | Run details |
| GET | `/api/run/{id}/logs` | SSE log stream |
| DELETE | `/api/run/{id}` | Delete run |

### DAGs
| Method | Path | Description |
|--------|------|-------------|
| GET/POST | `/api/dag` | List / create-or-update |
| GET/DELETE | `/api/dag/{id}` | CRUD |
| POST | `/api/dag/{id}/run` | Execute pipeline |
| GET | `/api/dag/{id}/run` | List DAG runs |

### Filesystem
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/fs/ls` | Directory listing |
| GET | `/api/fs/read` | Read file |
| POST | `/api/fs/write` | Write file |
| GET | `/api/fs/stream` | Stream download |
| DELETE | `/api/fs/delete` | Delete file/dir |

### Monitor
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/monitor` | Resource snapshot |
| GET | `/api/monitor/stream` | SSE metrics stream |

## @function Decorator

```python
from yggdrasil.node import function, dag

@function
def process(data: list) -> dict:
    import pandas as pd
    return {"count": len(data)}

# Execute — returns FunctionRun (Future-like)
run = process([1, 2, 3])
result = run.wait()           # block until done
result = run.wait(wait=10)    # 10s timeout

# Target specific environment or node
run = process.with_env("ml-env")([1, 2, 3])
run = process.on("http://node-2:8100")([1, 2, 3])

# DAG chaining
pipeline = dag("etl", extract >> transform >> load)
run = pipeline()
```

## NodePath

```python
from yggdrasil.node import NodePath

# Local files
p = NodePath("data/input.csv")
content = p.read_text()
p.write_text("hello")

# Remote files (via HTTP)
remote = NodePath("data/input.csv", node_url="http://node-2:8100")
content = remote.read_bytes()

# Copy between nodes
local.copy_to(remote)

# Streaming for large files
for chunk in p.stream_read(chunk_size=65536):
    process(chunk)
```

## ID System

All resource IDs are int64 using xxhash:

```python
from yggdrasil.node.ids import make_id
func_id = make_id("my_function")  # xxh32(key) << 32 | timestamp_ms
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `YGG_NODE_PORT` | 8100 | Server port |
| `YGG_NODE_HOST` | 0.0.0.0 | Bind address |
| `YGG_NODE_HOME` | ~/.ygg/{key} | Data directory |
| `YGG_NODE_FRONT_PORT` | 3000 | Frontend port |
| `YGG_NODE_FRONT_HOME` | next/ygg | Frontend directory |
