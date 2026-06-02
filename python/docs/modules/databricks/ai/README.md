# yggdrasil.databricks.ai

Databricks AI services, reached through `client.ai`:

- **`client.ai.vector_search`** — Vector Search endpoints and indexes.
- **`client.ai.serving`** — Model Serving for LLMs, Mosaic AI agents,
  foundation models, external models, and classic ML.

(Genie conversational analytics lives one level up at
[`client.genie`](../genie/README.md).)

## One-liner

```python
from yggdrasil.databricks import DatabricksClient

ai = DatabricksClient().ai

ai.vector_search.endpoint("my-endpoint").state          # Vector Search
ai.serving.endpoint("databricks-claude-sonnet-4").chat("Hi!").text  # Model Serving
```

---

# Model serving

`client.ai.serving` is a `ModelServing` service that fronts LLMs, agents,
foundation models, external models, and classic ML behind stable
endpoints. The defaults lean **maximal**: scale-to-zero on, AI Gateway
usage tracking on, inference-table payload capture on (when a
catalog/schema resolves) — a bare `serve_openai(...)` lands a fully
governed endpoint.

## Query a foundation model

Built-in, pay-per-token foundation models need no create step — just
query them by name:

```python
serving = DatabricksClient().ai.serving

# Chat (llm/v1/chat)
r = serving.endpoint("databricks-meta-llama-3-3-70b-instruct").chat(
    "Summarise the CAP theorem in one sentence.",
    max_tokens=128, temperature=0.0,
)
print(r.text)            # assistant text of the first choice
print(r.message)         # {"role": "assistant", "content": "…"}
print(r.usage)           # {"prompt_tokens": …, "total_tokens": …}

# A system + user turn
serving.endpoint("databricks-claude-sonnet-4").chat([
    {"role": "system", "content": "Answer with digits only."},
    {"role": "user",   "content": "What is 21 + 21?"},
]).text

# Completions (llm/v1/completions) and embeddings (llm/v1/embeddings)
serving.endpoint("my-completions").complete("Once upon", max_tokens=10).text
emb = serving.endpoint("databricks-gte-large-en").embed(["hello", "world"])
emb.embeddings   # [[…], […]]   one vector per input
```

## Serve an external LLM

External models put a Databricks endpoint in front of OpenAI, Anthropic,
Bedrock, Cohere, or Vertex. API keys travel by **secret reference**
(`{{secrets/scope/key}}`) — never plaintext.

```python
serving = DatabricksClient().ai.serving

# OpenAI
serving.endpoint("gpt-4o").serve_openai(
    "gpt-4o", api_key_secret="llm/openai_key", wait=True,
)

# Anthropic
serving.endpoint("claude").serve_anthropic(
    "claude-3-5-sonnet", api_key_secret="llm/anthropic_key", wait=True,
)

print(serving.endpoint("gpt-4o").chat("Hello!").text)
```

## Serve a Unity Catalog model or agent

```python
# A custom model or Mosaic AI agent registered in Unity Catalog
serving.endpoint("rag-agent").serve_uc_model("main.agents.rag", 3, wait=True)
```

## Build the served entities by hand

`Served` builds `ServedEntityInput`s for full control (multi-entity
traffic splits, provisioned throughput, workload sizing):

```python
from yggdrasil.databricks.ai import Served

serving.endpoint("multi").create(
    served_entities=[
        Served.uc_model("main.ml.model", 5, workload_size="Medium"),
        Served.openai("gpt-4o", api_key_secret="llm/openai_key"),
    ],
    wait=True,
)
```

`Served` covers `uc_model`, `openai`, `anthropic`, `amazon_bedrock`,
`cohere`, `google_vertex`, and a generic `external`.

## Lifecycle & ops

```python
ep = serving.endpoint("gpt-4o")

ep.exists()
ep.is_ready                 # True when state.ready == READY
ep.state                    # "NOT_UPDATING" / "IN_PROGRESS" / …
ep.endpoint_url
ep.served_entity_names
ep.wait_ready(wait=True)

ep.update_config(served_entities=[Served.openai("gpt-4o-mini", api_key_secret="llm/k")])
ep.add_tags({"env": "prod"})
ep.delete(missing_ok=True)

# Observability
ep.logs()        # model-server logs for the first served entity
ep.build_logs()  # container build logs
ep.metrics()     # Prometheus export
ep.openapi()     # OpenAPI schema

for ep in serving.list_endpoints():
    print(ep.name)
```

## Serving defaults

```python
from dataclasses import replace

serving.defaults = replace(
    serving.defaults,
    workload_size="Medium",
    scale_to_zero_enabled=True,
    enable_usage_tracking=True,
    enable_inference_table=True,
    inference_table_catalog="main",
    inference_table_schema="serving_logs",
    rate_limit_calls=600,       # per minute, via AI Gateway
)
```

---

# Vector Search

## Vector Search service

```python
from yggdrasil.databricks import DatabricksClient

client = DatabricksClient()
vs     = client.ai.vector_search   # VectorSearch service
```

## Endpoints

```python
# List all endpoints
for ep in vs.endpoints():
    print(ep.name, ep.state, ep.num_indexes)

# Look up one endpoint
ep = vs.endpoint("my-endpoint")

# Metadata
print(ep.name)
print(ep.state)          # "ONLINE" / "PROVISIONING" / ...
print(ep.is_online)
print(ep.endpoint_type)  # "STANDARD"
print(ep.num_indexes)
print(ep.explore_url)    # Databricks UI link

# Create an endpoint and wait until online
ep = vs.endpoint("my-endpoint")
ep.create(endpoint_type="STANDARD", missing_ok=True)
ep.wait_online()

# Delete
ep.delete(missing_ok=True)
```

## Indexes — Delta Sync

Delta Sync indexes stay automatically in sync with a Unity Catalog Delta table. The index is rebuilt incrementally as the source table changes.

```python
from yggdrasil.databricks import DatabricksClient

client = DatabricksClient()
vs     = client.ai.vector_search
ep     = vs.endpoint("my-endpoint")

# Create a Delta Sync index
idx = ep.index("main.ml.products_idx")
idx.create_delta_sync(
    source_table="main.ml.products",           # Delta table with embeddings column
    primary_key="product_id",
    embedding_vector_column="embedding",
    embedding_dimension=1536,
    index_type="DELTA_SYNC",
    pipeline_type="TRIGGERED",                 # or "CONTINUOUS"
    missing_ok=True,
)
idx.wait_online()
print(idx.indexed_row_count)
```

## Indexes — Direct Access

Direct Access indexes let you push embeddings yourself without a source Delta table:

```python
import pyarrow as pa

idx = ep.index("main.ml.articles_idx")
idx.create_direct_access(
    primary_key="article_id",
    embedding_dimension=768,
    schema=pa.schema([
        pa.field("article_id", pa.int64()),
        pa.field("title",      pa.string()),
        pa.field("embedding",  pa.list_(pa.float32(), 768)),
    ]),
    missing_ok=True,
)

# Upsert rows
import numpy as np
rows = [
    {"article_id": 1, "title": "AI breakthrough", "embedding": np.random.rand(768).tolist()},
    {"article_id": 2, "title": "Market news",     "embedding": np.random.rand(768).tolist()},
]
idx.upsert(rows)
```

## Query an index

```python
from yggdrasil.databricks import DatabricksClient
import numpy as np

client = DatabricksClient()
vs     = client.ai.vector_search
idx    = vs.endpoint("my-endpoint").index("main.ml.products_idx")

# Similarity search
query_embedding = np.random.rand(1536).tolist()
result = idx.similarity_search(
    columns=["product_id", "name", "score"],
    query_vector=query_embedding,
    num_results=10,
)

# result is a VectorSearchQueryResult
for row in result.to_pylist():
    print(row["product_id"], row["name"], row["score"])

# Arrow output
arrow_table = result.to_arrow_table()
```

## Inspect and manage an index

```python
idx = vs.endpoint("my-endpoint").index("main.ml.products_idx")

print(idx.name)
print(idx.exists())
print(idx.is_ready)
print(idx.indexed_row_count)
print(idx.primary_key)
print(idx.index_type)       # "DELTA_SYNC" or "DIRECT_ACCESS"
print(idx.source_table)     # set for Delta Sync indexes
print(idx.explore_url)

# Refresh / wait until ready
idx.refresh()
idx.wait_online()

# Delete
idx.delete(missing_ok=True)
```

## Default configuration

Set defaults once for all subsequent calls:

```python
from yggdrasil.databricks.ai import VectorSearchDefaults

vs.defaults = VectorSearchDefaults(
    endpoint_name="my-endpoint",
    pipeline_type="TRIGGERED",
    embedding_dimension=1536,
)
# All subsequent calls inherit these unless overridden inline
```

---

## API reference

- Model Serving:
  [`service`](../../../reference/yggdrasil/databricks/ai/serving/service.md) ·
  [`resources`](../../../reference/yggdrasil/databricks/ai/serving/resources.md)
- Vector Search:
  [`service`](../../../reference/yggdrasil/databricks/ai/vector_search/service.md) ·
  [`resources`](../../../reference/yggdrasil/databricks/ai/vector_search/resources.md)
- Genie lives at [`client.genie`](../genie/README.md).
