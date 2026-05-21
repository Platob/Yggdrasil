# yggdrasil.databricks.ai

Databricks AI services — Vector Search endpoints and indexes. Model serving and model registry support is planned.

## One-liner

```python
from yggdrasil.databricks import DatabricksClient

vs = DatabricksClient().ai.vector_search
ep = vs.endpoint("my-endpoint")
print(ep.state, ep.num_indexes)
```

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
ep.create(endpoint_type="STANDARD", if_not_exists=True)
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
    if_not_exists=True,
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
    if_not_exists=True,
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
print(idx.exists)
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
