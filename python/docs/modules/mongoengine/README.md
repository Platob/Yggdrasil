# yggdrasil.mongoengine

MongoEngine integration — connection management helpers and a Databricks-aware
`with_mongo_connection` decorator for notebooks and jobs.

## Surface map

| Symbol | Use for |
|---|---|
| `with_mongo_connection` | Decorator that opens/closes a MongoEngine connection around a function |
| `mongoengine.*` | All MongoEngine symbols re-exported (use the guard to stay base-install-safe) |

> **Guard import** — MongoEngine is an optional dependency. Always import
> via the guard so base installs remain functional:
>
> ```python
> from yggdrasil.mongoengine.lib import mongoengine
> ```

---

## 1) One-liner

```python
from yggdrasil.mongoengine import with_mongo_connection

@with_mongo_connection(host="mongodb://localhost:27017", db="mydb")
def list_users():
    return User.objects.count()

print(list_users())
```

---

## 2) Guard import — base-install-safe usage

```python
from yggdrasil.mongoengine.lib import mongoengine   # raises helpful error if missing

class Event(mongoengine.Document):
    name = mongoengine.StringField()
    ts   = mongoengine.DateTimeField()
    meta = {"collection": "events"}
```

---

## 3) `with_mongo_connection` — decorator pattern

Manages the MongoEngine `connect` / `disconnect` lifecycle around a
callable. Works on plain functions, Databricks notebook cells, and job
callables.

```python
from yggdrasil.mongoengine import with_mongo_connection
from yggdrasil.mongoengine.lib import mongoengine

class Order(mongoengine.Document):
    order_id = mongoengine.StringField(primary_key=True)
    amount   = mongoengine.FloatField()
    meta     = {"collection": "orders"}

@with_mongo_connection(host="mongodb://mongo:27017", db="sales")
def get_order(order_id: str) -> dict:
    return Order.objects(order_id=order_id).first().to_mongo().to_dict()

result = get_order("ORD-001")
```

### Multiple aliases (for multi-tenant setups)

```python
from yggdrasil.mongoengine import with_mongo_connection

@with_mongo_connection(
    host="mongodb://mongo:27017",
    db="tenant_a",
    alias="tenant_a",
)
def tenant_a_count():
    return MyDoc.objects.using("tenant_a").count()
```

---

## 4) Databricks-aware routing

When the connection URL is a Databricks secret reference, `with_mongo_connection`
resolves it at runtime using `DatabricksClient`:

```python
from yggdrasil.mongoengine import with_mongo_connection

# The host is fetched from Databricks Secrets at call time
@with_mongo_connection(
    host_secret="mongo/connection-url",   # scope/key
    db="production",
)
def ingest(records: list) -> int:
    from yggdrasil.mongoengine.lib import mongoengine
    class Event(mongoengine.Document):
        meta = {"collection": "events"}
    for r in records:
        Event(**r).save()
    return len(records)
```

---

## 5) Direct MongoEngine usage (no decorator)

```python
from yggdrasil.mongoengine.lib import mongoengine, get_connection_settings
from yggdrasil.io.url import URL

# Parse connection settings from a URL
settings = get_connection_settings(URL.from_str("mongodb://user:pass@host:27017/mydb"))

# Connect manually
mongoengine.connect(**settings)

try:
    class Log(mongoengine.Document):
        msg = mongoengine.StringField()
        meta = {"collection": "logs"}

    Log(msg="hello").save()
    print(Log.objects.count())
finally:
    mongoengine.disconnect()
```

---

## 6) Full pipeline: Databricks SQL → MongoDB

```python
from yggdrasil.databricks import DatabricksClient
from yggdrasil.mongoengine import with_mongo_connection
from yggdrasil.mongoengine.lib import mongoengine

client = DatabricksClient()

class Product(mongoengine.Document):
    product_id = mongoengine.StringField(primary_key=True)
    name       = mongoengine.StringField()
    price      = mongoengine.FloatField()
    meta       = {"collection": "products"}

@with_mongo_connection(host="mongodb://mongo:27017", db="catalog")
def sync_products():
    stmt = client.sql.execute("SELECT product_id, name, price FROM main.catalog.products")
    for row in stmt.to_arrow_table().to_pylist():
        Product.objects(product_id=row["product_id"]).update_one(
            upsert=True,
            set__name=row["name"],
            set__price=row["price"],
        )
    return stmt.num_rows

count = sync_products()
print(f"Synced {count} products to MongoDB")
```
