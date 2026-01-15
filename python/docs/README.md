# Yggdrasil Python documentation

Welcome to the Python documentation hub for Yggdrasil. This directory curates entry points to the core packages in `python/src/yggdrasil` and highlights the most common workflows (type-aware casting, Arrow schema inference, Databricks helpers, and utility tooling).

## Table of contents
- [Module overview](modules.md) – high-level map of the subpackages.
- [Module index](modules/README.md) – direct links to per-module docs.
- [Developer templates](developer-templates.md) – common snippets for everyday use.
- [Python utilities](pyutils.md) – retry, parallelism, environment helpers.

## Prerequisites
- Python **3.10+**
- Install from the `python/` directory:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e .
```

Optional dependencies you may want to add:
- **PySpark** for Spark casting and Databricks runtime integrations.
- **msal** for Azure Active Directory client credential sessions.
- **zstandard** for optional compression in callable serialization.

## Quick start
```python
from yggdrasil import yggdataclass
from yggdrasil.types import convert, arrow_field_from_hint

@yggdataclass
class Order:
    order_id: int
    total: float

order = Order.__safe_init__("100", "42.5")
assert order.order_id == 100

arrow_field = arrow_field_from_hint(Order)
normalized = convert({"order_id": "7", "total": "9.5"}, Order)
```

## Where to go next
- Start with [yggdrasil.dataclasses](modules/dataclasses/README.md) if you want Arrow-aware dataclass helpers.
- Use [yggdrasil.types](modules/types/README.md) for casting, Arrow inference, and schema-aware conversions.
- Explore [yggdrasil.databricks](modules/databricks/README.md) for workspace, SQL, and cluster helpers.
