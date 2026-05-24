# yggdrasil.databricks.jobs

Typed Databricks Jobs parameter parsing with `NotebookConfig`, plus lower-level job introspection and dependency sniffing utilities.

---

## Surface map

| Symbol | Use for |
|---|---|
| `NotebookConfig` | Typed dataclass for job/widget parameters — auto-parses from environment |
| `read_job_parameters()` | Raw `dict` of all task parameters in the current run |
| `read_widgets(dbutils)` | Read all widget values into a dict |
| `get_dbutils()` | Acquire `dbutils` handle (works in notebooks and Databricks Connect) |
| `sniff_imports(fn)` | AST-walk a function to find all `import` statements |
| `resolve_module_dependency(mod)` | Resolve a module name to its PyPI distribution name |
| `dependencies_to_pip_specs(deps)` | Convert module names to `pip` install specifiers |

---

## One-liner

```python
from yggdrasil.databricks.jobs import NotebookConfig

cfg = NotebookConfig.from_environment()
```

---

## 1) Typed config with defaults

```python
from dataclasses import dataclass, field
from yggdrasil.databricks.jobs import NotebookConfig

@dataclass
class IngestConfig(NotebookConfig):
    catalog: str = "main"
    schema: str = "raw"
    table: str = "events"
    dry_run: bool = False
    batch_size: int = 1000

cfg = IngestConfig.from_environment()
print(cfg.catalog, cfg.schema, cfg.table, cfg.dry_run)
```

`from_environment()` reads widget values in a notebook, job task parameters in a Databricks run, and plain environment variables otherwise — always in that priority order.

---

## 2) Initialise widgets in a notebook

Call `init_widgets()` at the top of a notebook cell to register widgets with their defaults. This makes the notebook runnable interactively with sliders/dropdowns:

```python
from dataclasses import dataclass
from yggdrasil.databricks.jobs import NotebookConfig

@dataclass
class ReportConfig(NotebookConfig):
    catalog: str = "main"
    schema: str = "curated"
    start_date: str = "2026-01-01"
    end_date: str = "2026-05-01"
    geo: bool = False

# Registers text/bool widgets if in a notebook; no-op in jobs
cfg = ReportConfig.init_widgets()
print(cfg.start_date, cfg.end_date)
```

---

## 3) Production job: `from_environment` + DatabricksClient

```python
from dataclasses import dataclass
from yggdrasil.databricks.jobs import NotebookConfig
from yggdrasil.databricks import DatabricksClient

@dataclass
class ETLConfig(NotebookConfig):
    catalog: str = "main"
    schema: str = "raw"
    table: str = "orders"
    mode: str = "append"

def main():
    cfg = ETLConfig.from_environment()
    c = DatabricksClient()
    target = c.sql.table(cfg.catalog, cfg.schema, cfg.table)
    print(f"Running ETL → {target} (mode={cfg.mode})")
    # … ingestion logic

main()
```

---

## 4) Read raw parameters

```python
from yggdrasil.databricks.jobs import read_job_parameters

params = read_job_parameters()
# {'catalog': 'main', 'schema': 'raw', 'table': 'orders', 'dry_run': 'false'}
print(params.get("catalog", "default_catalog"))
```

---

## 5) dbutils access (notebooks + Connect)

```python
from yggdrasil.databricks.jobs import get_dbutils

dbutils = get_dbutils()
if dbutils is not None:
    secret = dbutils.secrets.get(scope="my-scope", key="api-token")
    print("Got secret:", len(secret), "chars")
```

---

## 6) Dependency sniffing for Databricks job task packaging

When you ship a Python callable as a Databricks Job task, the executor auto-walks the function's AST to find imports and converts them to pip specifiers:

```python
from yggdrasil.databricks.jobs.introspect import sniff_imports, dependencies_to_pip_specs

def my_task():
    import polars as pl
    import pyarrow as pa
    from yggdrasil.fxrate import FxRate
    return FxRate().latest([("EUR", "USD")])

imports = sniff_imports(my_task)
# {'polars', 'pyarrow', 'yggdrasil'}

specs = dependencies_to_pip_specs(imports)
# ['polars', 'pyarrow', 'ygg[data,polars]']
```

---

## 7) Complex config: multi-source ingestion job

```python
from dataclasses import dataclass, field as dc_field
from typing import List
from yggdrasil.databricks.jobs import NotebookConfig
from yggdrasil.databricks import DatabricksClient

@dataclass
class MultiSourceConfig(NotebookConfig):
    catalog: str = "main"
    schemas: str = "raw,staging"          # comma-separated
    parallelism: int = 4
    dry_run: bool = False
    notify_on_failure: bool = True

    @property
    def schema_list(self) -> List[str]:
        return [s.strip() for s in self.schemas.split(",")]

def run_multi():
    cfg = MultiSourceConfig.from_environment()
    c = DatabricksClient()
    for schema in cfg.schema_list:
        print(f"Processing {cfg.catalog}.{schema} (dry_run={cfg.dry_run})")
        if not cfg.dry_run:
            c.sql.execute(f"OPTIMIZE {cfg.catalog}.{schema}.events")

run_multi()
```

---

## 8) Widget type hints

`NotebookConfig` maps Python types to Databricks widget types automatically:

| Python type | Widget type |
|---|---|
| `str` | text |
| `int`, `float` | text (numeric) |
| `bool` | checkbox |
| `list[str]` | multiselect |
| `Enum` subclass | dropdown |
