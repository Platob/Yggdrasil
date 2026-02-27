# yggdrasil.databricks.jobs

`NotebookConfig` — typed dataclass for Databricks notebook widgets and job parameters.

Define fields once; values are read from widgets (interactive notebooks), job parameters (scheduled jobs), or environment variables — and cast to the correct Python types automatically.

## Key export

```python
from yggdrasil.databricks.jobs import NotebookConfig
```

---

## Bootstrap: basic typed config

```python
from dataclasses import dataclass
from yggdrasil.databricks.jobs import NotebookConfig

@dataclass
class IngestConfig(NotebookConfig):
    catalog: str
    schema: str
    source_path: str
    dry_run: bool = False

cfg = IngestConfig.from_environment()
print(cfg)
# IngestConfig(catalog='main', schema='analytics', source_path='...', dry_run=False)
```

---

## Bootstrap: dates and lists

```python
from dataclasses import dataclass
import datetime as dt
from yggdrasil.databricks.jobs import NotebookConfig

@dataclass
class ReportConfig(NotebookConfig):
    run_date: dt.date          # parsed from "2024-01-15"
    markets: list[str]         # parsed from "US,FR,DE" (comma-separated widget)
    lookback_days: int = 7

cfg = ReportConfig.from_environment()
print(cfg.run_date)    # datetime.date(2024, 1, 15)
print(cfg.markets)     # ['US', 'FR', 'DE']
```

---

## Bootstrap: enum for controlled choices

```python
from dataclasses import dataclass
from enum import Enum
from yggdrasil.databricks.jobs import NotebookConfig

class RunMode(Enum):
    FULL = "full"
    INCREMENTAL = "incremental"

@dataclass
class PipelineConfig(NotebookConfig):
    mode: RunMode
    target_table: str

cfg = PipelineConfig.from_environment()
if cfg.mode == RunMode.FULL:
    rebuild_table(cfg.target_table)
```

---

## Bootstrap: initialize notebook widgets

Call at the top of a notebook cell to create widgets for interactive use:

```python
from dataclasses import dataclass
from yggdrasil.databricks.jobs import NotebookConfig

@dataclass
class FeatureConfig(NotebookConfig):
    lookback_days: int = 7
    include_debug: bool = False
    env: str = "prod"

# Creates text/dropdown widgets in the notebook UI
FeatureConfig.init_widgets()
```

Widgets are skipped if they already exist (`skip_existing=True` by default).

---

## Bootstrap: production entrypoint pattern

`init_job()` = `init_widgets()` + Spark session tuning + `from_environment()` in one call:

```python
from dataclasses import dataclass
from yggdrasil.databricks.jobs import NotebookConfig
from yggdrasil.databricks.sql import SQLEngine

@dataclass
class JobConfig(NotebookConfig):
    catalog: str
    schema: str
    table: str
    days_back: int = 1

cfg = JobConfig.init_job()   # recommended for scheduled Databricks jobs

engine = SQLEngine(catalog_name=cfg.catalog, schema_name=cfg.schema)
engine.execute(f"""
    DELETE FROM {cfg.catalog}.{cfg.schema}.{cfg.table}
    WHERE event_date >= date_sub(current_date(), {cfg.days_back})
""")
```

---

## `NotebookConfig` class methods

| Method | When to use |
|---|---|
| `MyConfig.from_environment()` | Load from widgets / job params / env vars |
| `MyConfig.init_widgets()` | Create notebook widgets (interactive notebooks) |
| `MyConfig.init_job()` | Full job bootstrap: widgets + Spark tuning + load |

## Widget type mapping

| Field type | Widget created |
|---|---|
| `str`, `int`, `float` | Text widget |
| `bool` | Dropdown: `true` / `false` |
| `Enum` subclass | Dropdown with enum values |
| `list[str]` / `set[str]` | Multiselect |
| `datetime.date` / `datetime.datetime` | Text (ISO format) |
