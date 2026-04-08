# yggdrasil.databricks.jobs

Typed Databricks Jobs parameter parsing with `NotebookConfig`.

## Recommended one-liner

```python
from yggdrasil.databricks.jobs import NotebookConfig

cfg = NotebookConfig.from_environment()
```

## Feature examples

```python
from dataclasses import dataclass
from yggdrasil.databricks.jobs import NotebookConfig
```

- Parse widget/job parameters into a typed object: `cfg = MyConfig.from_environment()`
- Auto-register widgets for local notebook execution: `cfg = MyConfig.init_widgets()`
- Initialize from Databricks job runtime: `cfg = MyConfig.init_job()`
- Access `dbutils` helper when available: `dbutils = MyConfig.get_dbutils()`

Full typed example:

```python
@dataclass
class IngestConfig(NotebookConfig):
    catalog: str
    schema: str
    table: str
    dry_run: bool = False

cfg = IngestConfig.from_environment()
print(cfg.catalog, cfg.schema, cfg.table, cfg.dry_run)
```

## Extended example: typed widgets + defaults

```python
from dataclasses import dataclass
from yggdrasil.databricks.jobs import NotebookConfig

@dataclass
class TrainConfig(NotebookConfig):
    catalog: str = "main"
    schema: str = "ml"
    model_name: str = "churn_v1"
    max_rows: int = 50000
    dry_run: bool = True

cfg = TrainConfig.init_widgets()  # local notebooks
# cfg = TrainConfig.from_environment()  # jobs
print(cfg)
```
