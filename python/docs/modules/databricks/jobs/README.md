# yggdrasil.databricks.jobs

Notebook/job configuration helpers for Databricks workflows.

## When to use
- You want to map widget/job parameters into dataclasses.
- You need a consistent, typed way to parse notebook inputs.

## Core APIs
- `NotebookConfig` is a dataclass base that reads values from Databricks widgets or job parameters.
- `WidgetType` enumerates supported widget types for Databricks widgets.

```python
from yggdrasil.databricks.jobs import NotebookConfig

class DemoConfig(NotebookConfig):
    run_id: str

config = DemoConfig.from_environment()
```

## Use cases
### Parse widget inputs into a typed config
```python
from yggdrasil.databricks.jobs import NotebookConfig

class IngestConfig(NotebookConfig):
    source: str
    target_table: str

config = IngestConfig.from_environment()
```

### Access job parameter defaults
```python
from yggdrasil.databricks.jobs import NotebookConfig

class JobConfig(NotebookConfig):
    run_id: str = "manual"

config = JobConfig.from_environment()
```

## Related modules
- [yggdrasil.databricks.workspaces](../workspaces/README.md) for workspace helpers that supply SDK clients.
