# yggdrasil.databricks.jobs

Typed job/notebook parameter parsing with `NotebookConfig`.

## Exports

```python
from yggdrasil.databricks.jobs import NotebookConfig, WidgetType
```

## Example

```python
from dataclasses import dataclass
from yggdrasil.databricks.jobs import NotebookConfig

@dataclass
class IngestConfig(NotebookConfig):
    catalog: str
    schema: str
    dry_run: bool = False

cfg = IngestConfig.from_environment()
print(cfg)
```
