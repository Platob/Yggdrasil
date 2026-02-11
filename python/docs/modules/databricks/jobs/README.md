# yggdrasil.databricks.jobs

This module helps you build **typed notebook/job configuration contracts**.

Instead of manually reading strings from widgets and converting them in ad hoc code, you define dataclasses once and load values consistently.

---

## Core APIs

- `NotebookConfig`: base dataclass for widget and environment-driven config.
- `WidgetType`: enum describing widget rendering semantics.

---

## Bootstrap: strongly-typed job arguments

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
```

---

## Bootstrap: list and datetime values

```python
from dataclasses import dataclass
import datetime as dt
from yggdrasil.databricks.jobs import NotebookConfig

@dataclass
class ReportConfig(NotebookConfig):
    markets: list[str]
    run_date: dt.date

cfg = ReportConfig.from_environment()
```

---

## Bootstrap: enums for controlled inputs

```python
from dataclasses import dataclass
from enum import Enum
from yggdrasil.databricks.jobs import NotebookConfig

class Mode(Enum):
    FULL = "full"
    INCREMENTAL = "incremental"

@dataclass
class PipelineConfig(NotebookConfig):
    mode: Mode
    target_table: str

cfg = PipelineConfig.from_environment()
```

---

## Bootstrap: generate widgets from dataclass schema

```python
from dataclasses import dataclass
from yggdrasil.databricks.jobs import NotebookConfig

@dataclass
class FeatureConfig(NotebookConfig):
    lookback_days: int = 7
    include_debug: bool = False

# In notebook setup cell (where dbutils is available)
FeatureConfig.init_widgets()
```

---

## Best practices

- Keep one config dataclass per notebook entrypoint.
- Use defaults for safe reruns and local notebook testing.
- Prefer enums for finite-mode choices.
- Validate config once, then pass object through all business functions.
