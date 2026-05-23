# yggdrasil.cli

Command-line interface tooling for Yggdrasil. Provides a base class for Databricks-aware CLIs and a ready-made conversational Genie agent CLI (`ygg-genie`).

**Install:** `pip install "ygg[databricks]"` — the CLI entry points are registered automatically.

---

## One-liner

```bash
ygg-genie --host https://<workspace> --token <token> --space-id <space-id> "Top 10 customers by revenue"
```

---

## 1) Entry points

| Command | Module | Description |
|---|---|---|
| `ygg-api` | `yggdrasil.fastapi.main` | Start the FastAPI REST service (used by the Power Query connector) |
| `ygg-genie` | `yggdrasil.cli.databricks.genie` | Interactive Genie conversational analytics |

---

## 2) `ygg-genie` — Genie conversational CLI

Runs a one-shot or interactive question against a Databricks Genie space.

### Authentication (same flags as `DatabricksClient`)

```bash
# PAT token
ygg-genie --host https://<workspace> --token <token> \
          --space-id <space-id> "Weekly cloud cost trend"

# OAuth client credentials
ygg-genie --host https://<workspace> \
          --client-id <id> --client-secret <secret> \
          --space-id <space-id> "Break down by team"

# Environment-driven (DATABRICKS_HOST / DATABRICKS_TOKEN)
ygg-genie --space-id <space-id> "Revenue by region this month"

# Profile from ~/.databrickscfg
ygg-genie --profile prod --space-id <space-id> "Top 5 SKUs by margin"
```

### Flags

| Flag | Description |
|---|---|
| `--host` | Databricks workspace URL |
| `--token` | Personal access token |
| `--client-id` / `--client-secret` | OAuth M2M credentials |
| `--profile` | Profile from `~/.databrickscfg` |
| `--space-id` | Genie space ID (required) |
| `--output json` | Output result as JSON (default: pretty-print) |
| `--wait` | Wait for the answer before returning (default: true) |

---

## 3) `DatabricksCLI` — base class for custom CLIs

Extend this to build your own Databricks-aware sub-command:

```python
from yggdrasil.cli.databricks import DatabricksCLI
from yggdrasil.databricks import DatabricksClient
import argparse

class MyServiceCLI(DatabricksCLI):
    """Run MyService operations from the command line."""

    def add_service_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--catalog", default="main", help="UC catalog")
        parser.add_argument("--table", required=True, help="Table to inspect")

    def run(self, client: DatabricksClient, args: argparse.Namespace) -> None:
        tbl = client.catalogs[args.catalog].schema("default").table(args.table)
        print(tbl.exists, tbl.full_name())

if __name__ == "__main__":
    MyServiceCLI().main()
```

Register it as a console script in `pyproject.toml`:

```toml
[project.scripts]
my-service = "my_package.cli:MyServiceCLI.entrypoint"
```

### `DatabricksCLI` interface

```python
from yggdrasil.cli.databricks import DatabricksCLI

class DatabricksCLI:
    def add_service_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Override to add service-specific flags."""
        ...

    def run(self, client: DatabricksClient, args: argparse.Namespace) -> None:
        """Override to implement the command logic."""
        ...

    def main(self, argv: list[str] | None = None) -> None:
        """Parse args, build DatabricksClient, call run()."""
        ...

    @classmethod
    def entrypoint(cls) -> None:
        """Module-level entry point for console_scripts."""
        cls().main()
```

The base class automatically adds `--host`, `--token`, `--client-id`, `--client-secret`, `--profile`, and `--config-file` arguments and constructs a `DatabricksClient` before calling `run()`.

---

## 4) Full pipeline CLI — ETL trigger from the shell

```python
# my_package/cli.py
from dataclasses import dataclass
from yggdrasil.cli.databricks import DatabricksCLI
from yggdrasil.databricks import DatabricksClient
from yggdrasil.data.enums import Mode
import argparse, pyarrow as pa

class IngestCLI(DatabricksCLI):
    """Trigger a data ingest into a Unity Catalog table."""

    def add_service_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--catalog", default="main")
        parser.add_argument("--schema", default="raw")
        parser.add_argument("--table", required=True)
        parser.add_argument("--dry-run", action="store_true")

    def run(self, client: DatabricksClient, args: argparse.Namespace) -> None:
        tbl = client.catalogs[args.catalog][args.schema][args.table]

        if args.dry_run:
            print(f"[dry-run] would ingest into {tbl.full_name()}")
            return

        data = pa.table({"id": [1, 2], "v": ["a", "b"]})
        tbl.write_arrow_table(data, mode=Mode.APPEND)
        print(f"Ingested {len(data)} rows into {tbl.full_name()}")

if __name__ == "__main__":
    IngestCLI().main()
```

```bash
python -m my_package.cli --catalog main --schema raw --table events --dry-run
python -m my_package.cli --catalog main --schema raw --table events
```
