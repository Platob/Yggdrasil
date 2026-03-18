# Yggdrasil Power Query Connector

Custom Power Query connector for Excel and Power BI that connects to the
**Yggdrasil FastAPI** service's Excel router (`yggdrasil.fastapi.routers.excel`).

## Features

| Function | Description |
|---|---|
| `Yggdrasil.Execute` | Run Python code and return a DataFrame as a Power Query table (real-time JSON) |
| `Yggdrasil.Prepare` | Run Python code with caching — returns manifest + cached parquet metadata |
| `Yggdrasil.Health` | Check service health |
| `Yggdrasil.Contents` | Navigation table entry point (shows up in the **Get Data** dialog) |

## Prerequisites

1. The Yggdrasil FastAPI service must be running:

   ```bash
   python -m yggdrasil.fastapi.main
   ```

   By default it listens on `http://127.0.0.1:8000`.

2. Excel 2016+ or Power BI Desktop.

## Installation

### Quick install (PowerShell)

```powershell
# From a local clone
.\install.ps1

# Or install directly from GitHub (no clone needed)
.\install.ps1 -Source GitHub
```

The `install.ps1` script builds the `.mez` and copies it to the correct
connector folders automatically.

| Parameter | Values | Default | Description |
|---|---|---|---|
| `-Source` | `Local`, `GitHub` | `Local` | Where to get the connector sources |
| `-Target` | `PowerBI`, `Excel`, `Both` | `Both` | Which app to install for |
| `-Branch` | any branch name | `main` | GitHub branch (only with `-Source GitHub`) |
| `-Repo` | `owner/repo` | `Platob/Yggdrasil` | GitHub repo (only with `-Source GitHub`) |

**Examples:**

```powershell
# Install for Power BI only, from local sources
.\install.ps1 -Source Local -Target PowerBI

# Install for Excel only, from GitHub main branch
.\install.ps1 -Source GitHub -Target Excel

# Install from a specific branch
.\install.ps1 -Source GitHub -Branch develop
```

### One-liner install from GitHub (no clone required)

```powershell
irm https://raw.githubusercontent.com/Platob/Yggdrasil/main/powerquery/install.ps1 | iex
```

### Manual install

1. Build the `.mez` file:
   ```powershell
   cd powerquery
   .\build.ps1
   ```
2. Copy `Yggdrasil.mez` to the custom connectors folder:
   - **Power BI Desktop**: `%USERPROFILE%\Documents\Power BI Desktop\Custom Connectors\`
   - **Excel**: `%USERPROFILE%\Documents\Microsoft Power Query\Custom Connectors\`
3. In Power BI Desktop go to *File → Options → Security → Data Extensions* and
   select **Allow any extension to load without validation or warning**.
4. Restart. The connector appears under *Get Data → Other → Yggdrasil (Beta)*.

## Usage in Power Query / M

```powerquery-m
// Real-time execute — returns a table
let
    Source = Yggdrasil.Execute(
        "import pandas as pd#(lf)df = pd.DataFrame({'x': [1,2,3], 'y': ['a','b','c']})",
        [Packages = {"pandas"}, DfName = "df"]
    )
in
    Source

// Cached prepare — returns manifest record
let
    Source = Yggdrasil.Prepare(
        "import pandas as pd#(lf)df = pd.DataFrame({'x': [1,2,3]})",
        [Packages = {"pandas"}, ForceRefresh = false]
    )
in
    Source
```

## Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `YGG_FASTAPI_HOST` | `127.0.0.1` | API host |
| `YGG_FASTAPI_PORT` | `8000` | API port |
| `YGG_FASTAPI_API_PREFIX` | `/api` | API route prefix |
| `YGG_FASTAPI_PYTHON_PREFIX` | `/python` | Python router prefix |
| `YGG_FASTAPI_EXCEL_PREFIX` | `/excel` | Excel router prefix |

The connector defaults to `http://127.0.0.1:8000`. Pass a different base URL to
any function to override.

## License

Same as the parent Yggdrasil repository — see `../LICENSE`.

