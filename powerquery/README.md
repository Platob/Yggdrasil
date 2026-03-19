# Yggdrasil Power Query Connector

Power Query connector for **Excel** and **Power BI** that connects to the
Yggdrasil FastAPI service (`yggdrasil.fastapi.routers.excel`).

Two variants are provided:

| File | For | How |
|---|---|---|
| `YggdrasilExcel.pq` | **Excel** | Paste into the Advanced Editor (no install) |
| `Yggdrasil.mez` | **Power BI Desktop** | Custom connector via Get Data dialog |

## Features

| Function | Description |
|---|---|
| `Yggdrasil.Execute` | Run Python code and return a DataFrame as a Power Query table |
| `Yggdrasil.Prepare` | Run Python code with server-side caching (parquet) |
| `Yggdrasil.DatabricksSQL` | Execute SQL on Databricks and return the result as a table (server-side cached) |
| `Yggdrasil.Health` | Check service health |
| `Yggdrasil.Contents` | Navigation table entry point (Power BI only) |

## Prerequisites

1. The Yggdrasil FastAPI service must be running:

   ```bash
   python -m yggdrasil.fastapi.main
   ```

   By default it listens on `http://127.0.0.1:8000`.

2. Excel 2016+ or Power BI Desktop.

## Excel Usage (no install required)

1. Open Excel, go to **Data > Get Data > From Other Sources > Blank Query**.
2. Click **Advanced Editor**.
3. Paste the contents of `YggdrasilExcel.pq`.
4. Replace the final `in Yggdrasil` with your query, for example:

```powerquery-m
in Yggdrasil[Execute](
    "import pyarrow as pa#(lf)df = pa.table({'x': [1,2,3], 'y': ['a','b','c']})",
    [Packages = {"pyarrow"}, DfName = "df"]
)
```

Or use it as a reusable query: save the module as a query named `Yggdrasil`,
then reference it from other queries:

```powerquery-m
let
    Ygg    = Yggdrasil,  // reference the saved query
    Result = Ygg[Execute](
        "import pyarrow as pa#(lf)df = pa.table({'x': list(range(100))})",
        [Packages = {"pyarrow"}, MaxRows = 10]
    )
in
    Result
```

### Databricks SQL from Excel

Query Databricks directly — results are cached server-side so repeated
refreshes don't flood your warehouse:

```powerquery-m
in Yggdrasil[DatabricksSQL](
    "SELECT * FROM main.default.my_table LIMIT 100"
)
```

With connection and cache options:

```powerquery-m
let
    Ygg = Yggdrasil,
    Result = Ygg[DatabricksSQL](
        "SELECT id, name, amount FROM main.analytics.transactions WHERE amount > 0",
        [
            CatalogName   = "main",
            SchemaName    = "analytics",
            WarehouseName = "Starter Warehouse",
            MaxRows       = 500,
            CacheTtl      = 600,
            ForceRefresh  = false
        ]
    )
in
    Result
```

| Option | Type | Default | Description |
|---|---|---|---|
| `Host` | text | `null` | Databricks workspace URL (env default if null) |
| `Token` | text | `null` | Personal Access Token (env default if null) |
| `CatalogName` | text | `null` | Unity Catalog name |
| `SchemaName` | text | `null` | Schema / database name |
| `WarehouseId` | text | `null` | SQL warehouse ID |
| `WarehouseName` | text | `null` | SQL warehouse name (alternative to ID) |
| `DfName` | text | `"df"` | Payload name |
| `MaxRows` | number | `null` | Limit returned rows |
| `CacheTtl` | number | `null` | Cache lifetime in seconds (`null` → server default 300 s) |
| `ForceRefresh` | logical | `false` | Bypass the cache and re-execute |

## Power BI Installation (.mez)

### Quick install (PowerShell)

```powershell
# From a local clone
.\install.ps1 -Target PowerBI

# Or install directly from GitHub (no clone needed)
.\install.ps1 -Source GitHub -Target PowerBI
```

| Parameter | Values | Default | Description |
|---|---|---|---|
| `-Source` | `Local`, `GitHub` | `Local` | Where to get the connector sources |
| `-Target` | `PowerBI`, `Excel`, `Both` | `Both` | Which app to install for |
| `-Branch` | any branch name | `main` | GitHub branch (only with `-Source GitHub`) |
| `-Repo` | `owner/repo` | `Platob/Yggdrasil` | GitHub repo (only with `-Source GitHub`) |

### Manual install

1. Build: `.\build.ps1`
2. Copy `Yggdrasil.mez` to `%USERPROFILE%\Documents\Power BI Desktop\Custom Connectors\`
3. In Power BI: *File > Options > Security > Data Extensions* >
   **Allow any extension to load without validation or warning**.
4. Restart. The connector appears under *Get Data > Other > Yggdrasil (Beta)*.

### Power BI M usage

```powerquery-m
// Execute - returns a typed table
let
    Source = Yggdrasil.Execute(
        "import pyarrow as pa#(lf)df = pa.table({'x': [1,2,3], 'y': ['a','b','c']})",
        [Packages = {"pyarrow"}, DfName = "df"]
    )
in
    Source

// Prepare (cached) - returns a manifest record
let
    Source = Yggdrasil.Prepare(
        "import pyarrow as pa#(lf)df = pa.table({'x': [1,2,3]})",
        [Packages = {"pyarrow"}, ForceRefresh = false]
    )
in
    Source

// Databricks SQL - query Databricks and return a typed table (server-side cached)
let
    Source = Yggdrasil.DatabricksSQL(
        "SELECT id, name, amount FROM main.analytics.transactions LIMIT 100",
        [
            CatalogName   = "main",
            SchemaName    = "analytics",
            WarehouseName = "Starter Warehouse",
            MaxRows       = 100,
            CacheTtl      = 300,
            ForceRefresh  = false
        ]
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
| `YGG_FASTAPI_DATABRICKS_CACHE_MAX_SIZE` | `128` | Max cached Databricks SQL results in memory |
| `YGG_FASTAPI_DATABRICKS_CACHE_DEFAULT_TTL` | `300` | Default TTL (seconds) for Databricks SQL cache |

The connector defaults to `http://127.0.0.1:8000`. Pass a different base URL to
any function to override.

## License

Same as the parent Yggdrasil repository — see `../LICENSE`.

