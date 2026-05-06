# Yggdrasil Power Query Connector

Connect **Excel** and **Power BI** to the Yggdrasil FastAPI service (`yggdrasil.fastapi.routers.excel`). Run Python on the server, get a typed Power Query table back. Includes a server-side cache for repeated refreshes.

| File | For | How to use |
|---|---|---|
| `YggdrasilExcel.pq` | Excel | Paste into Advanced Editor — no install |
| `Yggdrasil.mez`     | Power BI Desktop | Custom connector via Get Data |

---

## Functions exposed

| Function | What it does |
|---|---|
| `Yggdrasil.Execute` | Run Python code, return a DataFrame as a Power Query table |
| `Yggdrasil.Prepare` | Run Python code with server-side parquet caching |
| `Yggdrasil.DatabricksSQL` | Execute SQL on Databricks, return a typed table (server-side cached) |
| `Yggdrasil.Health` | Check service health |
| `Yggdrasil.Contents` | Navigation table entry point (Power BI only) |

---

## Step 0 — start the FastAPI service

```bash
pip install "ygg[api]"
python -m yggdrasil.fastapi.main     # or: ygg-api
```

By default the service listens on `http://127.0.0.1:8000`. Override with the env vars below.

| Env var | Default | Description |
|---|---|---|
| `YGG_FASTAPI_HOST`         | `127.0.0.1`   | API host |
| `YGG_FASTAPI_PORT`         | `8000`        | API port |
| `YGG_FASTAPI_API_PREFIX`   | `/api`        | API route prefix |
| `YGG_FASTAPI_PYTHON_PREFIX`| `/python`     | Python router prefix |
| `YGG_FASTAPI_EXCEL_PREFIX` | `/excel`      | Excel router prefix |
| `YGG_FASTAPI_DATABRICKS_CACHE_MAX_SIZE`    | `128` | Max cached Databricks SQL results |
| `YGG_FASTAPI_DATABRICKS_CACHE_DEFAULT_TTL` | `300` | Default TTL (seconds) |

---

## Excel — no install

1. **Data > Get Data > From Other Sources > Blank Query**
2. **Advanced Editor**, paste `YggdrasilExcel.pq`
3. Replace the trailing `in Yggdrasil` with one of the patterns below.

### Run a Python snippet

```powerquery-m
in Yggdrasil[Execute](
    "import pyarrow as pa#(lf)df = pa.table({'x': [1,2,3], 'y': ['a','b','c']})",
    [Packages = {"pyarrow"}, DfName = "df"]
)
```

### Reuse Yggdrasil as a saved query

Save the module as a query named `Yggdrasil`, then reference it:

```powerquery-m
let
    Ygg    = Yggdrasil,
    Result = Ygg[Execute](
        "import pyarrow as pa#(lf)df = pa.table({'x': list(range(100))})",
        [Packages = {"pyarrow"}, MaxRows = 10]
    )
in
    Result
```

### Cached snippet (`Prepare`)

```powerquery-m
let
    Ygg    = Yggdrasil,
    Source = Ygg[Prepare](
        "import polars as pl#(lf)df = pl.DataFrame({'a': [1,2,3]})",
        [Packages = {"polars"}, ForceRefresh = false]
    )
in
    Source
```

### Databricks SQL from Excel

Server-side cached so refreshing your sheet doesn't re-hit your warehouse:

```powerquery-m
in Yggdrasil[DatabricksSQL](
    "SELECT * FROM main.default.my_table LIMIT 100"
)
```

With explicit connection + cache options:

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

---

## Power BI Desktop — install the `.mez`

### One-line install (PowerShell)

```powershell
# From a local clone
.\install.ps1 -Target PowerBI

# Or directly from GitHub (no clone needed)
.\install.ps1 -Source GitHub -Target PowerBI
```

| Parameter | Values | Default | Description |
|---|---|---|---|
| `-Source` | `Local`, `GitHub` | `Local` | Where to get the connector sources |
| `-Target` | `PowerBI`, `Excel`, `Both` | `Both` | Which app to install for |
| `-Branch` | any branch | `main` | GitHub branch (only with `-Source GitHub`) |
| `-Repo`   | `owner/repo` | `Platob/Yggdrasil` | GitHub repo (only with `-Source GitHub`) |

### Manual install

1. `.\build.ps1` to produce `Yggdrasil.mez`.
2. Copy to `%USERPROFILE%\Documents\Power BI Desktop\Custom Connectors\`.
3. **File > Options > Security > Data Extensions** → *Allow any extension to load without validation or warning*.
4. Restart Power BI. Connector appears under **Get Data > Other > Yggdrasil (Beta)**.

### Power BI M usage

Run a Python snippet:

```powerquery-m
let
    Source = Yggdrasil.Execute(
        "import pyarrow as pa#(lf)df = pa.table({'x': [1,2,3], 'y': ['a','b','c']})",
        [Packages = {"pyarrow"}, DfName = "df"]
    )
in
    Source
```

Cached snippet (returns a manifest record):

```powerquery-m
let
    Source = Yggdrasil.Prepare(
        "import pyarrow as pa#(lf)df = pa.table({'x': [1,2,3]})",
        [Packages = {"pyarrow"}, ForceRefresh = false]
    )
in
    Source
```

Databricks SQL → typed table (server-side cached):

```powerquery-m
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

---

## Troubleshooting

- **Refresh hangs / 503**: confirm the FastAPI service is running on the expected port (`Yggdrasil.Health(...)` returns the build info).
- **`No module named 'pyarrow'`**: list every package you import in the `Packages = {...}` option so the server installs them lazily.
- **Databricks 401/403**: ensure `Host` + `Token` are set (option, env, or `~/.databrickscfg` profile) and the warehouse exists.
- **Stale results in Power BI**: pass `ForceRefresh = true` on the next refresh, or shorten `CacheTtl`.

---

## License

Same as the parent Yggdrasil repository — see [`../LICENSE`](../LICENSE).
