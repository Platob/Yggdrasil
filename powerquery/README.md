# Yggdrasil Power Query Connector

Connect **Excel** and **Power BI** to a Yggdrasil **node** and pull data
straight from it — run Python, read remote files, and walk the node's
filesystem. Results come back as **Parquet** and are decoded natively
with Power Query's `Parquet.Document`, so columns arrive fully typed.

| File | For | How to use |
|---|---|---|
| `YggdrasilExcel.pq` | Excel | Paste into the Advanced Editor — no install |
| `Yggdrasil.mez`     | Power BI Desktop | Custom connector via **Get Data** |

> Targets the node Excel service at `/api/v2/excel` (default
> `http://127.0.0.1:8100`). Writing data back to the node is the job of
> the **Office.js add-in** (`nextjs` `/excel`), since Power Query is a
> read/ingest surface.

---

## Functions

| Function | What it does |
|---|---|
| `Yggdrasil.Info` | Node identity + capability card |
| `Yggdrasil.Execute` | Run a Python snippet, return the named dataframe as a table |
| `Yggdrasil.ReadFile` | Read a remote file (parquet/csv/json/arrow) as a typed table |
| `Yggdrasil.Files` | Browse the node filesystem as a drill-down navigation table |
| `Yggdrasil.Contents` | Navigation entry point (Power BI **Get Data**) |

---

## Step 0 — start a node

```bash
pip install "ygg[node]"
ygg node serve         # node API on http://127.0.0.1:8100
```

---

## Excel — no install

1. **Data > Get Data > From Other Sources > Blank Query**
2. **Advanced Editor**, paste `YggdrasilExcel.pq`
3. Replace the trailing `in Yggdrasil` with a call:

### Run a Python snippet → table

```powerquery-m
in Yggdrasil[Execute](
    "import pandas as pd#(lf)df = pd.DataFrame({'x':[1,2,3], 'y':['a','b','c']})",
    [Packages = {"pandas"}]
)
```

### Run inside a named PyEnv, capped rows

```powerquery-m
in Yggdrasil[Execute](
    "import polars as pl#(lf)df = pl.DataFrame({'n': list(range(1000))})",
    [Env = "ml-env", MaxRows = 100]
)
```

### Read a remote file as a typed table

```powerquery-m
in Yggdrasil[ReadFile]("data/sales.parquet")
```

### Walk the remote filesystem

```powerquery-m
in Yggdrasil[Files]()        // drill into folders; files load as tables
```

### Options

`Execute` options record: `Env` (PyEnv name), `DfName` (default `"df"`),
`Packages` (list, lazily `pip`-installed), `MaxRows`, `Timeout`.
`ReadFile` options: `SourceFormat` (override parse; default by extension).

---

## Power BI Desktop — install the `.mez`

```powershell
# from a local clone
.\install.ps1 -Target PowerBI
# or straight from GitHub
.\install.ps1 -Source GitHub -Target PowerBI
```

Manual: `.\build.ps1` → copy `Yggdrasil.mez` to
`%USERPROFILE%\Documents\Power BI Desktop\Custom Connectors\` → enable
**File > Options > Security > Data Extensions → Allow any extension** →
restart. Appears under **Get Data > Other > Yggdrasil (Beta)**.

```powerquery-m
let
    Source = Yggdrasil.Execute(
        "import pandas as pd#(lf)df = pd.DataFrame({'x':[1,2,3]})",
        [Packages = {"pandas"}]
    )
in
    Source
```

---

## Transport

Tabular endpoints accept `?format=parquet|arrow|json`. The connector
always requests `parquet` and reads it with `Parquet.Document` — full
type fidelity, no row-by-row JSON. (The Office.js add-in uses `arrow`,
decoded by apache-arrow in JS.)

---

## Troubleshooting

- **Refresh hangs / connection refused**: confirm the node is up —
  `Yggdrasil.Info()` returns the node card.
- **`No module named X`**: list every imported package in
  `Packages = {...}` so the node installs it before running.
- **`snippet did not define 'df'`**: name your result `df` (or pass
  `DfName`).
- **Cross-machine**: pass the node URL as the last argument, e.g.
  `Yggdrasil[Execute](code, [], "http://other-host:8100")`.

---

## License

Same as the parent Yggdrasil repository — see [`../LICENSE`](../LICENSE).
