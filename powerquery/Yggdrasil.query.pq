/**
 * Yggdrasil.query.pq - Test queries
 *
 * For Power BI (with .mez installed):
 *   Uncomment any Test block below.
 *
 * For Excel (no .mez):
 *   Use YggdrasilExcel.pq instead - paste it into the Advanced Editor.
 *
 * The Yggdrasil FastAPI service must be running on localhost:8000.
 */

// ---- Test 1: Health check ----
let
    Health = Yggdrasil.Health()
in
    Health;

// ---- Test 2: Execute Python code ----
// let
//     Result = Yggdrasil.Execute(
//         "import pyarrow as pa#(lf)df = pa.table({'id': [1,2,3], 'value': [10.5, 20.3, 30.1]})",
//         [Packages = {"pyarrow"}, DfName = "df"]
//     )
// in
//     Result;

// ---- Test 3: Execute with max rows ----
// let
//     Result = Yggdrasil.Execute(
//         "import pyarrow as pa#(lf)df = pa.table({'x': list(range(1000))})",
//         [Packages = {"pyarrow"}, MaxRows = 10]
//     )
// in
//     Result;

// ---- Test 4: Prepare (cached) ----
// let
//     Result = Yggdrasil.Prepare(
//         "import pyarrow as pa#(lf)df = pa.table({'a': [1,2,3], 'b': ['x','y','z']})",
//         [Packages = {"pyarrow"}, ForceRefresh = false]
//     )
// in
//     Result;

// ---- Test 5: Databricks SQL ----
// let
//     Result = Yggdrasil.DatabricksSQL(
//         "SELECT * FROM main.default.my_table LIMIT 100",
//         [CacheTtl = 300, ForceRefresh = false]
//     )
// in
//     Result;

// ---- Test 6: Databricks SQL with connection options ----
// let
//     Result = Yggdrasil.DatabricksSQL(
//         "SELECT id, name, amount FROM main.analytics.transactions LIMIT 50",
//         [
//             CatalogName   = "main",
//             SchemaName    = "analytics",
//             WarehouseName = "Starter Warehouse",
//             MaxRows       = 50,
//             CacheTtl      = 600,
//             ForceRefresh  = false
//         ]
//     )
// in
//     Result;

// ---- Test 7: Navigation table ----
// let
//     Nav = Yggdrasil.Contents()
// in
//     Nav;

