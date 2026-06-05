/**
 * Yggdrasil.query.pq - Test queries for the .mez connector
 *
 * For Power BI (with .mez installed): uncomment any Test block below.
 * For Excel (no .mez): paste YggdrasilExcel.pq into the Advanced Editor.
 *
 * A Yggdrasil node must be running (default http://127.0.0.1:8100).
 */

// ---- Test 1: Node info ----
let
    Info = Yggdrasil.Info()
in
    Info;

// ---- Test 2: Run Python -> table ----
// let
//     Result = Yggdrasil.Execute(
//         "import pandas as pd#(lf)df = pd.DataFrame({'id':[1,2,3], 'value':[10.5,20.3,30.1]})",
//         [Packages = {"pandas"}, DfName = "df"]
//     )
// in
//     Result;

// ---- Test 3: Run Python with row cap, in a named PyEnv ----
// let
//     Result = Yggdrasil.Execute(
//         "import pyarrow as pa#(lf)df = pa.table({'x': list(range(1000))})",
//         [Env = "ml-env", MaxRows = 10]
//     )
// in
//     Result;

// ---- Test 4: Read a remote file as a typed table ----
// let
//     Result = Yggdrasil.ReadFile("data/sales.parquet")
// in
//     Result;

// ---- Test 5: Walk the remote filesystem (drill-down) ----
// let
//     Nav = Yggdrasil.Files()
// in
//     Nav;

// ---- Test 6: Full navigation entry point ----
// let
//     Nav = Yggdrasil.Contents()
// in
//     Nav;
