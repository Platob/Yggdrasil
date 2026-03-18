/**
 * Yggdrasil.query.pq — Sample queries for testing in Power Query SDK.
 *
 * Open this file in the Power Query SDK to test the connector interactively.
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
//         "import pandas as pd#(lf)df = pd.DataFrame({'id': [1,2,3], 'value': [10.5, 20.3, 30.1]})",
//         [Packages = {"pandas"}, DfName = "df"]
//     )
// in
//     Result;

// ---- Test 3: Execute with max rows ----
// let
//     Result = Yggdrasil.Execute(
//         "import pandas as pd#(lf)df = pd.DataFrame({'x': range(1000)})",
//         [Packages = {"pandas"}, MaxRows = 10]
//     )
// in
//     Result;

// ---- Test 4: Prepare (cached) ----
// let
//     Result = Yggdrasil.Prepare(
//         "import pandas as pd#(lf)df = pd.DataFrame({'a': [1,2,3], 'b': ['x','y','z']})",
//         [Packages = {"pandas"}, ForceRefresh = false]
//     )
// in
//     Result;

// ---- Test 5: Navigation table ----
// let
//     Nav = Yggdrasil.Contents()
// in
//     Nav;

