/**
 * Yggdrasil Power Query Connector (Power BI Desktop / Excel .mez)
 *
 * Talks to a Yggdrasil **node** (default http://127.0.0.1:8100) through the
 * node's Excel service (`/api/v2/excel`). Tabular results are returned as
 * Parquet and decoded natively with `Parquet.Document`, so types survive
 * the round trip with no hand-rolled schema mapping.
 *
 * Endpoints consumed:
 *   GET  /api/v2/excel/info      - node identity + capabilities
 *   POST /api/v2/excel/python    - run a snippet, get the named df as Parquet
 *   GET  /api/v2/excel/fs/read   - read a file as a typed Parquet table
 *   GET  /api/v2/excel/fs/tree   - walk the filesystem (navigation table)
 */

section Yggdrasil;

DefaultBaseUrl = "http://127.0.0.1:8100";
ApiPrefix      = "api/v2/excel";

// ---------------------------------------------------------------------------
// HTTP helpers
// ---------------------------------------------------------------------------

Yggdrasil.GetJson = (base as text, relPath as text, optional query as record) as any =>
    let
        Raw = Web.Contents(base, [
            RelativePath = relPath,
            Query = if query <> null then query else [],
            Headers = [#"Accept" = "application/json"],
            ManualStatusHandling = {400, 403, 404, 500}
        ]),
        Buffered = Binary.Buffer(Raw),
        Status = Value.Metadata(Raw)[Response.Status],
        Parsed = Json.Document(Buffered)
    in
        if Status >= 400
        then error Error.Record("Yggdrasil.APIError", "HTTP " & Number.ToText(Status), Parsed)
        else Parsed;

Yggdrasil.GetParquet = (base as text, relPath as text, optional query as record) as table =>
    let
        Q = Record.AddField(if query <> null then query else [], "format", "parquet"),
        Raw = Web.Contents(base, [RelativePath = relPath, Query = Q])
    in
        Parquet.Document(Raw);

Yggdrasil.PostParquet = (base as text, relPath as text, body as record) as table =>
    let
        Raw = Web.Contents(base, [
            RelativePath = relPath,
            Query = [format = "parquet"],
            Content = Json.FromValue(body),
            Headers = [#"Content-Type" = "application/json"]
        ])
    in
        Parquet.Document(Raw);

BaseOr = (baseUrl as nullable text) as text =>
    Text.TrimEnd(if baseUrl <> null then baseUrl else DefaultBaseUrl, "/");

// ---------------------------------------------------------------------------
// Yggdrasil.Info
// ---------------------------------------------------------------------------

shared Yggdrasil.Info = Value.ReplaceType(
    (optional baseUrl as text) as record =>
        Yggdrasil.GetJson(BaseOr(baseUrl), ApiPrefix & "/info"),
    type function (optional baseUrl as (type text meta [
        Documentation.FieldCaption = "Base URL",
        Documentation.SampleValues = {"http://127.0.0.1:8100"}
    ])) as record meta [
        Documentation.Name = "Yggdrasil.Info",
        Documentation.LongDescription = "Node identity + capability card."
    ]
);

// ---------------------------------------------------------------------------
// Yggdrasil.Execute  (run Python -> table)
// ---------------------------------------------------------------------------

shared Yggdrasil.Execute = Value.ReplaceType(
    Yggdrasil.ExecuteImpl,
    type function (
        code as (type text meta [
            Documentation.FieldCaption     = "Python Code",
            Documentation.FieldDescription = "Snippet that defines a dataframe named 'df' (pandas / polars / pyarrow / list / dict)",
            Documentation.SampleValues     = {"import pandas as pd#(lf)df = pd.DataFrame({'x':[1,2,3]})"}
        ]),
        optional options as (type record meta [Documentation.FieldCaption = "Options"]),
        optional baseUrl as (type text meta [Documentation.FieldCaption = "Base URL"])
    ) as table meta [
        Documentation.Name = "Yggdrasil.Execute",
        Documentation.LongDescription = "Run Python on the node and return the named dataframe as a table."
    ]
);

Yggdrasil.ExecuteImpl = (code as text, optional options as record, optional baseUrl as text) as table =>
    let
        Opts = if options <> null then options else [],
        Body = [
            code     = code,
            env      = Record.FieldOrDefault(Opts, "Env", null),
            df_name  = Record.FieldOrDefault(Opts, "DfName", "df"),
            packages = Record.FieldOrDefault(Opts, "Packages", {}),
            max_rows = Record.FieldOrDefault(Opts, "MaxRows", null),
            timeout  = Record.FieldOrDefault(Opts, "Timeout", null)
        ]
    in
        Yggdrasil.PostParquet(BaseOr(baseUrl), ApiPrefix & "/python", Body);

// ---------------------------------------------------------------------------
// Yggdrasil.ReadFile  (file -> table)
// ---------------------------------------------------------------------------

shared Yggdrasil.ReadFile = Value.ReplaceType(
    Yggdrasil.ReadFileImpl,
    type function (
        path as (type text meta [
            Documentation.FieldCaption = "Path",
            Documentation.SampleValues = {"data/sales.parquet"}
        ]),
        optional options as (type record meta [Documentation.FieldCaption = "Options"]),
        optional baseUrl as (type text meta [Documentation.FieldCaption = "Base URL"])
    ) as table meta [
        Documentation.Name = "Yggdrasil.ReadFile",
        Documentation.LongDescription = "Read a file on the node (parquet/csv/json/arrow) as a typed table."
    ]
);

Yggdrasil.ReadFileImpl = (path as text, optional options as record, optional baseUrl as text) as table =>
    let
        Opts = if options <> null then options else [],
        SF = Record.FieldOrDefault(Opts, "SourceFormat", null),
        Q = if SF <> null then [path = path, source_format = SF] else [path = path]
    in
        Yggdrasil.GetParquet(BaseOr(baseUrl), ApiPrefix & "/fs/read", Q);

// ---------------------------------------------------------------------------
// Yggdrasil.Files  (filesystem navigation table)
// ---------------------------------------------------------------------------

shared Yggdrasil.Files = Value.ReplaceType(
    Yggdrasil.FilesImpl,
    type function (
        optional path as (type text meta [Documentation.FieldCaption = "Path"]),
        optional baseUrl as (type text meta [Documentation.FieldCaption = "Base URL"])
    ) as table meta [
        Documentation.Name = "Yggdrasil.Files",
        Documentation.LongDescription = "Browse the node filesystem; folders expand lazily, files load as tables."
    ]
);

Yggdrasil.FilesImpl = (optional path as text, optional baseUrl as text) as table =>
    let
        Base = BaseOr(baseUrl),
        P = if path <> null then path else "",
        Resp = Yggdrasil.GetJson(Base, ApiPrefix & "/fs/tree", [path = P, depth = "1"]),
        Rows = List.Transform(Resp[tree], (e) =>
            [
                Name = e[name],
                Path = e[path],
                Data = if e[is_dir] then @Yggdrasil.FilesImpl(e[path], Base)
                       else Yggdrasil.ReadFileImpl(e[path], null, Base),
                ItemKind = if e[is_dir] then "Folder" else "Table",
                ItemName = if e[is_dir] then "Folder" else "Table",
                IsLeaf = not e[is_dir]
            ]
        ),
        AsTable = Table.FromRecords(
            Rows,
            type table [Name = text, Path = text, Data = any, ItemKind = text, ItemName = text, IsLeaf = logical]
        )
    in
        Table.ToNavigationTable(AsTable, {"Path"}, "Name", "Data", "ItemKind", "ItemName", "IsLeaf");

// ---------------------------------------------------------------------------
// Yggdrasil.Contents  (Get Data entry point)
// ---------------------------------------------------------------------------

shared Yggdrasil.Contents = Value.ReplaceType(
    Yggdrasil.ContentsImpl,
    type function (optional baseUrl as (type text meta [
        Documentation.FieldCaption = "Base URL",
        Documentation.SampleValues = {"http://127.0.0.1:8100"}
    ])) as table meta [
        Documentation.Name = "Yggdrasil",
        Documentation.LongDescription = "Connect to a Yggdrasil node."
    ]
);

Yggdrasil.ContentsImpl = (optional baseUrl as text) as table =>
    let
        Base = BaseOr(baseUrl),
        Objects = #table(
            {"Name",        "Key",      "Data",                       "ItemKind", "ItemName", "IsLeaf"},
            {
                {"Files",         "Files",    Yggdrasil.FilesImpl(null, Base), "Folder",   "Folder",   false},
                {"Run Python",    "Execute",  Yggdrasil.Execute,               "Function", "Function", true},
                {"Read File",     "ReadFile", Yggdrasil.ReadFile,              "Function", "Function", true},
                {"Node Info",     "Info",     Yggdrasil.Info(Base),            "Record",   "Record",   true}
            }
        )
    in
        Table.ToNavigationTable(Objects, {"Key"}, "Name", "Data", "ItemKind", "ItemName", "IsLeaf");

// ---------------------------------------------------------------------------
// Navigation table helper
// ---------------------------------------------------------------------------

Table.ToNavigationTable = (
    table as table,
    keyColumns as list,
    nameColumn as text,
    dataColumn as text,
    itemKindColumn as text,
    itemNameColumn as text,
    isLeafColumn as text
) as table =>
    let
        tableType = Value.Type(table),
        newTableType = Type.AddTableKey(tableType, keyColumns, true) meta [
            NavigationTable.NameColumn   = nameColumn,
            NavigationTable.DataColumn   = dataColumn,
            NavigationTable.ItemKindColumn = itemKindColumn,
            Preview.DelayColumn          = itemNameColumn,
            NavigationTable.IsLeafColumn = isLeafColumn
        ],
        navigationTable = Value.ReplaceType(table, newTableType)
    in
        navigationTable;

// ---------------------------------------------------------------------------
// Data Source Kind  (record name MUST match the section name)
// ---------------------------------------------------------------------------

Yggdrasil = [
    TestConnection = (dataSourcePath) => { "Yggdrasil.Contents", DefaultBaseUrl },
    Authentication = [ Implicit = [] ],
    Label = "Yggdrasil Node Connector"
];

Yggdrasil.Publish = [
    Beta            = true,
    Category        = "Other",
    ButtonText      = { "Yggdrasil", "Connect to a Yggdrasil node — files, Python, remote data" },
    LearnMoreUrl    = "https://github.com/Platob/Yggdrasil",
    SourceImage     = Yggdrasil.Icons,
    SourceTypeImage = Yggdrasil.Icons
];

Yggdrasil.Icons = [
    Icon16 = { Extension.Contents("Yggdrasil16.png") },
    Icon20 = { Extension.Contents("Yggdrasil20.png") },
    Icon24 = { Extension.Contents("Yggdrasil24.png") },
    Icon32 = { Extension.Contents("Yggdrasil32.png") }
];
