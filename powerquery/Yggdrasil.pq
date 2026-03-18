/**
 * Yggdrasil Power Query Connector
 *
 * Custom data connector for Excel / Power BI that communicates with the
 * Yggdrasil FastAPI service (yggdrasil.fastapi.routers.excel).
 *
 * Endpoints consumed:
 *   POST /api/python/excel/execute   — real-time JSON DataFrame
 *   POST /api/python/excel/prepare   — cached parquet execution
 *   GET  /api/system/healthz         — service health check
 */

section Yggdrasil;

// ---------------------------------------------------------------------------
// Default configuration
// ---------------------------------------------------------------------------

DefaultBaseUrl = "http://127.0.0.1:8000";
ApiPrefix      = "/api";
ExcelPath      = "/python/excel";
SystemPath     = "/system";

// ---------------------------------------------------------------------------
// Shared HTTP helpers
// ---------------------------------------------------------------------------

/**
 * Build a full endpoint URL.
 */
Yggdrasil.BuildUrl = (baseUrl as text, path as text) as text =>
    Text.TrimEnd(baseUrl, "/") & path;

/**
 * POST JSON to the Yggdrasil API and return the parsed response.
 */
Yggdrasil.PostJson = (url as text, body as record) as record =>
    let
        JsonBody   = Json.FromValue(body),
        RawResponse = Web.Contents(
            url,
            [
                Content  = JsonBody,
                Headers  = [
                    #"Content-Type" = "application/json",
                    #"Accept"       = "application/json"
                ],
                ManualStatusHandling = {400, 403, 404, 500}
            ]
        ),
        BufferedResponse = Binary.Buffer(RawResponse),
        Parsed = Json.Document(BufferedResponse),
        Meta   = Value.Metadata(RawResponse),
        Status = Meta[Response.Status],
        Result = if Status >= 400 then
                     error Error.Record(
                         "Yggdrasil.APIError",
                         "API request failed (HTTP " & Number.ToText(Status) & ")",
                         Parsed
                     )
                 else
                     Parsed
    in
        Result;

/**
 * GET from the Yggdrasil API and return the parsed JSON.
 */
Yggdrasil.GetJson = (url as text) as record =>
    let
        RawResponse = Web.Contents(
            url,
            [
                Headers = [
                    #"Accept" = "application/json"
                ],
                ManualStatusHandling = {400, 403, 404, 500}
            ]
        ),
        BufferedResponse = Binary.Buffer(RawResponse),
        Parsed = Json.Document(BufferedResponse),
        Meta   = Value.Metadata(RawResponse),
        Status = Meta[Response.Status],
        Result = if Status >= 400 then
                     error Error.Record(
                         "Yggdrasil.APIError",
                         "API request failed (HTTP " & Number.ToText(Status) & ")",
                         Parsed
                     )
                 else
                     Parsed
    in
        Result;

// ---------------------------------------------------------------------------
// Schema helpers
// ---------------------------------------------------------------------------

/**
 * Map a Python/pandas dtype string to a Power Query M type.
 */
Yggdrasil.MapDtype = (dtype as text) as type =>
    let
        Lower = Text.Lower(dtype),
        Mapped =
            if Text.StartsWith(Lower, "int")    then Int64.Type    else
            if Text.StartsWith(Lower, "uint")   then Int64.Type    else
            if Text.StartsWith(Lower, "float")  then Double.Type   else
            if Text.StartsWith(Lower, "bool")   then Logical.Type  else
            if Text.StartsWith(Lower, "datetime") then DateTimeZone.Type else
            if Text.StartsWith(Lower, "date")   then Date.Type     else
            if Text.StartsWith(Lower, "time")   then Time.Type     else
            if Lower = "object"                  then Text.Type     else
            if Text.StartsWith(Lower, "str")     then Text.Type     else
            if Text.StartsWith(Lower, "category") then Text.Type   else
            Any.Type
    in
        Mapped;

/**
 * Build a typed Table.Type from the schema list returned by the API.
 * Each schema entry is a record with "name" and "dtype" fields.
 */
Yggdrasil.BuildTableType = (schemaList as list) as type =>
    let
        Columns = List.Transform(
            schemaList,
            each {[name], Yggdrasil.MapDtype([dtype])}
        ),
        TableType = type table (Type.ForRecord(
            Record.FromList(
                List.Transform(Columns, each _{1}),
                List.Transform(Columns, each _{0})
            ),
            false
        ))
    in
        TableType;

/**
 * Convert the API rows (list of records) into a typed Power Query table.
 */
Yggdrasil.RowsToTable = (rows as list, schemaList as list, columns as list) as table =>
    let
        RawTable  = Table.FromRecords(rows, columns, MissingField.UseNull),
        // Apply column types from schema
        TypePairs = List.Transform(
            schemaList,
            each {[name], Yggdrasil.MapDtype([dtype])}
        ),
        Typed = Table.TransformColumnTypes(RawTable, TypePairs)
    in
        Typed;

// ---------------------------------------------------------------------------
// Public API functions
// ---------------------------------------------------------------------------

/**
 * Yggdrasil.Health
 *
 * Check the Yggdrasil service health.
 * Returns a record with { ok, env_home, python_executable }.
 */
shared Yggdrasil.Health = Value.ReplaceType(
    Yggdrasil.Health.Impl,
    Yggdrasil.Health.Type
);

Yggdrasil.Health.Type = type function (
    optional baseUrl as (
        type text meta [
            Documentation.FieldCaption     = "Base URL",
            Documentation.FieldDescription = "Yggdrasil API base URL",
            Documentation.SampleValues     = {"http://127.0.0.1:8000"}
        ]
    )
) as record meta [
    Documentation.Name             = "Yggdrasil.Health",
    Documentation.LongDescription  = "Check the Yggdrasil FastAPI service health.",
    Documentation.Examples = {[
        Description = "Default health check",
        Code        = "Yggdrasil.Health()",
        Result      = "[ok = true, env_home = ""..."", python_executable = ""...""]"
    ]}
];

Yggdrasil.Health.Impl = (optional baseUrl as text) as record =>
    let
        Base = if baseUrl <> null then baseUrl else DefaultBaseUrl,
        Url  = Yggdrasil.BuildUrl(Base, ApiPrefix & SystemPath & "/healthz")
    in
        Yggdrasil.GetJson(Url);

// ---------------------------------------------------------------------------

/**
 * Yggdrasil.Execute
 *
 * Execute Python code and return the resulting DataFrame as a Power Query table.
 * Calls POST /api/python/excel/execute.
 *
 * Options record fields:
 *   Identifier       text      "current"  — Python env identifier
 *   Packages         list      {}         — pip packages to ensure
 *   DfName           text      "df"       — name of the DataFrame variable
 *   MaxRows          number    null       — limit returned rows
 *   PreferUv         logical   true       — prefer uv over pip
 *   CreateIfMissing  logical   false      — create env if not found
 *   IncludeResult    logical   false      — include CommandResult in metadata
 *   AutoInstall      logical   false      — auto-install missing imports
 */
shared Yggdrasil.Execute = Value.ReplaceType(
    Yggdrasil.Execute.Impl,
    Yggdrasil.Execute.Type
);

Yggdrasil.Execute.Type = type function (
    code as (
        type text meta [
            Documentation.FieldCaption     = "Python Code",
            Documentation.FieldDescription = "Python source code that produces a pandas DataFrame",
            Documentation.SampleValues     = {"import pandas as pd#(lf)df = pd.DataFrame({'x': [1,2,3]})"}
        ]
    ),
    optional options as (
        type record meta [
            Documentation.FieldCaption     = "Options",
            Documentation.FieldDescription = "Optional settings for execution"
        ]
    ),
    optional baseUrl as (
        type text meta [
            Documentation.FieldCaption     = "Base URL",
            Documentation.FieldDescription = "Yggdrasil API base URL",
            Documentation.SampleValues     = {"http://127.0.0.1:8000"}
        ]
    )
) as table meta [
    Documentation.Name            = "Yggdrasil.Execute",
    Documentation.LongDescription = "Execute Python code via the Yggdrasil FastAPI service and return the resulting pandas DataFrame as a Power Query table.",
    Documentation.Examples = {[
        Description = "Simple DataFrame",
        Code        = "Yggdrasil.Execute(""import pandas as pd#(lf)df = pd.DataFrame({'col': [1,2,3]})"")",
        Result      = "#table({""col""}, {{1},{2},{3}})"
    ]}
];

Yggdrasil.Execute.Impl = (code as text, optional options as record, optional baseUrl as text) as table =>
    let
        Base = if baseUrl <> null then baseUrl else DefaultBaseUrl,
        Url  = Yggdrasil.BuildUrl(Base, ApiPrefix & ExcelPath & "/execute"),
        Opts = if options <> null then options else [],

        Body = Record.Combine({
            [
                code              = code,
                identifier        = Record.FieldOrDefault(Opts, "Identifier",      "current"),
                prefer_uv         = Record.FieldOrDefault(Opts, "PreferUv",         true),
                create_if_missing = Record.FieldOrDefault(Opts, "CreateIfMissing",  false),
                packages          = Record.FieldOrDefault(Opts, "Packages",         {}),
                df_name           = Record.FieldOrDefault(Opts, "DfName",           "df"),
                auto_install      = Record.FieldOrDefault(Opts, "AutoInstall",      false),
                raise_error       = true,
                include_result    = Record.FieldOrDefault(Opts, "IncludeResult",    false)
            ],
            // Conditionally include max_rows only when provided
            if Record.HasFields(Opts, {"MaxRows"}) then
                [max_rows = Record.Field(Opts, "MaxRows")]
            else
                []
        }),

        Response    = Yggdrasil.PostJson(Url, Body),
        DataPayload = Response[data],
        Rows        = DataPayload[rows],
        Columns     = DataPayload[columns],
        Schema      = DataPayload[schema],

        ResultTable = Yggdrasil.RowsToTable(Rows, Schema, Columns)
    in
        ResultTable;

// ---------------------------------------------------------------------------

/**
 * Yggdrasil.Prepare
 *
 * Execute Python code with server-side caching (parquet).
 * Calls POST /api/python/excel/prepare.
 *
 * Options record fields:
 *   Identifier       text      "default"  — Python env identifier
 *   Packages         list      {}         — pip packages to ensure
 *   DfName           text      "df"       — name of the DataFrame variable
 *   MaxRows          number    null       — limit returned rows
 *   PreferUv         logical   true       — prefer uv over pip
 *   ForceRefresh     logical   false      — bypass cache
 */
shared Yggdrasil.Prepare = Value.ReplaceType(
    Yggdrasil.Prepare.Impl,
    Yggdrasil.Prepare.Type
);

Yggdrasil.Prepare.Type = type function (
    code as (
        type text meta [
            Documentation.FieldCaption     = "Python Code",
            Documentation.FieldDescription = "Python source code that produces a pandas DataFrame",
            Documentation.SampleValues     = {"import pandas as pd#(lf)df = pd.DataFrame({'x': [1,2,3]})"}
        ]
    ),
    optional options as (
        type record meta [
            Documentation.FieldCaption     = "Options",
            Documentation.FieldDescription = "Optional settings for preparation"
        ]
    ),
    optional baseUrl as (
        type text meta [
            Documentation.FieldCaption     = "Base URL",
            Documentation.FieldDescription = "Yggdrasil API base URL",
            Documentation.SampleValues     = {"http://127.0.0.1:8000"}
        ]
    )
) as record meta [
    Documentation.Name            = "Yggdrasil.Prepare",
    Documentation.LongDescription = "Execute Python code with server-side parquet caching via the Yggdrasil FastAPI service. Returns a manifest record with cache metadata.",
    Documentation.Examples = {[
        Description = "Prepare with caching",
        Code        = "Yggdrasil.Prepare(""import pandas as pd#(lf)df = pd.DataFrame({'x': [1,2,3]})"")",
        Result      = "[ok = true, data = [cache_hit = false, ...]]"
    ]}
];

Yggdrasil.Prepare.Impl = (code as text, optional options as record, optional baseUrl as text) as record =>
    let
        Base = if baseUrl <> null then baseUrl else DefaultBaseUrl,
        Url  = Yggdrasil.BuildUrl(Base, ApiPrefix & ExcelPath & "/prepare"),
        Opts = if options <> null then options else [],

        Body = Record.Combine({
            [
                code              = code,
                identifier        = Record.FieldOrDefault(Opts, "Identifier",      "default"),
                prefer_uv         = Record.FieldOrDefault(Opts, "PreferUv",         true),
                create_if_missing = Record.FieldOrDefault(Opts, "CreateIfMissing",  false),
                packages          = Record.FieldOrDefault(Opts, "Packages",         {}),
                df_name           = Record.FieldOrDefault(Opts, "DfName",           "df"),
                force_refresh     = Record.FieldOrDefault(Opts, "ForceRefresh",     false)
            ],
            if Record.HasFields(Opts, {"MaxRows"}) then
                [max_rows = Record.Field(Opts, "MaxRows")]
            else
                []
        }),

        Response = Yggdrasil.PostJson(Url, Body)
    in
        Response;

// ---------------------------------------------------------------------------
// Navigation table (Get Data entry point)
// ---------------------------------------------------------------------------

/**
 * Yggdrasil.Contents
 *
 * Navigation table shown in the "Get Data" dialog of Excel / Power BI.
 */
shared Yggdrasil.Contents = Value.ReplaceType(
    Yggdrasil.Contents.Impl,
    Yggdrasil.Contents.Type
);

Yggdrasil.Contents.Type = type function (
    optional baseUrl as (
        type text meta [
            Documentation.FieldCaption     = "Base URL",
            Documentation.FieldDescription = "Yggdrasil API base URL",
            Documentation.SampleValues     = {"http://127.0.0.1:8000"}
        ]
    )
) as table meta [
    Documentation.Name            = "Yggdrasil",
    Documentation.LongDescription = "Connect to a Yggdrasil Python execution service to run Python code and return DataFrames in Excel or Power BI."
];

Yggdrasil.Contents.Impl = (optional baseUrl as text) as table =>
    let
        Objects = #table(
            {"Name",             "Key",      "Data",                                    "ItemKind", "ItemName", "IsLeaf"},
            {
                {"Execute Python",  "Execute",  Yggdrasil.Execute,                        "Function", "Function", true},
                {"Prepare (Cached)","Prepare",  Yggdrasil.Prepare,                        "Function", "Function", true},
                {"Service Health",  "Health",   Yggdrasil.Health(baseUrl),                "Record",   "Record",   true}
            }
        ),
        NavTable = Table.ToNavigationTable(Objects, {"Key"}, "Name", "Data", "ItemKind", "ItemName", "IsLeaf")
    in
        NavTable;

// ---------------------------------------------------------------------------
// Navigation table helper  (standard PQ SDK pattern)
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
            NavigationTable.NameColumn  = nameColumn,
            NavigationTable.DataColumn  = dataColumn,
            NavigationTable.ItemKindColumn = itemKindColumn,
            Preview.DelayColumn         = itemNameColumn,
            NavigationTable.IsLeafColumn = isLeafColumn
        ],
        navigationTable = Value.ReplaceType(table, newTableType)
    in
        navigationTable;

// ---------------------------------------------------------------------------
// Data Source Kind declaration
// ---------------------------------------------------------------------------

Yggdrasil = [
    TestConnection = (dataSourcePath) =>
        let
            json  = Json.Document(dataSourcePath),
            base  = if Record.HasFields(json, {"baseUrl"}) then json[baseUrl] else DefaultBaseUrl
        in
            { "Yggdrasil.Contents", base },
    Authentication = [
        Implicit = []
    ],
    Label = "Yggdrasil Python Connector"
];

// ---------------------------------------------------------------------------
// Data Source UI metadata
// ---------------------------------------------------------------------------

Yggdrasil.Publish = [
    Beta            = true,
    Category        = "Other",
    ButtonText      = { "Yggdrasil", "Connect to a Yggdrasil Python execution service" },
    LearnMoreUrl    = "https://github.com/your-org/Yggdrasil",
    SourceImage     = Yggdrasil.Icons,
    SourceTypeImage = Yggdrasil.Icons
];

Yggdrasil.Icons = [
    Icon16 = Yggdrasil.Icon16,
    Icon20 = Yggdrasil.Icon20,
    Icon24 = Yggdrasil.Icon24,
    Icon32 = Yggdrasil.Icon32
];

// Placeholder 1×1 PNG (replace with real icons)
Yggdrasil.Icon16 = Extension.Contents("Yggdrasil16.png");
Yggdrasil.Icon20 = Extension.Contents("Yggdrasil20.png");
Yggdrasil.Icon24 = Extension.Contents("Yggdrasil24.png");
Yggdrasil.Icon32 = Extension.Contents("Yggdrasil32.png");

