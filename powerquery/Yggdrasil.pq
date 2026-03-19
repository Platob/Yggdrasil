/**
 * Yggdrasil Power Query Connector
 *
 * Custom data connector for Power BI Desktop that communicates with the
 * Yggdrasil FastAPI service (yggdrasil.fastapi.routers.excel).
 *
 * Endpoints consumed:
 *   POST /api/python/excel/execute        - real-time JSON DataFrame
 *   POST /api/python/excel/prepare        - cached parquet execution
 *   POST /api/python/excel/databricks/sql - Databricks SQL query (cached)
 *   GET  /api/system/healthz              - service health check
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

Yggdrasil.BuildUrl = (baseUrl as text, path as text) as text =>
    Text.TrimEnd(baseUrl, "/") & path;

Yggdrasil.PostJson = (url as text, body as record) as any =>
    let
        JsonBody    = Json.FromValue(body),
        RawResponse = Web.Contents(
            url,
            [
                Content = JsonBody,
                Headers = [
                    #"Content-Type" = "application/json",
                    #"Accept"       = "application/json"
                ],
                ManualStatusHandling = {400, 403, 404, 500}
            ]
        ),
        Buffered = Binary.Buffer(RawResponse),
        Meta     = Value.Metadata(RawResponse),
        Status   = Meta[Response.Status],
        Parsed   = Json.Document(Buffered),
        Result   = if Status >= 400 then
                       error Error.Record(
                           "Yggdrasil.APIError",
                           "API returned HTTP " & Number.ToText(Status),
                           Parsed
                       )
                   else
                       Parsed
    in
        Result;

Yggdrasil.GetJson = (url as text) as any =>
    let
        RawResponse = Web.Contents(
            url,
            [
                Headers = [#"Accept" = "application/json"],
                ManualStatusHandling = {400, 403, 404, 500}
            ]
        ),
        Buffered = Binary.Buffer(RawResponse),
        Meta     = Value.Metadata(RawResponse),
        Status   = Meta[Response.Status],
        Parsed   = Json.Document(Buffered),
        Result   = if Status >= 400 then
                       error Error.Record(
                           "Yggdrasil.APIError",
                           "API returned HTTP " & Number.ToText(Status),
                           Parsed
                       )
                   else
                       Parsed
    in
        Result;

// ---------------------------------------------------------------------------
// Schema helpers
// ---------------------------------------------------------------------------

Yggdrasil.MapDtype = (dtype as text) as type =>
    let
        L = Text.Lower(dtype),
        T = // Arrow types
            if L = "string" or L = "utf8" or L = "large_string" or L = "large_utf8"
                                           then Text.Type        else
            if L = "int8" or L = "int16" or L = "int32" or L = "int64"
                                           then Int64.Type       else
            if L = "uint8" or L = "uint16" or L = "uint32" or L = "uint64"
                                           then Int64.Type       else
            if L = "float" or L = "float16" or L = "float32" or L = "double" or L = "float64" or L = "halffloat"
                                           then Double.Type      else
            if L = "bool" or L = "boolean" then Logical.Type     else
            if L = "date32" or L = "date32[day]"
                                           then Date.Type        else
            if Text.StartsWith(L, "timestamp")
                                           then DateTimeZone.Type else
            if Text.StartsWith(L, "date")  then Date.Type        else
            if Text.StartsWith(L, "time")  then Time.Type        else
            if Text.StartsWith(L, "duration")
                                           then Duration.Type    else
            if L = "binary" or L = "large_binary"
                                           then Binary.Type      else
            if L = "null"                  then Any.Type         else
            if Text.StartsWith(L, "decimal")
                                           then Number.Type      else
            // pandas fallbacks
            if Text.StartsWith(L, "int")       then Int64.Type       else
            if Text.StartsWith(L, "uint")      then Int64.Type       else
            if Text.StartsWith(L, "float")     then Double.Type      else
            if Text.StartsWith(L, "bool")      then Logical.Type     else
            if Text.StartsWith(L, "datetime")  then DateTimeZone.Type else
            if L = "object"                    then Text.Type        else
            if Text.StartsWith(L, "str")       then Text.Type        else
            if Text.StartsWith(L, "category")  then Text.Type        else
            Any.Type
    in
        T;

Yggdrasil.RowsToTable = (rows as list, schemaList as list, columns as list) as table =>
    let
        Raw       = Table.FromRecords(rows, columns, MissingField.UseNull),
        TypePairs = List.Transform(schemaList, each {[name], Yggdrasil.MapDtype([dtype])}),
        Typed     = Table.TransformColumnTypes(Raw, TypePairs)
    in
        Typed;

// ---------------------------------------------------------------------------
// Yggdrasil.Health
// ---------------------------------------------------------------------------

shared Yggdrasil.Health = Value.ReplaceType(
    Yggdrasil.HealthImpl,
    type function (
        optional baseUrl as (type text meta [
            Documentation.FieldCaption = "Base URL",
            Documentation.SampleValues = {"http://127.0.0.1:8000"}
        ])
    ) as record meta [
        Documentation.Name = "Yggdrasil.Health",
        Documentation.LongDescription = "Check the Yggdrasil service health."
    ]
);

Yggdrasil.HealthImpl = (optional baseUrl as text) as record =>
    let
        Base = if baseUrl <> null then baseUrl else DefaultBaseUrl,
        Url  = Yggdrasil.BuildUrl(Base, ApiPrefix & SystemPath & "/healthz")
    in
        Yggdrasil.GetJson(Url);

// ---------------------------------------------------------------------------
// Yggdrasil.Execute
// ---------------------------------------------------------------------------

shared Yggdrasil.Execute = Value.ReplaceType(
    Yggdrasil.ExecuteImpl,
    type function (
        code as (type text meta [
            Documentation.FieldCaption     = "Python Code",
            Documentation.FieldDescription = "Python code that produces a pyarrow.Table or pandas DataFrame named 'df'",
            Documentation.SampleValues     = {"import pyarrow as pa#(lf)df = pa.table({'x': [1,2,3]})"}
        ]),
        optional options as (type record meta [
            Documentation.FieldCaption = "Options"
        ]),
        optional baseUrl as (type text meta [
            Documentation.FieldCaption = "Base URL",
            Documentation.SampleValues = {"http://127.0.0.1:8000"}
        ])
    ) as table meta [
        Documentation.Name = "Yggdrasil.Execute",
        Documentation.LongDescription = "Execute Python code and return the DataFrame as a table."
    ]
);

Yggdrasil.ExecuteImpl = (code as text, optional options as record, optional baseUrl as text) as table =>
    let
        Base = if baseUrl <> null then baseUrl else DefaultBaseUrl,
        Url  = Yggdrasil.BuildUrl(Base, ApiPrefix & ExcelPath & "/execute"),
        Opts = if options <> null then options else [dummy = null],

        Body = [
            code              = code,
            identifier        = Record.FieldOrDefault(Opts, "Identifier",     "current"),
            prefer_uv         = Record.FieldOrDefault(Opts, "PreferUv",        true),
            create_if_missing = Record.FieldOrDefault(Opts, "CreateIfMissing", false),
            packages          = Record.FieldOrDefault(Opts, "Packages",        {}),
            df_name           = Record.FieldOrDefault(Opts, "DfName",          "df"),
            auto_install      = Record.FieldOrDefault(Opts, "AutoInstall",     false),
            raise_error       = true,
            include_result    = Record.FieldOrDefault(Opts, "IncludeResult",   false),
            max_rows          = Record.FieldOrDefault(Opts, "MaxRows",         null)
        ],

        Response    = Yggdrasil.PostJson(Url, Body),
        DataPayload = Response[data],
        ResultTable = Yggdrasil.RowsToTable(DataPayload[rows], DataPayload[schema], DataPayload[columns])
    in
        ResultTable;

// ---------------------------------------------------------------------------
// Yggdrasil.Prepare
// ---------------------------------------------------------------------------

shared Yggdrasil.Prepare = Value.ReplaceType(
    Yggdrasil.PrepareImpl,
    type function (
        code as (type text meta [
            Documentation.FieldCaption     = "Python Code",
            Documentation.FieldDescription = "Python code that produces a pyarrow.Table or pandas DataFrame named 'df'",
            Documentation.SampleValues     = {"import pyarrow as pa#(lf)df = pa.table({'x': [1,2,3]})"}
        ]),
        optional options as (type record meta [
            Documentation.FieldCaption = "Options"
        ]),
        optional baseUrl as (type text meta [
            Documentation.FieldCaption = "Base URL",
            Documentation.SampleValues = {"http://127.0.0.1:8000"}
        ])
    ) as record meta [
        Documentation.Name = "Yggdrasil.Prepare",
        Documentation.LongDescription = "Execute Python code with server-side caching. Returns a manifest record."
    ]
);

Yggdrasil.PrepareImpl = (code as text, optional options as record, optional baseUrl as text) as record =>
    let
        Base = if baseUrl <> null then baseUrl else DefaultBaseUrl,
        Url  = Yggdrasil.BuildUrl(Base, ApiPrefix & ExcelPath & "/prepare"),
        Opts = if options <> null then options else [dummy = null],

        Body = [
            code              = code,
            identifier        = Record.FieldOrDefault(Opts, "Identifier",     "default"),
            prefer_uv         = Record.FieldOrDefault(Opts, "PreferUv",        true),
            create_if_missing = Record.FieldOrDefault(Opts, "CreateIfMissing", false),
            packages          = Record.FieldOrDefault(Opts, "Packages",        {}),
            df_name           = Record.FieldOrDefault(Opts, "DfName",          "df"),
            force_refresh     = Record.FieldOrDefault(Opts, "ForceRefresh",    false),
            max_rows          = Record.FieldOrDefault(Opts, "MaxRows",         null)
        ],

        Response = Yggdrasil.PostJson(Url, Body)
    in
        Response;

// ---------------------------------------------------------------------------
// Yggdrasil.DatabricksSQL
// ---------------------------------------------------------------------------

shared Yggdrasil.DatabricksSQL = Value.ReplaceType(
    Yggdrasil.DatabricksSQLImpl,
    type function (
        statement as (type text meta [
            Documentation.FieldCaption     = "SQL Statement",
            Documentation.FieldDescription = "SQL query to execute on Databricks",
            Documentation.SampleValues     = {"SELECT * FROM main.default.my_table LIMIT 100"}
        ]),
        optional options as (type record meta [
            Documentation.FieldCaption = "Options"
        ]),
        optional baseUrl as (type text meta [
            Documentation.FieldCaption = "Base URL",
            Documentation.SampleValues = {"http://127.0.0.1:8000"}
        ])
    ) as table meta [
        Documentation.Name = "Yggdrasil.DatabricksSQL",
        Documentation.LongDescription = "Execute a SQL query on Databricks and return the result as a table. Results are cached server-side to avoid redundant calls."
    ]
);

Yggdrasil.DatabricksSQLImpl = (statement as text, optional options as record, optional baseUrl as text) as table =>
    let
        Base = if baseUrl <> null then baseUrl else DefaultBaseUrl,
        Url  = Yggdrasil.BuildUrl(Base, ApiPrefix & ExcelPath & "/databricks/sql"),
        Opts = if options <> null then options else [dummy = null],

        Body = [
            statement      = statement,
            host           = Record.FieldOrDefault(Opts, "Host",          null),
            token          = Record.FieldOrDefault(Opts, "Token",         null),
            catalog_name   = Record.FieldOrDefault(Opts, "CatalogName",   null),
            schema_name    = Record.FieldOrDefault(Opts, "SchemaName",    null),
            warehouse_id   = Record.FieldOrDefault(Opts, "WarehouseId",   null),
            warehouse_name = Record.FieldOrDefault(Opts, "WarehouseName", null),
            df_name        = Record.FieldOrDefault(Opts, "DfName",        "df"),
            max_rows       = Record.FieldOrDefault(Opts, "MaxRows",       null),
            cache_ttl      = Record.FieldOrDefault(Opts, "CacheTtl",      null),
            force_refresh  = Record.FieldOrDefault(Opts, "ForceRefresh",  false)
        ],

        Response    = Yggdrasil.PostJson(Url, Body),
        DataPayload = Response[data],
        ResultTable = Yggdrasil.RowsToTable(DataPayload[rows], DataPayload[schema], DataPayload[columns])
    in
        ResultTable;

// ---------------------------------------------------------------------------
// Navigation table  (Get Data entry point)
// ---------------------------------------------------------------------------

shared Yggdrasil.Contents = Value.ReplaceType(
    Yggdrasil.ContentsImpl,
    type function (
        optional baseUrl as (type text meta [
            Documentation.FieldCaption = "Base URL",
            Documentation.SampleValues = {"http://127.0.0.1:8000"}
        ])
    ) as table meta [
        Documentation.Name = "Yggdrasil",
        Documentation.LongDescription = "Connect to a Yggdrasil Python execution service."
    ]
);

Yggdrasil.ContentsImpl = (optional baseUrl as text) as table =>
    let
        Objects = #table(
            {"Name",              "Key",      "Data",                       "ItemKind", "ItemName", "IsLeaf"},
            {
                {"Execute Python",   "Execute",       Yggdrasil.Execute,       "Function", "Function", true},
                {"Prepare (Cached)", "Prepare",       Yggdrasil.Prepare,       "Function", "Function", true},
                {"Databricks SQL",   "DatabricksSQL", Yggdrasil.DatabricksSQL, "Function", "Function", true},
                {"Service Health",   "Health",         Yggdrasil.Health(baseUrl), "Record", "Record",   true}
            }
        ),
        NavTable = Table.ToNavigationTable(Objects, {"Key"}, "Name", "Data", "ItemKind", "ItemName", "IsLeaf")
    in
        NavTable;

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
// Data Source Kind  (the record name MUST match the section name)
// ---------------------------------------------------------------------------

Yggdrasil = [
    TestConnection = (dataSourcePath) =>
        let
            base = DefaultBaseUrl
        in
            { "Yggdrasil.Contents", base },
    Authentication = [
        Implicit = []
    ],
    Label = "Yggdrasil Python Connector"
];

// ---------------------------------------------------------------------------
// Publish metadata  (Power BI Get Data dialog)
// ---------------------------------------------------------------------------

Yggdrasil.Publish = [
    Beta            = true,
    Category        = "Other",
    ButtonText      = { "Yggdrasil", "Connect to a Yggdrasil Python execution service" },
    LearnMoreUrl    = "https://github.com/Platob/Yggdrasil",
    SourceImage     = Yggdrasil.Icons,
    SourceTypeImage = Yggdrasil.Icons
];

Yggdrasil.Icons = [
    Icon16 = Extension.Contents("Yggdrasil16.png"),
    Icon20 = Extension.Contents("Yggdrasil20.png"),
    Icon24 = Extension.Contents("Yggdrasil24.png"),
    Icon32 = Extension.Contents("Yggdrasil32.png")
];

