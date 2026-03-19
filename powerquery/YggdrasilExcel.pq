/**
 * Yggdrasil for Excel - Power Query M module
 *
 * Paste this entire file into Excel's Power Query Advanced Editor,
 * or load it as a blank query. No .mez installation required.
 *
 * The Yggdrasil FastAPI service must be running (default http://127.0.0.1:8000).
 *
 * Usage:
 *   1. In Excel: Data > Get Data > From Other Sources > Blank Query
 *   2. Click "Advanced Editor"
 *   3. Paste this file (or just the Yggdrasil.Execute call below)
 *   4. Click Done
 *
 * Quick example (replace the let/in block at the bottom):
 *   let
 *       Source = Yggdrasil.Execute(
 *           "import pyarrow as pa#(lf)df = pa.table({'x': [1,2,3], 'y': ['a','b','c']})"
 *       )
 *   in
 *       Source
 */
let
    // -- Configuration --
    DefaultBaseUrl = "http://127.0.0.1:8000",
    ApiPrefix      = "/api",
    ExcelPath      = "/python/excel",
    SystemPath     = "/system",

    // -- HTTP helpers --
    BuildUrl = (baseUrl as text, path as text) as text =>
        Text.TrimEnd(baseUrl, "/") & path,

    PostJson = (url as text, body as record) as any =>
        let
            RawResponse = Web.Contents(
                url,
                [
                    Content = Json.FromValue(body),
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
            Result,

    GetJson = (url as text) as any =>
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
            Result,

    // -- Schema helpers --
    MapDtype = (dtype as text) as type =>
        let
            L = Text.Lower(dtype),
            T = if L = "string" or L = "utf8" or L = "large_string" or L = "large_utf8"
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
            T,

    RowsToTable = (rows as list, schemaList as list, columns as list) as table =>
        let
            Raw       = Table.FromRecords(rows, columns, MissingField.UseNull),
            TypePairs = List.Transform(schemaList, each {[name], MapDtype([dtype])}),
            Typed     = Table.TransformColumnTypes(Raw, TypePairs)
        in
            Typed,

    // =====================================================================
    // Public functions
    // =====================================================================

    /**
     * Yggdrasil.Health - check service health
     */
    Yggdrasil.Health = (optional baseUrl as text) as record =>
        let
            Base = if baseUrl <> null then baseUrl else DefaultBaseUrl,
            Url  = BuildUrl(Base, ApiPrefix & SystemPath & "/healthz")
        in
            GetJson(Url),

    /**
     * Yggdrasil.Execute - run Python code, return DataFrame as table
     *
     * Options (optional record):
     *   Identifier      text     "current"
     *   Packages        list     {}
     *   DfName          text     "df"
     *   MaxRows         number   null
     *   PreferUv        logical  true
     *   CreateIfMissing logical  false
     *   AutoInstall     logical  false
     *   IncludeResult   logical  false
     */
    Yggdrasil.Execute = (code as text, optional options as record, optional baseUrl as text) as table =>
        let
            Base = if baseUrl <> null then baseUrl else DefaultBaseUrl,
            Url  = BuildUrl(Base, ApiPrefix & ExcelPath & "/execute"),
            Opts = if options <> null then options else [_dummy = null],

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

            Response    = PostJson(Url, Body),
            DataPayload = Response[data],
            ResultTable = RowsToTable(DataPayload[rows], DataPayload[schema], DataPayload[columns])
        in
            ResultTable,

    /**
     * Yggdrasil.Prepare - run Python code with server-side caching
     *
     * Options (optional record):
     *   Identifier      text     "default"
     *   Packages        list     {}
     *   DfName          text     "df"
     *   MaxRows         number   null
     *   PreferUv        logical  true
     *   CreateIfMissing logical  false
     *   ForceRefresh    logical  false
     */
    Yggdrasil.Prepare = (code as text, optional options as record, optional baseUrl as text) as record =>
        let
            Base = if baseUrl <> null then baseUrl else DefaultBaseUrl,
            Url  = BuildUrl(Base, ApiPrefix & ExcelPath & "/prepare"),
            Opts = if options <> null then options else [_dummy = null],

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

            Response = PostJson(Url, Body)
        in
            Response,

    /**
     * Yggdrasil.DatabricksSQL - execute SQL on Databricks, return table
     *
     * Options (optional record):
     *   Host           text     null  (Databricks host, null = env default)
     *   Token          text     null  (PAT token, null = env default)
     *   CatalogName    text     null
     *   SchemaName     text     null
     *   WarehouseId    text     null
     *   WarehouseName  text     null
     *   DfName         text     "df"
     *   MaxRows        number   null
     *   CacheTtl       number   null  (seconds, null = server default 300s)
     *   ForceRefresh   logical  false
     */
    Yggdrasil.DatabricksSQL = (statement as text, optional options as record, optional baseUrl as text) as table =>
        let
            Base = if baseUrl <> null then baseUrl else DefaultBaseUrl,
            Url  = BuildUrl(Base, ApiPrefix & ExcelPath & "/databricks/sql"),
            Opts = if options <> null then options else [_dummy = null],

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

            Response    = PostJson(Url, Body),
            DataPayload = Response[data],
            ResultTable = RowsToTable(DataPayload[rows], DataPayload[schema], DataPayload[columns])
        in
            ResultTable,

    // =====================================================================
    // Exported record  (Yggdrasil "namespace")
    // =====================================================================
    Yggdrasil = [
        Execute       = Yggdrasil.Execute,
        Prepare       = Yggdrasil.Prepare,
        DatabricksSQL = Yggdrasil.DatabricksSQL,
        Health        = Yggdrasil.Health
    ]
in
    Yggdrasil

