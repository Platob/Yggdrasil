/**
 * Yggdrasil for Excel — Power Query M module (no install)
 *
 * Connects Excel to a Yggdrasil **node** (default http://127.0.0.1:8100)
 * via the node's Excel service (`/api/v2/excel`). Results come back as
 * Parquet and are decoded natively with `Parquet.Document`, so columns
 * land fully typed — no hand-rolled schema mapping.
 *
 * Usage:
 *   1. Excel: Data > Get Data > From Other Sources > Blank Query
 *   2. Advanced Editor — paste this whole file
 *   3. Replace the trailing `in Yggdrasil` with a call, e.g.:
 *
 *        in Yggdrasil[Execute](
 *            "import pandas as pd#(lf)df = pd.DataFrame({'x':[1,2,3]})",
 *            [Packages = {"pandas"}]
 *        )
 *
 *   Or save this as a query named `Yggdrasil` and reference Yggdrasil[...]
 *   from other queries.
 *
 * Functions:
 *   Yggdrasil[Info](optional baseUrl)               -> record   (node card)
 *   Yggdrasil[Execute](code, optional opts, optional baseUrl) -> table
 *   Yggdrasil[ReadFile](path, optional opts, optional baseUrl) -> table
 *   Yggdrasil[Files](optional path, optional baseUrl)         -> navtable
 */
let
    // -- Configuration --------------------------------------------------
    DefaultBaseUrl = "http://127.0.0.1:8100",
    ApiPrefix      = "api/v2/excel",

    BaseOr = (baseUrl as nullable text) as text =>
        Text.TrimEnd(if baseUrl <> null then baseUrl else DefaultBaseUrl, "/"),

    // -- HTTP helpers ---------------------------------------------------
    // GET JSON with explicit error surfacing.
    GetJson = (base as text, relPath as text, optional query as record) as any =>
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
            else Parsed,

    // GET a Parquet body and decode it into a typed table.
    GetParquet = (base as text, relPath as text, optional query as record) as table =>
        let
            Q = Record.AddField(if query <> null then query else [], "format", "parquet"),
            Raw = Web.Contents(base, [RelativePath = relPath, Query = Q]),
            Tbl = Parquet.Document(Raw)
        in
            Tbl,

    // POST a JSON body, get a Parquet body back, decode to a table.
    PostParquet = (base as text, relPath as text, body as record) as table =>
        let
            Raw = Web.Contents(base, [
                RelativePath = relPath,
                Query = [format = "parquet"],
                Content = Json.FromValue(body),
                Headers = [#"Content-Type" = "application/json"]
            ]),
            Tbl = Parquet.Document(Raw)
        in
            Tbl,

    // -- Navigation-table helper (standard M idiom) ---------------------
    ToNav = (table as table) as table =>
        let
            tableType = Value.Type(table),
            newType = Type.AddTableKey(tableType, {"Path"}, true) meta [
                NavigationTable.NameColumn = "Name",
                NavigationTable.DataColumn = "Data",
                NavigationTable.ItemKindColumn = "ItemKind",
                Preview.DelayColumn = "Data",
                NavigationTable.IsLeafColumn = "IsLeaf"
            ]
        in
            Value.ReplaceType(table, newType),

    // =====================================================================
    // Public functions
    // =====================================================================

    /** Node identity + capability card. */
    Yggdrasil.Info = (optional baseUrl as text) as record =>
        GetJson(BaseOr(baseUrl), ApiPrefix & "/info"),

    /**
     * Run a Python snippet on the node; return the named dataframe.
     * Options: [Env, DfName, Packages, MaxRows, Timeout]
     */
    Yggdrasil.Execute = (code as text, optional options as record, optional baseUrl as text) as table =>
        let
            Opts = if options <> null then options else [],
            Body = [
                code      = code,
                env       = Record.FieldOrDefault(Opts, "Env", null),
                df_name   = Record.FieldOrDefault(Opts, "DfName", "df"),
                packages  = Record.FieldOrDefault(Opts, "Packages", {}),
                max_rows  = Record.FieldOrDefault(Opts, "MaxRows", null),
                timeout   = Record.FieldOrDefault(Opts, "Timeout", null),
                env_vars  = Record.FieldOrDefault(Opts, "EnvVars", [])
            ]
        in
            PostParquet(BaseOr(baseUrl), ApiPrefix & "/python", Body),

    /**
     * Read a file on the node (parquet/csv/json/arrow) as a typed table.
     * Options: [SourceFormat]  (override parse; default by extension)
     */
    Yggdrasil.ReadFile = (path as text, optional options as record, optional baseUrl as text) as table =>
        let
            Opts = if options <> null then options else [],
            SF = Record.FieldOrDefault(Opts, "SourceFormat", null),
            Q = if SF <> null then [path = path, source_format = SF] else [path = path]
        in
            GetParquet(BaseOr(baseUrl), ApiPrefix & "/fs/read", Q),

    /**
     * Browse the node filesystem as a drill-down navigation table.
     * Folders expand lazily; files load via ReadFile when clicked.
     */
    Yggdrasil.Files = (optional path as text, optional baseUrl as text) as table =>
        let
            Base = BaseOr(baseUrl),
            P = if path <> null then path else "",
            Resp = GetJson(Base, ApiPrefix & "/fs/tree", [path = P, depth = "1"]),
            Entries = Resp[tree],
            Rows = List.Transform(Entries, (e) =>
                [
                    Name = e[name],
                    Path = e[path],
                    Data = if e[is_dir]
                           then Yggdrasil.Files(e[path], Base)
                           else Yggdrasil.ReadFile(e[path], null, Base),
                    ItemKind = if e[is_dir] then "Folder" else "Table",
                    IsLeaf = not e[is_dir]
                ]
            ),
            AsTable = Table.FromRecords(
                Rows,
                type table [Name = text, Path = text, Data = any, ItemKind = text, IsLeaf = logical]
            )
        in
            ToNav(AsTable),

    Yggdrasil = [
        Info     = Yggdrasil.Info,
        Execute  = Yggdrasil.Execute,
        ReadFile = Yggdrasil.ReadFile,
        Files    = Yggdrasil.Files
    ]
in
    Yggdrasil
