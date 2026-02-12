// Target: .NET Framework 4.8
// NuGet: ParquetSharp + ParquetSharp.Native (x64 recommended)
//
// Matrix is row-major: matrix[row, col]
//
// Write: schema inferred from first non-null value per column (value types become nullable if nulls exist)
// Write: column names come from optional headers[]; otherwise default to c0..cN (and de-duped)
//
// Read: returns object[,] with optional header row at result[0, *] containing Parquet column names (prefer full dot-path)
//
// Read: repeated (list) columns ARE supported:
//   - Each cell becomes List<object> (or null if the list itself is null).
//   - Empty lists become empty List<object>.
//   - Elements are mapped using the same logical/physical type mapping as scalars.
//
// Read (optional): collapseStructs=true packs leaf columns with shared dot-prefix into Dictionary<string,object> per row.
// Example: person.name + person.age => single column "person" holding { "name": ..., "age": ... }
//
// Write (struct-like + list-like): if a cell contains IDictionary, POCO, or IEnumerable (except string/byte[]),
// it is written as JSON string (portable scalar).

using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Web.Script.Serialization;
using ParquetSharp;
using ParquetSharp.IO;

public static class ParquetMatrixIO
{
    // ---------------- Public Read ----------------

    public static object[,] ReadParquetToMatrix(string filePath, bool writeHeaderRow = true, bool collapseStructs = false)
    {
        if (string.IsNullOrWhiteSpace(filePath)) throw new ArgumentNullException(nameof(filePath));

        using (var file = new ParquetFileReader(filePath))
        {
            return ReadParquetToMatrixCore(file, writeHeaderRow, collapseStructs);
        }
    }

    public static object[,] ReadParquetToMatrix(Stream parquetStream, bool writeHeaderRow = true, bool collapseStructs = false)
    {
        if (parquetStream == null) throw new ArgumentNullException(nameof(parquetStream));
        if (!parquetStream.CanRead) throw new ArgumentException("Stream must be readable.", nameof(parquetStream));

        // ManagedRandomAccessFile works across ParquetSharp versions and supports non-seekable streams
        // (internally buffers if needed).
        using (var raf = new ManagedRandomAccessFile(parquetStream))
        using (var file = new ParquetFileReader(raf))
        {
            return ReadParquetToMatrixCore(file, writeHeaderRow, collapseStructs);
        }
    }

    // ---------------- Public Write ----------------

    public static void WriteMatrixToParquet(object[,] matrix, string parquetFilePath, string[] headers = null)
    {
        if (matrix == null) throw new ArgumentNullException(nameof(matrix));
        if (string.IsNullOrWhiteSpace(parquetFilePath)) throw new ArgumentNullException(nameof(parquetFilePath));

        var plan = BuildWritePlan(matrix, headers);

        using (var file = new ParquetFileWriter(parquetFilePath, plan.Columns))
        {
            WriteRowGroup(file, plan);
            file.Close();
        }
    }

    public static void WriteMatrixToParquet(object[,] matrix, Stream stream, string[] headers = null)
    {
        if (matrix == null) throw new ArgumentNullException(nameof(matrix));
        if (stream == null) throw new ArgumentNullException(nameof(stream));
        if (!stream.CanWrite) throw new ArgumentException("Stream must be writable.", nameof(stream));

        var plan = BuildWritePlan(matrix, headers);

        using (var file = new ParquetFileWriter(stream, plan.Columns))
        {
            WriteRowGroup(file, plan);
            file.Close(); // important for stream output
        }
    }

    // ---------------- Core Read ----------------

    private static object[,] ReadParquetToMatrixCore(ParquetFileReader file, bool writeHeaderRow, bool collapseStructs)
    {
        var meta = file.FileMetaData;
        int numRowGroups = meta.NumRowGroups;
        int numCols = meta.Schema.NumColumns;

        int totalRows = GetTotalRows(file, numRowGroups);

        int headerOffset = writeHeaderRow ? 1 : 0;
        var result = new object[totalRows + headerOffset, numCols];

        // Prefer full dot-path headers for nested structs
        string[] headerNames = null;
        if (writeHeaderRow || collapseStructs)
        {
            headerNames = new string[numCols];
            for (int c = 0; c < numCols; c++)
                headerNames[c] = TryGetColumnPath(meta.Schema, c) ?? meta.Schema.Column(c).Name;

            if (writeHeaderRow)
            {
                for (int c = 0; c < numCols; c++)
                    result[0, c] = headerNames[c];
            }
        }
        else if (writeHeaderRow)
        {
            for (int c = 0; c < numCols; c++)
                result[0, c] = meta.Schema.Column(c).Name;
        }

        int rowOffset = 0;

        for (int rg = 0; rg < numRowGroups; rg++)
        {
            using (var rgReader = file.RowGroup(rg))
            {
                int groupRows = checked((int)rgReader.MetaData.NumRows);

                for (int c = 0; c < numCols; c++)
                {
                    using (var colReader = rgReader.Column(c))
                    {
                        var colData = ReadColumnAsObjects(colReader, groupRows);
                        for (int r = 0; r < groupRows; r++)
                        {
                            result[rowOffset + headerOffset + r, c] = colData[r];
                        }
                    }
                }

                rowOffset += groupRows;
            }
        }

        if (collapseStructs)
        {
            if (headerNames == null)
            {
                headerNames = new string[numCols];
                for (int c = 0; c < numCols; c++)
                    headerNames[c] = meta.Schema.Column(c).Name;
            }

            return CollapseStructColumns(result, headerNames, writeHeaderRow);
        }

        return result;
    }

    private static int GetTotalRows(ParquetFileReader file, int numRowGroups)
    {
        long totalRowsLong = 0;

        for (int rg = 0; rg < numRowGroups; rg++)
        {
            using (var rgReader = file.RowGroup(rg))
                totalRowsLong += rgReader.MetaData.NumRows;
        }

        if (totalRowsLong > int.MaxValue)
            throw new InvalidOperationException("Total rows exceed Int32.MaxValue; object[,] cannot hold that many rows.");

        return (int)totalRowsLong;
    }

    private static string TryGetColumnPath(SchemaDescriptor schema, int colIndex)
    {
        // ParquetSharp API surface varies by version; reflection keeps it resilient.
        try
        {
            var col = schema.Column(colIndex);

            // Some versions expose .Path as string[]
            var pathProp = col.GetType().GetProperty("Path", BindingFlags.Public | BindingFlags.Instance);
            if (pathProp != null)
            {
                var path = pathProp.GetValue(col, null) as string[];
                if (path != null && path.Length > 0) return string.Join(".", path);
            }

            // Some versions expose a method returning a dot string
            var dotMethod = col.GetType().GetMethod("PathToDotString", BindingFlags.Public | BindingFlags.Instance);
            if (dotMethod != null)
            {
                var s = dotMethod.Invoke(col, null) as string;
                if (!string.IsNullOrWhiteSpace(s)) return s;
            }
        }
        catch { /* ignore */ }

        return null;
    }

    private static object[] ReadColumnAsObjects(ColumnReader colReader, int numRows)
    {
        var desc = colReader.ColumnDescriptor;

        bool nullable = desc.MaxDefinitionLevel > 0;
        bool repeated = desc.MaxRepetitionLevel > 0; // lists etc.

        if (repeated)
        {
            // Return List<object> per row in the matrix (or null if list itself is null)
            return ReadRepeatedColumnAsObjects(colReader, numRows);
        }

        var phys = desc.PhysicalType;
        var logical = desc.LogicalType;

        // Logical types first (more semantic) — string matching keeps this compatible across ParquetSharp versions.
        if (logical != null)
        {
            var logicalName = logical.GetType().Name; // e.g., TimestampLogicalType, StringLogicalType, DateLogicalType

            // Timestamp -> DateTime
            if (logicalName.IndexOf("Timestamp", StringComparison.OrdinalIgnoreCase) >= 0)
            {
                return nullable
                    ? Box(colReader.LogicalReader<DateTime?>().ReadAll(numRows))
                    : Box(colReader.LogicalReader<DateTime>().ReadAll(numRows));
            }

            // String
            if (logicalName.IndexOf("String", StringComparison.OrdinalIgnoreCase) >= 0)
            {
                return Box(colReader.LogicalReader<string>().ReadAll(numRows));
            }

            // Decimal
            if (logicalName.IndexOf("Decimal", StringComparison.OrdinalIgnoreCase) >= 0)
            {
                return nullable
                    ? Box(colReader.LogicalReader<decimal?>().ReadAll(numRows))
                    : Box(colReader.LogicalReader<decimal>().ReadAll(numRows));
            }

            // UUID (commonly fixed 16 bytes)
            if (logicalName.IndexOf("UUID", StringComparison.OrdinalIgnoreCase) >= 0)
            {
                var bytes = colReader.LogicalReader<byte[]>().ReadAll(numRows);
                var arr = new Guid?[numRows];

                for (int i = 0; i < numRows; i++)
                {
                    var b = bytes[i];
                    if (b == null) { arr[i] = null; continue; }
                    if (b.Length == 16) arr[i] = new Guid(b);
                    else arr[i] = null;
                }

                return Box(arr);
            }

            // DATE (int32 days since epoch) -> DateTime? (midnight)
            if (logicalName.Equals("DateLogicalType", StringComparison.OrdinalIgnoreCase) ||
                (logicalName.IndexOf("Date", StringComparison.OrdinalIgnoreCase) >= 0 && logicalName.IndexOf("Timestamp", StringComparison.OrdinalIgnoreCase) < 0))
            {
                var days = colReader.LogicalReader<int?>().ReadAll(numRows);
                var arr = new DateTime?[numRows];
                var epoch = new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc);

                for (int i = 0; i < numRows; i++)
                {
                    if (!days[i].HasValue) { arr[i] = null; continue; }
                    arr[i] = epoch.AddDays(days[i].Value).Date;
                }

                return Box(arr);
            }

            // TIME -> TimeSpan? (best-effort; unit depends on writer)
            if (logicalName.IndexOf("Time", StringComparison.OrdinalIgnoreCase) >= 0)
            {
                // Try int64 first (micros/nanos), then int32 (millis)
                try
                {
                    var v = colReader.LogicalReader<long?>().ReadAll(numRows);
                    var ts = new TimeSpan?[numRows];

                    for (int i = 0; i < numRows; i++)
                    {
                        if (!v[i].HasValue) { ts[i] = null; continue; }
                        // Most common: micros. Convert micros -> ticks (1 tick = 100ns)
                        ts[i] = TimeSpan.FromTicks(v[i].Value * 10);
                    }

                    return Box(ts);
                }
                catch
                {
                    var v = colReader.LogicalReader<int?>().ReadAll(numRows);
                    var ts = new TimeSpan?[numRows];

                    for (int i = 0; i < numRows; i++)
                    {
                        if (!v[i].HasValue) { ts[i] = null; continue; }
                        ts[i] = TimeSpan.FromMilliseconds(v[i].Value);
                    }

                    return Box(ts);
                }
            }

            // ENUM -> string
            if (logicalName.IndexOf("Enum", StringComparison.OrdinalIgnoreCase) >= 0)
            {
                try { return Box(colReader.LogicalReader<string>().ReadAll(numRows)); }
                catch { /* fall through */ }
            }

            // JSON -> string
            if (logicalName.IndexOf("Json", StringComparison.OrdinalIgnoreCase) >= 0)
            {
                return Box(colReader.LogicalReader<string>().ReadAll(numRows));
            }
        }

        // Physical types fallback
        switch (phys)
        {
            case PhysicalType.Boolean:
                return nullable
                    ? Box(colReader.LogicalReader<bool?>().ReadAll(numRows))
                    : Box(colReader.LogicalReader<bool>().ReadAll(numRows));

            case PhysicalType.Int32:
                return nullable
                    ? Box(colReader.LogicalReader<int?>().ReadAll(numRows))
                    : Box(colReader.LogicalReader<int>().ReadAll(numRows));

            case PhysicalType.Int64:
                return nullable
                    ? Box(colReader.LogicalReader<long?>().ReadAll(numRows))
                    : Box(colReader.LogicalReader<long>().ReadAll(numRows));

            case PhysicalType.Float:
                return nullable
                    ? Box(colReader.LogicalReader<float?>().ReadAll(numRows))
                    : Box(colReader.LogicalReader<float>().ReadAll(numRows));

            case PhysicalType.Double:
                return nullable
                    ? Box(colReader.LogicalReader<double?>().ReadAll(numRows))
                    : Box(colReader.LogicalReader<double>().ReadAll(numRows));

            case PhysicalType.ByteArray:
                // Could be string or raw bytes; try string first, fall back to byte[]
                try { return Box(colReader.LogicalReader<string>().ReadAll(numRows)); }
                catch { return Box(colReader.LogicalReader<byte[]>().ReadAll(numRows)); }

            case PhysicalType.FixedLenByteArray:
                return Box(colReader.LogicalReader<byte[]>().ReadAll(numRows));

            case PhysicalType.Int96:
                try
                {
                    return nullable
                        ? Box(colReader.LogicalReader<DateTime?>().ReadAll(numRows))
                        : Box(colReader.LogicalReader<DateTime>().ReadAll(numRows));
                }
                catch
                {
                    return Box(colReader.LogicalReader<byte[]>().ReadAll(numRows));
                }

            default:
                throw new NotSupportedException("Unsupported Parquet column type for matrix read. Extend ReadColumnAsObjects().");
        }
    }

    private static object[] Box<T>(T[] data)
    {
        var boxed = new object[data.Length];
        for (int i = 0; i < data.Length; i++) boxed[i] = data[i];
        return boxed;
    }

    // ---------------- Repeated (LIST) support (Read) ----------------

    private enum ParquetValueKind
    {
        DateTime,
        String,
        Decimal,
        GuidFromBytes,
        DateFromDays,
        TimeFromMicrosOrMillis,
        Bool,
        Int32,
        Int64,
        Float,
        Double,
        ByteArray,
        FixedLenByteArray,
        Int96AsDateTimeOrBytes,
        Unknown
    }

    private static ParquetValueKind GetValueKind(ColumnReader colReader)
    {
        var desc = colReader.ColumnDescriptor;
        var logical = desc.LogicalType;
        var phys = desc.PhysicalType;

        if (logical != null)
        {
            var logicalName = logical.GetType().Name;

            if (logicalName.IndexOf("Timestamp", StringComparison.OrdinalIgnoreCase) >= 0)
                return ParquetValueKind.DateTime;

            if (logicalName.IndexOf("String", StringComparison.OrdinalIgnoreCase) >= 0)
                return ParquetValueKind.String;

            if (logicalName.IndexOf("Decimal", StringComparison.OrdinalIgnoreCase) >= 0)
                return ParquetValueKind.Decimal;

            if (logicalName.IndexOf("UUID", StringComparison.OrdinalIgnoreCase) >= 0)
                return ParquetValueKind.GuidFromBytes;

            if (logicalName.Equals("DateLogicalType", StringComparison.OrdinalIgnoreCase) ||
                (logicalName.IndexOf("Date", StringComparison.OrdinalIgnoreCase) >= 0 &&
                 logicalName.IndexOf("Timestamp", StringComparison.OrdinalIgnoreCase) < 0))
                return ParquetValueKind.DateFromDays;

            if (logicalName.IndexOf("Time", StringComparison.OrdinalIgnoreCase) >= 0)
                return ParquetValueKind.TimeFromMicrosOrMillis;

            if (logicalName.IndexOf("Enum", StringComparison.OrdinalIgnoreCase) >= 0)
                return ParquetValueKind.String;

            if (logicalName.IndexOf("Json", StringComparison.OrdinalIgnoreCase) >= 0)
                return ParquetValueKind.String;
        }

        switch (phys)
        {
            case PhysicalType.Boolean: return ParquetValueKind.Bool;
            case PhysicalType.Int32: return ParquetValueKind.Int32;
            case PhysicalType.Int64: return ParquetValueKind.Int64;
            case PhysicalType.Float: return ParquetValueKind.Float;
            case PhysicalType.Double: return ParquetValueKind.Double;
            case PhysicalType.ByteArray: return ParquetValueKind.ByteArray;
            case PhysicalType.FixedLenByteArray: return ParquetValueKind.FixedLenByteArray;
            case PhysicalType.Int96: return ParquetValueKind.Int96AsDateTimeOrBytes;
            default: return ParquetValueKind.Unknown;
        }
    }

    private static object[] ReadRepeatedColumnAsObjects(ColumnReader colReader, int numRows)
    {
        var kind = GetValueKind(colReader);

        switch (kind)
        {
            case ParquetValueKind.DateTime:
                return ReadRepeatedGeneric<DateTime>(colReader, numRows, v => v);

            case ParquetValueKind.String:
                return ReadRepeatedGeneric<string>(colReader, numRows, v => v);

            case ParquetValueKind.Decimal:
                return ReadRepeatedGeneric<decimal>(colReader, numRows, v => v);

            case ParquetValueKind.GuidFromBytes:
                return ReadRepeatedGeneric<byte[]>(colReader, numRows, b =>
                {
                    if (b == null || b.Length != 16) return null;
                    return (object)new Guid(b);
                });

            case ParquetValueKind.DateFromDays:
                {
                    var epoch = new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc);
                    return ReadRepeatedGeneric<int>(colReader, numRows, days => (object)epoch.AddDays(days).Date);
                }

            case ParquetValueKind.TimeFromMicrosOrMillis:
                try
                {
                    return ReadRepeatedGeneric<long>(colReader, numRows, v => (object)TimeSpan.FromTicks(v * 10));
                }
                catch
                {
                    return ReadRepeatedGeneric<int>(colReader, numRows, v => (object)TimeSpan.FromMilliseconds(v));
                }

            case ParquetValueKind.Bool:
                return ReadRepeatedGeneric<bool>(colReader, numRows, v => v);

            case ParquetValueKind.Int32:
                return ReadRepeatedGeneric<int>(colReader, numRows, v => v);

            case ParquetValueKind.Int64:
                return ReadRepeatedGeneric<long>(colReader, numRows, v => v);

            case ParquetValueKind.Float:
                return ReadRepeatedGeneric<float>(colReader, numRows, v => v);

            case ParquetValueKind.Double:
                return ReadRepeatedGeneric<double>(colReader, numRows, v => v);

            case ParquetValueKind.ByteArray:
                try { return ReadRepeatedGeneric<string>(colReader, numRows, v => v); }
                catch { return ReadRepeatedGeneric<byte[]>(colReader, numRows, v => v); }

            case ParquetValueKind.FixedLenByteArray:
                return ReadRepeatedGeneric<byte[]>(colReader, numRows, v => v);

            case ParquetValueKind.Int96AsDateTimeOrBytes:
                try { return ReadRepeatedGeneric<DateTime>(colReader, numRows, v => v); }
                catch { return ReadRepeatedGeneric<byte[]>(colReader, numRows, v => v); }

            default:
                throw new NotSupportedException("Unsupported repeated Parquet column type. Extend GetValueKind()/ReadRepeatedColumnAsObjects().");
        }
    }

    private static object[] ReadRepeatedGeneric<TLeaf>(ColumnReader colReader, int numRows, Func<TLeaf, object> map)
    {
        if (map == null) throw new ArgumentNullException(nameof(map));

        var desc = colReader.ColumnDescriptor;
        int maxDef = desc.MaxDefinitionLevel;

        var perRow = new List<object>[numRows];
        var isNullList = new bool[numRows];
        for (int r = 0; r < numRows; r++) perRow[r] = new List<object>();

        // Reflection: LogicalReader<TLeaf>() then ReadBatch(int, def[], rep[], values[])
        object logicalReader = typeof(ColumnReader)
            .GetMethods(BindingFlags.Public | BindingFlags.Instance)
            .First(m => m.Name == "LogicalReader" && m.IsGenericMethodDefinition && m.GetParameters().Length == 0)
            .MakeGenericMethod(typeof(TLeaf))
            .Invoke(colReader, null);

        var readBatch = logicalReader.GetType()
            .GetMethods(BindingFlags.Public | BindingFlags.Instance)
            .FirstOrDefault(m =>
            {
                if (m.Name != "ReadBatch") return false;
                var ps = m.GetParameters();
                if (ps.Length != 4) return false;
                if (ps[0].ParameterType != typeof(int)) return false;
                if (!ps[1].ParameterType.IsArray) return false; // def
                if (!ps[2].ParameterType.IsArray) return false; // rep
                if (!ps[3].ParameterType.IsArray) return false; // values
                return true;
            });

        if (readBatch == null)
            throw new NotSupportedException("ParquetSharp LogicalReader<T>.ReadBatch(...) not found; cannot read repeated columns in this version.");

        var ps2 = readBatch.GetParameters();
        Type levelElemType = ps2[1].ParameterType.GetElementType(); // int/short depending on ParquetSharp build
        int batchSize = 8192;

        Array defLevels = Array.CreateInstance(levelElemType, batchSize);
        Array repLevels = Array.CreateInstance(levelElemType, batchSize);
        var values = new TLeaf[batchSize];

        int currentRow = -1;

        while (true)
        {
            int valuesRead = (int)readBatch.Invoke(logicalReader, new object[] { batchSize, defLevels, repLevels, values });
            if (valuesRead <= 0) break;

            for (int i = 0; i < valuesRead; i++)
            {
                int rep = Convert.ToInt32(repLevels.GetValue(i), CultureInfo.InvariantCulture);
                int def = Convert.ToInt32(defLevels.GetValue(i), CultureInfo.InvariantCulture);

                if (rep == 0) currentRow++;

                if (currentRow < 0) currentRow = 0;
                if (currentRow >= numRows) break;

                // def == 0 => null list (common)
                if (def == 0)
                {
                    isNullList[currentRow] = true;
                    perRow[currentRow].Clear();
                    continue;
                }

                // def < maxDef => empty list marker or null element depending on schema
                // We choose: do nothing (keeps empty lists empty, avoids inventing nulls)
                if (def < maxDef) continue;

                // def == maxDef => actual element
                perRow[currentRow].Add(map(values[i]));
            }
        }

        var boxed = new object[numRows];
        for (int r = 0; r < numRows; r++)
            boxed[r] = isNullList[r] ? null : (object)perRow[r];

        return boxed;
    }

    // ---------------- Struct collapsing (Read) ----------------

    private static object[,] CollapseStructColumns(object[,] flat, string[] flatHeaders, bool hasHeaderRow)
    {
        int rowCount = flat.GetLength(0);
        int colCount = flat.GetLength(1);

        var groups = new Dictionary<string, List<(int idx, string leaf)>>(StringComparer.OrdinalIgnoreCase);
        for (int c = 0; c < colCount; c++)
        {
            var h = flatHeaders[c] ?? ("c" + c);
            int dot = h.LastIndexOf('.');
            if (dot <= 0) continue;

            var prefix = h.Substring(0, dot);
            var leaf = h.Substring(dot + 1);

            if (!groups.TryGetValue(prefix, out var list))
            {
                list = new List<(int, string)>();
                groups[prefix] = list;
            }
            list.Add((c, leaf));
        }

        var collapsedPrefixes = groups.Where(kvp => kvp.Value.Count >= 2)
                                      .Select(kvp => (prefix: kvp.Key, cols: kvp.Value))
                                      .OrderBy(x => x.cols.Min(y => y.idx))
                                      .ToList();

        var consumed = new bool[colCount];
        foreach (var cp in collapsedPrefixes)
            foreach (var (idx, _) in cp.cols)
                consumed[idx] = true;

        var outCols = new List<(string name, Func<int, object> getter)>();

        for (int c = 0; c < colCount; c++)
        {
            if (consumed[c]) continue;
            string name = flatHeaders[c] ?? ("c" + c);
            int cc = c;
            outCols.Add((name, (r) => flat[r, cc]));
        }

        foreach (var cp in collapsedPrefixes)
        {
            string structName = cp.prefix;
            var cols = cp.cols;

            outCols.Add((structName, (r) =>
            {
                var dict = new Dictionary<string, object>(StringComparer.OrdinalIgnoreCase);
                foreach (var (idx, leaf) in cols)
                {
                    dict[leaf] = flat[r, idx];
                }
                return dict;
            }
            ));
        }

        var result = new object[rowCount, outCols.Count];

        if (hasHeaderRow)
        {
            for (int oc = 0; oc < outCols.Count; oc++)
                result[0, oc] = outCols[oc].name;
        }

        for (int r = hasHeaderRow ? 1 : 0; r < rowCount; r++)
        {
            for (int oc = 0; oc < outCols.Count; oc++)
                result[r, oc] = outCols[oc].getter(r);
        }

        return result;
    }

    // ---------------- Core Write ----------------

    private sealed class WritePlan
    {
        public Column[] Columns { get; }
        public Action<RowGroupWriter>[] Writers { get; }

        public WritePlan(Column[] columns, Action<RowGroupWriter>[] writers)
        {
            Columns = columns ?? throw new ArgumentNullException(nameof(columns));
            Writers = writers ?? throw new ArgumentNullException(nameof(writers));
        }
    }

    private static WritePlan BuildWritePlan(object[,] matrix, string[] headers)
    {
        int cols = matrix.GetLength(1);
        var names = BuildColumnNames(headers, cols);

        var columns = new Column[cols];
        var actions = new Action<RowGroupWriter>[cols];

        for (int c = 0; c < cols; c++)
        {
            var inf = InferColumnType(matrix, c, names[c]);
            columns[c] = inf.Column;
            actions[c] = inf.WriteAction;
        }

        return new WritePlan(columns, actions);
    }

    private static void WriteRowGroup(ParquetFileWriter file, WritePlan plan)
    {
        using (var rowGroup = file.AppendRowGroup())
        {
            for (int c = 0; c < plan.Writers.Length; c++)
                plan.Writers[c](rowGroup);
        }
    }

    private sealed class ColumnInference
    {
        public Column Column;
        public Action<RowGroupWriter> WriteAction;
    }

    private static ColumnInference InferColumnType(object[,] matrix, int colIndex, string name)
    {
        int rows = matrix.GetLength(0);

        object firstNonNull = null;
        bool hasNulls = false;

        for (int r = 0; r < rows; r++)
        {
            var v = matrix[r, colIndex];
            if (v == null || v == DBNull.Value) { hasNulls = true; continue; }
            firstNonNull = v;
            break;
        }

        if (firstNonNull == null)
        {
            return new ColumnInference
            {
                Column = new Column<string>(name),
                WriteAction = (rowGroup) =>
                {
                    using (var w = rowGroup.NextColumn().LogicalWriter<string>())
                    {
                        w.WriteBatch(new string[rows]); // all nulls
                    }
                }
            };
        }

        // If it's list-like/struct-like, write as JSON string (portable scalar)
        if (LooksLikeStructValue(firstNonNull))
        {
            return BuildWriterClass<string>(matrix, colIndex, name, v => v == null ? null : ToJson(v));
        }

        var t = NormalizeType(firstNonNull.GetType());

        if (t == typeof(DateTime)) return BuildWriterStruct<DateTime>(matrix, colIndex, hasNulls, name);
        if (t == typeof(int)) return BuildWriterStruct<int>(matrix, colIndex, hasNulls, name);
        if (t == typeof(long)) return BuildWriterStruct<long>(matrix, colIndex, hasNulls, name);
        if (t == typeof(float)) return BuildWriterStruct<float>(matrix, colIndex, hasNulls, name);
        if (t == typeof(double)) return BuildWriterStruct<double>(matrix, colIndex, hasNulls, name);
        if (t == typeof(bool)) return BuildWriterStruct<bool>(matrix, colIndex, hasNulls, name);
        if (t == typeof(decimal)) return BuildWriterStruct<decimal>(matrix, colIndex, hasNulls, name);

        if (t == typeof(string)) return BuildWriterClass<string>(matrix, colIndex, name);
        if (t == typeof(byte[])) return BuildWriterClass<byte[]>(matrix, colIndex, name);

        if (t.IsEnum)
            return BuildWriterClass<string>(matrix, colIndex, name, v => v?.ToString());

        if (t == typeof(DateTimeOffset))
            return BuildWriterStruct<DateTime>(matrix, colIndex, hasNulls, name, v => ((DateTimeOffset)v).UtcDateTime);

        if (t == typeof(TimeSpan))
            return BuildWriterClass<string>(matrix, colIndex, name, v => v == null ? null : ((TimeSpan)v).ToString());

        if (t == typeof(Guid))
            return BuildWriterClass<string>(matrix, colIndex, name, v => v == null ? null : ((Guid)v).ToString("D"));

        throw new NotSupportedException(
            "Unsupported matrix column type: " + t.FullName +
            ". Extend InferColumnType() to map it to a ParquetSharp Column<T>.");
    }

    private static Type NormalizeType(Type t)
    {
        if (t.IsGenericType && t.GetGenericTypeDefinition() == typeof(Nullable<>))
            t = Nullable.GetUnderlyingType(t);

        if (t == typeof(short) || t == typeof(byte) || t == typeof(sbyte) || t == typeof(ushort))
            return typeof(int);

        if (t == typeof(uint) || t == typeof(ulong))
            return typeof(long);

        if (t == typeof(char))
            return typeof(string);

        return t;
    }

    private static bool LooksLikeStructValue(object v)
    {
        if (v == null) return false;

        // Lists/arrays/enumerables -> JSON scalar on write (portable)
        if (v is IEnumerable && !(v is string) && !(v is byte[]))
            return true;

        if (v is IDictionary) return true;

        var t = v.GetType();

        if (t == typeof(string) || t == typeof(byte[])) return false;

        if (t.IsClass && t != typeof(object))
        {
            var props = t.GetProperties(BindingFlags.Public | BindingFlags.Instance);
            return props.Any(p => p.CanRead && p.GetIndexParameters().Length == 0);
        }

        return false;
    }

    private static string ToJson(object v)
    {
        var ser = new JavaScriptSerializer();
        return ser.Serialize(v);
    }

    private static string ColName(int colIndex)
    {
        return "c" + colIndex.ToString(CultureInfo.InvariantCulture);
    }

    // ---- Writer builders (structs) ----

    private static ColumnInference BuildWriterStruct<T>(object[,] matrix, int colIndex, bool hasNulls, string name)
        where T : struct
    {
        int rows = matrix.GetLength(0);

        if (hasNulls)
        {
            return new ColumnInference
            {
                Column = new Column<T?>(name),
                WriteAction = (rowGroup) =>
                {
                    using (var w = rowGroup.NextColumn().LogicalWriter<T?>())
                    {
                        var arr = new T?[rows];
                        for (int r = 0; r < rows; r++)
                        {
                            var v = matrix[r, colIndex];
                            if (v == null || v == DBNull.Value) arr[r] = null;
                            else arr[r] = (T)Convert.ChangeType(v, typeof(T), CultureInfo.InvariantCulture);
                        }
                        w.WriteBatch(arr);
                    }
                }
            };
        }

        return new ColumnInference
        {
            Column = new Column<T>(name),
            WriteAction = (rowGroup) =>
            {
                using (var w = rowGroup.NextColumn().LogicalWriter<T>())
                {
                    var arr = new T[rows];
                    for (int r = 0; r < rows; r++)
                    {
                        var v = matrix[r, colIndex];
                        if (v == null || v == DBNull.Value)
                            throw new InvalidOperationException($"Null found in non-nullable column {name} at row {r}.");

                        arr[r] = (T)Convert.ChangeType(v, typeof(T), CultureInfo.InvariantCulture);
                    }
                    w.WriteBatch(arr);
                }
            }
        };
    }

    private static ColumnInference BuildWriterStruct<T>(object[,] matrix, int colIndex, bool hasNulls, string name, Func<object, T> convert)
        where T : struct
    {
        if (convert == null) throw new ArgumentNullException(nameof(convert));
        int rows = matrix.GetLength(0);

        if (hasNulls)
        {
            return new ColumnInference
            {
                Column = new Column<T?>(name),
                WriteAction = (rowGroup) =>
                {
                    using (var w = rowGroup.NextColumn().LogicalWriter<T?>())
                    {
                        var arr = new T?[rows];
                        for (int r = 0; r < rows; r++)
                        {
                            var v = matrix[r, colIndex];
                            if (v == null || v == DBNull.Value) arr[r] = null;
                            else arr[r] = convert(v);
                        }
                        w.WriteBatch(arr);
                    }
                }
            };
        }

        return new ColumnInference
        {
            Column = new Column<T>(name),
            WriteAction = (rowGroup) =>
            {
                using (var w = rowGroup.NextColumn().LogicalWriter<T>())
                {
                    var arr = new T[rows];
                    for (int r = 0; r < rows; r++)
                    {
                        var v = matrix[r, colIndex];
                        if (v == null || v == DBNull.Value)
                            throw new InvalidOperationException($"Null found in non-nullable column {name} at row {r}.");

                        arr[r] = convert(v);
                    }
                    w.WriteBatch(arr);
                }
            }
        };
    }

    // ---- Writer builders (classes) ----

    private static ColumnInference BuildWriterClass<T>(object[,] matrix, int colIndex, string name)
        where T : class
    {
        int rows = matrix.GetLength(0);

        return new ColumnInference
        {
            Column = new Column<T>(name),
            WriteAction = (rowGroup) =>
            {
                using (var w = rowGroup.NextColumn().LogicalWriter<T>())
                {
                    var arr = new T[rows];
                    for (int r = 0; r < rows; r++)
                    {
                        var v = matrix[r, colIndex];
                        if (v == null || v == DBNull.Value) arr[r] = null;
                        else arr[r] = (T)v;
                    }
                    w.WriteBatch(arr);
                }
            }
        };
    }

    private static ColumnInference BuildWriterClass<TOut>(object[,] matrix, int colIndex, string name, Func<object, TOut> convert)
        where TOut : class
    {
        if (convert == null) throw new ArgumentNullException(nameof(convert));
        int rows = matrix.GetLength(0);

        return new ColumnInference
        {
            Column = new Column<TOut>(name),
            WriteAction = (rowGroup) =>
            {
                using (var w = rowGroup.NextColumn().LogicalWriter<TOut>())
                {
                    var arr = new TOut[rows];
                    for (int r = 0; r < rows; r++)
                    {
                        var v = matrix[r, colIndex];
                        if (v == null || v == DBNull.Value) arr[r] = null;
                        else arr[r] = convert(v);
                    }
                    w.WriteBatch(arr);
                }
            }
        };
    }

    // ---------------- Header helpers ----------------

    private static string[] BuildColumnNames(string[] headers, int cols)
    {
        string[] names;

        if (headers == null || headers.Length != cols)
        {
            names = new string[cols];
            for (int c = 0; c < cols; c++) names[c] = ColName(c);
            return names;
        }

        names = (string[])headers.Clone();
        for (int c = 0; c < cols; c++)
            if (string.IsNullOrWhiteSpace(names[c])) names[c] = ColName(c);

        return MakeHeadersUnique(names);
    }

    private static string[] MakeHeadersUnique(string[] headers)
    {
        var seen = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);

        for (int i = 0; i < headers.Length; i++)
        {
            var baseName = string.IsNullOrWhiteSpace(headers[i]) ? ColName(i) : headers[i].Trim();

            if (!seen.TryGetValue(baseName, out int n))
            {
                seen[baseName] = 0;
                headers[i] = baseName;
            }
            else
            {
                n++;
                seen[baseName] = n;
                headers[i] = baseName + "_" + n.ToString(CultureInfo.InvariantCulture);
            }
        }

        return headers;
    }
}
