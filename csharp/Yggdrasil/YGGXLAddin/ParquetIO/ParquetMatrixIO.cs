// Target: .NET Framework 4.8
// NuGet: ParquetSharp + ParquetSharp.Native (x64 recommended)
//
// Matrix is row-major: matrix[row, col]
// Write: schema inferred from first non-null value per column (value types become nullable if nulls exist)
// Write: column names come from optional headers[]; otherwise default to c0..cN (and de-duped)
//
// Read: returns object[,] with optional header row at result[0, *] containing Parquet column names
// Read: repeated (list) columns are NOT supported (object[,] can't represent them cleanly)

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using ParquetSharp;
using ParquetSharp.IO;

public static class ParquetMatrixIO
{
    // ---------------- Public Read ----------------

    public static object[,] ReadParquetToMatrix(string filePath, bool writeHeaderRow = true)
    {
        if (string.IsNullOrWhiteSpace(filePath)) throw new ArgumentNullException(nameof(filePath));

        using (var file = new ParquetFileReader(filePath))
        {
            return ReadParquetToMatrixCore(file, writeHeaderRow);
        }
    }

    public static object[,] ReadParquetToMatrix(Stream parquetStream, bool writeHeaderRow = true)
    {
        if (parquetStream == null) throw new ArgumentNullException(nameof(parquetStream));
        if (!parquetStream.CanRead) throw new ArgumentException("Stream must be readable.", nameof(parquetStream));

        // ManagedRandomAccessFile works across ParquetSharp versions and supports non-seekable streams
        // (internally buffers if needed).
        using (var raf = new ManagedRandomAccessFile(parquetStream))
        using (var file = new ParquetFileReader(raf))
        {
            return ReadParquetToMatrixCore(file, writeHeaderRow);
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

    private static object[,] ReadParquetToMatrixCore(ParquetFileReader file, bool writeHeaderRow)
    {
        var meta = file.FileMetaData;
        int numRowGroups = meta.NumRowGroups;
        int numCols = meta.Schema.NumColumns;

        int totalRows = GetTotalRows(file, numRowGroups);

        int headerOffset = writeHeaderRow ? 1 : 0;
        var result = new object[totalRows + headerOffset, numCols];

        if (writeHeaderRow)
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

    private static object[] ReadColumnAsObjects(ColumnReader colReader, int numRows)
    {
        var desc = colReader.ColumnDescriptor;

        bool nullable = desc.MaxDefinitionLevel > 0;
        bool repeated = desc.MaxRepetitionLevel > 0; // lists etc.

        if (repeated)
            throw new NotSupportedException("Repeated (list) columns are not supported for object[,] matrix.");

        var phys = desc.PhysicalType;
        var logical = desc.LogicalType;

        // Logical types first (more semantic)
        if (logical != null)
        {
            var logicalName = logical.GetType().Name;

            if (logicalName.IndexOf("Timestamp", StringComparison.OrdinalIgnoreCase) >= 0)
            {
                return nullable
                    ? Box(colReader.LogicalReader<DateTime?>().ReadAll(numRows))
                    : Box(colReader.LogicalReader<DateTime>().ReadAll(numRows));
            }

            if (logicalName.IndexOf("String", StringComparison.OrdinalIgnoreCase) >= 0)
            {
                return Box(colReader.LogicalReader<string>().ReadAll(numRows));
            }

            if (logicalName.IndexOf("Decimal", StringComparison.OrdinalIgnoreCase) >= 0)
            {
                return nullable
                    ? Box(colReader.LogicalReader<decimal?>().ReadAll(numRows))
                    : Box(colReader.LogicalReader<decimal>().ReadAll(numRows));
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

        throw new NotSupportedException(
            "Unsupported matrix column type: " + t.FullName +
            ". Extend InferColumnType() to map it to a ParquetSharp Column<T>.");
    }

    private static Type NormalizeType(Type t)
    {
        // Excel interop + boxed values can come in with small integer types
        if (t == typeof(short) || t == typeof(byte) || t == typeof(sbyte) || t == typeof(ushort))
            return typeof(int);

        if (t == typeof(uint))
            return typeof(long);

        return t;
    }

    private static string ColName(int colIndex)
    {
        return "c" + colIndex.ToString(CultureInfo.InvariantCulture);
    }

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
                        else arr[r] = (T)v; // keep as direct cast for ref types
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
        // Parquet schema cannot have duplicate column names.
        // Excel users absolutely will give you duplicates. We fix it deterministically.
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
