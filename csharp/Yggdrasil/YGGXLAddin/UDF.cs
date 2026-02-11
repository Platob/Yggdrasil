using ExcelDna.Integration;
using Microsoft.Office.Interop.Excel;
using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using YGGXLAddin.Python;

namespace YGGXLAddin
{
    public static class UDF
    {
        // ---------------------------
        // PYEXE
        // ---------------------------
        [ExcelFunction(Name = "PYEXE", Description = "Python execute code", IsThreadSafe = false)]
        public static object PyExe(
            object code,
            string environment = null,
            object input = null,
            object output = null,
            string workingDirectory = null)
        {
            workingDirectory = EnsureWorkingDir(workingDirectory);

            try
            {
                var pyCode = ResolveCodeFromExcelArg(code);

                var result = PyEnvManager.Instance.RunPythonCode(
                    code: pyCode,
                    environment: environment,
                    workingDirectory: workingDirectory);

                result?.ThrowIfFailed();
                return result.StdOut ?? "";
            }
            catch (Exception ex)
            {
                return $"#PYEXE! {ex.GetType().Name}: {ex.Message}";
            }
        }

        // ---------------------------
        // READ_DATA_PATH
        // DataPath -> temp parquet -> worksheet write (macro queued)
        // ---------------------------
        [ExcelFunction(Name = "READ_DATA_PATH", Description = "Read DataPath and write to worksheet (parquet->cells).", IsThreadSafe = false)]
        public static object ReadDataPath(
            string path,
            object output = null,
            string environment = null,
            string workingDirectory = null)
        {
            if (string.IsNullOrWhiteSpace(path))
                return "#VALUE! path is empty";

            workingDirectory = EnsureWorkingDir(workingDirectory);
            var outRange = ResolveRange(output);

            var tmp = TempParquet("read_data_path_");
            var pyCode = BuildPyReadDataPathToParquet(path, tmp);

            try
            {
                var result = RunPython(pyCode, environment, workingDirectory);

                ExcelAsyncUtil.QueueAsMacro(() =>
                {
                    try
                    {
                        ExcelParquetIO.WriteParquetToWorksheet(tmp, outRange, writeHeaderRow: true);
                    }
                    finally
                    {
                        SafeDelete(tmp);
                        ReleaseCom(outRange);
                    }
                });

                return result.StdOut ?? "QUEUED";
            }
            catch (Exception ex)
            {
                SafeDelete(tmp);
                return $"#PYEXE! {ex.GetType().Name}: {ex.Message}";
            }
        }

        [ExcelFunction(Name = "READ_DATA_SQL", Description = "Read DataPath and write to worksheet (parquet->cells).", IsThreadSafe = false)]
        public static object ReadDataSQL(
            string path,
            object query,
            object output = null,
            string environment = null,
            string workingDirectory = null)
        {
            string sqlQuery = ResolveCodeFromExcelArg(query);
            workingDirectory = EnsureWorkingDir(workingDirectory);
            var outRange = ResolveRange(output);

            var tmp = TempParquet("read_databricks_sql_");
            var pyCode = BuildPyReadDataPathSQLToParquet(srcPath: path, query: sqlQuery, dstPath: tmp);

            try
            {
                var result = RunPython(pyCode, environment, workingDirectory);

                ExcelAsyncUtil.QueueAsMacro(() =>
                {
                    try
                    {
                        ExcelParquetIO.WriteParquetToWorksheet(tmp, outRange, writeHeaderRow: true);
                    }
                    finally
                    {
                        SafeDelete(tmp);
                        ReleaseCom(outRange);
                    }
                });

                return result.StdOut ?? "QUEUED";
            }
            catch (Exception ex)
            {
                SafeDelete(tmp);
                return $"#PYEXE! {ex.GetType().Name}: {ex.Message}";
            }
        }

        // ---------------------------
        // WRITE_DATA_PATH
        // Worksheet -> temp parquet -> DataPath write (macro queued)
        // ---------------------------
        [ExcelFunction(Name = "WRITE_DATA_PATH", Description = "Write worksheet block to destination DataPath.", IsThreadSafe = false)]
        public static object WriteDataPath(
            object input = null,
            string path = null,
            int rowCount = 0,
            int columnCount = 0,
            string environment = null,
            string workingDirectory = null)
        {
            if (string.IsNullOrEmpty(path))
                throw new ArgumentNullException("path");

            if (string.IsNullOrWhiteSpace(path))
                return "#VALUE! path is empty";

            workingDirectory = EnsureWorkingDir(workingDirectory);

            var tmp = TempParquet("write_data_path_");
            var pyCode = BuildPyWriteParquetToDataPath(srcPath: tmp, dstPath: path);

            // Important: avoid COM access on UDF thread; do EVERYTHING COM-ish inside QueueAsMacro.
            ExcelAsyncUtil.QueueAsMacro(() =>
            {
                Range anchor = null;
                Range block = null;

                try
                {
                    anchor = ResolveRange(input);

                    int rows, cols;
                    GetBlock(anchor, rowCount, columnCount, out block, out rows, out cols);

                    // Excel -> Parquet (ParquetSharp matrix utils)
                    ExcelParquetIO.ReadWorksheetToParquetFile(
                        topLeftCell: anchor,
                        rowCount: rows,
                        columnCount: cols,
                        parquetFilePath: tmp,
                        hasHeaderRow: true);

                    RunPython(pyCode, environment, workingDirectory).ThrowIfFailed();
                }
                finally
                {
                    SafeDelete(tmp);
                    ReleaseCom(block);
                    ReleaseCom(anchor);
                }
            });

            return path;
        }

        [ExcelFunction(Name = "WRITE_DATA_SQL", Description = "Write worksheet block to destination DataPath.", IsThreadSafe = false)]
        public static object WriteDataSQL(
            object input = null,
            string path = null,
            int rowCount = 0,
            int columnCount = 0,
            string environment = null,
            string workingDirectory = null)
        {
            if (string.IsNullOrEmpty(path))
                throw new ArgumentNullException("path");

            if (string.IsNullOrWhiteSpace(path))
                return "#VALUE! path is empty, must be like catalog_name.schema_name.table_name";

            workingDirectory = EnsureWorkingDir(workingDirectory);

            var tmp = TempParquet("write_databricks_sql_");
            var pyCode = BuildPyWriteParquetToDatabricksSQL(tmp, path);

            // Important: avoid COM access on UDF thread; do EVERYTHING COM-ish inside QueueAsMacro.
            ExcelAsyncUtil.QueueAsMacro(() =>
            {
                Range anchor = null;
                Range block = null;

                try
                {
                    anchor = ResolveRange(input);

                    int rows, cols;
                    GetBlock(anchor, rowCount, columnCount, out block, out rows, out cols);

                    // Excel -> Parquet (ParquetSharp matrix utils)
                    ExcelParquetIO.ReadWorksheetToParquetFile(
                        topLeftCell: anchor,
                        rowCount: rows,
                        columnCount: cols,
                        parquetFilePath: tmp,
                        hasHeaderRow: true);

                    RunPython(pyCode, environment, workingDirectory).ThrowIfFailed();
                }
                finally
                {
                    SafeDelete(tmp);
                    ReleaseCom(block);
                    ReleaseCom(anchor);
                }
            });

            return path;
        }

        // ---------------------------
        // Helpers (COM + temp + python)
        // ---------------------------

        private static string EnsureWorkingDir(string workingDirectory)
        {
            return string.IsNullOrEmpty(workingDirectory) ? GetCurrentWorkbookDir() : workingDirectory;
        }

        private static string TempParquet(string prefix)
        {
            return Path.Combine(Path.GetTempPath(), $"{prefix}{Guid.NewGuid():N}.parquet");
        }

        private static dynamic RunPython(string pyCode, string environment, string workingDirectory)
        {
            var result = PyEnvManager.Instance.RunPythonCode(
                code: pyCode,
                environment: environment,
                workingDirectory: workingDirectory);

            result?.ThrowIfFailed();
            return result;
        }

        private static string BuildPyReadDataPathToParquet(
            string srcPath,
            string dstParquetFile)
        {
            string pySrc = EscapePyRaw(srcPath);
            string pyDst = EscapePyRaw(dstParquetFile);

            return
$@"from yggdrasil.io.path import LocalDataPath

src = r""{pySrc}""
dst = r""{pyDst}""

df = LocalDataPath(src).read_polars()
df.write_parquet(dst)
";
        }

        private static string BuildPyReadDataPathSQLToParquet(string srcPath, string query, string dstPath)
        {
            if (string.IsNullOrWhiteSpace(srcPath))
                throw new ArgumentException("srcPath is empty", nameof(srcPath));
            if (string.IsNullOrWhiteSpace(query))
                throw new ArgumentException("query is empty", nameof(query));
            if (string.IsNullOrWhiteSpace(dstPath))
                throw new ArgumentException("dstParquetFile is empty", nameof(dstPath));

            string pySrc = EscapePyRaw(srcPath);
            string pyDst = EscapePyRaw(dstPath);
            string pyQuery = ToPythonStringLiteral(query);

            return
        $@"from yggdrasil.io.path import LocalDataPath

src = r""{pySrc}""
dst = r""{pyDst}""
query = {pyQuery}

df = LocalDataPath(src).sql_engine().execute(query, row_limit=1024 * 1024).to_polars()
df.write_parquet(dst)
";
        }

        private static string BuildPyWriteParquetToDataPath(
            string srcPath,
            string dstPath)
        {
            string pySrc = EscapePyRaw(srcPath);
            string pyDst = EscapePyRaw(dstPath);

            return
$@"from yggdrasil.io.path import LocalDataPath
from yggdrasil.polars import polars as pl

src = r""{pySrc}""
dst = r""{pyDst}""

LocalDataPath(dst).write_polars(pl.read_parquet(src))
";
        }

        private static string BuildPyWriteParquetToDatabricksSQL(
            string srcPath,
            string dstPath)
        {
            string pySrc = EscapePyRaw(srcPath);
            string pyDst = EscapePyRaw(dstPath);

            return
$@"from yggdrasil.io.path import LocalDataPath

src = r""{pySrc}""
dst = r""{pyDst}""
dst_path = LocalDataPath(dst)

# normalize empty strings too
catalog_name, schema_name, table_name, _ = dst_path.sql_volume_or_table_parts()

parts = (catalog_name or None, schema_name or None, table_name or None)

if all(p is None for p in parts):
    raise ValueError(
        f""dst_path must resolve to a catalog/schema/table (or volume/table target), ""
        f""got nothing from sql_volume_or_table_parts(): {{dst_path}}""
    )

df = LocalDataPath(src).read_polars()
dst_path.sql_engine().insert_into(
    df,
    catalog_name=catalog_name, schema_name=schema_name,
    table_name=table_name
)
";
        }

        private static string EscapePyRaw(string s)
        {
            return (s ?? "").Replace("\\", "\\\\");
        }

        private static void GetBlock(Range anchor, int rowCount, int columnCount, out Range block, out int rows, out int cols)
        {
            if (anchor == null) throw new ArgumentNullException(nameof(anchor));

            if (rowCount > 0 && columnCount > 0)
            {
                block = anchor.Worksheet.Range[
                    anchor.Worksheet.Cells[anchor.Row, anchor.Column],
                    anchor.Worksheet.Cells[anchor.Row + rowCount - 1, anchor.Column + columnCount - 1]];
                rows = rowCount;
                cols = columnCount;
                return;
            }

            block = anchor.CurrentRegion;
            rows = block.Rows.Count;
            cols = block.Columns.Count; // fixed: don't drop last column
        }

        private static Range ResolveRange(object arg, Worksheet worksheet = null)
        {
            var app = (Application)ExcelDnaUtil.Application;
            worksheet = worksheet ?? (Worksheet)app.ActiveSheet;

            if (arg == null || arg is ExcelMissing || arg is ExcelEmpty)
                arg = XlCall.Excel(XlCall.xlfCaller);

            if (arg is ExcelReference xref)
            {
                var address = (string)XlCall.Excel(XlCall.xlfReftext, xref, true);
                return app.Range[address];
            }

            if (arg is Range rng)
                return rng;

            if (arg is string s && !string.IsNullOrWhiteSpace(s))
            {
                s = s.Trim();
                return (s.IndexOf('!') >= 0) ? app.Range[s] : worksheet.Range[s];
            }

            throw new ArgumentException("Argument must be a cell/range reference (or omitted).");
        }

        private static string GetCurrentWorkbookDir()
        {
            try
            {
                var app = (Application)ExcelDnaUtil.Application;
                var wb = app.ActiveWorkbook;
                if (wb != null && !string.IsNullOrEmpty(wb.Path))
                    return wb.Path;
            }
            catch { }

            return Environment.CurrentDirectory;
        }

        private static void SafeDelete(string file)
        {
            try
            {
                if (!string.IsNullOrWhiteSpace(file) && File.Exists(file))
                    File.Delete(file);
            }
            catch { }
        }

        private static void ReleaseCom(object o)
        {
            if (o == null) return;
            try { Marshal.FinalReleaseComObject(o); } catch { }
        }

        // ---------------------------
        // Excel arg -> python code string helpers (unchanged, just cleaned)
        // ---------------------------

        private static string ResolveCodeFromExcelArg(object arg)
        {
            if (arg == null || arg is ExcelEmpty || arg is ExcelMissing)
                return "";

            if (arg is ExcelError err)
                throw new ArgumentException($"ExcelError passed as code: {err}");

            if (arg is ExcelReference xref)
            {
                var v = XlCall.Excel(XlCall.xlCoerce, xref);
                return CoerceToString(v);
            }

            if (arg is object[,] arr)
                return ResolveFromArray(arr);

            if (arg is string s0)
                return s0;

            return Convert.ToString(arg) ?? "";
        }

        private static string CoerceToString(object v)
        {
            if (v == null || v is ExcelEmpty || v is ExcelMissing) return "";
            if (v is ExcelError err) throw new ArgumentException($"ExcelError in referenced cell: {err}");
            if (v is object[,] arr) return ResolveFromArray(arr);
            return v as string ?? Convert.ToString(v) ?? "";
        }

        private static string ResolveFromArray(object[,] arr)
        {
            if (arr == null || arr.Length == 0) return "";

            // ExcelDna arrays are 0-based
            int rLen = arr.GetLength(0);
            int cLen = arr.GetLength(1);

            if (rLen == 1 && cLen == 1)
                return CoerceToString(arr[0, 0]);

            var sb = new StringBuilder();
            for (int r = 0; r < rLen; r++)
                for (int c = 0; c < cLen; c++)
                {
                    var s = CoerceToString(arr[r, c]);
                    if (string.IsNullOrEmpty(s)) continue;

                    if (sb.Length > 0) sb.AppendLine();
                    sb.Append(s);
                }

            return sb.ToString();
        }

        /// <summary>
        /// Converts a C# string to a safe Python string literal.
        /// Uses a triple-quoted Python string unless the content contains '''.
        /// Falls back to a double-quoted escaped literal if needed.
        /// </summary>
        private static string ToPythonStringLiteral(string s)
        {
            if (s == null) return "None";

            // Normalize line endings to keep output stable.
            s = s.Replace("\r\n", "\n").Replace("\r", "\n");

            // Prefer triple single quotes (keeps SQL readable).
            if (s.IndexOf("'''", StringComparison.Ordinal) < 0)
            {
                return "r'''" + s + "'''";
                // NOTE: raw triple quotes are fine for SQL; backslashes stay literal.
                // If your SQL can end with a single backslash, raw strings can be tricky,
                // but that’s rare for SQL.
            }

            // If it contains ''', use standard double-quoted escaping.
            var sb = new StringBuilder(s.Length + 16);
            sb.Append('"');

            for (int i = 0; i < s.Length; i++)
            {
                char ch = s[i];
                switch (ch)
                {
                    case '\\': sb.Append("\\\\"); break;
                    case '"': sb.Append("\\\""); break;
                    case '\n': sb.Append("\\n"); break;
                    case '\t': sb.Append("\\t"); break;
                    case '\0': sb.Append("\\0"); break;
                    default:
                        // Escape other control chars explicitly
                        if (ch < 32)
                        {
                            sb.Append("\\x");
                            sb.Append(((int)ch).ToString("x2"));
                        }
                        else
                        {
                            sb.Append(ch);
                        }
                        break;
                }
            }

            sb.Append('"');
            return sb.ToString();
        }
    }
}
