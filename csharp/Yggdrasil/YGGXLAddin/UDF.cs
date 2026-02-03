using System;
using ExcelDna.Integration;
using Microsoft.Office.Interop.Excel;
using YGGXLAddin.Python;

namespace YGGXLAddin
{
    public static class UDF
    {
        [ExcelFunction(
            Name = "PYEXE",
            Description = "Python execute code")]
        public static object PyExe(
            object code,
            string pyVariable = null,
            string environment = null,
            string workingDirectory = null)
        {
            if (string.IsNullOrEmpty(workingDirectory))
            {
                workingDirectory = GetCurrentWorkbookDir();
            }

            try
            {
                // If user passed a cell/range, pull its displayed value.
                // If they passed a literal string, use it as-is.
                var pyCode = ResolveCodeFromExcelArg(code);

                var result = PyEnvManager.Instance.RunPythonCode(
                    code: pyCode,
                    environment: environment,
                    workingDirectory: workingDirectory);

                result?.ThrowIfFailed();

                // RunPythonCode returns stdout (string). If you want stderr too, change RunPythonCode signature.
                return result.StdOut;
            }
            catch (Exception ex)
            {
                // Don’t throw into Excel; return a clean error.
                return $"#PYEXE! {ex.GetType().Name}: {ex.Message}";
            }
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
            catch (Exception ex)
            {
                // Optional: log ex somewhere if you care
                // Debug.WriteLine(ex);
            }

            // Fallback for unsaved workbooks or COM tantrums
            return Environment.CurrentDirectory;
        }

        private static string ResolveCodeFromExcelArg(object arg)
        {
            if (arg == null || arg is ExcelEmpty)
                return "";

            if (arg is ExcelError err)
                throw new ArgumentException($"ExcelError passed as code: {err}");

            // Excel range / reference -> read the value from Excel
            if (arg is ExcelReference xref)
            {
                var v = XlCall.Excel(XlCall.xlCoerce, xref);
                return CoerceToString(v);
            }

            // Sometimes Excel-DNA hands you an object[,] (multi-cell)
            if (arg is object[,] arr)
            {
                if (arr.Length == 0) return "";

                // If it's 1x1, just treat it as a single value.
                if (arr.GetLength(0) == 1 && arr.GetLength(1) == 1)
                    return CoerceToString(arr[0, 0]);

                // If it's bigger, join cells by newline (nice for multi-line code laid out vertically).
                // Row-major join.
                var sb = new System.Text.StringBuilder();
                for (int r = 0; r < arr.GetLength(0); r++)
                {
                    for (int c = 0; c < arr.GetLength(1); c++)
                    {
                        var s = CoerceToString(arr[r, c]);
                        if (string.IsNullOrEmpty(s)) continue;

                        if (sb.Length > 0) sb.AppendLine();
                        sb.Append(s);
                    }
                }
                return sb.ToString();
            }

            // If it's already a string, great.
            if (arg is string s0)
                return s0;

            // Numbers/bools etc: stringify
            return Convert.ToString(arg) ?? "";
        }

        private static string CoerceToString(object v)
        {
            if (v == null || v is ExcelEmpty) return "";
            if (v is ExcelError err) throw new ArgumentException($"ExcelError in referenced cell: {err}");
            if (v is object[,] arr) return ResolveFromArray(arr);
            return v as string ?? Convert.ToString(v) ?? "";
        }

        private static string ResolveFromArray(object[,] arr)
        {
            if (arr.Length == 0) return "";

            if (arr.GetLength(0) == 1 && arr.GetLength(1) == 1)
                return CoerceToString(arr[0, 0]);

            var sb = new System.Text.StringBuilder();
            for (int r = 0; r < arr.GetLength(0); r++)
            {
                for (int c = 0; c < arr.GetLength(1); c++)
                {
                    var s = CoerceToString(arr[r, c]);
                    if (string.IsNullOrEmpty(s)) continue;

                    if (sb.Length > 0) sb.AppendLine();
                    sb.Append(s);
                }
            }
            return sb.ToString();
        }
    }
}
