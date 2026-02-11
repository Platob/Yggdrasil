using Microsoft.Office.Interop.Excel;
using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;

namespace YGGXLAddin
{
    public static class ExcelParquetIO
    {
        public static int WriteParquetToWorksheet(string parquetFilePath, Range topLeftCell, bool writeHeaderRow = true)
        {
            if (string.IsNullOrWhiteSpace(parquetFilePath))
                throw new ArgumentException("Parquet file path is required.", nameof(parquetFilePath));
            if (!File.Exists(parquetFilePath))
                throw new FileNotFoundException("Parquet file not found.", parquetFilePath);
            if (topLeftCell == null)
                throw new ArgumentNullException(nameof(topLeftCell));

            Worksheet ws = null;
            Range headerRange = null;
            Range dataRange = null;

            try
            {
                ws = topLeftCell.Worksheet;
                if (ws == null) throw new InvalidOperationException("Top-left cell is not associated with a worksheet.");

                var matrix = ParquetMatrixIO.ReadParquetToMatrix(parquetFilePath, writeHeaderRow: writeHeaderRow);
                int rows = matrix.GetLength(0);
                int cols = matrix.GetLength(1);
                if (cols == 0) return 0;

                int startRow = topLeftCell.Row;
                int startCol = topLeftCell.Column;

                int writeRow = startRow;

                if (rows > 0)
                {
                    var excelValues = ToExcelValue2(matrix);

                    dataRange = ws.Range[
                        ws.Cells[writeRow, startCol],
                        ws.Cells[writeRow + rows - 1, startCol + cols - 1]];
                    dataRange.Value2 = excelValues;
                }

                return rows;
            }
            finally
            {
                if (headerRange != null) Marshal.FinalReleaseComObject(headerRange);
                if (dataRange != null) Marshal.FinalReleaseComObject(dataRange);
                if (ws != null) Marshal.FinalReleaseComObject(ws);
            }
        }

        public static void ReadWorksheetToParquetFile(
    Range topLeftCell,
    int rowCount,
    int columnCount,
    string parquetFilePath,
    bool hasHeaderRow = true)
        {
            if (topLeftCell == null) throw new ArgumentNullException(nameof(topLeftCell));
            if (rowCount <= 0) throw new ArgumentOutOfRangeException(nameof(rowCount));
            if (columnCount <= 0) throw new ArgumentOutOfRangeException(nameof(columnCount));
            if (string.IsNullOrWhiteSpace(parquetFilePath))
                throw new ArgumentException("Parquet file path is required.", nameof(parquetFilePath));

            Worksheet ws = null;
            Range block = null;

            try
            {
                ws = topLeftCell.Worksheet ?? throw new InvalidOperationException("Top-left cell is not associated with a worksheet.");

                int startRow = topLeftCell.Row;
                int startCol = topLeftCell.Column;

                block = ws.Range[
                    ws.Cells[startRow, startCol],
                    ws.Cells[startRow + rowCount - 1, startCol + columnCount - 1]];

                object raw = block.Value2;
                object[,] values1Based = NormalizeTo1Based(raw, rowCount, columnCount);

                // ---- Header detection: build headers[] once, cleanly ----
                string[] headers = BuildHeaders(values1Based, columnCount, hasHeaderRow);

                int dataStartR = (hasHeaderRow && headers != null) ? 2 : 1; // 1-based into values1Based
                int dataRows = rowCount - (dataStartR - 1);
                if (dataRows < 0) dataRows = 0;

                var matrix = new object[dataRows, columnCount];
                for (int r = 0; r < dataRows; r++)
                {
                    for (int c = 0; c < columnCount; c++)
                    {
                        object v = values1Based[dataStartR + r, 1 + c];
                        matrix[r, c] = CleanExcelValue(v);
                    }
                }

                using (var fs = new FileStream(parquetFilePath, FileMode.Create, FileAccess.Write, FileShare.None))
                {
                    // overload this to accept headers; otherwise store them in your schema builder step
                    ParquetMatrixIO.WriteMatrixToParquet(matrix, fs, headers);
                }
            }
            finally
            {
                if (block != null) Marshal.FinalReleaseComObject(block);
                if (ws != null) Marshal.FinalReleaseComObject(ws);
            }
        }

        private static string[] BuildHeaders(object[,] values1Based, int columnCount, bool hasHeaderRow)
        {
            if (!hasHeaderRow) return null;

            var headers = new string[columnCount];
            bool any = false;

            for (int c = 0; c < columnCount; c++)
            {
                object v = values1Based[1, 1 + c];
                v = CleanExcelValue(v);

                if (v is string s && !string.IsNullOrWhiteSpace(s))
                {
                    headers[c] = s.Trim();
                    any = true;
                }
                else
                {
                    headers[c] = "c" + c.ToString(); // fallback for blank/non-string header cell
                }
            }

            // If the "header row" is actually empty/garbage, treat as no header row
            return any ? MakeHeadersUnique(headers) : null;
        }

        private static string[] MakeHeadersUnique(string[] headers)
        {
            // Ensures Parquet schema doesn't get duplicate names (Excel loves duplicates)
            var seen = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
            for (int i = 0; i < headers.Length; i++)
            {
                var baseName = string.IsNullOrWhiteSpace(headers[i]) ? "c" + i.ToString() : headers[i].Trim();

                if (!seen.TryGetValue(baseName, out int n))
                {
                    seen[baseName] = 0;
                    headers[i] = baseName;
                }
                else
                {
                    n++;
                    seen[baseName] = n;
                    headers[i] = $"{baseName}_{n}";
                }
            }
            return headers;
        }

        // --------- Helpers ---------

        private static object[,] ToExcelValue2(object[,] matrix)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);

            var excel = new object[rows, cols];

            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++)
                {
                    var v = matrix[r, c];
                    if (v == null || v == DBNull.Value) { excel[r, c] = null; continue; }

                    // If Parquet had DateTime, Excel Value2 wants OADate double (usually).
                    if (v is DateTime dt) { excel[r, c] = dt.ToOADate(); continue; }

                    excel[r, c] = v;
                }

            return excel;
        }

        private static object[,] NormalizeTo1Based(object raw, int rows, int cols)
        {
            if (raw is object[,] arr) return arr;

            // 1-cell scalar case
            var normalized = new object[rows + 1, cols + 1];
            normalized[1, 1] = raw;
            return normalized;
        }

        private static object CleanExcelValue(object value)
        {
            if (value == null || value == DBNull.Value) return null;

            if (value is string s)
            {
                var t = s.Trim();
                if (t.Length == 0) return null;
                if (t.Equals("N/A", StringComparison.OrdinalIgnoreCase)) return null;
                if (t.Equals("None", StringComparison.OrdinalIgnoreCase)) return null;
                return t;
            }

            // Keep double as double (Value2 does that). If you WANT dates, do column-based conversion instead.
            return value;
        }
    }
}
