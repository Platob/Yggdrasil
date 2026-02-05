using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.Office.Interop.Excel;
using Parquet;
using Parquet.Data;
using Parquet.Schema;

namespace YGGXLAddin
{
    public static class ExcelParquetIO
    {
        public static int WriteParquetToWorksheet(string parquetFile, Range topLeftCell)
        {
            if (topLeftCell == null)
                throw new ArgumentNullException(nameof(topLeftCell));
            
            var worksheet = topLeftCell.Worksheet;
            if (worksheet == null)
                throw new InvalidOperationException("Top-left cell is not associated with a worksheet.");

            using (var stream = File.OpenRead(parquetFile))
            using (var reader = ParquetReader.CreateAsync(stream).Result)
            {
                var fields = reader.Schema.GetDataFields();
                if (fields.Length == 0)
                    return 0;

                var startRow = topLeftCell.Row;
                var startCol = topLeftCell.Column;

                var header = new object[1, fields.Length];
                for (var c = 0; c < fields.Length; c++)
                {
                    header[0, c] = fields[c].Name;
                }

                var headerRange = worksheet.Range[
                    worksheet.Cells[startRow, startCol],
                    worksheet.Cells[startRow, startCol + fields.Length - 1]];
                headerRange.Value2 = header;

                var rowOffset = 0;

                for (var g = 0; g < reader.RowGroupCount; g++)
                {
                    using (var group = reader.OpenRowGroupReader(g))
                    {
                        var dataColumns = fields.Select(field => group.ReadColumnAsync(field).Result).ToArray();
                        var rows = dataColumns.Length == 0 ? 0 : dataColumns[0].Data.Length;

                        if (rows == 0)
                            continue;

                        var values = new object[rows, fields.Length];

                        for (var c = 0; c < fields.Length; c++)
                        {
                            var data = dataColumns[c].Data;
                            for (var r = 0; r < rows; r++)
                            {
                                values[r, c] = data.GetValue(r) ?? "";
                            }
                        }

                        var dataRange = worksheet.Range[
                            worksheet.Cells[startRow + 1 + rowOffset, startCol],
                            worksheet.Cells[startRow + rowOffset + rows, startCol + fields.Length - 1]];
                        dataRange.Value2 = values;

                        rowOffset += rows;
                    }
                }

                return rowOffset;
            }
        }

        public static void ReadWorksheetToParquet(
            Range topLeftCell,
            int rowCount,
            int columnCount,
            string parquetFile)
        {
            if (topLeftCell == null)
                throw new ArgumentNullException(nameof(topLeftCell));
            if (rowCount <= 0)
                throw new ArgumentOutOfRangeException(nameof(rowCount), "Row count must be positive.");
            if (columnCount <= 0)
                throw new ArgumentOutOfRangeException(nameof(columnCount), "Column count must be positive.");
            if (string.IsNullOrWhiteSpace(parquetFile))
                throw new ArgumentException("Parquet file path is required.", nameof(parquetFile));

            var worksheet = topLeftCell.Worksheet;
            if (worksheet == null)
                throw new InvalidOperationException("Top-left cell is not associated with a worksheet.");

            var startRow = topLeftCell.Row;
            var startCol = topLeftCell.Column;

            var dataRange = worksheet.Range[
                worksheet.Cells[startRow, startCol],
                worksheet.Cells[startRow + rowCount - 1, startCol + columnCount - 1]];

            var values = dataRange.Value2 as object[,];
            if (values == null)
                throw new InvalidOperationException("Worksheet range is empty.");

            var fields = new DataField[columnCount];
            var columnSpecs = new ColumnSpec[columnCount];
            for (var c = 0; c < columnCount; c++)
            {
                var headerValue = values[1, c + 1];
                var header = headerValue == null || headerValue == DBNull.Value
                    ? $"Column{c + 1}"
                    : Convert.ToString(headerValue);

                var cleanedValues = new List<object>();
                for (var r = 2; r <= rowCount; r++)
                {
                    cleanedValues.Add(CleanCellValue(values[r, c + 1]));
                }

                var inferredType = InferColumnType(cleanedValues);
                columnSpecs[c] = new ColumnSpec(inferredType, cleanedValues);
                fields[c] = CreateField(inferredType, header);
            }

            var schema = new ParquetSchema(fields);
            using (var fileStream = File.Create(parquetFile))
            using (var writer = ParquetWriter.CreateAsync(schema, fileStream).Result)
            {
                using (var rowGroupWriter = writer.CreateRowGroup())
                {
                    for (var c = 0; c < columnCount; c++)
                    {
                        var column = BuildColumn(fields[c], columnSpecs[c]);
                        rowGroupWriter.WriteColumnAsync(column).RunSynchronously();
                    }
                }
            }
        }

        private static object CleanCellValue(object value)
        {
            if (value == null || value == DBNull.Value)
                return null;

            if (value is string s)
            {
                var trimmed = s.Trim();
                if (string.IsNullOrEmpty(trimmed))
                    return null;

                if (trimmed.Equals("N/A", StringComparison.OrdinalIgnoreCase) ||
                    trimmed.Equals("None", StringComparison.OrdinalIgnoreCase))
                {
                    return null;
                }

                return trimmed;
            }

            return value;
        }

        private static ColumnType InferColumnType(IEnumerable<object> values)
        {
            var hasValue = false;
            var allDateTime = true;
            var allDouble = true;
            var allInt64 = true;

            foreach (var value in values)
            {
                if (value == null)
                    continue;

                hasValue = true;

                if (value is DateTime)
                {
                    allDouble = false;
                    allInt64 = false;
                    continue;
                }

                if (IsNumeric(value))
                {
                    allDateTime = false;
                    var numericValue = Convert.ToDouble(value);
                    if (Math.Abs(numericValue - Math.Round(numericValue)) > 0.0000001)
                    {
                        allInt64 = false;
                    }

                    continue;
                }

                allDateTime = false;
                allDouble = false;
                allInt64 = false;
            }

            if (!hasValue)
                return ColumnType.String;

            if (allDateTime)
                return ColumnType.DateTime;

            if (allInt64)
                return ColumnType.Int64;

            if (allDouble)
                return ColumnType.Double;

            return ColumnType.String;
        }

        private static bool IsNumeric(object value)
        {
            return value is byte ||
                   value is sbyte ||
                   value is short ||
                   value is ushort ||
                   value is int ||
                   value is uint ||
                   value is long ||
                   value is ulong ||
                   value is float ||
                   value is double ||
                   value is decimal;
        }

        private static DataField CreateField(ColumnType columnType, string name)
        {
            switch (columnType)
            {
                case ColumnType.Int64:
                    return new DataField<long?>(name);
                case ColumnType.Double:
                    return new DataField<double?>(name);
                case ColumnType.DateTime:
                    return new DataField<DateTime?>(name);
                default:
                    return new DataField<string>(name);
            }
        }

        private static DataColumn BuildColumn(DataField field, ColumnSpec spec)
        {
            switch (spec.ColumnType)
            {
                case ColumnType.Int64:
                    return new DataColumn(field, spec.Values.Select(ToNullableInt64).ToArray());
                case ColumnType.Double:
                    return new DataColumn(field, spec.Values.Select(ToNullableDouble).ToArray());
                case ColumnType.DateTime:
                    return new DataColumn(field, spec.Values.Select(ToNullableDateTime).ToArray());
                default:
                    return new DataColumn(field, spec.Values.Select(ToStringValue).ToArray());
            }
        }

        private static long? ToNullableInt64(object value)
        {
            if (value == null)
                return null;

            return Convert.ToInt64(value);
        }

        private static double? ToNullableDouble(object value)
        {
            if (value == null)
                return null;

            return Convert.ToDouble(value);
        }

        private static DateTime? ToNullableDateTime(object value)
        {
            if (value == null)
                return null;

            if (value is DateTime dt)
                return dt;

            return null;
        }

        private static string ToStringValue(object value)
        {
            if (value == null)
                return null;

            return Convert.ToString(value);
        }

        private struct ColumnSpec
        {
            public ColumnSpec(ColumnType columnType, List<object> values)
            {
                ColumnType = columnType;
                Values = values;
            }

            public ColumnType ColumnType { get; }
            public List<object> Values { get; }
        }

        private enum ColumnType
        {
            String,
            Int64,
            Double,
            DateTime
        }
    }
}
