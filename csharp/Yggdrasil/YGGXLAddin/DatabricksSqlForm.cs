using System;
using System.Drawing;
using System.IO;
using System.Runtime.InteropServices;
using System.Windows.Forms;
using YGGXLAddin.Python;
using Excel = Microsoft.Office.Interop.Excel;

namespace YGGXLAddin
{
    public sealed class DatabricksSqlForm : Form
    {
        private readonly TextBox _queryInput;
        private readonly Button _runButton;
        private readonly Label _statusLabel;
        private bool _isRunning;

        public DatabricksSqlForm()
        {
            Text = "Databricks SQL";
            MinimumSize = new Size(720, 420);
            StartPosition = FormStartPosition.CenterParent;

            var layout = new TableLayoutPanel
            {
                Dock = DockStyle.Fill,
                ColumnCount = 1,
                RowCount = 4,
                Padding = new Padding(12)
            };

            layout.RowStyles.Add(new RowStyle(SizeType.AutoSize));
            layout.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
            layout.RowStyles.Add(new RowStyle(SizeType.AutoSize));
            layout.RowStyles.Add(new RowStyle(SizeType.AutoSize));

            layout.Controls.Add(new Label
            {
                AutoSize = true,
                Text = "SQL Query:",
                TextAlign = ContentAlignment.MiddleLeft
            }, 0, 0);

            _queryInput = new TextBox
            {
                Dock = DockStyle.Fill,
                Multiline = true,
                ScrollBars = ScrollBars.Both,
                Font = new Font("Consolas", 10f),
                AcceptsReturn = true,
                AcceptsTab = true,
                WordWrap = false
            };
            layout.Controls.Add(_queryInput, 0, 1);

            var actionPanel = new FlowLayoutPanel
            {
                Dock = DockStyle.Top,
                AutoSize = true,
                FlowDirection = FlowDirection.LeftToRight
            };

            _runButton = new Button
            {
                Text = "Run SQL",
                AutoSize = true
            };
            _runButton.Click += (_, __) => RunQuery();
            actionPanel.Controls.Add(_runButton);

            layout.Controls.Add(actionPanel, 0, 2);

            _statusLabel = new Label
            {
                AutoSize = true,
                Text = "Ready."
            };
            layout.Controls.Add(_statusLabel, 0, 3);

            Controls.Add(layout);
        }

        private void RunQuery()
        {
            if (_isRunning)
                return;

            var statement = _queryInput.Text;

            if (string.IsNullOrWhiteSpace(statement))
            {
                MessageBox.Show(this, "Please enter a SQL query to run.", "Missing SQL", MessageBoxButtons.OK,
                    MessageBoxIcon.Information);
                return;
            }

            string tempFile = null;

            try
            {
                _isRunning = true;
                _runButton.Enabled = false;
                _statusLabel.Text = "Running Databricks SQL query...";

                tempFile = Path.Combine(Path.GetTempPath(), $"ygg_sql_{Guid.NewGuid():N}.parquet");

                var pyCode = BuildPythonCode(statement, tempFile);

                var result = PyEnvManager.Instance.RunPythonCode(
                    code: pyCode,
                    workingDirectory: Path.GetTempPath());

                result?.ThrowIfFailed("Databricks SQL query failed.");

                Excel.Application excelApp;

                try
                {
                    // Attach to already-running Excel
                    excelApp = (Excel.Application)Marshal.GetActiveObject("Excel.Application");
                }
                catch (COMException)
                {
                    // Excel not running ? start a new one
                    excelApp = new Excel.Application();
                    excelApp.Visible = true;
                }

                if (excelApp == null)
                    throw new InvalidOperationException("Excel application is not available.");

                var worksheet = excelApp.ActiveSheet as Microsoft.Office.Interop.Excel.Worksheet;
                if (worksheet == null)
                    throw new InvalidOperationException("No active worksheet found.");

                var startCell = excelApp.ActiveCell ?? worksheet.Cells[1, 1];

                var rows = ExcelParquetIO.WriteParquetToWorksheet(tempFile, startCell);

                _statusLabel.Text = $"Completed. Loaded {rows} rows.";
            }
            catch (Exception ex)
            {
                _statusLabel.Text = "Failed.";
                MessageBox.Show(this, ex.Message, "Databricks SQL Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            finally
            {
                _runButton.Enabled = true;
                _isRunning = false;

                if (!string.IsNullOrWhiteSpace(tempFile) && File.Exists(tempFile))
                {
                    try
                    {
                        File.Delete(tempFile);
                    }
                    catch
                    {
                        // ignore cleanup failures
                    }
                }
            }
        }

        private static string BuildPythonCode(string statement, string tempFile)
        {
            var statementLiteral = ToPythonStringLiteral(statement);
            var tempFileLiteral = ToPythonStringLiteral(tempFile);

            return $@"from yggdrasil.databricks.workspaces import Workspace

workspace = Workspace(host=""dbc-e646c5f9-8a44.cloud.databricks.com"")
engine = workspace.sql()
__tempfile__ = {tempFileLiteral}
statement = {statementLiteral}
result = engine.execute(statement=statement)
result.to_polars().write_parquet(__tempfile__)
";
        }

        private static string ToPythonStringLiteral(string value)
        {
            if (value == null)
                return "\"\"";

            var escaped = value
                .Replace("\\", "\\\\")
                .Replace("\"", "\\\"")
                .Replace("\r", "\\r")
                .Replace("\n", "\\n");

            return $"\"{escaped}\"";
        }
    }
}
