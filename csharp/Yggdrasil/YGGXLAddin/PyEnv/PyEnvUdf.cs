using System;
using System.Runtime.InteropServices;
using System.Windows.Forms;
using Microsoft.Office.Interop.Excel;
using YGGXLAddin.PyEnv;

namespace YGGXLAddin
{
    [ComVisible(true)]
    [ClassInterface(ClassInterfaceType.AutoDual)]
    public static class PyEnvUdf
    {
        public static string YGGPython(object codeOrCell, bool isCell = false, bool showMessage = false)
        {
            try
            {
                var code = ResolveCode(codeOrCell, isCell);
                if (string.IsNullOrWhiteSpace(code))
                    return "";

                var env = PyEnvManager.SystemDefault();
                var result = env.RunCode(code);
                var output = result.ExitCode == 0
                    ? (result.StdOut ?? "").TrimEnd()
                    : $"#PYERROR: {(result.StdErr ?? "").TrimEnd()}";

                if (showMessage)
                {
                    var detail = result.ExitCode == 0
                        ? $"Python output:\n{result.StdOut}"
                        : $"Python error:\n{result.StdErr}";
                    MessageBox.Show(detail, "YGGPython", MessageBoxButtons.OK,
                        result.ExitCode == 0 ? MessageBoxIcon.Information : MessageBoxIcon.Error);
                }

                return output;
            }
            catch (Exception ex)
            {
                if (showMessage)
                    MessageBox.Show(ex.Message, "YGGPython error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return $"#PYERROR: {ex.Message}";
            }
        }

        private static string ResolveCode(object codeOrCell, bool isCell)
        {
            if (codeOrCell == null)
                return "";

            if (codeOrCell is Range range)
                return Convert.ToString(range.Value2 ?? "");

            var text = Convert.ToString(codeOrCell ?? "");
            if (!isCell || string.IsNullOrWhiteSpace(text))
                return text;

            try
            {
                var app = Globals.ThisAddIn?.Application;
                if (app == null)
                    return text;

                var sheet = app.ActiveSheet as Worksheet;
                var cell = sheet?.Range[text];
                return Convert.ToString(cell?.Value2 ?? "");
            }
            catch
            {
                return text;
            }
        }
    }
}
