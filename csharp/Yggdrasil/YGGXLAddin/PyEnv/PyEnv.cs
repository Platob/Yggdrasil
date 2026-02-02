using System;
using System.Diagnostics;
using System.IO;
using System.Text;

namespace YGGXLAddin.PyEnv
{
    public readonly struct PyEnvMetadata
    {
        public readonly string Name;
        public readonly string ExePath; // python executable path (typically venv python)
        public readonly PyVersion Version;

        public PyEnvMetadata(string name, string exePath, PyVersion version)
        {
            if (string.IsNullOrWhiteSpace(name)) throw new ArgumentException("Name is required.", nameof(name));
            if (string.IsNullOrWhiteSpace(exePath)) throw new ArgumentException("ExePath is required.", nameof(exePath));

            Name = name.Trim();
            ExePath = exePath.Trim();
            Version = version;
        }

        public override string ToString() => $"{Name} ({Version}) @ {ExePath}";
    }

    public sealed class PyEnv
    {
        public PyEnvMetadata Metadata { get; }

        public string Name => Metadata.Name;
        public string ExePath => Metadata.ExePath;
        public PyVersion Version => Metadata.Version;

        private PyEnv(PyEnvMetadata metadata)
        {
            Metadata = metadata;
        }

        /// <summary>
        /// Create a PyEnv by validating exePath and detecting Python version.
        /// </summary>
        public static PyEnv Create(string name, string exePath, TimeSpan? timeout = null)
        {
            if (string.IsNullOrWhiteSpace(exePath))
                throw new ArgumentException("exePath is required.", nameof(exePath));

            exePath = exePath.Trim();

            // For explicit paths, validate existence. (SystemDefault resolves to a real path anyway.)
            if (exePath.IndexOf(Path.DirectorySeparatorChar) >= 0 || exePath.IndexOf(Path.AltDirectorySeparatorChar) >= 0)
            {
                if (!File.Exists(exePath))
                    throw new FileNotFoundException("Python executable not found.", exePath);
            }

            var ver = DetectVersion(exePath, timeout ?? TimeSpan.FromSeconds(10));
            return new PyEnv(new PyEnvMetadata(name, exePath, ver));
        }

        /// <summary>
        /// Run "python {arguments}" and capture stdout/stderr.
        /// </summary>
        public PyProcessResult Run(string arguments, string workingDirectory = null, TimeSpan? timeout = null)
        {
            return RunProcess(
                fileName: ExePath,
                arguments: arguments ?? "",
                workingDirectory: workingDirectory,
                timeout: timeout ?? TimeSpan.FromMinutes(5));
        }

        /// <summary>
        /// Run python -c "..."
        /// </summary>
        public PyProcessResult RunCode(string pythonCode, string workingDirectory = null, TimeSpan? timeout = null)
        {
            if (pythonCode == null) throw new ArgumentNullException(nameof(pythonCode));
            var args = "-c " + QuoteArg(pythonCode);
            return Run(args, workingDirectory, timeout);
        }

        private static PyVersion DetectVersion(string exePath, TimeSpan timeout)
        {
            // python --version sometimes writes to stderr. yes, really.
            var res = RunProcess(exePath, "--version", workingDirectory: null, timeout: timeout);

            var text = (res.StdOut ?? "").Trim();
            if (string.IsNullOrWhiteSpace(text))
                text = (res.StdErr ?? "").Trim();

            if (!PyVersion.TryParse(text, out var v))
            {
                throw new InvalidOperationException(
                    "Failed to parse Python version from output.\n" +
                    $"Exe: {exePath}\n" +
                    $"Exit: {res.ExitCode}\n" +
                    $"StdOut: '{res.StdOut?.Trim()}'\n" +
                    $"StdErr: '{res.StdErr?.Trim()}'");
            }

            return v;
        }

        /// <summary>
        /// Install packages into this environment.
        /// - Default: python -m pip install ...
        /// - If useUV=true: uv pip install ...  (uv will be installed via pip if missing)
        ///
        /// Notes:
        /// - Adds "--disable-pip-version-check" and "--no-input" for less noise / no prompts.
        /// - upgradePip only applies when useUV=false.
        /// </summary>
        public PyProcessResult PipInstall(
            string packageSpec,
            bool upgrade = false,
            bool forceReinstall = false,
            bool noDeps = false,
            bool upgradePip = false,
            bool useUV = false,
            string extraIndexUrl = null,
            string indexUrl = null,
            string trustedHost = null,
            string requirementsFile = null,
            string targetDirectory = null,
            string cacheDir = null,
            TimeSpan? timeout = null)
        {
            if (string.IsNullOrWhiteSpace(packageSpec) && string.IsNullOrWhiteSpace(requirementsFile))
                throw new ArgumentException("Provide packageSpec and/or requirementsFile.");

            timeout = timeout ?? TimeSpan.FromMinutes(5);

            if (useUV)
            {
                var uvPath = FindUVPath(installIfMissing: true, timeout: timeout);

                var sbUv = new StringBuilder();
                sbUv.Append("pip install ");
                sbUv.Append("--no-input ");

                if (upgrade) sbUv.Append("--upgrade ");
                if (forceReinstall) sbUv.Append("--reinstall ");
                if (noDeps) sbUv.Append("--no-deps ");

                if (!string.IsNullOrWhiteSpace(indexUrl))
                    sbUv.Append("--index-url ").Append(QuoteArg(indexUrl.Trim())).Append(' ');

                if (!string.IsNullOrWhiteSpace(extraIndexUrl))
                    sbUv.Append("--extra-index-url ").Append(QuoteArg(extraIndexUrl.Trim())).Append(' ');

                if (!string.IsNullOrWhiteSpace(trustedHost))
                    sbUv.Append("--trusted-host ").Append(QuoteArg(trustedHost.Trim())).Append(' ');

                if (!string.IsNullOrWhiteSpace(cacheDir))
                    sbUv.Append("--cache-dir ").Append(QuoteArg(cacheDir.Trim())).Append(' ');

                if (!string.IsNullOrWhiteSpace(targetDirectory))
                    sbUv.Append("--target ").Append(QuoteArg(targetDirectory.Trim())).Append(' ');

                if (!string.IsNullOrWhiteSpace(requirementsFile))
                    sbUv.Append("-r ").Append(QuoteArg(requirementsFile.Trim())).Append(' ');

                if (!string.IsNullOrWhiteSpace(packageSpec))
                    sbUv.Append(packageSpec.Trim()).Append(' ');

                return RunProcess(
                    fileName: uvPath,
                    arguments: sbUv.ToString().TrimEnd(),
                    workingDirectory: null,
                    timeout: timeout.Value);
            }

            if (upgradePip)
            {
                var up = Run("-m pip install --upgrade pip --disable-pip-version-check --no-input", timeout: timeout);
                if (up.ExitCode != 0)
                    return up;
            }

            var sb = new StringBuilder();
            sb.Append("-m pip install ");
            sb.Append("--disable-pip-version-check --no-input ");

            if (upgrade) sb.Append("--upgrade ");
            if (forceReinstall) sb.Append("--force-reinstall ");
            if (noDeps) sb.Append("--no-deps ");

            if (!string.IsNullOrWhiteSpace(indexUrl))
                sb.Append("--index-url ").Append(QuoteArg(indexUrl.Trim())).Append(' ');

            if (!string.IsNullOrWhiteSpace(extraIndexUrl))
                sb.Append("--extra-index-url ").Append(QuoteArg(extraIndexUrl.Trim())).Append(' ');

            if (!string.IsNullOrWhiteSpace(trustedHost))
                sb.Append("--trusted-host ").Append(QuoteArg(trustedHost.Trim())).Append(' ');

            if (!string.IsNullOrWhiteSpace(cacheDir))
                sb.Append("--cache-dir ").Append(QuoteArg(cacheDir.Trim())).Append(' ');

            if (!string.IsNullOrWhiteSpace(targetDirectory))
                sb.Append("--target ").Append(QuoteArg(targetDirectory.Trim())).Append(' ');

            if (!string.IsNullOrWhiteSpace(requirementsFile))
                sb.Append("-r ").Append(QuoteArg(requirementsFile.Trim())).Append(' ');

            if (!string.IsNullOrWhiteSpace(packageSpec))
                sb.Append(packageSpec.Trim()).Append(' ');

            return Run(sb.ToString().TrimEnd(), timeout: timeout);
        }

        public PyProcessResult PipInstall(
            string[] packageSpecs,
            bool upgrade = false,
            bool forceReinstall = false,
            bool noDeps = false,
            bool upgradePip = false,
            bool useUV = false,
            string extraIndexUrl = null,
            string indexUrl = null,
            string trustedHost = null,
            string requirementsFile = null,
            string targetDirectory = null,
            string cacheDir = null,
            TimeSpan? timeout = null)
        {
            if (packageSpecs == null || packageSpecs.Length == 0)
            {
                return PipInstall(
                    packageSpec: null,
                    upgrade: upgrade,
                    forceReinstall: forceReinstall,
                    noDeps: noDeps,
                    upgradePip: upgradePip,
                    useUV: useUV,
                    extraIndexUrl: extraIndexUrl,
                    indexUrl: indexUrl,
                    trustedHost: trustedHost,
                    requirementsFile: requirementsFile,
                    targetDirectory: targetDirectory,
                    cacheDir: cacheDir,
                    timeout: timeout);
            }

            var joined = string.Join(" ", packageSpecs);

            return PipInstall(
                packageSpec: joined,
                upgrade: upgrade,
                forceReinstall: forceReinstall,
                noDeps: noDeps,
                upgradePip: upgradePip,
                useUV: useUV,
                extraIndexUrl: extraIndexUrl,
                indexUrl: indexUrl,
                trustedHost: trustedHost,
                requirementsFile: requirementsFile,
                targetDirectory: targetDirectory,
                cacheDir: cacheDir,
                timeout: timeout);
        }

        private static PyProcessResult RunProcess(string fileName, string arguments, string workingDirectory, TimeSpan timeout)
        {
            var psi = new ProcessStartInfo
            {
                FileName = fileName,
                Arguments = arguments ?? "",
                WorkingDirectory = string.IsNullOrWhiteSpace(workingDirectory)
                    ? Environment.CurrentDirectory
                    : workingDirectory,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true,
                StandardOutputEncoding = Encoding.UTF8,
                StandardErrorEncoding = Encoding.UTF8
            };

            using (var p = new Process { StartInfo = psi })
            {
                var stdout = new StringBuilder();
                var stderr = new StringBuilder();

                p.OutputDataReceived += (_, e) => { if (e.Data != null) stdout.AppendLine(e.Data); };
                p.ErrorDataReceived += (_, e) => { if (e.Data != null) stderr.AppendLine(e.Data); };

                if (!p.Start())
                    throw new InvalidOperationException("Failed to start process: " + fileName);

                p.BeginOutputReadLine();
                p.BeginErrorReadLine();

                if (!p.WaitForExit((int)timeout.TotalMilliseconds))
                {
                    try { p.Kill(); } catch { /* ignore */ }
                    throw new TimeoutException($"Process timed out after {timeout.TotalSeconds:n0}s: {fileName} {arguments}");
                }

                p.WaitForExit();
                return new PyProcessResult(p.ExitCode, stdout.ToString(), stderr.ToString());
            }
        }

        public string FindUVPath(bool installIfMissing = true, TimeSpan? timeout = null)
        {
            timeout = timeout ?? TimeSpan.FromMinutes(5);

            var direct = TryFindUvNextToPython();
            if (!string.IsNullOrEmpty(direct))
                return direct;

            var which = TryWhichUvViaPython(timeout.Value);
            if (!string.IsNullOrEmpty(which))
                return which;

            if (!installIfMissing)
                return null;

            var install = PipInstall(
                packageSpec: "uv",
                upgrade: false,
                forceReinstall: false,
                noDeps: false,
                upgradePip: false,
                useUV: false,
                timeout: timeout);

            if (install.ExitCode != 0)
                throw new InvalidOperationException("uv was not found and pip install uv failed.\n" + install.ToString());

            direct = TryFindUvNextToPython();
            if (!string.IsNullOrEmpty(direct))
                return direct;

            which = TryWhichUvViaPython(timeout.Value);
            if (!string.IsNullOrEmpty(which))
                return which;

            throw new FileNotFoundException(
                "uv was installed but the executable could not be found in this environment. " +
                "Tried Scripts/bin next to python and shutil.which('uv').");
        }

        private string TryFindUvNextToPython()
        {
            var pyDir = Path.GetDirectoryName(ExePath);
            if (string.IsNullOrWhiteSpace(pyDir) || !Directory.Exists(pyDir))
                return null;

            var win = Path.Combine(pyDir, "uv.exe");
            if (File.Exists(win))
                return win;

            var nix = Path.Combine(pyDir, "uv");
            if (File.Exists(nix))
                return nix;

            return null;
        }

        private string TryWhichUvViaPython(TimeSpan timeout)
        {
            const string code =
                "import shutil\n" +
                "p = shutil.which('uv')\n" +
                "print(p if p else '')\n";

            var res = RunCode(code, timeout: timeout);
            if (res.ExitCode != 0)
                return null;

            var path = (res.StdOut ?? "").Trim();
            if (string.IsNullOrWhiteSpace(path))
                return null;

            if (File.Exists(path))
                return path;

            return path;
        }

        private static string QuoteArg(string s)
        {
            return "\"" + (s ?? "").Replace("\"", "\\\"") + "\"";
        }
    }

    public sealed class PyProcessResult
    {
        public int ExitCode { get; }
        public string StdOut { get; }
        public string StdErr { get; }

        public PyProcessResult(int exitCode, string stdOut, string stdErr)
        {
            ExitCode = exitCode;
            StdOut = stdOut ?? "";
            StdErr = stdErr ?? "";
        }

        public override string ToString()
            => $"Exit={ExitCode}\n--- stdout ---\n{StdOut}\n--- stderr ---\n{StdErr}";
    }
}
