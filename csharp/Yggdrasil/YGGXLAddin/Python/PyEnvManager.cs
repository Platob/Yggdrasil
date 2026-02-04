using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;

namespace YGGXLAddin.Python
{
    /// <summary>
    /// Manages Python virtual environments under:
    ///   %USERPROFILE%\.local\python\envs\<name>
    ///
    /// Create uses uv:
    ///   uv venv --python 3.11.6 <envDir>
    ///
    /// Update uses uv inside the env:
    ///   uv pip install ...
    /// </summary>
    public sealed class PyEnvManager
    {   
        public static PyEnvManager Instance = new PyEnvManager();

        private readonly Dictionary<string, PyEnv> _envs =
            new Dictionary<string, PyEnv>(StringComparer.OrdinalIgnoreCase);

        // -----------------------------
        // Manager default (optional)
        // -----------------------------
        private readonly object _defaultLock = new object();
        private PyEnv _default;

        /// <summary>
        /// Returns a singleton representing the "system default" Python on PATH.
        /// It resolves python via PATH (Windows: python.exe / py.exe; Unix: python3/python),
        /// then detects version and caches the result.
        /// </summary>
        public static PyEnv SystemDefault(TimeSpan? timeout = null)
        {
            var exePath = ResolveSystemPythonExe(timeout ?? TimeSpan.FromSeconds(10));
            return PyEnv.Create("system", exePath, timeout);
        }

        /// <summary>
        /// Returns manager default if set, otherwise system default.
        /// </summary>
        public PyEnv Default(TimeSpan? timeout = null)
        {
            lock (_defaultLock)
            {
                return _default ?? SystemDefault(timeout);
            }
        }

        /// <summary>
        /// Clear manager default (falls back to system default).
        /// </summary>
        public void ResetDefault()
        {
            lock (_defaultLock) { _default = null; }
        }

        /// <summary>
        /// Set manager default by cached env name or python exe file.
        /// </summary>
        public void SetDefault(string nameOrFile)
        {
            if (string.IsNullOrWhiteSpace(nameOrFile))
                throw new ArgumentException("name is required.", nameof(nameOrFile));

            nameOrFile = nameOrFile.Trim();

            if (!_envs.TryGetValue(nameOrFile, out var env))
            {
                if (!File.Exists(nameOrFile))
                    throw new KeyNotFoundException($"Env '{nameOrFile}' not found in cache.");

                env = PyEnv.Create(name: null, exePath: nameOrFile);

                _envs[env.Name] = env;
            }

            lock (_defaultLock) { _default = env; }
        }

        private static string ResolveSystemPythonExe(TimeSpan timeout)
        {
            // Strategy:
            // 1) Try "python" from PATH (works on most)
            // 2) On Windows, also try "py -3 -c ..." to get interpreter path
            // 3) Fallback "python3" (common on Unix)
            //
            // We do NOT assume File.Exists for "python" because ProcessStartInfo can resolve it via PATH.
            // But we still want a real path: we ask python itself for sys.executable.

            // 1) Try "python"
            var p = TryGetExecutablePathFromInterpreter("python", timeout);
            if (!string.IsNullOrWhiteSpace(p)) return p;

            // 2) Windows launcher: py
            if (IsWindows())
            {
                // Prefer Python 3 via launcher
                // py -3 -c "import sys; print(sys.executable)"
                var res = RunProcess("py", "-3 -c " + QuoteArg("import sys; print(sys.executable)"), null, timeout);
                if (res.ExitCode == 0)
                {
                    var path = (res.StdOut ?? "").Trim();
                    if (!string.IsNullOrWhiteSpace(path) && File.Exists(path))
                        return path;
                }
            }

            // 3) Try "python3"
            p = TryGetExecutablePathFromInterpreter("python3", timeout);
            if (!string.IsNullOrWhiteSpace(p)) return p;

            throw new FileNotFoundException(
                "Could not resolve system Python. Tried: python, py -3 (Windows), python3.");
        }

        private static string TryGetExecutablePathFromInterpreter(string interpreterCommand, TimeSpan timeout)
        {
            var res = RunProcess(interpreterCommand, "-c " + QuoteArg("import sys; print(sys.executable)"), null, timeout);
            if (res.ExitCode != 0) return null;

            var path = (res.StdOut ?? "").Trim();

            // Some shells might return empty; bail.
            if (string.IsNullOrWhiteSpace(path)) return null;

            // sys.executable should be an actual file path. Validate when possible.
            if (File.Exists(path)) return path;

            // If it's not a file, still return it as last-ditch (but Create() will fail fast).
            return path;
        }

        private static bool IsWindows()
        {
            var p = Environment.OSVersion.Platform;
            return p == PlatformID.Win32NT || p == PlatformID.Win32S || p == PlatformID.Win32Windows || p == PlatformID.WinCE;
        }

        public string BaseDir { get; }

        public PyEnvManager(string baseDir = null)
        {
            BaseDir = string.IsNullOrWhiteSpace(baseDir)
                ? Path.Combine(
                    Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                    ".local", "python", "envs")
                : baseDir;

            Directory.CreateDirectory(BaseDir);
            Reload();
        }

        /// <summary>Snapshot of current environments.</summary>
        public IReadOnlyDictionary<string, PyEnv> Envs => _envs;

        /// <summary>Indexer like a dict. Throws if not found.</summary>
        public PyEnv this[string name] => _envs[name];

        public bool TryGet(string name, out PyEnv env) => _envs.TryGetValue(name, out env);

        public IEnumerable<string> Names() => _envs.Keys.OrderBy(x => x);

        public string GetEnvDir(string name)
        {
            if (string.IsNullOrWhiteSpace(name)) throw new ArgumentException("name is required.", nameof(name));
            return Path.Combine(BaseDir, name.Trim());
        }

        /// <summary>
        /// Re-scan BaseDir and rebuild internal dict.
        /// </summary>
        public void Reload()
        {
            _envs.Clear();

            var dftEnv = SystemDefault();

            _envs[$"system-{dftEnv.Version}"] = dftEnv;

            if (!Directory.Exists(BaseDir))
                return;

            foreach (var dir in Directory.GetDirectories(BaseDir))
            {
                var name = Path.GetFileName(dir);
                var pyExe = ResolvePythonExePath(dir);
                if (pyExe == null || !File.Exists(pyExe))
                    continue;

                try
                {
                    var env = PyEnv.Create(name, pyExe);
                    _envs[name] = env;
                }
                catch
                {
                    // Skip broken envs (missing python, invalid exe, etc.)
                }
            }
        }

        /// <summary>
        /// Create a new environment (or recreate if overwrite=true) with the given Python version.
        /// Uses uv to create the venv in BaseDir/name.
        /// </summary>
        public PyEnv Create(
            string name,
            string pythonVersion,
            bool overwrite = false,
            TimeSpan? timeout = null)
        {
            name = NormalizeName(name);

            var envDir = GetEnvDir(name);
            EnsureEnvDirReady(name, envDir, overwrite);

            var t = timeout ?? TimeSpan.FromMinutes(5);

            CreateVenvWithSystemUv(envDir, pythonVersion.ToString(), t);

            var pyExe = RequireVenvPython(envDir);

            // Register env first (so callers can see it even if bootstrap fails)
            var env = PyEnv.Create(name, pyExe, t);
            _envs[name] = env;

            BootstrapPipAndUv(envDir, pyExe, t);

            return env;
        }

        private static string NormalizeName(string name)
        {
            if (string.IsNullOrWhiteSpace(name))
                throw new ArgumentException("name is required.", nameof(name));
            return name.Trim();
        }

        private void EnsureEnvDirReady(string name, string envDir, bool overwrite)
        {
            if (Directory.Exists(envDir))
            {
                if (!overwrite)
                    throw new InvalidOperationException($"Env '{name}' already exists at: {envDir}");
                Delete(name);
            }
        }

        private void CreateVenvWithSystemUv(string envDir, string pythonVersion, TimeSpan timeout)
        {
            var uv = SystemDefault().FindUVPath();
            var args = $"venv --python {pythonVersion} {QuoteArg(envDir)}";

            var env = new System.Collections.Generic.Dictionary<string, string>
            {
                ["RUST_TLS_ALLOW_INVALID_CERTS"] = "1",
                ["UV_INSECURE_HOSTS"] = "github.com",
            };

            var res = RunProcess(uv, args, workingDirectory: BaseDir, timeout: timeout, env: env);

            if (res.ExitCode != 0)
            {
                if (IsWindows() && PyVersion.TryParse(pythonVersion, out var v))
                {
                    var localPython = $"C:\\Program Files\\Python{v.Major}{v.Minor}\\python.exe";
                    args = $"venv --python {QuoteArg(localPython)} {QuoteArg(envDir)}";

                    res = RunProcess(uv, args, workingDirectory: BaseDir, timeout: timeout, env: env);
                }
            }

            if (res.ExitCode != 0)
            {
                throw new InvalidOperationException(
                    "uv venv failed.\n" +
                    $"Cmd: {uv} {args}\n" +
                    res.ToString());
            }
        }

        private static string RequireVenvPython(string envDir)
        {
            var pyExe = ResolvePythonExePath(envDir);
            if (string.IsNullOrWhiteSpace(pyExe) || !File.Exists(pyExe))
                throw new FileNotFoundException("Created env but python executable not found.", pyExe ?? envDir);

            return pyExe;
        }

        private void BootstrapPipAndUv(string envDir, string pyExe, TimeSpan timeout)
        {
            // 1) Best-effort ensurepip (some distros disable it; don’t brick the env if it's missing)
            var ensurePip = RunProcess(pyExe, "-m ensurepip --upgrade", workingDirectory: envDir, timeout: timeout);

            // 2) Upgrade pip tooling (this should generally work if pip exists)
            var pipUp = RunProcess(
                pyExe,
                "-m pip install --upgrade pip setuptools wheel --disable-pip-version-check --no-input",
                workingDirectory: envDir,
                timeout: timeout);

            if (pipUp.ExitCode != 0)
            {
                throw new InvalidOperationException(
                    "Upgrading pip tooling failed.\n" +
                    $"Cmd: {pyExe} -m pip install --upgrade pip setuptools wheel\n" +
                    pipUp.ToString());
            }

            // 3) Install uv into the venv so future updates can use uv pip
            var uvInVenv = RunProcess(
                pyExe,
                "-m pip install --upgrade uv ygg --disable-pip-version-check --no-input",
                workingDirectory: envDir,
                timeout: timeout);

            if (uvInVenv.ExitCode != 0)
            {
                throw new InvalidOperationException(
                    "Installing uv into venv failed.\n" +
                    $"Cmd: {pyExe} -m pip install --upgrade uv\n" +
                    uvInVenv.ToString());
            }
        }

        /// <summary>
        /// Delete an environment directory and remove it from the manager.
        /// </summary>
        public void Delete(string name)
        {
            if (string.IsNullOrWhiteSpace(name)) throw new ArgumentException("name is required.", nameof(name));
            name = name.Trim();

            var envDir = GetEnvDir(name);

            // Remove from dict first to avoid stale state if delete fails partially.
            _envs.Remove(name);

            if (!Directory.Exists(envDir))
                return;

            // Nuke it.
            Directory.Delete(envDir, recursive: true);
        }

        /// <summary>
        /// Update an environment using uv pip (fast resolver).
        /// Under the hood this just calls env.PipInstall(... useUV:true ...)
        /// </summary>
        public PyProcessResult Update(
            string name,
            string[] packageSpecs = null,
            string requirementsFile = null,
            bool upgrade = true,
            bool forceReinstall = false,
            bool noDeps = false,
            string extraIndexUrl = null,
            string indexUrl = null,
            string trustedHost = null,
            string targetDirectory = null,
            string cacheDir = null,
            TimeSpan? timeout = null)
        {
            if (!_envs.TryGetValue(name, out var env))
                throw new KeyNotFoundException($"Env '{name}' not found.");

            // Ensure uv exists inside the env (auto-install via pip if missing).
            _ = env.FindUVPath(installIfMissing: true, timeout: timeout);

            if (packageSpecs != null && packageSpecs.Length > 0)
            {
                return env.PipInstall(
                    packageSpecs: packageSpecs,
                    upgrade: upgrade,
                    forceReinstall: forceReinstall,
                    noDeps: noDeps,
                    upgradePip: false,
                    useUV: true,
                    extraIndexUrl: extraIndexUrl,
                    indexUrl: indexUrl,
                    trustedHost: trustedHost,
                    requirementsFile: requirementsFile,
                    targetDirectory: targetDirectory,
                    cacheDir: cacheDir,
                    timeout: timeout);
            }

            // requirements-only update
            return env.PipInstall(
                packageSpec: null,
                upgrade: upgrade,
                forceReinstall: forceReinstall,
                noDeps: noDeps,
                upgradePip: false,
                useUV: true,
                extraIndexUrl: extraIndexUrl,
                indexUrl: indexUrl,
                trustedHost: trustedHost,
                requirementsFile: requirementsFile,
                targetDirectory: targetDirectory,
                cacheDir: cacheDir,
                timeout: timeout);
        }

        public PyProcessResult RunPythonCode(
            string code,
            object input = null,
            object output = null,
            string environment = null,
            string workingDirectory = null,
            TimeSpan? timeout = null)
        {
            PyEnv pyenv;

            if (string.IsNullOrEmpty(environment))
                pyenv = SystemDefault();
            else if (File.Exists(environment))
            {
                pyenv = PyEnv.Create(name: null, exePath: environment);
                _envs[pyenv.Name] = pyenv;
            }
            else
                pyenv = _envs[environment];

            
            var result = pyenv.RunCode(code, workingDirectory: workingDirectory, timeout: timeout);

            return result;
        }

        /// <summary>
        /// Resolve python executable path for a venv dir (Windows + Unix).
        /// </summary>
        private static string ResolvePythonExePath(string envDir)
        {
            // Windows venv layout
            var win = Path.Combine(envDir, "Scripts", "python.exe");
            if (File.Exists(win)) return win;

            // Unix venv layout
            var nix = Path.Combine(envDir, "bin", "python");
            if (File.Exists(nix)) return nix;

            // Some distros: python3
            var nix3 = Path.Combine(envDir, "bin", "python3");
            if (File.Exists(nix3)) return nix3;

            return null;
        }

        private static PyProcessResult RunProcess(
            string fileName, string arguments, string workingDirectory, TimeSpan timeout,
            IDictionary<string, string> env = null)
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

            // .NET Framework 4.8: use EnvironmentVariables (not psi.Environment)
            if (env != null)
            {
                foreach (var kv in env)
                {
                    if (string.IsNullOrWhiteSpace(kv.Key))
                        continue;

                    // If value is null -> remove the var (optional behavior)
                    if (kv.Value == null)
                    {
                        if (psi.EnvironmentVariables.ContainsKey(kv.Key))
                            psi.EnvironmentVariables.Remove(kv.Key);
                    }
                    else
                    {
                        psi.EnvironmentVariables[kv.Key] = kv.Value;
                    }
                }
            }

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

        private static string QuoteArg(string s) => "\"" + (s ?? "").Replace("\"", "\\\"") + "\"";
    }
}
