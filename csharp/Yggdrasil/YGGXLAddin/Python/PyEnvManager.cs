using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

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
    ///
    /// Cache is keyed by python exe path (ExePath).
    /// A secondary index maps env Name -> exe path key for convenience/back-compat.
    /// </summary>
    public sealed class PyEnvManager
    {
        // 🔥 set env var too (choose a name that your app/tooling expects)
        const string DEFAULT_PYTHON_ENVIRONMENT = "DEFAULT_PYTHON_ENVIRONMENT"; // or "PYTHONHOME", "PYTHON_EXE", etc.

        public static PyEnvManager Instance = new PyEnvManager();

        // Primary cache: exePath(full) -> PyEnv
        private readonly Dictionary<string, PyEnv> _envsByExe =
            new Dictionary<string, PyEnv>(StringComparer.OrdinalIgnoreCase);

        // Secondary index: name -> exePath(full)
        private readonly Dictionary<string, string> _envsByName =
            new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);

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
        /// Supports DEFAULT_PYTHON_ENVIRONMENT as either:
        ///  - a python exe path
        ///  - a cached env name
        /// </summary>
        public PyEnv Default(TimeSpan? timeout = null)
        {
            lock (_defaultLock)
            {
                if (_default != null)
                    return _default;

                var v = Environment.GetEnvironmentVariable(DEFAULT_PYTHON_ENVIRONMENT, EnvironmentVariableTarget.User);

                if (!string.IsNullOrWhiteSpace(v))
                {
                    v = v.Trim();

                    // 1) If env var looks like a path and exists -> use it (exe-keyed cache)
                    if (File.Exists(v))
                    {
                        var exeKey = NormalizeExeKey(v);

                        if (!TryGetByExe(exeKey, out var cached))
                        {
                            cached = PyEnv.Create(name: null, exePath: v, timeout: timeout);
                            RegisterEnv(cached);
                        }

                        return _default = cached;
                    }

                    // 2) Otherwise treat it as a loaded name/key
                    if (TryGetByName(v, out var envFromName))
                        return _default = envFromName;

                    // 3) Slow scan fallback (safe)
                    foreach (var e in _envsByExe.Values)
                    {
                        if (string.Equals(e.Name, v, StringComparison.OrdinalIgnoreCase))
                            return _default = e;
                    }
                }

                return _default = SystemDefault(timeout);
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

            PyEnv env = null;

            // Prefer: name lookup
            if (!TryGetByName(nameOrFile, out env))
            {
                // If they passed an exe path, use exe-keyed cache
                if (!File.Exists(nameOrFile))
                    throw new KeyNotFoundException($"Env '{nameOrFile}' not found in cache.");

                var exeKey = NormalizeExeKey(nameOrFile);

                if (!TryGetByExe(exeKey, out env))
                {
                    env = PyEnv.Create(name: null, exePath: nameOrFile);
                    RegisterEnv(env);
                }
            }

            SetDefault(env);
        }

        public void SetDefault(PyEnv env)
        {
            if (env == null) throw new ArgumentNullException(nameof(env));

            RegisterEnv(env);

            lock (_defaultLock)
            {
                Environment.SetEnvironmentVariable(DEFAULT_PYTHON_ENVIRONMENT, env.ExePath, EnvironmentVariableTarget.Process);

                _ = Task.Run(() =>
                {
                    try
                    {
                        Environment.SetEnvironmentVariable(DEFAULT_PYTHON_ENVIRONMENT, env.ExePath, EnvironmentVariableTarget.User);
                    }
                    catch { }
                });

                _default = env;
            }
        }

        private static string ResolveSystemPythonExe(TimeSpan timeout)
        {
            // 1) Try "python"
            var p = TryGetExecutablePathFromInterpreter("python", timeout);
            if (!string.IsNullOrWhiteSpace(p)) return p;

            // 2) Windows launcher: py
            if (IsWindows())
            {
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
            if (string.IsNullOrWhiteSpace(path)) return null;

            if (File.Exists(path)) return path;
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

        /// <summary>Snapshot of current environments, keyed by exe path.</summary>
        public IReadOnlyDictionary<string, PyEnv> Envs => _envsByExe;

        /// <summary>Indexer by env name (back-compat). Throws if not found.</summary>
        public PyEnv this[string name]
        {
            get
            {
                if (TryGetByName(name, out var env)) return env;
                throw new KeyNotFoundException($"Env '{name}' not found.");
            }
        }

        /// <summary>Try-get by env name (back-compat).</summary>
        public bool TryGet(string name, out PyEnv env) => TryGetByName(name, out env);

        /// <summary>Env names currently indexed.</summary>
        public IEnumerable<string> Names() => _envsByName.Keys.OrderBy(x => x);

        public string GetEnvDir(string name)
        {
            if (string.IsNullOrWhiteSpace(name)) throw new ArgumentException("name is required.", nameof(name));
            return Path.Combine(BaseDir, name.Trim());
        }

        /// <summary>
        /// Re-scan BaseDir and rebuild internal dicts.
        /// </summary>
        public void Reload()
        {
            _envsByExe.Clear();
            _envsByName.Clear();

            var dftEnv = SystemDefault();
            RegisterEnv(dftEnv);

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
                    RegisterEnv(env);
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
            RegisterEnv(env);

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
            var uv = Default().FindUVPath();
            var args = $"venv --python {pythonVersion} {QuoteArg(envDir)}";

            var env = new Dictionary<string, string>
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
            // 1) Best-effort ensurepip
            _ = RunProcess(pyExe, "-m ensurepip --upgrade", workingDirectory: envDir, timeout: timeout);

            // 2) Upgrade pip tooling
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

            // 3) Install/upgrade uv in the venv (so future updates can use uv pip)
            var uvInVenv = RunProcess(
                pyExe,
                "-m pip install --upgrade uv --disable-pip-version-check --no-input",
                workingDirectory: envDir,
                timeout: timeout);

            if (uvInVenv.ExitCode != 0)
            {
                throw new InvalidOperationException(
                    "Installing/upgrading uv into venv failed.\n" +
                    $"Cmd: {pyExe} -m pip install --upgrade uv\n" +
                    uvInVenv.ToString());
            }

            // 4) Install/upgrade ygg (latest) at bootstrap too (fast path)
            // If uv is present, we'll do a uv-based ensure later; this makes env usable even if uv call fails.
            var yggInVenv = RunProcess(
                pyExe,
                "-m pip install --upgrade ygg --disable-pip-version-check --no-input",
                workingDirectory: envDir,
                timeout: timeout);

            if (yggInVenv.ExitCode != 0)
            {
                throw new InvalidOperationException(
                    "Installing/upgrading ygg into venv failed.\n" +
                    $"Cmd: {pyExe} -m pip install --upgrade ygg\n" +
                    yggInVenv.ToString());
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

            // Remove from indexes
            if (_envsByName.TryGetValue(name, out var exeKey))
            {
                _envsByName.Remove(name);
                _envsByExe.Remove(exeKey);
            }

            if (!Directory.Exists(envDir))
                return;

            Directory.Delete(envDir, recursive: true);
        }

        /// <summary>
        /// Update an environment using uv pip (fast resolver).
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
            if (!TryGetByName(name, out var env))
                throw new KeyNotFoundException($"Env '{name}' not found.");

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
            {
                pyenv = Default(timeout);
            }
            else if (File.Exists(environment))
            {
                // exe path provided
                var exeKey = NormalizeExeKey(environment);

                if (!TryGetByExe(exeKey, out pyenv))
                {
                    pyenv = PyEnv.Create(name: null, exePath: environment, timeout: timeout);
                    RegisterEnv(pyenv);
                }
            }
            else
            {
                // name provided
                pyenv = this[environment];
            }

            var result = pyenv.RunCode(code, workingDirectory: workingDirectory, timeout: timeout);
            return result;
        }

        /// <summary>
        /// Resolve python executable path for a venv dir (Windows + Unix).
        /// </summary>
        private static string ResolvePythonExePath(string envDir)
        {
            var win = Path.Combine(envDir, "Scripts", "python.exe");
            if (File.Exists(win)) return win;

            var nix = Path.Combine(envDir, "bin", "python");
            if (File.Exists(nix)) return nix;

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

        // -----------------------------
        // Cache helpers (exe-keyed)
        // -----------------------------
        private static string NormalizeExeKey(string exePath)
        {
            if (string.IsNullOrWhiteSpace(exePath)) return null;
            try { return Path.GetFullPath(exePath.Trim()); }
            catch { return exePath.Trim(); } // fallback if path is weird
        }

        private void RegisterEnv(PyEnv env)
        {
            if (env == null) return;

            var exeKey = NormalizeExeKey(env.ExePath);
            if (string.IsNullOrWhiteSpace(exeKey))
                throw new ArgumentException("Env exe path is required.", nameof(env));

            _envsByExe[exeKey] = env;

            if (!string.IsNullOrWhiteSpace(env.Name))
                _envsByName[env.Name] = exeKey;
        }

        private bool TryGetByName(string name, out PyEnv env)
        {
            env = null;
            if (string.IsNullOrWhiteSpace(name)) return false;

            if (_envsByName.TryGetValue(name.Trim(), out var exeKey) &&
                _envsByExe.TryGetValue(exeKey, out env))
                return true;

            return false;
        }

        private bool TryGetByExe(string exePathKey, out PyEnv env)
        {
            env = null;
            if (string.IsNullOrWhiteSpace(exePathKey)) return false;

            var key = NormalizeExeKey(exePathKey);
            if (string.IsNullOrWhiteSpace(key)) return false;

            return _envsByExe.TryGetValue(key, out env);
        }
    }
}
