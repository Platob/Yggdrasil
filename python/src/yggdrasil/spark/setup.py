"""Spark local environment setup utilities for Windows (and other platforms).

This module provides helpers to bootstrap a working local Spark environment
from scratch — particularly on Windows, where Hadoop native binaries
(winutils.exe) are not shipped with PySpark and Java compatibility can be
tricky.

Usage::

    from yggdrasil.spark.setup import ensure_spark_env, install_spark

    # Full one-shot setup: installs pyspark, winutils, sets env vars,
    # and returns a ready SparkSession.
    spark = ensure_spark_env()

    # Or step by step:
    install_spark()                # pip-install pyspark if missing
    ensure_hadoop_home()           # download winutils and set HADOOP_HOME
    configure_java_compat()        # patch JVM args for Java 17+
    spark = create_local_session() # build a local SparkSession
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

__all__ = [
    "ensure_spark_env",
    "ensure_java",
    "install_spark",
    "ensure_hadoop_home",
    "configure_java_compat",
    "create_local_session",
    "get_java_version",
    "get_java_version_from_bin",
    "spark_home_dir",
]

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# winutils release we pull from — matches Hadoop 3.x which ships with
# PySpark 3.5+ / 4.x.
_WINUTILS_REPO = "cdarlint/winutils"
_WINUTILS_BRANCH = "master"
_WINUTILS_HADOOP_VERSION = "hadoop-3.3.6"
_WINUTILS_BASE_URL = (
    f"https://raw.githubusercontent.com/{_WINUTILS_REPO}/"
    f"{_WINUTILS_BRANCH}/{_WINUTILS_HADOOP_VERSION}/bin"
)
_WINUTILS_FILES = ["winutils.exe", "hadoop.dll"]

# ---------------------------------------------------------------------------
# Zulu JDK auto-download constants
# ---------------------------------------------------------------------------
# We pin Zulu JDK 21 (LTS) — the sweet spot for Spark compatibility.
# Java 17 works too, but 21 is the current long-term target.  Java 24+ is
# explicitly *not* supported because PySpark Python workers segfault on it.
_ZULU_JAVA_MAJOR = 21
_ZULU_VERSION = "21.38.21"
_ZULU_JDK_VERSION = "21.0.5"

# Azul CDN URL pattern:
#   https://cdn.azul.com/zulu/bin/zulu{zulu_ver}-ca-jdk{jdk_ver}-{os}_{arch}.{ext}
_ZULU_CDN_BASE = "https://cdn.azul.com/zulu/bin"

# Mapping from (platform.system(), platform.machine()) to Zulu archive naming.
_ZULU_PLATFORM_MAP: dict[tuple[str, str], tuple[str, str, str]] = {
    # (os_name, arch) → (zulu_os, zulu_arch, archive_ext)
    ("Windows", "AMD64"):   ("win",     "x64",     "zip"),
    ("Windows", "x86_64"):  ("win",     "x64",     "zip"),
    ("Linux", "x86_64"):    ("linux",   "x64",     "tar.gz"),
    ("Linux", "aarch64"):   ("linux",   "aarch64", "tar.gz"),
    ("Darwin", "x86_64"):   ("macosx",  "x64",     "tar.gz"),
    ("Darwin", "arm64"):    ("macosx",  "aarch64", "tar.gz"),
}

# Compatible Java versions for Spark — anything outside this range triggers
# an auto-download when ensure_java(auto_download=True) is called.
_JAVA_COMPATIBLE_VERSIONS = {17, 21}
_JAVA_MAX_COMPATIBLE = 21

# Where we stash the downloaded JDK, relative to the user's home directory.
_JAVA_HOME_RELATIVE = Path(".yggdrasil") / "java"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_windows() -> bool:
    return platform.system() == "Windows"


def spark_home_dir() -> Path:
    """Return the default Yggdrasil-managed Spark home directory.

    On Windows this is ``%LOCALAPPDATA%\\.yggdrasil_spark``.
    On other platforms it's ``~/.yggdrasil_spark``.
    """
    if _is_windows():
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path.home()
    return base / ".yggdrasil" / "spark"


def get_java_version() -> int | None:
    """Return the major Java version (e.g. 17, 21, 24), or None if Java isn't found."""
    java = shutil.which("java")
    if java is None:
        return None
    try:
        result = subprocess.run(
            [java, "-version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # Java prints version to stderr, e.g.:
        #   java version "24.0.2" 2025-07-15
        #   openjdk version "17.0.9" 2023-10-17
        output = result.stderr or result.stdout or ""
        for line in output.splitlines():
            line = line.strip()
            if "version" in line.lower():
                # Extract the version string between quotes.
                start = line.find('"')
                end = line.find('"', start + 1)
                if start >= 0 and end > start:
                    ver_str = line[start + 1:end]
                    # "24.0.2" → 24, "17.0.9" → 17, "1.8.0_292" → 8
                    major = ver_str.split(".")[0]
                    if major == "1":
                        return int(ver_str.split(".")[1])
                    return int(major)
        return None
    except Exception:
        return None


def get_java_version_from_bin(java_bin: str | Path) -> int | None:
    """Return the major Java version for a specific java binary, or None on failure."""
    java_bin = str(java_bin)
    try:
        result = subprocess.run(
            [java_bin, "-version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        output = result.stderr or result.stdout or ""
        for line in output.splitlines():
            line = line.strip()
            if "version" in line.lower():
                start = line.find('"')
                end = line.find('"', start + 1)
                if start >= 0 and end > start:
                    ver_str = line[start + 1:end]
                    major = ver_str.split(".")[0]
                    if major == "1":
                        return int(ver_str.split(".")[1])
                    return int(major)
        return None
    except Exception:
        return None


def _zulu_download_url() -> tuple[str, str]:
    """Return (download_url, archive_filename) for the current platform.

    Raises:
        RuntimeError: If the current OS/arch combo isn't supported.
    """
    key = (platform.system(), platform.machine())
    mapping = _ZULU_PLATFORM_MAP.get(key)
    if mapping is None:
        raise RuntimeError(
            f"No Zulu JDK download available for {key[0]} / {key[1]}. "
            f"Supported platforms: {list(_ZULU_PLATFORM_MAP.keys())}. "
            "Install Java 17 or 21 manually and make sure 'java -version' works."
        )
    zulu_os, zulu_arch, ext = mapping
    filename = f"zulu{_ZULU_VERSION}-ca-jdk{_ZULU_JDK_VERSION}-{zulu_os}_{zulu_arch}.{ext}"
    url = f"{_ZULU_CDN_BASE}/{filename}"
    return url, filename


def _managed_java_home() -> Path:
    """Return the path where we install a managed Zulu JDK."""
    return Path.home() / _JAVA_HOME_RELATIVE


def _find_java_bin(java_home: Path) -> Path | None:
    """Locate the java binary inside a JDK directory tree.

    Zulu archives unpack into a directory like ``zulu21.38.21-ca-jdk21.0.5-win_x64/``
    so we need to look one level deeper.
    """
    # Direct layout: java_home/bin/java(.exe)
    exe = "java.exe" if _is_windows() else "java"
    candidate = java_home / "bin" / exe
    if candidate.is_file():
        return candidate

    # One-level-deeper layout: java_home/<unpacked_dir>/bin/java(.exe)
    for child in java_home.iterdir():
        if child.is_dir():
            candidate = child / "bin" / exe
            if candidate.is_file():
                return candidate

    return None


def ensure_java(
    auto_download: bool = True,
    force: bool = False,
) -> Path:
    """Make sure a compatible Java is available, downloading Zulu JDK 21 if needed.

    Resolution order:
    1. Check the system ``java`` on PATH — if it's a compatible version, use it.
    2. Check the managed Yggdrasil Java at ``~/.yggdrasil/java`` — if present
       and valid, prepend it to PATH/JAVA_HOME and use it.
    3. If *auto_download* is True, download Azul Zulu JDK 21 into
       ``~/.yggdrasil/java`` and set it up.
    4. Otherwise, raise with guidance on what to install.

    Args:
        auto_download: Whether to download Zulu JDK 21 when no compatible Java
                       is found.  Defaults to True.
        force:         Re-download even if a managed JDK already exists.

    Returns:
        The resolved JAVA_HOME path.

    Raises:
        RuntimeError: If no compatible Java is available and *auto_download*
                      is False, or the download fails.
    """
    # 1. Check system Java.
    sys_ver = get_java_version()
    if sys_ver is not None and sys_ver <= _JAVA_MAX_COMPATIBLE:
        LOGGER.info("System Java %d is compatible — using it.", sys_ver)
        java_bin = shutil.which("java")
        if java_bin:
            java_home = Path(java_bin).resolve().parent.parent
            os.environ["JAVA_HOME"] = str(java_home)
            return java_home

    # 2. Check managed Java.
    managed_home = _managed_java_home()
    if managed_home.exists() and not force:
        java_bin = _find_java_bin(managed_home)
        if java_bin is not None:
            managed_ver = get_java_version_from_bin(java_bin)
            if managed_ver is not None and managed_ver <= _JAVA_MAX_COMPATIBLE:
                LOGGER.info(
                    "Found managed Zulu JDK %d at %s — using it.",
                    managed_ver, managed_home,
                )
                _activate_managed_java(java_bin.parent.parent)
                return java_bin.parent.parent

    # 3. Auto-download.
    if not auto_download:
        msg = (
            "No compatible Java found (need Java 17 or 21). "
            f"System Java version: {sys_ver or 'not found'}. "
            "Install a compatible JDK from https://www.azul.com/downloads/ "
            "or https://adoptium.net/, or call ensure_java(auto_download=True) "
            "to let Yggdrasil download Zulu JDK 21 automatically."
        )
        raise RuntimeError(msg)

    return _download_zulu_jdk(managed_home, force=force)


def _activate_managed_java(java_home: Path) -> None:
    """Set JAVA_HOME and prepend the JDK's bin dir to PATH."""
    os.environ["JAVA_HOME"] = str(java_home)
    bin_dir = str(java_home / "bin")
    path_dirs = os.environ.get("PATH", "").split(os.pathsep)
    if bin_dir not in path_dirs:
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
    LOGGER.info("JAVA_HOME set to %s", java_home)


def _download_zulu_jdk(dest: Path, force: bool = False) -> Path:
    """Download and extract Zulu JDK into *dest*.

    Returns the actual JAVA_HOME (the directory containing ``bin/java``).
    """
    import urllib.request

    url, filename = _zulu_download_url()
    dest.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Downloading Zulu JDK %s from %s ...", _ZULU_JDK_VERSION, url)

    # Download into a temp file, then extract.  This way a failed download
    # doesn't leave a half-baked directory behind.
    with tempfile.TemporaryDirectory() as tmpdir:
        archive_path = Path(tmpdir) / filename
        try:
            urllib.request.urlretrieve(url, str(archive_path))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download Zulu JDK from {url}: {exc}. "
                "Check your network connection, or install Java 17 or 21 manually."
            ) from exc

        # Extract — zip on Windows, tar.gz elsewhere.
        if filename.endswith(".zip"):
            with zipfile.ZipFile(archive_path) as zf:
                zf.extractall(dest)
        elif filename.endswith(".tar.gz"):
            import tarfile
            with tarfile.open(archive_path, "r:gz") as tf:
                tf.extractall(dest)
        else:
            raise RuntimeError(
                f"Unexpected archive format: {filename}. "
                "This is a bug in Yggdrasil — please file an issue."
            )

    # Find the actual java binary inside the extracted tree.
    java_bin = _find_java_bin(dest)
    if java_bin is None:
        raise RuntimeError(
            f"Downloaded and extracted Zulu JDK to {dest}, but could not find "
            "a java binary inside it. The archive layout may have changed. "
            "Install Java 17 or 21 manually as a workaround."
        )

    java_home = java_bin.parent.parent
    # Verify the downloaded JDK works.
    downloaded_ver = get_java_version_from_bin(java_bin)
    if downloaded_ver is None:
        LOGGER.warning(
            "Downloaded Zulu JDK but could not verify its version. "
            "Proceeding anyway — if Spark fails, install Java manually."
        )
    else:
        LOGGER.info("Zulu JDK %d installed at %s", downloaded_ver, java_home)

    _activate_managed_java(java_home)
    return java_home


# ---------------------------------------------------------------------------
# Step 1: Install PySpark
# ---------------------------------------------------------------------------

def install_spark(
    version: str | None = None,
    extras: list[str] | None = None,
) -> None:
    """Install PySpark via pip if it's not already importable.

    Args:
        version: PySpark version pin, e.g. ``"4.1.1"``. If ``None``, installs
                 the latest version compatible with the current Python.
        extras:  Additional pip packages to install alongside PySpark,
                 e.g. ``["delta-spark"]``.
    """
    try:
        import pyspark  # noqa: F401
        LOGGER.info("PySpark %s is already installed.", pyspark.__version__)
        return
    except ImportError:
        pass

    pkg = f"pyspark=={version}" if version else "pyspark"
    cmd = [sys.executable, "-m", "pip", "install", pkg]
    if extras:
        cmd.extend(extras)

    LOGGER.info("Installing PySpark: %s", " ".join(cmd))
    subprocess.check_call(cmd)

    # Verify it's importable now.
    import pyspark  # noqa: F401
    LOGGER.info("PySpark %s installed successfully.", pyspark.__version__)


# ---------------------------------------------------------------------------
# Step 2: Ensure Hadoop native binaries on Windows
# ---------------------------------------------------------------------------

def _download_file(url: str, dest: Path) -> None:
    """Download a file from *url* to *dest* using urllib (no extra deps)."""
    import urllib.request
    LOGGER.info("Downloading %s → %s", url, dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, str(dest))


def ensure_hadoop_home(
    hadoop_home: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Ensure HADOOP_HOME is set and contains winutils.exe (Windows only).

    On non-Windows platforms this is a no-op — Hadoop native libs aren't
    needed for local-mode Spark.

    Args:
        hadoop_home: Explicit directory to use.  If ``None``, uses the
                     Yggdrasil-managed default under ``spark_home_dir()``.
        force:       Re-download even if the binaries already exist.

    Returns:
        The resolved HADOOP_HOME path.
    """
    if not _is_windows():
        # Non-Windows: HADOOP_HOME is nice-to-have but not required for local mode.
        existing = os.environ.get("HADOOP_HOME", "")
        if existing:
            return Path(existing)
        LOGGER.debug("Non-Windows platform — skipping winutils setup.")
        return Path("")

    if hadoop_home is None:
        # Check if already set and valid.
        existing = os.environ.get("HADOOP_HOME", "")
        if existing and (Path(existing) / "bin" / "winutils.exe").is_file() and not force:
            LOGGER.info("HADOOP_HOME already set and valid: %s", existing)
            return Path(existing)
        hadoop_home = spark_home_dir() / "hadoop"

    hadoop_home = Path(hadoop_home)
    bin_dir = hadoop_home / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)

    winutils_path = bin_dir / "winutils.exe"

    if winutils_path.is_file() and not force:
        LOGGER.info("winutils.exe already exists at %s", winutils_path)
    else:
        for filename in _WINUTILS_FILES:
            url = f"{_WINUTILS_BASE_URL}/{filename}"
            dest = bin_dir / filename
            try:
                _download_file(url, dest)
                # Make executable (matters on some setups).
                dest.chmod(dest.stat().st_mode | stat.S_IEXEC)
            except Exception as exc:
                LOGGER.warning(
                    "Failed to download %s from %s: %s. "
                    "Spark may still work for basic local-mode operations, "
                    "but filesystem operations could fail.",
                    filename, url, exc,
                )

    # Set the environment variables so Spark picks them up.
    os.environ["HADOOP_HOME"] = str(hadoop_home)
    LOGGER.info("HADOOP_HOME set to %s", hadoop_home)

    # Also add bin to PATH so winutils.exe is discoverable.
    path_dirs = os.environ.get("PATH", "").split(os.pathsep)
    bin_str = str(bin_dir)
    if bin_str not in path_dirs:
        os.environ["PATH"] = bin_str + os.pathsep + os.environ.get("PATH", "")

    return hadoop_home


# ---------------------------------------------------------------------------
# Step 3: Configure Java compatibility
# ---------------------------------------------------------------------------

def configure_java_compat(java_version: int | None = None) -> list[str]:
    """Return (and set) JVM options needed for modern Java + Spark.

    Java 17+ restricts reflective access that Spark/Hadoop need.  PySpark 4.x
    handles most of this internally, but some edge cases (especially on Windows
    with older Hadoop jars) still need extra ``--add-opens`` flags.

    This function detects the Java version and sets ``PYSPARK_SUBMIT_ARGS``
    so the Spark driver JVM picks up the right flags.

    Args:
        java_version: Override auto-detected Java major version.

    Returns:
        The list of extra JVM flags that were applied.
    """
    if java_version is None:
        java_version = get_java_version()

    if java_version is None:
        LOGGER.warning(
            "Could not detect Java version. If Spark fails to start, "
            "make sure Java 17 or 21 is installed and on PATH."
        )
        return []

    LOGGER.info("Detected Java version: %d", java_version)

    extra_flags: list[str] = []

    if java_version >= 17:
        # Core modules Spark/Hadoop need reflective access to.
        # PySpark 4.x handles many of these, but we add the ones that
        # are still needed for local-mode on Windows.
        extra_flags.extend([
            "--add-opens=java.base/java.lang=ALL-UNNAMED",
            "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED",
            "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED",
            "--add-opens=java.base/java.io=ALL-UNNAMED",
            "--add-opens=java.base/java.net=ALL-UNNAMED",
            "--add-opens=java.base/java.nio=ALL-UNNAMED",
            "--add-opens=java.base/java.util=ALL-UNNAMED",
            "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED",
            "--add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED",
            "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
            "--add-opens=java.base/sun.nio.cs=ALL-UNNAMED",
            "--add-opens=java.base/sun.util.calendar=ALL-UNNAMED",
        ])


    if not extra_flags:
        return []

    # Build a single combined extraJavaOptions value.  Spark only reads one
    # value per config key, so we must combine all flags into one string.
    combined = " ".join(extra_flags)

    # PYSPARK_SUBMIT_ARGS is the env var PySpark reads before launching the
    # JVM gateway.  It must end with "pyspark-shell" (or similar command).
    # We inject a single --conf with all flags joined.
    submit_args = os.environ.get("PYSPARK_SUBMIT_ARGS", "")
    driver_conf = f'--conf "spark.driver.extraJavaOptions={combined}"'

    if driver_conf not in submit_args:
        # Strip any trailing command word so we can re-append it.
        tail = "pyspark-shell"
        base = submit_args.replace(tail, "").strip()
        os.environ["PYSPARK_SUBMIT_ARGS"] = f"{base} {driver_conf} {tail}".strip()
        LOGGER.info("Set PYSPARK_SUBMIT_ARGS: %s", os.environ["PYSPARK_SUBMIT_ARGS"])

    return extra_flags


# ---------------------------------------------------------------------------
# Step 4: Create a local SparkSession
# ---------------------------------------------------------------------------

def create_local_session(
    app_name: str = "yggdrasil-local",
    cores: str = "local[*]",
    extra_config: dict[str, str] | None = None,
    **kwargs: Any,
) -> "SparkSession":
    """Create a local-mode SparkSession with sensible defaults.

    Args:
        app_name:     Application name shown in the Spark UI.
        cores:        Master URL, default ``"local[*]"`` uses all cores.
        extra_config: Additional Spark config key/value pairs.
        **kwargs:     Passed through to ``SparkSession.builder.config``.

    Returns:
        A ready-to-use SparkSession.
    """
    from pyspark.sql import SparkSession

    builder = (
        SparkSession.builder
        .master(cores)
        .appName(app_name)
        # Arrow-based optimizations — these make Pandas ↔ Spark transfers fast.
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true")
        # Silence the noisy Hadoop warnings on Windows.
        .config("spark.sql.warehouse.dir", str(spark_home_dir() / "warehouse"))
        .config("spark.driver.host", "localhost")
    )

    if extra_config:
        for k, v in extra_config.items():
            builder = builder.config(k, v)

    for k, v in kwargs.items():
        builder = builder.config(k, str(v))

    session = builder.getOrCreate()
    session.sparkContext.setLogLevel("WARN")
    LOGGER.info(
        "SparkSession ready — version %s, master %s",
        session.version,
        cores,
    )
    return session


# ---------------------------------------------------------------------------
# All-in-one entry point
# ---------------------------------------------------------------------------

def ensure_spark_env(
    app_name: str = "yggdrasil-local",
    cores: str = "local[*]",
    install: bool = True,
    pyspark_version: str | None = None,
    hadoop_home: str | Path | None = None,
    extra_config: dict[str, str] | None = None,
    **kwargs: Any,
) -> "SparkSession":
    """One-shot bootstrap: install, configure, and return a local SparkSession.

    This is the easiest way to get Spark running on a fresh Windows machine.
    Call it once and you're good::

        from yggdrasil.spark.setup import ensure_spark_env
        spark = ensure_spark_env()

    It handles:
    1. Installing PySpark if missing
    2. Downloading winutils.exe for Hadoop on Windows
    3. Configuring JVM flags for Java 17+ / 24+ compatibility
    4. Creating a local SparkSession with Arrow optimizations

    Args:
        app_name:         Spark application name.
        cores:            Master URL (default: all local cores).
        install:          Whether to pip-install PySpark if missing.
        pyspark_version:  Pin a specific PySpark version.
        hadoop_home:      Custom HADOOP_HOME path (Windows only).
        extra_config:     Extra Spark config entries.
        **kwargs:         Passed to ``create_local_session``.

    Returns:
        A ready SparkSession.

    Raises:
        ImportError: If PySpark is not installed and ``install=False``.
        RuntimeError: If Java is not found on PATH.
    """
    # Check Java first — if missing or incompatible, download Zulu JDK 21.
    ensure_java(auto_download=True)

    # Step 1: Install PySpark.
    if install:
        install_spark(version=pyspark_version)

    # Step 2: Hadoop native binaries (Windows).
    ensure_hadoop_home(hadoop_home=hadoop_home)

    # Step 3: Java compatibility flags.
    configure_java_compat()

    # Step 4: Create the session.
    return create_local_session(
        app_name=app_name,
        cores=cores,
        extra_config=extra_config,
        **kwargs,
    )

