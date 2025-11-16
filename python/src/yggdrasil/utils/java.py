"""Java installation and configuration utilities for Yggdrasil.

This module provides utilities for installing and configuring Java, which is
required for using Spark with Yggdrasil.
"""

import logging
import os
import platform
import subprocess
import tempfile
import urllib.request
import zipfile
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Default Java version to install
DEFAULT_JAVA_VERSION = "17"

# JDK download URLs by version and platform
JDK_URLS = {
    "17": {
        "windows": "https://github.com/adoptium/temurin17-binaries/releases/download/jdk-17.0.10%2B7/OpenJDK17U-jdk_x64_windows_hotspot_17.0.10_7.zip",
        "linux": "https://github.com/adoptium/temurin17-binaries/releases/download/jdk-17.0.10%2B7/OpenJDK17U-jdk_x64_linux_hotspot_17.0.10_7.tar.gz",
        "darwin": "https://github.com/adoptium/temurin17-binaries/releases/download/jdk-17.0.10%2B7/OpenJDK17U-jdk_x64_mac_hotspot_17.0.10_7.tar.gz"
    },
    "11": {
        "windows": "https://github.com/adoptium/temurin11-binaries/releases/download/jdk-11.0.22%2B7/OpenJDK11U-jdk_x64_windows_hotspot_11.0.22_7.zip",
        "linux": "https://github.com/adoptium/temurin11-binaries/releases/download/jdk-11.0.22%2B7/OpenJDK11U-jdk_x64_linux_hotspot_11.0.22_7.tar.gz",
        "darwin": "https://github.com/adoptium/temurin11-binaries/releases/download/jdk-11.0.22%2B7/OpenJDK11U-jdk_x64_mac_hotspot_11.0.22_7.tar.gz"
    },
    "8": {
        "windows": "https://github.com/adoptium/temurin8-binaries/releases/download/jdk8u402-b06/OpenJDK8U-jdk_x64_windows_hotspot_8u402b06.zip",
        "linux": "https://github.com/adoptium/temurin8-binaries/releases/download/jdk8u402-b06/OpenJDK8U-jdk_x64_linux_hotspot_8u402b06.tar.gz",
        "darwin": "https://github.com/adoptium/temurin8-binaries/releases/download/jdk8u402-b06/OpenJDK8U-jdk_x64_mac_hotspot_8u402b06.tar.gz"
    }
}


def is_java_installed() -> Tuple[bool, Optional[str]]:
    """Check if Java is installed and get the version.

    Returns:
        A tuple of (is_installed, version_string)
    """
    try:
        java_process = subprocess.run(
            ["java", "-version"],
            capture_output=True,
            text=True,
            check=False
        )

        # Java prints version info to stderr for historical reasons
        output = java_process.stderr

        if java_process.returncode == 0 and output:
            # Extract version from output like: openjdk version "11.0.22"
            import re
            version_match = re.search(r'version "([^"]+)"', output)
            if version_match:
                return True, version_match.group(1)
            return True, None
        return False, None
    except FileNotFoundError:
        return False, None


def get_java_home() -> Optional[str]:
    """Get the current JAVA_HOME environment variable.

    Returns:
        The path to JAVA_HOME or None if not set
    """
    java_home = os.environ.get("JAVA_HOME")
    if java_home and os.path.exists(java_home):
        return java_home
    return None


def set_java_home(java_path: str, persist: bool = False) -> bool:
    """Set JAVA_HOME to the specified path.

    Args:
        java_path: Path to the Java installation
        persist: If True, attempt to make the change permanent in the user's environment

    Returns:
        True if successful, False otherwise
    """
    if not os.path.exists(java_path):
        logger.error(f"Cannot set JAVA_HOME: Path {java_path} does not exist")
        return False

    # Set for current process
    os.environ["JAVA_HOME"] = java_path

    # Add to PATH for current process
    bin_dir = os.path.join(java_path, "bin")
    if bin_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")

    if persist:
        try:
            if platform.system() == "Windows":
                # On Windows, use setx command to persist environment variables
                subprocess.run(["setx", "JAVA_HOME", java_path], check=True)
                # PATH is handled automatically by Windows when using JAVA_HOME
                return True
            elif platform.system() in ("Linux", "Darwin"):
                # For Linux/Mac, add to shell profile files
                home = os.path.expanduser("~")
                profile_files = []

                if platform.system() == "Linux":
                    profile_files = [
                        os.path.join(home, ".bashrc"),
                        os.path.join(home, ".bash_profile"),
                        os.path.join(home, ".profile")
                    ]
                else:  # Darwin (macOS)
                    profile_files = [
                        os.path.join(home, ".bash_profile"),
                        os.path.join(home, ".zshrc")
                    ]

                for profile in profile_files:
                    if os.path.exists(profile):
                        with open(profile, 'r') as f:
                            content = f.read()

                        # Add JAVA_HOME if not already present
                        if f"JAVA_HOME={java_path}" not in content:
                            with open(profile, 'a') as f:
                                f.write(f"\n# Added by Yggdrasil\nexport JAVA_HOME={java_path}\n")
                                f.write(f"export PATH=$JAVA_HOME/bin:$PATH\n")

                        return True

                logger.warning("Could not find any profile files to update")
                return False
            else:
                logger.warning(f"Persisting environment variables not supported on {platform.system()}")
                return False
        except Exception as e:
            logger.error(f"Error persisting JAVA_HOME: {e}")
            return False

    return True


def download_and_extract_jdk(version: str = DEFAULT_JAVA_VERSION,
                            dest_dir: Optional[str] = None) -> Optional[str]:
    """Download and extract the JDK.

    Args:
        version: Java version to download (8, 11, 17)
        dest_dir: Directory to install Java to (default: ~/.yggdrasil/java)

    Returns:
        Path to the extracted JDK or None if failed
    """
    system = platform.system().lower()

    if system not in ("windows", "linux", "darwin"):
        logger.error(f"Unsupported platform: {system}")
        return None

    if version not in JDK_URLS:
        logger.error(f"Unsupported Java version: {version}. Supported versions: {list(JDK_URLS.keys())}")
        return None

    if version not in JDK_URLS[version]:
        logger.error(f"No JDK URL found for version {version} on platform {system}")
        return None

    # Determine destination directory
    if dest_dir is None:
        home_dir = os.path.expanduser("~")
        dest_dir = os.path.join(home_dir, ".yggdrasil", "java")

    os.makedirs(dest_dir, exist_ok=True)

    # Download JDK
    jdk_url = JDK_URLS[version][system]
    jdk_filename = os.path.basename(jdk_url)
    download_path = os.path.join(tempfile.gettempdir(), jdk_filename)

    logger.info(f"Downloading JDK {version} from {jdk_url}")
    try:
        urllib.request.urlretrieve(jdk_url, download_path)
    except Exception as e:
        logger.error(f"Error downloading JDK: {e}")
        return None

    # Extract the archive
    logger.info(f"Extracting JDK to {dest_dir}")
    java_home = None

    try:
        if system == "windows":
            with zipfile.ZipFile(download_path, 'r') as zip_ref:
                zip_ref.extractall(dest_dir)

                # The first directory in the archive is the JDK home
                jdk_dir = next(
                    (item for item in zip_ref.namelist() if item.endswith('/')),
                    None
                )
                if jdk_dir:
                    java_home = os.path.join(dest_dir, jdk_dir.rstrip('/'))
        else:
            # For Linux/Mac, we need to extract the tar.gz file
            import tarfile
            with tarfile.open(download_path, 'r:gz') as tar:
                # Get the root directory name
                root_dirs = {item.name.split('/')[0] for item in tar.getmembers() if '/' in item.name}
                if root_dirs:
                    root_dir = next(iter(root_dirs))
                    tar.extractall(dest_dir)
                    java_home = os.path.join(dest_dir, root_dir)
    except Exception as e:
        logger.error(f"Error extracting JDK: {e}")
        return None
    finally:
        # Clean up the downloaded file
        if os.path.exists(download_path):
            os.remove(download_path)

    # Ensure we found the Java home directory
    if not java_home or not os.path.exists(java_home):
        logger.error("Could not determine Java home directory from extracted files")
        return None

    # Verify the installation by checking for the java executable
    java_bin = os.path.join(java_home, "bin", "java" + (".exe" if system == "windows" else ""))
    if not os.path.exists(java_bin):
        logger.error(f"Java executable not found at {java_bin}")
        return None

    return java_home


def install_java(version: str = DEFAULT_JAVA_VERSION,
                dest_dir: Optional[str] = None,
                set_env: bool = True,
                persist_env: bool = True) -> Optional[str]:
    """Install Java if not already installed.

    Args:
        version: Java version to install (8, 11, 17)
        dest_dir: Directory to install Java to (default: ~/.yggdrasil/java)
        set_env: If True, set JAVA_HOME environment variable
        persist_env: If True, attempt to make environment changes permanent

    Returns:
        Path to the Java installation or None if failed
    """
    # Check if Java is already installed
    java_installed, java_version = is_java_installed()
    if java_installed:
        logger.info(f"Java is already installed (version {java_version})")

        # If already installed, just set JAVA_HOME if requested
        java_home = get_java_home()
        if java_home:
            logger.info(f"JAVA_HOME is already set to {java_home}")
            return java_home
        elif set_env:
            # Try to locate Java and set JAVA_HOME
            system = platform.system()
            java_home = None

            try:
                if system == "Windows":
                    # Try to find Java in Program Files
                    for program_files in ["C:\\Program Files\\Java", "C:\\Program Files (x86)\\Java"]:
                        if os.path.exists(program_files):
                            # Look for jdk directories
                            jdk_dirs = [d for d in os.listdir(program_files)
                                      if os.path.isdir(os.path.join(program_files, d))
                                      and d.startswith(("jdk", "openjdk"))]
                            if jdk_dirs:
                                # Use the first one found
                                java_home = os.path.join(program_files, jdk_dirs[0])
                                break
                elif system in ("Linux", "Darwin"):
                    # Try common locations
                    common_locations = [
                        "/usr/lib/jvm",
                        "/usr/java",
                        "/opt/java",
                        "/Library/Java/JavaVirtualMachines"
                    ]

                    for location in common_locations:
                        if os.path.exists(location):
                            # Look for directories containing "jdk" or "openjdk"
                            for dir_name in os.listdir(location):
                                if "jdk" in dir_name or "openjdk" in dir_name:
                                    # On macOS, the actual JDK is in Contents/Home
                                    if system == "Darwin" and "JavaVirtualMachines" in location:
                                        potential_home = os.path.join(location, dir_name, "Contents/Home")
                                    else:
                                        potential_home = os.path.join(location, dir_name)

                                    # Verify it's a JDK by checking for javac
                                    javac_path = os.path.join(
                                        potential_home, "bin",
                                        "javac" + (".exe" if system == "Windows" else "")
                                    )
                                    if os.path.exists(javac_path):
                                        java_home = potential_home
                                        break

                            if java_home:
                                break

                if java_home:
                    logger.info(f"Found existing Java installation at {java_home}")
                    set_java_home(java_home, persist_env)
                    return java_home
                else:
                    logger.warning("Java is installed but could not determine JAVA_HOME")
            except Exception as e:
                logger.error(f"Error trying to locate Java: {e}")

    # Download and install Java
    logger.info(f"Installing Java {version}...")
    java_home = download_and_extract_jdk(version, dest_dir)

    if not java_home:
        logger.error("Failed to install Java")
        return None

    logger.info(f"Java {version} installed at {java_home}")

    # Set environment variables
    if set_env:
        set_java_home(java_home, persist_env)
        logger.info(f"Set JAVA_HOME to {java_home}")

    return java_home