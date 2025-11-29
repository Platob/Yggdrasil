# modules.py
import functools
import subprocess
import sys
from typing import Callable, Optional, Set


class ModuleInstallError(ImportError):
    """Raised when a module cannot be imported even after attempting installation."""


# Cache of pip packages we've already attempted to install
_INSTALLED_PACKAGES: Set[str] = set()


def install_package(
    package: str,
    *,
    upgrade: bool = False,
    quiet: bool = False,
) -> None:
    """
    Install a Python package at runtime using pip.

    Parameters
    ----------
    package:
        Name of the package to install (as used with pip).
    upgrade:
        Whether to pass --upgrade.
    quiet:
        Whether to pass --quiet to pip.
    """
    if package in _INSTALLED_PACKAGES:
        # We've already attempted this once in this process.
        return

    cmd = [sys.executable, "-m", "pip", "install"]
    if upgrade:
        cmd.append("--upgrade")
    if quiet:
        cmd.append("--quiet")
    cmd.append(package)

    try:
        subprocess.check_call(cmd)
        _INSTALLED_PACKAGES.add(package)
    except subprocess.CalledProcessError as e:
        raise ModuleInstallError(
            f"Failed to install package {package!r} via pip."
        ) from e


def _infer_package_name(
    exc: ImportError,
    module_names: tuple[str, ...],
    package: Optional[str],
) -> str:
    """
    Infer which pip package to install from:
    - explicit `package` argument, or
    - ImportError.name, or
    - first module_name, or
    - raise if we can't guess.
    """
    if package:
        return package

    # ImportError.name is set for "No module named 'xxx'" cases
    missing = getattr(exc, "name", None)
    if missing:
        return missing.split(".", 1)[0]

    if module_names:
        return module_names[0].split(".", 1)[0]

    raise ModuleInstallError(
        "Cannot infer which package to install from ImportError; "
        "provide `package=` to @require_modules."
    )


def require_modules(
    *module_names: str,
    package: Optional[str] = None,
    auto_install: bool = True,
    retries: int = 1,
) -> Callable:
    """
    Decorator that calls the wrapped function, and on ImportError:

    - infers which package to install (from `package`, ImportError.name,
      or module_names),
    - runs `pip install`,
    - retries the function call up to `retries` times.

    Behaviors
    ---------
    - `module_names` may be empty.
    - If both `module_names` and `package` are empty, the decorator
      will still catch ImportError, but can only auto-install if the
      ImportError has a valid `.name` attribute.
    - If `auto_install=False`, ImportError is just re-raised.

    Examples
    --------
    @require_modules("polars")
    def fn():
        import polars as pl
        ...

    @require_modules("pyspark.sql", package="pyspark")
    def fn():
        from pyspark.sql import SparkSession
        ...

    @require_modules(package="polars")
    def fn():
        import polars as pl
        ...

    @require_modules()  # allowed; only reacts if ImportError happens
    def fn():
        ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except ImportError as exc:
                    # If we don't auto-install, just bubble up
                    if not auto_install or attempt >= retries:
                        raise

                    pip_name = _infer_package_name(exc, module_names, package)
                    install_package(pip_name)
                    attempt += 1
                    # loop: retry calling the function after install

        return wrapper

    return decorator


__all__ = [
    "ModuleInstallError",
    "install_package",
    "require_modules",
]
