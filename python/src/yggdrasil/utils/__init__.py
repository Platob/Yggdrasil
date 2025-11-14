"""Utility modules for Yggdrasil."""

# Import key utilities for easier access
from yggdrasil.utils.java import (
    is_java_installed,
    get_java_home,
    set_java_home,
    install_java,
    DEFAULT_JAVA_VERSION
)

__all__ = [
    "is_java_installed",
    "get_java_home",
    "set_java_home",
    "install_java",
    "DEFAULT_JAVA_VERSION"
]