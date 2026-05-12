"""Build the Cython extensions that ship inside the ``ygg`` wheel.

Most of the package metadata lives in ``pyproject.toml`` — this script
only exists so setuptools picks up the ``ext_modules`` list. The .pyx
sources sit next to the Python modules they accelerate (e.g.
``yggdrasil/io/_url.pyx`` next to ``yggdrasil/io/url.py``); the
compiled ``.so`` lands at the same import path and the matching
``url.py`` reaches it via a guarded ``from . import _url`` so a
fallback to the pure-Python implementation still works when the
extension didn't compile.
"""
from __future__ import annotations

from setuptools import Extension, setup

try:
    from Cython.Build import cythonize
except ImportError:  # pragma: no cover — build-time only
    cythonize = None

# Each entry compiles to ``yggdrasil/<path>/<name>.so`` and is
# imported by the matching ``.py`` next to it.
CYTHON_SOURCES: list[tuple[str, str]] = [
    ("yggdrasil.io._url", "src/yggdrasil/io/_url.pyx"),
]

_extensions = [
    Extension(name=mod, sources=[src]) for mod, src in CYTHON_SOURCES
]

if cythonize is not None:
    ext_modules = cythonize(
        _extensions,
        language_level=3,
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "initializedcheck": False,
            "cdivision": True,
        },
    )
else:
    # No Cython available — the .pyx sources won't compile, but the
    # pure-Python fallbacks in ``yggdrasil.*`` still cover every kernel.
    ext_modules = []

setup(ext_modules=ext_modules)
