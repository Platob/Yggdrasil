"""Build the Cython extensions for `yggcy`.

Mirrors the layout of the sibling `rust/` package: each kernel lives
under `_yggcy.<sub>` so the bridge in `yggdrasil/cy.py` can re-publish
the compiled modules under the `yggdrasil.cy` namespace.
"""
from __future__ import annotations

from setuptools import Extension, setup
from Cython.Build import cythonize

EXTENSIONS = [
    Extension(
        name="_yggcy.io.url",
        sources=["src/_yggcy/io/url.pyx"],
    ),
]

setup(
    ext_modules=cythonize(
        EXTENSIONS,
        language_level=3,
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "initializedcheck": False,
            "cdivision": True,
        },
    ),
)
