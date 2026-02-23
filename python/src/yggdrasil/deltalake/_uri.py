"""
_uri.py — URI scheme detection and stripping for PyArrow filesystem paths.

PyArrow filesystem implementations (``S3FileSystem``, ``GcsFileSystem``,
``AzureBlobFileSystem``, ``LocalFileSystem``) all expect *bare* paths without
a URI scheme prefix.  These helpers normalise any supported scheme away so
the rest of the package doesn't need to branch on which cloud is in use.

Internal module — not part of the public API.
"""

from __future__ import annotations

__all__: list[str] = []  # internal; nothing exported

# All URI scheme prefixes recognised by this package.
URI_SCHEMES: tuple[str, ...] = (
    "s3://", "s3a://",          # AWS S3 (s3a is the Hadoop-compatible alias)
    "gs://", "gcs://",          # Google Cloud Storage
    "abfs://", "abfss://",      # Azure Blob / ADLS Gen2 (non-TLS / TLS)
    "az://", "adl://",          # Azure Data Lake (legacy)
)


def strip_uri_scheme(path: str) -> str:
    """Strip a URI scheme prefix from *path*, returning a bare filesystem path.

    PyArrow filesystems expect bare paths (``bucket/key``) rather than full
    URIs.  This function normalises any supported scheme away so callers
    don't need to know which filesystem is in use.

    Args:
        path: A URI such as ``"s3://bucket/key"`` or a bare path
              ``"bucket/key"``.  Any leading/trailing whitespace is
              **not** stripped — pass a clean string.

    Returns:
        The path with the first matching URI scheme prefix removed.
        Unrecognised schemes and bare paths are returned unchanged.

    Examples:
        >>> strip_uri_scheme("s3://my-bucket/trading/crude_oil/")
        'my-bucket/trading/crude_oil/'
        >>> strip_uri_scheme("gs://my-bucket/data")
        'my-bucket/data'
        >>> strip_uri_scheme("my-bucket/data")   # already bare
        'my-bucket/data'
    """
    for scheme in URI_SCHEMES:
        if path.startswith(scheme):
            return path[len(scheme):]
    return path


def has_uri_scheme(path: str) -> bool:
    """Return ``True`` if *path* starts with a recognised URI scheme.

    Args:
        path: Any string path or URI.

    Returns:
        ``True`` when *path* has a known cloud storage URI prefix;
        ``False`` for bare paths and unrecognised schemes.

    Examples:
        >>> has_uri_scheme("s3://bucket/key")
        True
        >>> has_uri_scheme("bucket/key")
        False
        >>> has_uri_scheme("file:///local/path")  # not in URI_SCHEMES
        False
    """
    return any(path.startswith(s) for s in URI_SCHEMES)
