"""AWS filesystem submodule.

Currently S3-only. The structure (``service.py`` + ``path.py``) is
ready to grow other AWS-backed filesystems (EFS, FSx) without
disturbing the API.
"""

from __future__ import annotations

from .path import S3Bucket, S3Path
from .service import S3Service


__all__ = [
    "S3Bucket",
    "S3Path",
    "S3Service",
]