"""In-memory :class:`VolumePath` for unit tests.

Subclass of :class:`yggdrasil.databricks.fs.volume_path.VolumePath` whose
SDK transport hooks read/write a class-level dict instead of calling the
Databricks Files API. Lets us exercise the path machinery —
``open_io``, transaction buffer, format-leaf dispatch by extension,
``TabularIO`` surface — without a live workspace.

Lives under ``_helpers`` (leading underscore) so pytest doesn't try to
collect it as a test module.
"""

from __future__ import annotations

from typing import ClassVar, Dict, Iterator

from yggdrasil.databricks.fs.volume_path import VolumePath
from yggdrasil.io.path_stat import PathKind, PathStats


__all__ = ["InMemoryVolumePath"]


class InMemoryVolumePath(VolumePath):
    """:class:`VolumePath` backed by a class-level dict.

    Overrides exactly the SDK seam (``_remote_download``,
    ``_remote_upload``, ``write_stream``) plus the filesystem metadata
    hooks (``_mkdir``, ``_stat``, ``_ls``) so the rest of the path
    machinery — URL handling, ``open_io`` mode dispatch, the BytesIO
    transaction buffer, format-leaf dispatch by extension — runs
    unmodified.
    """

    __slots__ = ()

    _STORE: ClassVar[Dict[str, bytes]] = {}
    upload_count: ClassVar[int] = 0
    download_count: ClassVar[int] = 0

    @classmethod
    def reset(cls) -> None:
        cls._STORE.clear()
        cls.upload_count = 0
        cls.download_count = 0

    # -- SDK transport ----------------------------------------------------
    def _remote_download(self, allow_not_found: bool = False) -> bytes:
        type(self).download_count += 1
        key = self.full_path()
        if key in self._STORE:
            return self._STORE[key]
        if allow_not_found:
            return b""
        raise FileNotFoundError(key)

    def _remote_upload(self, payload) -> None:
        type(self).upload_count += 1
        self._STORE[self.full_path()] = self._coerce_payload(payload)

    def write_stream(
        self,
        src,
        *,
        batch_size: int = 4 * 1024 * 1024,
        parents: bool = True,
    ) -> None:
        # The BytesIO commit path calls write_stream directly (not
        # _remote_upload), so we have to intercept it here too.
        type(self).upload_count += 1
        self._STORE[self.full_path()] = self._coerce_payload(src)

    @staticmethod
    def _coerce_payload(payload) -> bytes:
        if hasattr(payload, "seek"):
            try:
                payload.seek(0)
            except Exception:
                pass
        if hasattr(payload, "read"):
            return payload.read()
        return bytes(payload)

    # -- Filesystem metadata ---------------------------------------------
    def _mkdir(self, parents: bool = True, exist_ok: bool = True) -> None:
        return

    def _stat(self) -> PathStats:
        key = self.full_path()
        if key in self._STORE:
            return PathStats(
                kind=PathKind.FILE,
                size=len(self._STORE[key]),
                mtime=None,
            )
        prefix = key.rstrip("/") + "/"
        if any(k.startswith(prefix) for k in self._STORE):
            return PathStats(kind=PathKind.DIRECTORY, size=0, mtime=None)
        return PathStats(kind=PathKind.MISSING, size=0, mtime=None)

    def _ls(
        self,
        recursive: bool = False,
        allow_not_found: bool = True,
    ) -> Iterator["InMemoryVolumePath"]:
        prefix = self.full_path().rstrip("/") + "/"
        seen: set = set()
        for k in self._STORE:
            if not k.startswith(prefix):
                continue
            tail = k[len(prefix):]
            head = tail.split("/", 1)[0]
            child = prefix + head
            if child in seen:
                continue
            seen.add(child)
            yield type(self)(child)

    def _from_url(self, url):
        return type(self)(url=url)

    def invalidate_mirror(self) -> None:
        return  # No mirror in the fake store.

    @property
    def connected(self) -> bool:
        return True
