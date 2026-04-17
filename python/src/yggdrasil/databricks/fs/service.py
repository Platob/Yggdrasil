from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

from .path import DatabricksPath, DatabricksStatResult
from ..client import DatabricksService

__all__ = [
    "FileSystem",
]


@dataclass(frozen=True)
class FileSystem(DatabricksService):
    """OS-style filesystem helpers backed by ``DatabricksPath``."""

    sep: str = "/"

    def path(
        self,
        path: Any,
        *,
        temporary: bool = False,
    ) -> DatabricksPath:
        return DatabricksPath.parse(path, client=self.client, temporary=temporary)

    def join(self, *parts: Any, temporary: bool = False) -> DatabricksPath:
        if not parts:
            raise ValueError("At least one path part is required.")
        head, *tail = parts
        path = self.path(head, temporary=temporary)
        return path.joinpath(*tail) if tail else path

    def open(
        self,
        path: Any,
        mode: str = "rb",
        buffering: int = -1,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ):
        return self.path(path).open(
            mode=mode,
            buffering=buffering,
            encoding=encoding,
            errors=errors,
            newline=newline,
        )

    def exists(self, path: Any) -> bool:
        return self.path(path).exists()

    def isfile(self, path: Any) -> bool:
        return bool(self.path(path).is_file())

    def isdir(self, path: Any) -> bool:
        return bool(self.path(path).is_dir())

    def stat(self, path: Any) -> DatabricksStatResult:
        return self.path(path).stat()

    def abspath(self, path: Any) -> str:
        return self.path(path).full_path()

    def iterdir(
        self,
        path: Any,
        *,
        recursive: bool = False,
        allow_not_found: bool = True,
    ) -> Iterator[DatabricksPath]:
        yield from self.path(path).ls(
            recursive=recursive,
            allow_not_found=allow_not_found,
        )

    def ls(
        self,
        path: Any,
        *,
        fetch_size: int | None = None,
        recursive: bool = False,
        allow_not_found: bool = True,
    ):
        yield from self.path(path=path).ls(
            fetch_size=fetch_size,
            recursive=recursive,
            allow_not_found=allow_not_found
        )

    def listdir(
        self,
        path: Any,
        *,
        recursive: bool = False,
        allow_not_found: bool = True,
    ) -> list[str]:
        return [
            child.name
            for child in self.iterdir(
                path,
                recursive=recursive,
                allow_not_found=allow_not_found,
            )
        ]

    def glob(self, path: Any, pattern: str) -> list[DatabricksPath]:
        return list(self.path(path).glob(pattern))

    def rglob(self, path: Any, pattern: str) -> list[DatabricksPath]:
        return list(self.path(path).rglob(pattern))

    def mkdir(self, path: Any, mode: int = 0o777) -> DatabricksPath:
        return self.path(path).mkdir(mode=mode, parents=False, exist_ok=False)

    def makedirs(
        self,
        path: Any,
        mode: int = 0o777,
        exist_ok: bool = False,
    ) -> DatabricksPath:
        return self.path(path).mkdir(mode=mode, parents=True, exist_ok=exist_ok)

    def touch(self, path: Any, mode: int = 0o666, exist_ok: bool = True) -> DatabricksPath:
        target = self.path(path)
        target.touch(mode=mode, exist_ok=exist_ok)
        return target

    def remove(self, path: Any, missing_ok: bool = False) -> None:
        self.path(path).rmfile(allow_not_found=missing_ok)

    def unlink(self, path: Any, missing_ok: bool = False) -> None:
        self.remove(path, missing_ok=missing_ok)

    def rmdir(
        self,
        path: Any,
        *,
        recursive: bool = False,
        allow_not_found: bool = False,
    ) -> DatabricksPath:
        return self.path(path).rmdir(
            recursive=recursive,
            allow_not_found=allow_not_found,
        )

    def delete(
        self,
        path: Any,
        *,
        recursive: bool = True,
        allow_not_found: bool = False,
    ) -> DatabricksPath:
        return self.path(path).remove(
            recursive=recursive,
            allow_not_found=allow_not_found,
        )

    def rename(self, src: Any, dst: Any) -> DatabricksPath:
        return self.path(src).rename(self.path(dst))

    def replace(self, src: Any, dst: Any) -> DatabricksPath:
        return self.rename(src, dst)

    def copy(self, src: Any, dst: Any) -> DatabricksPath:
        source = self.path(src)
        dest = self.path(dst)
        source.copy_to(dest)
        return dest

    def read_bytes(self, path: Any, *, use_cache: bool = False) -> bytes:
        return self.path(path).read_bytes(use_cache=use_cache)

    def write_bytes(self, path: Any, data: Any) -> DatabricksPath:
        target = self.path(path)
        target.write_bytes(data)
        return target

    def read_text(
        self,
        path: Any,
        *,
        encoding: str = "utf-8",
        errors: str | None = None,
    ) -> str:
        return self.path(path).read_text(encoding=encoding, errors=errors)

    def write_text(
        self,
        path: Any,
        data: str,
        *,
        encoding: str = "utf-8",
        errors: str | None = None,
        newline: str | None = None,
    ) -> DatabricksPath:
        target = self.path(path)
        target.write_text(data, encoding=encoding, errors=errors, newline=newline)
        return target

    def copytree(
        self,
        src: Any,
        dst: Any,
        *,
        dirs_exist_ok: bool = True,
    ) -> DatabricksPath:
        source = self.path(src)
        dest = self.path(dst)
        if not dirs_exist_ok and dest.exists():
            raise FileExistsError(f"Destination already exists: {dest}")
        source.copy_to(dest)
        return dest

    def walk(
        self,
        path: Any,
        *,
        allow_not_found: bool = True,
    ) -> Iterator[tuple[DatabricksPath, list[DatabricksPath], list[DatabricksPath]]]:
        root = self.path(path)
        by_parent: dict[str, tuple[DatabricksPath, list[DatabricksPath], list[DatabricksPath]]] = {}

        def ensure(node: DatabricksPath) -> tuple[DatabricksPath, list[DatabricksPath], list[DatabricksPath]]:
            key = node.full_path()
            existing = by_parent.get(key)
            if existing is None:
                existing = (node, [], [])
                by_parent[key] = existing
            return existing

        ensure(root)
        for child in root.ls(recursive=True, allow_not_found=allow_not_found):
            parent_entry = ensure(child.parent)
            if child.is_dir():
                parent_entry[1].append(child)
                ensure(child)
            else:
                parent_entry[2].append(child)

        for key in sorted(by_parent):
            yield by_parent[key]
