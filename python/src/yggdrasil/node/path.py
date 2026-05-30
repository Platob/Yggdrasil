"""NodePath -- pathlib-like interface for node filesystem access.

Uses ``npfs://host:port/path`` URLs. When the URL resolves to the local
node (same host:port), operations go directly to the filesystem. When
remote, operations use HTTP against the node's ``/api/v2/fs/*`` endpoints.

Usage::

    from yggdrasil.node.path import NodePath

    # Local node files (direct filesystem)
    p = NodePath("data/input.csv")
    content = p.read_text()

    # Remote node files (via HTTP)
    remote = NodePath("data/input.csv", node_url="http://node-2:8100")
    content = remote.read_bytes()

    # From npfs:// URL
    p = NodePath.from_url("npfs://node-2:8100/data/input.csv")
"""
from __future__ import annotations

import base64
import json
import os
import shutil
import urllib.request
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Iterator
from urllib.parse import quote, urlparse


def _is_self(host: str, port: int) -> bool:
    """True when host:port points back to this node."""
    local_port = int(os.environ.get("YGG_NODE_PORT", "8100"))
    if port != local_port:
        return False
    return host in ("127.0.0.1", "localhost", "::1", "0.0.0.0", "")


class NodePath:
    """Path-like object for node filesystem access.

    When ``node_url`` is None (or resolves to self), operates directly
    on the local filesystem within the node's data directory.  When
    ``node_url`` is set and remote, uses HTTP calls to the remote
    node's ``/api/v2/fs`` endpoints.

    Supports ``npfs://`` protocol URLs::

        npfs://node-host:port/path/to/content
        npfs://localhost:8100/data/input.csv
        npfs:///local/path  (triple slash = local node)
    """

    __slots__ = ("_path", "_node_url", "_root")

    def __init__(
        self,
        path: str = "",
        *,
        node_url: str | None = None,
        _root: Path | None = None,
    ) -> None:
        self._path = PurePosixPath(path.lstrip("/"))
        if node_url:
            parsed = urlparse(node_url)
            host = parsed.hostname or ""
            port = parsed.port or 8100
            if _is_self(host, port):
                node_url = None
        self._node_url = node_url
        if _root is None and node_url is None:
            from .config import get_settings
            self._root = get_settings().data_root / "files"
        else:
            self._root = _root

    # -- NPFS URL support -----------------------------------------------------

    @classmethod
    def from_url(cls, url: str) -> NodePath:
        """Parse ``npfs://host:port/path`` URL.

        - ``npfs://host:port/path`` -- remote node
        - ``npfs:///path`` -- local node (triple slash)
        - Plain paths (no scheme) -- treated as local
        """
        if url.startswith("npfs://"):
            parsed = urlparse(url)
            host = parsed.hostname or ""
            port = parsed.port or 8100
            path = parsed.path.lstrip("/")
            if not host or _is_self(host, port):
                return cls(path)
            return cls(path, node_url=f"http://{host}:{port}")
        return cls(url)

    def to_url(self) -> str:
        """Convert to ``npfs://`` URL."""
        if self.is_local:
            return f"npfs:///{self._path}"
        parsed = urlparse(self._node_url)
        return f"npfs://{parsed.hostname}:{parsed.port or 8100}/{self._path}"

    @classmethod
    def from_(cls, value) -> NodePath | None:
        """Polymorphic constructor -- accepts str, NodePath, None."""
        if value is None:
            return None
        if isinstance(value, cls):
            return value
        return cls.from_url(str(value))

    # -- Properties -----------------------------------------------------------

    @property
    def is_local(self) -> bool:
        return self._node_url is None

    @property
    def name(self) -> str:
        return self._path.name or ""

    @property
    def parent(self) -> NodePath:
        p = str(self._path.parent) if str(self._path) != "." else ""
        return NodePath(p, node_url=self._node_url, _root=self._root)

    @property
    def suffix(self) -> str:
        return self._path.suffix

    @property
    def stem(self) -> str:
        return self._path.stem

    def __truediv__(self, other: str) -> NodePath:
        return NodePath(
            str(self._path / other),
            node_url=self._node_url,
            _root=self._root,
        )

    def __str__(self) -> str:
        return str(self._path)

    def __repr__(self) -> str:
        target = self._node_url or "local"
        return f"NodePath({str(self._path)!r}, node={target!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NodePath):
            return NotImplemented
        return self._path == other._path and self._node_url == other._node_url

    def __hash__(self) -> int:
        return hash((str(self._path), self._node_url))

    # -- Local path resolution ------------------------------------------------

    def _local_path(self) -> Path:
        resolved = (self._root / str(self._path)).resolve()
        if not str(resolved).startswith(str(self._root.resolve())):
            raise PermissionError("Path traversal not allowed")
        return resolved

    # -- Read operations ------------------------------------------------------

    def exists(self) -> bool:
        if self.is_local:
            return self._local_path().exists()
        try:
            self.stat()
            return True
        except Exception:
            return False

    def is_dir(self) -> bool:
        if self.is_local:
            return self._local_path().is_dir()
        try:
            info = self.stat()
            return info.get("is_dir", False)
        except Exception:
            return False

    def is_file(self) -> bool:
        if self.is_local:
            return self._local_path().is_file()
        return self.exists() and not self.is_dir()

    def stat(self) -> dict:
        if self.is_local:
            p = self._local_path()
            s = p.stat()
            return {
                "path": str(self._path),
                "name": p.name,
                "is_dir": p.is_dir(),
                "size": s.st_size,
                "modified_at": datetime.fromtimestamp(
                    s.st_mtime, tz=timezone.utc
                ).isoformat(),
            }
        return _get(f"{self._node_url}/api/v2/fs/stat?path={_quote(str(self._path))}")

    def read_text(self, encoding: str = "utf-8") -> str:
        if self.is_local:
            return self._local_path().read_text(encoding=encoding)
        # Stream the whole file so the JSON /read window (4 MB) never truncates it.
        return b"".join(self.stream_read()).decode(encoding)

    def read_bytes(self, offset: int = 0, length: int | None = None) -> bytes:
        """Read the whole file, or a byte range ``[offset, offset+length)``.

        A ranged read seeks locally and uses the node's ``/fs/read?offset=&
        max_bytes=`` window remotely — so paging through a large remote file
        never pulls the whole thing across the network. A full remote read
        (no range) streams via ``/fs/stream`` so it is never capped by the
        JSON ``/read`` window (4 MB) — large files come back intact.
        """
        if self.is_local:
            if offset == 0 and length is None:
                return self._local_path().read_bytes()
            with open(self._local_path(), "rb") as f:
                if offset:
                    f.seek(offset)
                return f.read(length if length is not None else -1)
        if offset == 0 and length is None:
            return b"".join(self.stream_read())
        url = f"{self._node_url}/api/v2/fs/read?path={_quote(str(self._path))}&offset={offset}"
        if length is not None:
            url += f"&max_bytes={length}"
        resp = _get(url)
        content = resp.get("content", "")
        if resp.get("encoding") == "base64":
            return base64.b64decode(content)
        return content.encode()

    def iterdir(self, offset: int = 0, limit: int | None = None) -> Iterator[NodePath]:
        """Yield child paths, optionally a single page ``[offset, offset+limit)``.

        Remote listings page through ``/fs/ls?offset=&limit=`` so a directory
        with a million entries streams a window at a time instead of one giant
        response.
        """
        if self.is_local:
            children = sorted(self._local_path().iterdir())
            window = children[offset:offset + limit] if limit is not None else children[offset:]
            for child in window:
                yield NodePath(
                    str(self._path / child.name),
                    node_url=self._node_url,
                    _root=self._root,
                )
            return
        url = f"{self._node_url}/api/v2/fs/ls?path={_quote(str(self._path))}&offset={offset}"
        if limit is not None:
            url += f"&limit={limit}"
        resp = _get(url)
        for entry in resp.get("entries", []):
            yield NodePath(
                entry["path"],
                node_url=self._node_url,
                _root=self._root,
            )

    # -- Write operations -----------------------------------------------------

    def write_text(self, content: str, encoding: str = "utf-8") -> None:
        if self.is_local:
            p = self._local_path()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding=encoding)
            return
        _post(f"{self._node_url}/api/v2/fs/write", {
            "path": str(self._path),
            "content": content,
            "encoding": "utf-8",
            "mkdir": True,
        })

    def write_bytes(self, data: bytes) -> None:
        if self.is_local:
            p = self._local_path()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
            return
        _post(f"{self._node_url}/api/v2/fs/write", {
            "path": str(self._path),
            "content": base64.b64encode(data).decode(),
            "encoding": "base64",
            "mkdir": True,
        })

    def mkdir(self, parents: bool = True, exist_ok: bool = True) -> None:
        if self.is_local:
            self._local_path().mkdir(parents=parents, exist_ok=exist_ok)
            return
        _post(f"{self._node_url}/api/v2/fs/mkdir?path={_quote(str(self._path))}", {})

    def unlink(self, missing_ok: bool = False) -> None:
        if self.is_local:
            p = self._local_path()
            if p.is_dir():
                shutil.rmtree(p)
            elif p.exists():
                p.unlink()
            elif not missing_ok:
                raise FileNotFoundError(str(self._path))
            return
        try:
            _delete(f"{self._node_url}/api/v2/fs/delete?path={_quote(str(self._path))}")
        except Exception:
            if not missing_ok:
                raise

    def rename(self, target: str) -> NodePath:
        if self.is_local:
            src = self._local_path()
            dst = (self._root / target.lstrip("/")).resolve()
            src.rename(dst)
            return NodePath(target, node_url=self._node_url, _root=self._root)
        _post(f"{self._node_url}/api/v2/fs/move", {
            "source": str(self._path),
            "destination": target,
        })
        return NodePath(target, node_url=self._node_url, _root=self._root)

    # -- Streaming ------------------------------------------------------------

    def stream_read(self, chunk_size: int = 65536) -> Iterator[bytes]:
        if self.is_local:
            with open(self._local_path(), "rb") as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk
            return
        req = urllib.request.Request(
            f"{self._node_url}/api/v2/fs/stream?path={_quote(str(self._path))}",
            headers={"Accept": "application/octet-stream"},
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                yield chunk

    def stream_write(self, data: Iterator[bytes]) -> None:
        """Send streaming data to the remote upload endpoint.

        The iterator is handed straight to the request body — urllib emits it
        with ``Transfer-Encoding: chunked`` and the node's ``/fs/upload``
        endpoint writes each chunk as it arrives, so neither side ever holds
        the whole file in memory.
        """
        if self.is_local:
            p = self._local_path()
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "wb") as f:
                for chunk in data:
                    f.write(chunk)
            return
        req = urllib.request.Request(
            f"{self._node_url}/api/v2/fs/upload?path={_quote(str(self._path))}",
            data=iter(data),
            headers={"Content-Type": "application/octet-stream"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=300):
            pass

    # -- Cross-node copy ------------------------------------------------------

    def copy_to(self, target: NodePath) -> NodePath:
        """Copy this file/directory to another NodePath (possibly on a different node)."""
        if self.is_dir():
            target.mkdir()
            for child in self.iterdir():
                child.copy_to(target / child.name)
            return target

        if self.is_local and target.is_local:
            shutil.copy2(str(self._local_path()), str(target._local_path()))
        else:
            # Pipe the source byte stream straight into the destination's
            # streaming writer — even a remote→remote copy of a huge file
            # flows chunk-by-chunk and never lands wholly in memory.
            target.stream_write(self.stream_read())
        return target


# -- HTTP helpers -------------------------------------------------------------

def _quote(s: str) -> str:
    return quote(s, safe="")


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def _post(url: str, data: dict) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def _delete(url: str) -> None:
    req = urllib.request.Request(url, method="DELETE")
    with urllib.request.urlopen(req, timeout=30):
        pass
