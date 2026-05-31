"""In-memory remote path for hermetic Delta tests.

:class:`MemRemotePath` is a dict-backed :class:`yggdrasil.path.remote_path.RemotePath`
— it implements yggdrasil's *own* IO contract (``_stat_uncached`` / ``_read_mv``
/ ``_upload`` / ``_ls`` / ``_remove_*``) directly, the same surface
:class:`yggdrasil.aws.fs.path.S3Path` implements. That's enough to drive the
full :class:`~yggdrasil.io.delta.DeltaFolder` protocol over a non-local path
without simulating the boto3 S3 client wire (head_object / get_object /
upload_fileobj / list_objects_v2 paginator). It mirrors how ``VolumePath`` is
unit-tested over a staged local tree: exercise the Path/IO abstraction, not the
cloud SDK.

Keys are stored ``"<bucket>/<key>"`` (so tests can inspect / inject objects),
and every primitive bumps :attr:`MemRemotePath.calls` so the latency / call-count
benchmarks can assert that the Delta caches cut *our* remote round-trips
(reads, lists, stats) — the metric that actually matters, backend-agnostic.
"""
from __future__ import annotations

import time
from collections import Counter
from typing import Any, ClassVar, Iterator

from yggdrasil.dataclasses import WaitingConfig
from yggdrasil.dataclasses.expiring import ExpiringDict
from yggdrasil.io.io_stats import IOKind, IOStats
from yggdrasil.path.remote_path import RemotePath
from yggdrasil.url import URL

__all__ = ["MemRemotePath", "mem_delta_folder"]


class MemRemotePath(RemotePath):
    """Dict-backed ``s3://``-shaped RemotePath. Storage + call counts are
    process-global (mirroring real singleton paths); call :meth:`reset`
    in ``setUp`` to isolate each test."""

    #: ``"<bucket>/<key>" -> bytes``. Inspectable + injectable from tests.
    objects: ClassVar[dict[str, bytes]] = {}
    #: Per-primitive call counts ("read", "list", "stat", "upload", "delete").
    calls: ClassVar[Counter] = Counter()
    _INSTANCES: ClassVar[ExpiringDict] = ExpiringDict(default_ttl=300.0, max_size=10_000)
    _ACCEPTED_SCHEMES: ClassVar[frozenset[str]] = frozenset({"s3", "s3a", "s3n"})

    def __init__(self, data: Any = None, *, url: URL | None = None,
                 singleton_ttl: Any = ..., **kwargs: Any) -> None:
        del singleton_ttl
        if getattr(self, "_initialized", False):
            return
        if url is None and isinstance(data, str):
            url = URL.from_(data)
        elif url is None and isinstance(data, URL):
            url, data = data, None
        if url is not None and url.scheme in self._ACCEPTED_SCHEMES and url.scheme != "s3":
            url = URL(scheme="s3", host=url.host, path=url.path,
                      port=url.port, query=url.query, fragment=url.fragment)
        super().__init__(data=data, url=url, singleton_ttl=False, **kwargs)
        self._initialized = True

    # -- identity ----------------------------------------------------------
    @classmethod
    def reset(cls) -> None:
        cls.objects.clear()
        cls.calls.clear()

    @property
    def bucket(self) -> str:
        if not self.url.host:
            raise ValueError(f"path has no bucket: {self.url!r}")
        return self.url.host

    @property
    def key(self) -> str:
        return (self.url.path or "").lstrip("/")

    def full_path(self) -> str:
        key = self.key
        return f"s3://{self.bucket}/{key}" if key else f"s3://{self.bucket}/"

    # -- backend primitives (yggdrasil's IO contract) ----------------------
    def _stat_uncached(self) -> IOStats:
        self.calls["stat"] += 1
        sk = f"{self.bucket}/{self.key}"
        if sk in self.objects:
            return IOStats(size=len(self.objects[sk]), kind=IOKind.FILE, mtime=time.time())
        prefix = sk.rstrip("/") + "/"
        if any(k.startswith(prefix) for k in self.objects):
            return IOStats(size=0, kind=IOKind.DIRECTORY, mtime=0.0)
        return IOStats(size=0, kind=IOKind.MISSING, mtime=0.0)

    def _read_mv(self, n: int, pos: int) -> memoryview:
        self.calls["read"] += 1
        data = self.objects.get(f"{self.bucket}/{self.key}")
        if data is None:
            raise FileNotFoundError(self.full_path())
        return memoryview(data[pos:]) if n < 0 else memoryview(data[pos : pos + n])

    def _upload(self, content: bytes) -> int:
        self.calls["upload"] += 1
        self.objects[f"{self.bucket}/{self.key}"] = bytes(content)
        self._persist_stat_cache(IOStats(size=len(content), kind=IOKind.FILE, mtime=time.time()))
        return len(content)

    def _ls(self, recursive: bool = False, *, singleton_ttl: Any = False) -> "Iterator[MemRemotePath]":
        self.calls["list"] += 1
        prefix = f"{self.bucket}/{self.key}"
        if prefix and not prefix.endswith("/"):
            prefix += "/"
        seen: set[str] = set()
        for k in sorted(self.objects):
            if not k.startswith(prefix):
                continue
            remainder = k[len(prefix):]
            if recursive:
                child_key = k
            else:
                top = remainder.split("/")[0]
                if top in seen:
                    continue
                seen.add(top)
                child_key = prefix + top
            cut = child_key.index("/")
            yield MemRemotePath(url=URL(scheme="s3", host=child_key[:cut],
                                        path="/" + child_key[cut + 1:]), singleton_ttl=False)

    def _mkdir(self, parents: bool, exist_ok: bool) -> None:
        pass  # object store — no directory concept

    def _remove_file(self, missing_ok: bool, wait: WaitingConfig) -> None:
        self.calls["delete"] += 1
        sk = f"{self.bucket}/{self.key}"
        if sk in self.objects:
            del self.objects[sk]
            self.invalidate_singleton()
        elif not missing_ok:
            raise FileNotFoundError(self.full_path())

    def _remove_dir(self, recursive: bool, missing_ok: bool, wait: WaitingConfig) -> None:
        self.calls["delete"] += 1
        prefix = f"{self.bucket}/{self.key}"
        if prefix and not prefix.endswith("/"):
            prefix += "/"
        victims = [k for k in self.objects if k.startswith(prefix)]
        if not victims and not missing_ok:
            raise FileNotFoundError(self.full_path())
        for k in victims:
            del self.objects[k]
        self.invalidate_singleton()

    def _from_url(self, url: URL) -> "MemRemotePath":
        return MemRemotePath(url=url, singleton_ttl=False)

    def __repr__(self) -> str:
        return f"MemRemotePath({self.full_path()!r})"


# Match the RemotePath scheme convention (used for child-URL building, not
# global ``Path.from_`` dispatch — that stays bound to the real S3Path).
MemRemotePath.scheme = "s3"


def mem_delta_folder(bucket: str, prefix: str):
    """A :class:`DeltaFolder` rooted at an in-memory ``s3://`` path."""
    from yggdrasil.io.delta import DeltaFolder

    return DeltaFolder(path=MemRemotePath(f"s3://{bucket}/{prefix}"))
