"""HTTP(S) byte holder — a :class:`RemotePath` over an ``http://`` /
``https://`` URL.

:class:`HTTPPath` plugs the URL-scheme registry so any code that
constructs a :class:`yggdrasil.io.holder.Holder` /
:class:`yggdrasil.io.path.path.Path` from an HTTP URL lands here
instead of falling back to :class:`Memory`. The holder primitives
(:meth:`_read_mv` ranged GET, :meth:`_upload` whole-resource PUT) issue
real requests through an attached :class:`HTTPSession`;
:meth:`_stat_uncached` does a HEAD probe for ``Content-Length`` /
``Last-Modified`` / ``Content-Type``.

There is no filesystem surface (HTTP has no directory listing),
so :meth:`_ls`, :meth:`_mkdir`, :meth:`_remove_dir` raise.
:meth:`_remove_file` issues a DELETE for callers that genuinely
want it.

Inside :class:`HTTPResponse`, :attr:`HTTPResponse.path` returns an
:class:`HTTPPath` bound to the request URL — sharing the response's
own session when one is attached so the connection pool is reused.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Iterator

from yggdrasil.dataclasses import ExpiringDict, WaitingConfig
from yggdrasil.enums import Scheme
from yggdrasil.io.io_stats import IOKind, IOStats
from yggdrasil.path import RemotePath
from yggdrasil.path.remote_path import _STAT_CACHE_TTL
from yggdrasil.url import URL

if TYPE_CHECKING:
    from .session import HTTPSession


__all__ = ["HTTPPath"]


class HTTPPath(RemotePath):
    """:class:`Path` over an ``http://`` / ``https://`` URL.

    Construction shapes::

        HTTPPath("https://example.com/data.csv")
        HTTPPath(url=URL("http://api.example/v1"), session=my_session)

    The ``session`` kwarg accepts any :class:`HTTPSession`; when
    omitted, a default one is lazy-built on first access (mirroring
    :meth:`PreparedRequest._send` for orphan requests). Sharing a
    single session across paths reuses the connection pool.
    """

    scheme: ClassVar[Scheme] = Scheme.HTTPS

    #: URL schemes accepted on input. Both keep their original spelling
    #: (unlike :class:`S3Path` which normalizes ``s3a`` / ``s3n`` to
    #: ``s3``); ``http`` and ``https`` are not interchangeable on the
    #: wire, so the holder remembers which the caller asked for.
    _ACCEPTED_SCHEMES: ClassVar[frozenset[str]] = frozenset({"http", "https"})

    # Per-class singleton cache — partitions HTTP construction
    # contention away from S3Path / DatabricksPath / future
    # backends. No companion lock —
    # :class:`ExpiringDict.get_or_set` is GIL-atomic.
    _INSTANCES: ClassVar[ExpiringDict] = ExpiringDict(
        default_ttl=_STAT_CACHE_TTL,
        max_size=10_000,
    )

    def __init__(
        self,
        data: Any = None,
        *,
        url: URL | None = None,
        session: "HTTPSession | None" = None,
        **kwargs: Any,
    ) -> None:
        if url is None and isinstance(data, (str, URL)):
            url = URL.from_(data)
            data = None
        super().__init__(data=data, url=url, **kwargs)
        self._session: "HTTPSession | None" = session

    @property
    def url(self) -> URL:
        from yggdrasil.io.holder import Holder
        return Holder.url.fget(self)  # type: ignore[attr-defined]

    @url.setter
    def url(self, value: Any) -> None:
        u = URL.from_(value)
        # Preserve ``http`` vs ``https`` instead of letting the base
        # setter rewrite everything to the class :attr:`scheme`.
        if u.scheme not in self._ACCEPTED_SCHEMES:
            u = u.with_scheme(self.scheme)
        self._url = u

    @property
    def session(self) -> "HTTPSession":
        """Attached :class:`HTTPSession`, building a default one on miss."""
        if self._session is None:
            from .session import HTTPSession
            self._session = HTTPSession()
        return self._session

    def attach_session(self, session: "HTTPSession") -> "HTTPPath":
        self._session = session
        return self

    # ==================================================================
    # Backend-native string form
    # ==================================================================

    def full_path(self) -> str:
        return self.url.to_string()

    # ==================================================================
    # Stat — HEAD probe
    # ==================================================================

    def _stat_uncached(self) -> IOStats:
        from .request import HTTPRequest
        from ..send_config import SendConfig

        req = HTTPRequest.prepare("HEAD", self.url)
        try:
            resp = self.session.send(req, SendConfig(raise_error=False))
        except Exception:
            return IOStats(
                size=0, mtime=0.0, kind=IOKind.MISSING,
                media_type=self.url.infer_media_type(default=None),
            )

        if resp.status_code >= 400:
            return IOStats(
                size=0, mtime=0.0, kind=IOKind.MISSING,
                media_type=self.url.infer_media_type(default=None),
            )

        size_hdr = resp.headers.get("Content-Length") if resp.headers else None
        try:
            size = int(size_hdr) if size_hdr is not None else 0
        except (TypeError, ValueError):
            size = 0

        mtime = 0.0
        last_mod = resp.headers.get("Last-Modified") if resp.headers else None
        if last_mod:
            try:
                from email.utils import parsedate_to_datetime
                mtime = float(parsedate_to_datetime(last_mod).timestamp())
            except Exception:
                mtime = 0.0

        return IOStats(
            size=size, mtime=mtime, kind=IOKind.FILE,
            media_type=resp.media_type,
        )

    # ==================================================================
    # Read / write — GET / PUT
    # ==================================================================

    def _read_mv(self, n: int, pos: int) -> memoryview:
        """Ranged GET → ``memoryview``. ``n < 0`` reads to EOF."""
        if n == 0:
            return memoryview(b"")
        from .request import HTTPRequest

        headers: dict[str, str] = {}
        if pos > 0 or n >= 0:
            end = "" if n < 0 else str(pos + n - 1)
            headers["Range"] = f"bytes={pos}-{end}"

        req = HTTPRequest.prepare("GET", self.url, headers=headers or None)
        resp = self.session.send(req)
        return memoryview(resp.content or b"")

    def _upload(self, content: bytes) -> int:
        """Whole-resource PUT. HTTP has no positional write — the body
        replaces the resource (matching the whole-blob remote contract)."""
        from .request import HTTPRequest

        body = bytes(content)
        req = HTTPRequest.prepare("PUT", self.url, body=body)
        resp = self.session.send(req)
        if resp.status_code >= 400:
            resp.raise_for_status()
        self.invalidate_singleton()
        return len(body)

    # ==================================================================
    # Filesystem surface — HTTP has no directory listing
    # ==================================================================

    def _ls(
        self,
        recursive: bool = False,
        *,
        singleton_ttl: Any = False,
    ) -> Iterator["HTTPPath"]:
        del recursive, singleton_ttl
        return iter(())

    def _mkdir(self, parents: bool, exist_ok: bool) -> None:
        del parents, exist_ok
        raise NotImplementedError(
            f"{type(self).__name__} cannot create directories — HTTP has "
            "no directory concept. Use a path scheme that does (file://, "
            "s3://, dbfs:/) for the destination."
        )

    def _remove_file(self, missing_ok: bool, wait: WaitingConfig) -> None:
        from .request import HTTPRequest
        from ..send_config import SendConfig

        del wait
        req = HTTPRequest.prepare("DELETE", self.url)
        resp = self.session.send(req, SendConfig(raise_error=False))
        if resp.status_code == 404:
            if not missing_ok:
                raise FileNotFoundError(self.url.to_string())
        elif resp.status_code >= 400:
            resp.raise_for_status()
        self.invalidate_singleton()

    def _remove_dir(
        self, recursive: bool, missing_ok: bool, wait: WaitingConfig,
    ) -> None:
        del recursive, missing_ok, wait
        raise NotImplementedError(
            f"{type(self).__name__} cannot remove directories — HTTP has "
            "no directory concept."
        )


# Register the ``http`` scheme alongside ``https`` so plain-HTTP URLs
# also dispatch here. :meth:`URLBased.__init_subclass__` only registers
# the single :attr:`scheme` ClassVar; HTTP is the only scheme pair in
# the codebase where a single backend serves two scheme spellings, so
# we slot the alias in directly rather than introducing a multi-scheme
# registration mechanism nothing else needs.
from yggdrasil.url import _URL_BASED_REGISTRY as _HTTP_SCHEMES
_HTTP_SCHEMES.setdefault(Scheme.HTTP, HTTPPath)
del _HTTP_SCHEMES
