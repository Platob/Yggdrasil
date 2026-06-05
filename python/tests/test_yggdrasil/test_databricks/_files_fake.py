"""Shared fake Files-API HTTP transport for :class:`VolumePath` unit tests.

:class:`VolumePath` issues Databricks Files-API traffic over yggdrasil's
own :class:`HTTPSession` (``/api/2.0/fs/files`` /
``/api/2.0/fs/directories``) instead of the SDK's ``workspace.files``
client. To keep the unit suites focused on :class:`VolumePath`'s own
logic — URL building, status → exception translation, pagination,
parent auto-creation — without standing up a real workspace,
:func:`wire_files_session` points a client's ``files_session`` at a
:class:`_FakeFilesSession` that translates each Files-API HTTP call back
onto the same ``workspace.files`` MagicMock the tests already configure.

Real wire retry / stream-resume lives in the :class:`HTTPSession` tests;
the volume ↔ session integration is exercised against a real session
elsewhere.
"""

from __future__ import annotations

import io
import json as _json
from urllib.parse import unquote, urlsplit

from yggdrasil.url import URL

__all__ = ["FakeResp", "FakeFilesSession", "wire_files_session"]

_FILES_PREFIX = "/api/2.0/fs/files"
_DIRS_PREFIX = "/api/2.0/fs/directories"


class FakeResp:
    """Minimal :class:`HTTPResponse`-shaped object for the fake session."""

    def __init__(self, status, *, headers=None, body=b"", json_data=None):
        self.status = status
        self.status_code = status
        self.ok = 200 <= status < 400
        self.headers = dict(headers or {})
        self._body = bytes(body) if isinstance(body, (bytes, bytearray)) else b""
        self._json = json_data

    @property
    def data(self):
        return self._body

    @property
    def text(self):
        return self._body.decode() if self._body else ""

    def json(self, *a, **k):
        if self._json is not None:
            return self._json
        return _json.loads(self.text or "{}")

    def drain_conn(self):
        pass

    def release_conn(self):
        pass


def _exc_to_status(exc):
    """Map an SDK-style exception onto the Files-API wire status it would carry."""
    name = type(exc).__name__
    msg = str(exc)
    low = msg.lower()
    if (
        name in ("NotFound", "ResourceDoesNotExist")
        or isinstance(exc, FileNotFoundError)
        or "does not exist" in low
    ):
        return 404, msg
    if name in ("PermissionDenied", "Forbidden", "Unauthorized", "AccessDenied"):
        return 403, msg
    if name in ("AlreadyExists", "ResourceAlreadyExists") or "already exists" in low:
        return 409, msg
    return 500, msg


def _meta_headers(meta):
    headers = {}
    cl = getattr(meta, "content_length", None)
    if isinstance(cl, (int, str)):
        headers["Content-Length"] = str(cl)
    ct = getattr(meta, "content_type", None) or getattr(meta, "mime_type", None)
    if isinstance(ct, str):
        headers["Content-Type"] = ct
    return headers


def _download_headers(resp):
    headers = {}
    ct = getattr(resp, "content_type", None)
    if isinstance(ct, str):
        headers["Content-Type"] = ct
    lm = getattr(resp, "last_modified", None)
    if isinstance(lm, str):
        headers["Last-Modified"] = lm
    return headers


def _entry_dict(entry):
    out = {}
    for key in ("path", "name", "is_directory", "file_size", "last_modified"):
        value = getattr(entry, key, None)
        if value is not None:
            out[key] = value
    return out


class FakeFilesSession:
    """Routes Files-API HTTP calls onto a ``workspace.files`` SDK mock.

    *honor_range* toggles whether a ``Range`` request header yields a
    206 slice + ``Content-Range`` (the real Files-API behaviour) or is
    ignored with a 200 full body (to exercise VolumePath's local-slice
    fallback). ``bytes_served`` accumulates the body bytes returned, so
    benchmarks can assert random-seek reads transfer only the slice.
    """

    def __init__(self, files, *, honor_range=True):
        self._files = files
        self._honor_range = honor_range
        self.calls = []
        self.bytes_served = 0

    def fetch(self, method, url, *, headers=None, body=None,
              preload_content=True, **_kw):
        raw = urlsplit(str(url)).path
        self.calls.append((method, raw))
        range_header = (headers or {}).get("Range")
        if raw.startswith(_DIRS_PREFIX):
            kind, api_path = "directories", unquote(raw[len(_DIRS_PREFIX):])
        elif raw.startswith(_FILES_PREFIX):
            kind, api_path = "files", unquote(raw[len(_FILES_PREFIX):])
        else:
            return FakeResp(404)

        files = self._files
        try:
            if method == "HEAD" and kind == "files":
                meta = files.get_metadata(api_path)
                return FakeResp(200, headers=_meta_headers(meta))
            if method == "HEAD" and kind == "directories":
                files.get_directory_metadata(api_path)
                return FakeResp(200)
            if method == "GET" and kind == "directories":
                entries = files.list_directory_contents(api_path)
                contents = [_entry_dict(e) for e in entries]
                return FakeResp(
                    200,
                    headers={"Content-Type": "application/json"},
                    json_data={"contents": contents},
                )
            if method == "GET" and kind == "files":
                resp = files.download(api_path)
                contents = getattr(resp, "contents", None)
                if contents is not None and hasattr(contents, "read"):
                    data = contents.read()
                elif hasattr(resp, "read"):
                    data = resp.read()
                else:
                    data = bytes(contents) if contents is not None else b""
                headers = _download_headers(resp)
                if range_header and self._honor_range and range_header.startswith("bytes="):
                    spec = range_header[len("bytes="):].split("-", 1)
                    start = int(spec[0]) if spec[0].strip().isdigit() else 0
                    end = spec[1].strip() if len(spec) > 1 else ""
                    total = len(data)
                    stop = int(end) + 1 if end.isdigit() else total
                    slice_ = data[start:stop]
                    headers["Content-Range"] = (
                        f"bytes {start}-{start + len(slice_) - 1}/{total}"
                    )
                    self.bytes_served += len(slice_)
                    return FakeResp(206, headers=headers, body=slice_)
                self.bytes_served += len(data)
                return FakeResp(200, headers=headers, body=data)
            if method == "PUT" and kind == "files":
                # A streaming body arrives as a yggdrasil Holder (the real
                # session would iter_mv it onto the wire); read it here.
                if hasattr(body, "read_bytes"):
                    data = body.read_bytes()
                elif hasattr(body, "read"):
                    data = body.read()
                else:
                    data = bytes(body or b"")
                files.upload(
                    file_path=api_path,
                    contents=io.BytesIO(data),
                    overwrite=True,
                )
                return FakeResp(204)
            if method == "PUT" and kind == "directories":
                files.create_directory(api_path)
                return FakeResp(204)
            if method == "DELETE" and kind == "files":
                files.delete(api_path)
                return FakeResp(204)
            if method == "DELETE" and kind == "directories":
                files.delete_directory(api_path)
                return FakeResp(204)
        except Exception as exc:  # noqa: BLE001 — translate to wire status
            status, message = _exc_to_status(exc)
            return FakeResp(
                status,
                headers={"Content-Type": "application/json"},
                json_data={"message": message},
            )
        return FakeResp(404)


def wire_files_session(client, *, host: str = "https://test.databricks.com",
                       honor_range: bool = True):
    """Point *client*'s Files-API transport at a :class:`FakeFilesSession`.

    Sets ``base_url`` / ``files_authorization`` / ``files_session`` so
    :class:`VolumePath` reaches a fake session backed by the client's
    existing ``workspace_client().files`` mock. Returns the client; the
    session is reachable via ``client.files_session.return_value``.
    """
    client.base_url = URL.from_(host)
    client.files_authorization.return_value = "Bearer test-token"
    client.files_session.return_value = FakeFilesSession(
        client.workspace_client.return_value.files,
        honor_range=honor_range,
    )
    return client
