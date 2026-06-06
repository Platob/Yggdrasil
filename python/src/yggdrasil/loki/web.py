"""Loki on the internet — fetch, browse, tables, images.

Loki reaches the web through yggdrasil's own stack, not a bolted-on client:
every request rides :class:`~yggdrasil.http_.HTTPSession` (its pooling, retry,
and response cache), and every tabular body is parsed through the io tabular
handlers — :meth:`HTTPResponse.to_polars` auto-detects CSV / JSON / Parquet /
Arrow / XLSX. So "look it up on the internet" and "parse that table" are the
same two project abstractions the rest of yggdrasil runs on.

    from yggdrasil.loki import web

    web.read_text("https://example.com")          # browse → readable text + links
    web.read_table("https://…/data.csv")           # → polars DataFrame (any format)
    web.read_json("https://api.example.com/x")      # → decoded JSON
    web.read_image("https://…/chart.png")           # → bytes + dims + content-type
"""
from __future__ import annotations

import html.parser
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    import polars as pl

    from yggdrasil.http_ import HTTPResponse

__all__ = [
    "session",
    "fetch",
    "read_text",
    "read_table",
    "read_json",
    "read_image",
]

#: URL-extension → mime name, to force the right tabular/image leaf when a
#: server mislabels the body (``text/plain`` CSVs, octet-stream parquet, …).
EXT_MIME: dict[str, str] = {
    "csv": "CSV", "tsv": "CSV", "json": "JSON", "ndjson": "NDJSON",
    "jsonl": "NDJSON", "parquet": "PARQUET", "pq": "PARQUET",
    "arrow": "ARROW_STREAM", "feather": "ARROW_STREAM", "xlsx": "XLSX",
    "xls": "XLSX", "png": "PNG", "jpg": "JPEG", "jpeg": "JPEG",
    "gif": "GIF", "webp": "WEBP",
}


def session() -> "HTTPSession":
    """The shared (singleton-cached) :class:`HTTPSession`."""
    from yggdrasil.http_ import HTTPSession

    return HTTPSession()


def fetch(
    url: str,
    *,
    params: Optional[dict[str, str]] = None,
    headers: Optional[dict[str, str]] = None,
    timeout: float = 30.0,
) -> "HTTPResponse":
    """GET *url* through the yggdrasil HTTP session → an :class:`HTTPResponse`."""
    return session().get(url, params=params, headers=headers, timeout=timeout)


def _forced_media(url: str, resp: "HTTPResponse"):
    """Resolve a media type for *resp*, falling back to the URL extension when
    the server's content-type is generic/wrong."""
    from yggdrasil.enums import MediaType, MimeTypes

    mt = resp.media_type
    mime = mt.mime_type
    # A confidently-typed tabular/columnar body — trust it.
    if getattr(mime, "is_tabular", False):
        return None
    ext = url.rsplit("?", 1)[0].rsplit(".", 1)[-1].lower()
    name = EXT_MIME.get(ext)
    if name is None:
        return None
    return MediaType.from_mime(getattr(MimeTypes, name))


def read_table(url: str, *, fmt: Optional[str] = None, **fetch_kwargs: Any) -> "pl.DataFrame":
    """Fetch *url* and parse it into a polars DataFrame via the io handlers.

    Handles every tabular format the io layer knows (CSV / JSON / Parquet /
    Arrow / XLSX). ``fmt`` (e.g. ``"csv"``) forces the leaf when the URL has no
    extension and the server mislabels the body.
    """
    from yggdrasil.enums import MediaType, MimeTypes

    resp = fetch(url, **fetch_kwargs)
    forced = None
    if fmt is not None:
        forced = MediaType.from_mime(getattr(MimeTypes, EXT_MIME.get(fmt.lower(), fmt.upper())))
    else:
        forced = _forced_media(url, resp)
    if forced is not None:
        resp.media_type = forced
    return resp.to_polars()


def read_json(url: str, **fetch_kwargs: Any) -> Any:
    """Fetch *url* and decode its JSON body."""
    return fetch(url, **fetch_kwargs).json()


def read_image(url: str, *, save_to: Optional[str] = None, **fetch_kwargs: Any) -> dict[str, Any]:
    """Fetch an image; report its content type, byte size, and (if Pillow is
    present) pixel dimensions. Optionally save the bytes to *save_to*."""
    import pathlib

    resp = fetch(url, **fetch_kwargs)
    data = resp.content
    info: dict[str, Any] = {
        "url": url,
        "status": resp.status_code,
        "content_type": str(resp.media_type.mime_type.value),
        "bytes": len(data),
    }
    try:  # dimensions are a nicety, not a requirement
        import io as _io

        from PIL import Image

        with Image.open(_io.BytesIO(data)) as im:
            info["width"], info["height"], info["mode"] = im.width, im.height, im.mode
    except Exception:
        pass
    if save_to:
        path = pathlib.Path(save_to)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        info["saved_to"] = str(path)
    return info


class _Reader(html.parser.HTMLParser):
    """Minimal HTML → readable text + anchor links (a lightweight 'browser')."""

    _SKIP = {"script", "style", "head", "noscript", "svg"}

    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []
        self._skip = 0
        self.links: list[tuple[str, str]] = []
        self._href: Optional[str] = None
        self._anchor: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        if tag in self._SKIP:
            self._skip += 1
        elif tag == "a":
            self._href = dict(attrs).get("href")
            self._anchor = []
        elif tag in ("p", "br", "div", "li", "tr", "h1", "h2", "h3", "h4"):
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP and self._skip:
            self._skip -= 1
        elif tag == "a" and self._href:
            text = " ".join("".join(self._anchor).split())
            self.links.append((text, self._href))
            self._href = None

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        self._chunks.append(data)
        if self._href is not None:
            self._anchor.append(data)

    def text(self) -> str:
        joined = "".join(self._chunks)
        lines = [ln.strip() for ln in joined.splitlines()]
        return "\n".join(ln for ln in lines if ln)


def read_text(url: str, *, max_chars: int = 4000, **fetch_kwargs: Any) -> dict[str, Any]:
    """Browse *url*: fetch it and return readable text + links.

    HTML is stripped to readable text (scripts/styles dropped) and its anchors
    collected — a lightweight browser view. Non-HTML bodies come back as text.
    """
    resp = fetch(url, **fetch_kwargs)
    mime = str(resp.media_type.mime_type.value)
    out: dict[str, Any] = {"url": url, "status": resp.status_code, "content_type": mime}
    if "html" in mime:
        reader = _Reader()
        reader.feed(resp.text)
        text = reader.text()
        out["links"] = [{"text": t, "href": h} for t, h in reader.links[:50]]
    else:
        text = resp.text
    out["truncated"] = len(text) > max_chars
    out["text"] = text[:max_chars]
    return out
