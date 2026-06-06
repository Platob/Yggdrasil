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
import os
import re
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
    "scrape",
    "discover_apis",
    "Browser",
    "browser_available",
    "fill_form",
    "interact",
]

#: One realistic browser header profile for this process, from the http_
#: user-agent utilities — so sites serve the content they serve a browser.
#: Stable per run (a session shouldn't flip identity mid-scrape); override the
#: UA with ``YGG_LOKI_USER_AGENT``. Normal scraping hygiene, not evasion.
_BROWSER_HEADERS: "Optional[dict[str, str]]" = None


def _browser_headers() -> dict[str, str]:
    global _BROWSER_HEADERS
    if _BROWSER_HEADERS is None:
        from yggdrasil.http_.user_agents import random_browser_headers

        _BROWSER_HEADERS = dict(random_browser_headers())
        ua = os.getenv("YGG_LOKI_USER_AGENT")
        if ua:
            _BROWSER_HEADERS["User-Agent"] = ua
    return _BROWSER_HEADERS

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
    """GET *url* through the yggdrasil HTTP session → an :class:`HTTPResponse`.

    Sends a realistic browser header profile (from the http_ user-agent utils)
    so pages render normally, unless the caller overrides specific headers.
    """
    h = dict(_browser_headers())
    if headers:
        h.update(headers)
    return session().get(url, params=params, headers=h, timeout=timeout)


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


def scrape(url: str, *, max_chars: int = 6000, **fetch_kwargs: Any) -> dict[str, Any]:
    """Scrape a page into structured pieces — title, meta, text, links, tables.

    A richer :func:`read_text`: pulls the ``<title>`` and meta description, the
    readable text, the anchor links, the first HTML ``<table>`` as records (if
    any), and any embedded JSON-LD structured data. Legitimate extraction of a
    public page's content — it does not defeat access controls.
    """
    resp = fetch(url, **fetch_kwargs)
    html = resp.text
    out: dict[str, Any] = {"url": url, "status": resp.status_code,
                           "content_type": str(resp.media_type.mime_type.value)}
    title = re.search(r"<title[^>]*>(.*?)</title>", html, re.S | re.I)
    out["title"] = (title.group(1).strip() if title else "")[:200]
    desc = re.search(r'<meta[^>]+name=["\']description["\'][^>]+content=["\'](.*?)["\']',
                     html, re.S | re.I)
    out["description"] = (desc.group(1).strip() if desc else "")[:400]
    reader = _Reader()
    reader.feed(html)
    out["links"] = [{"text": t, "href": h} for t, h in reader.links[:60]]
    text = reader.text()
    out["text"] = text[:max_chars]
    out["json_ld"] = _json_ld(html)[:5]
    try:
        df = read_table(url, **fetch_kwargs)
        out["table_preview"] = df.head(10).to_dicts()
        out["table_shape"] = list(df.shape)
    except Exception:
        out["table_preview"] = None
    return out


def _json_ld(html: str) -> list:
    import json as _json

    blocks = []
    for m in re.finditer(
        r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        html, re.S | re.I,
    ):
        try:
            blocks.append(_json.loads(m.group(1).strip()))
        except Exception:
            pass
    return blocks


# -- interactive browser automation ----------------------------------------
#
# Reading a page (above) is a plain HTTP fetch; *interacting* with one — typing
# into fields, ticking boxes, clicking buttons, submitting forms and reading
# what the page becomes — needs a real browser. That's a headless Playwright
# session, imported lazily so everything above keeps working without it.


def browser_available() -> bool:
    """True when Playwright (and a browser binary) is installed for automation."""
    import importlib.util

    if importlib.util.find_spec("playwright") is None:
        return False
    try:  # the package can be present without the browser binaries downloaded
        from playwright.sync_api import sync_playwright

        with sync_playwright() as pw:
            return bool(pw.chromium.executable_path)
    except Exception:
        return False


class Browser:
    """A headless browser session for interacting with a live page.

    Drive a page the way a person would — :meth:`goto`, :meth:`fill` a field,
    :meth:`type`, :meth:`check` a box, :meth:`select_option`, :meth:`click` a
    button, :meth:`submit`, then read the result (:meth:`text`, :attr:`url`,
    :meth:`title`). Backed by Playwright (Chromium by default), imported only
    when you open one. Use as a context manager so the browser always closes::

        with web.Browser() as b:
            b.goto("https://example.com/login")
            b.fill("#user", "me").fill("#pass", "secret").submit("button[type=submit]")
            print(b.url, b.text())

    The action methods return ``self`` so calls chain.
    """

    def __init__(self, *, headless: bool = True, browser: str = "chromium",
                 timeout: float = 30000) -> None:
        self._kind = browser
        self._headless = headless
        self._timeout = timeout
        self._pw = None
        self._browser = None
        self.page: Any = None

    def __enter__(self) -> "Browser":
        from playwright.sync_api import sync_playwright

        self._pw = sync_playwright().start()
        self._browser = getattr(self._pw, self._kind).launch(headless=self._headless)
        ctx = self._browser.new_context(user_agent=_browser_headers().get("User-Agent"))
        self.page = ctx.new_page()
        self.page.set_default_timeout(self._timeout)
        return self

    def __exit__(self, *exc: Any) -> bool:
        try:
            if self._browser is not None:
                self._browser.close()
        finally:
            if self._pw is not None:
                self._pw.stop()
        return False

    def goto(self, url: str) -> "Browser":
        self.page.goto(url)
        return self

    def fill(self, selector: str, value: str) -> "Browser":
        """Set an input/textarea's value (clears it first)."""
        self.page.fill(selector, value)
        return self

    def type(self, selector: str, text: str, *, delay: float = 0) -> "Browser":
        """Type *text* key by key (fires keypress handlers, unlike :meth:`fill`)."""
        self.page.type(selector, text, delay=delay)
        return self

    def click(self, selector: str) -> "Browser":
        self.page.click(selector)
        return self

    def check(self, selector: str, value: bool = True) -> "Browser":
        (self.page.check if value else self.page.uncheck)(selector)
        return self

    def select_option(self, selector: str, value: str) -> "Browser":
        self.page.select_option(selector, value)
        return self

    def press(self, selector: str, key: str) -> "Browser":
        self.page.press(selector, key)
        return self

    def submit(self, selector: Optional[str] = None) -> "Browser":
        """Submit a form — click *selector* if given, else press Enter."""
        if selector:
            self.page.click(selector)
        else:
            self.page.keyboard.press("Enter")
        return self

    def wait_for(self, selector: str) -> "Browser":
        self.page.wait_for_selector(selector)
        return self

    @property
    def url(self) -> str:
        return self.page.url

    def title(self) -> str:
        return self.page.title()

    def value(self, selector: str) -> str:
        return self.page.input_value(selector)

    def text(self, max_chars: int = 4000) -> str:
        return self.page.inner_text("body")[:max_chars]

    def screenshot(self, path: str) -> str:
        self.page.screenshot(path=path)
        return path


def fill_form(
    url: str,
    fields: dict[str, str],
    *,
    submit: Optional[str] = None,
    headless: bool = True,
    browser: str = "chromium",
    wait_for: Optional[str] = None,
    screenshot: Optional[str] = None,
    max_chars: int = 4000,
) -> dict[str, Any]:
    """Open *url*, fill *fields* (CSS selector → value), optionally submit, and
    return the resulting page state — the high-level "fill in this form".

    ``submit`` is the selector of the submit button (omit to skip submitting);
    ``wait_for`` waits for a selector to appear after submit (e.g. a results
    panel); ``screenshot`` saves a PNG of the final page.
    """
    with Browser(headless=headless, browser=browser) as b:
        b.goto(url)
        for selector, value in fields.items():
            b.fill(selector, str(value))
        if submit:
            b.submit(submit)
        if wait_for:
            b.wait_for(wait_for)
        out: dict[str, Any] = {"url": b.url, "title": b.title(),
                               "filled": list(fields), "text": b.text(max_chars)}
        if screenshot:
            out["screenshot"] = b.screenshot(screenshot)
        return out


def interact(
    url: str,
    steps: list[dict[str, Any]],
    *,
    headless: bool = True,
    browser: str = "chromium",
    screenshot: Optional[str] = None,
    max_chars: int = 4000,
) -> dict[str, Any]:
    """Drive a page through a sequence of interaction *steps*, return its final
    state. Each step is one action dict — the page is a thing you operate::

        web.interact("https://shop.example/search", [
            {"type": ["#q", "wireless headphones"]},
            {"press": ["#q", "Enter"]},
            {"wait_for": ".results"},
            {"click": ".results a:first-child"},
        ])

    Actions: ``goto`` (url), ``fill``/``type``/``select``/``press`` (``[selector,
    value]``), ``click``/``check``/``wait_for`` (selector), ``submit`` (button
    selector or ``null`` for Enter).
    """
    log: list[dict[str, Any]] = []
    with Browser(headless=headless, browser=browser) as b:
        b.goto(url)
        for step in steps:
            for action, arg in step.items():
                if action == "goto":
                    b.goto(arg)
                elif action == "fill":
                    b.fill(*arg)
                elif action == "type":
                    b.type(*arg)
                elif action == "select":
                    b.select_option(*arg)
                elif action == "press":
                    b.press(*arg)
                elif action == "click":
                    b.click(arg)
                elif action == "check":
                    b.check(arg) if isinstance(arg, str) else b.check(*arg)
                elif action == "wait_for":
                    b.wait_for(arg)
                elif action == "submit":
                    b.submit(arg if isinstance(arg, str) else None)
                else:
                    raise ValueError(f"unknown interaction step {action!r}")
                log.append({action: arg})
        out: dict[str, Any] = {"url": b.url, "title": b.title(),
                               "steps": log, "text": b.text(max_chars)}
        if screenshot:
            out["screenshot"] = b.screenshot(screenshot)
        return out


def discover_apis(url: str, *, limit: int = 40, **fetch_kwargs: Any) -> dict[str, Any]:
    """Discover the data APIs a page already uses — its underlying endpoints.

    Inspects a public page for the data sources it embeds or calls: JSON-LD
    structured data, ``<script type="application/json">`` payloads (e.g.
    ``__NEXT_DATA__``), and candidate endpoint URLs referenced in markup/scripts
    (``/api/…``, ``*.json``/``*.csv``, ``fetch(...)`` targets). This surfaces the
    *documented/embedded* APIs to pull data from — it does not bypass auth; any
    endpoint that needs credentials still needs them.
    """
    import json as _json

    resp = fetch(url, **fetch_kwargs)
    html = resp.text
    blocks = []
    for m in re.finditer(
        r'<script[^>]+type=["\']application/json["\'][^>]*>(.*?)</script>',
        html, re.S | re.I,
    ):
        try:
            data = _json.loads(m.group(1).strip())
            blocks.append({"keys": list(data.keys())[:20] if isinstance(data, dict) else "array"})
        except Exception:
            pass
    endpoints: set[str] = set()
    # data files (*.json / *.csv), and paths with a real /api/ or /v<N>/ segment.
    for m in re.finditer(
        r'["\']('
        r'https?://[^"\'\s]+?\.(?:json|csv)(?:\?[^"\']*)?'
        r'|(?:https?:)?//[^"\'\s]+?/(?:api|graphql|v\d+)/[^"\'\s]*'
        r'|/(?:api|graphql|v\d+)/[^"\'\s]*'
        r')["\']',
        html,
    ):
        endpoints.add(m.group(1))
    for m in re.finditer(r'fetch\(\s*["\']([^"\']+)["\']', html):
        endpoints.add(m.group(1))
    title = re.search(r"<title[^>]*>(.*?)</title>", html, re.S | re.I)
    return {
        "url": url,
        "status": resp.status_code,
        "title": (title.group(1).strip() if title else "")[:200],
        "json_ld": _json_ld(html)[:5],
        "json_blocks": blocks[:10],
        "endpoints": sorted(endpoints)[:limit],
        "note": "embedded/public endpoints only — credentials are not bypassed",
    }
