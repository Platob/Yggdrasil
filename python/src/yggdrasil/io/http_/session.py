"""Concrete HTTP/HTTPS session backed by ``urllib3``.

The session also exposes lazy browser-style features — User-Agent generation,
Chromium client-hint headers, a cookie jar, navigation helpers — that stay
dormant until a caller invokes :meth:`HTTPSession.get` / :meth:`post` /
:meth:`navigate` / :meth:`submit_form`. None of those features pay any cost
unless they are actually used; the plain :meth:`send` path is unchanged.
"""
from __future__ import annotations

import datetime as dt
import re
from itertools import takewhile
from typing import Any, Mapping, Optional
from urllib.parse import urlencode, urlsplit

import urllib3

from yggdrasil.concurrent.threading import Job, JobPoolExecutor
from yggdrasil.dataclasses import WaitingConfig
from yggdrasil.dataclasses.waiting import DEFAULT_WAITING_CONFIG
from yggdrasil.data.enums import MediaTypes
from yggdrasil.io import BytesIO
from yggdrasil.io.primitive import ArrowIPCIO
from yggdrasil.io.url import URL
from yggdrasil.io.user_agents import UserAgentGenerator

from ..request import PreparedRequest
from ..send_config import SendConfig
from ..session import Session
from .cookies import Cookies
from .response import HTTPResponse

__all__ = ["HTTPSession"]


# Backoff tuning. 429s get a longer, gentler schedule than 5xx because rate
# limits often need real wall-clock time to clear; transient server errors
# usually resolve faster.
_RETRY_TOTAL = 6
_RETRY_CONNECT = 3
_RETRY_READ = 3

# 5xx schedule: 1, 2, 4, 8, 16, 32 (capped at backoff_max)
_BACKOFF_5XX_FACTOR = 1.0
_BACKOFF_5XX_MAX = 60.0

# 429 schedule: 4, 8, 16, 32, 64, 128 (capped at backoff_max)
# Server-supplied Retry-After always wins over this when present.
_BACKOFF_429_FACTOR = 4.0
_BACKOFF_429_MAX = 300.0

_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


# ── Browser-mode constants ────────────────────────────────────────────────

_BROWSER_ACCEPT = (
    "text/html,application/xhtml+xml,application/xml;q=0.9,"
    "image/avif,image/webp,image/apng,*/*;q=0.8,"
    "application/signed-exchange;v=b3;q=0.7"
)
_BROWSER_ACCEPT_ENCODING = "gzip, deflate, br, zstd"

_CHROME_RE = re.compile(r"Chrome/(\d+)\.")
_EDGE_RE = re.compile(r"Edg(?:e)?/(\d+)\.")
_FIREFOX_RE = re.compile(r"Firefox/(\d+)\.")
_SAFARI_RE = re.compile(r"\bSafari/")
_WINDOWS_RE = re.compile(r"Windows NT")
_MAC_RE = re.compile(r"Macintosh|Mac OS X")
_LINUX_RE = re.compile(r"Linux")
_ANDROID_RE = re.compile(r"Android")
_IOS_RE = re.compile(r"iPhone|iPad|iPod")
_MOBILE_RE = re.compile(r"Mobile|Android")

_PLATFORM_METHOD_MAP: dict[str, str] = {
    "windows": "_desktop_windows",
    "mac": "_desktop_mac",
    "linux": "_desktop_linux",
    "android": "_android",
    "ios": "_ios",
}


class _TieredRetry(urllib3.Retry):
    """``urllib3.Retry`` variant with status-aware backoff.

    Standard ``Retry`` exposes a single ``backoff_factor`` shared by every
    retry, so 429 (rate limit) and 503 (transient outage) get the same
    schedule. This subclass branches on the most recent response status:

    * **429** uses a longer, gentler exponential schedule, since rate-limit
      windows are typically wall-clock bound and respond poorly to tight
      retries.
    * **Everything else** (5xx, transport errors) uses a shorter schedule.
    * The server's ``Retry-After`` header — when present and respected via
      ``respect_retry_after_header=True`` — always overrides this, because
      ``urllib3`` checks ``get_retry_after`` before ``get_backoff_time``.
    """

    def get_backoff_time(self) -> float:  # type: ignore[override]
        # Mirror urllib3's own short-circuit: no backoff before the second
        # consecutive error. ``history`` is a tuple of RequestHistory entries.
        consecutive_errors = list(
            takewhile(lambda x: x.redirect_location is None, reversed(self.history))
        )
        if len(consecutive_errors) <= 1:
            return 0.0

        last_status = consecutive_errors[0].status

        if last_status == 429:
            # Count *consecutive* 429s only — if the last attempt was a 503,
            # we want the 5xx schedule, not a 429 schedule inflated by older
            # rate-limit hits.
            n = 0
            for h in consecutive_errors:
                if h.status == 429:
                    n += 1
                else:
                    break
            backoff = _BACKOFF_429_FACTOR * (2 ** (n - 1))
            return float(min(_BACKOFF_429_MAX, backoff))

        # Default 5xx / transport-error schedule, mirroring urllib3's formula
        # but with our own factor and cap.
        backoff = _BACKOFF_5XX_FACTOR * (2 ** (len(consecutive_errors) - 1))
        return float(min(_BACKOFF_5XX_MAX, backoff))


class HTTPSession(Session):
    """HTTP/HTTPS session backed by a ``urllib3`` connection pool.

    Pure-transport callers can keep using :meth:`send` and :meth:`send_many`
    exactly as before. The browser-style helpers — :meth:`get`, :meth:`post`,
    :meth:`head`, :meth:`navigate`, :meth:`follow_link`, :meth:`submit_form`,
    plus the :attr:`cookies` jar and :attr:`user_agent` machinery — are fully
    lazy: the cookie manager and user-agent generator are not constructed
    until something actually reads them.
    """

    def __init__(
        self,
        base_url: Optional[URL | str] = None,
        verify: bool = True,
        pool_maxsize: int = 10,
        send_headers: Optional[dict[str, str]] = None,
        waiting: WaitingConfig = DEFAULT_WAITING_CONFIG,
        *,
        key: str = "",
        user_agent: Optional[str] = None,
        accept: str = _BROWSER_ACCEPT,
        accept_language: str = "en-US,en;q=0.9",
        accept_encoding: str = _BROWSER_ACCEPT_ENCODING,
        ua_seed: Optional[int] = None,
        cookies: Optional[Cookies | Mapping[str, str]] = None,
        browser_mode: bool = False,
    ) -> None:
        if getattr(self, "_initialized", False):
            return
        # ``urllib3`` connection pools cap out at 8 hosts comfortably for
        # our typical workloads; clamp here so a caller passing the legacy
        # default (10) does not blow past the urllib3 sweet spot.
        pool_maxsize = min(8, int(pool_maxsize)) if pool_maxsize else 8
        super().__init__(
            base_url=base_url,
            verify=verify,
            pool_maxsize=pool_maxsize,
            send_headers=send_headers,
            waiting=waiting,
            key=key,
        )
        self._http_pool: Optional[urllib3.PoolManager] = self._build_http_pool()

        # Browser-mode configuration. Stored verbatim; the actual
        # ``UserAgentGenerator`` / ``Cookies`` objects are created lazily on
        # first read so a plain transport-only session never pays for them.
        # ``browser_mode`` gates whether the verb helpers (get/head/post/
        # navigate/submit_form/follow_link) layer on User-Agent, Accept,
        # Sec-Fetch-*, sec-ch-ua-*, Cookie, and Referer. Off by default —
        # the verbs then send only ``send_headers`` plus per-call extras,
        # which is what most API clients want. Each verb also accepts
        # ``browser_mode=`` to override the session-level flag per call.
        self.browser_mode: bool = browser_mode
        self.user_agent: Optional[str] = user_agent
        self.accept: str = accept
        self.accept_language: str = accept_language
        self.accept_encoding: str = accept_encoding
        self.ua_seed: Optional[int] = ua_seed
        self._referrer: Optional[str] = None
        self._cookies: Optional[Cookies] = (
            cookies if isinstance(cookies, Cookies)
            else Cookies(cookies) if cookies
            else None
        )
        self._ua_generator: Optional[UserAgentGenerator] = None

    _TRANSIENT_STATE_ATTRS = Session._TRANSIENT_STATE_ATTRS | {"_http_pool"}

    def __setstate__(self, state):
        # Singleton hit: ``Session.__setstate__`` returns early and the
        # cached pool stays attached — skip the rebuild so we don't drop it.
        if getattr(self, "_initialized", False):
            return
        super().__setstate__(state)
        self._http_pool = self._build_http_pool()


    def _build_retry(self) -> urllib3.Retry:
        """Build the :class:`urllib3.Retry` policy used by the connection pool.

        Subclasses can override to swap the policy entirely, or call
        ``super()._build_retry().new(...)`` to tweak a single field.
        """
        return _TieredRetry(
            total=_RETRY_TOTAL,
            connect=_RETRY_CONNECT,
            read=_RETRY_READ,
            status=_RETRY_TOTAL,
            other=2,
            status_forcelist=_RETRY_STATUSES,
            allowed_methods=None,  # retry every method, incl. POST/PATCH
            respect_retry_after_header=True,
            raise_on_status=False,
            raise_on_redirect=False,
            # backoff_factor/backoff_max are unused — _TieredRetry overrides
            # get_backoff_time entirely — but we set sane defaults so any
            # fallback path (e.g. .new() that drops back to base behavior) is
            # still well-behaved.
            backoff_factor=_BACKOFF_5XX_FACTOR,
            backoff_max=_BACKOFF_429_MAX,
        )

    def _build_http_pool(self) -> urllib3.PoolManager:
        return urllib3.PoolManager(
            num_pools=self.pool_maxsize,
            maxsize=self.pool_maxsize,
            block=True,
            retries=self._build_retry(),
            cert_reqs="CERT_REQUIRED" if self.verify else "CERT_NONE",
            ca_certs=None,
        )

    @property
    def http_pool(self):
        if self._http_pool is None:
            with self._lock:
                if self._http_pool is None:
                    self._http_pool = self._build_http_pool()
        return self._http_pool

    # ------------------------------------------------------------------
    # Lazy browser-mode state
    # ------------------------------------------------------------------

    @property
    def cookies(self) -> Cookies:
        """Cookie jar managed by :class:`Cookies`. Built on first access."""
        if self._cookies is None:
            with self._lock:
                if self._cookies is None:
                    self._cookies = Cookies()
        return self._cookies

    @property
    def ua_generator(self) -> UserAgentGenerator:
        """:class:`UserAgentGenerator` instance, built on first access."""
        if self._ua_generator is None:
            with self._lock:
                if self._ua_generator is None:
                    self._ua_generator = UserAgentGenerator(seed=self.ua_seed)
        return self._ua_generator

    @property
    def referrer(self) -> Optional[str]:
        """URL of the last successfully navigated page, sent as ``Referer``."""
        return self._referrer

    def set_referrer(self, url: Optional[str]) -> None:
        self._referrer = str(url) if url else None

    def clear_referrer(self) -> None:
        self._referrer = None

    def get_user_agent(self) -> str:
        """Return the active User-Agent, generating one lazily if unset."""
        if self.user_agent is None:
            self.user_agent = self.ua_generator.random()
        return self.user_agent

    def set_user_agent(self, ua: str) -> None:
        self.user_agent = str(ua)

    def rotate_user_agent(self, seed: Optional[int] = None) -> str:
        """Generate and apply a new random ``User-Agent`` string."""
        self._ua_generator = UserAgentGenerator(seed=seed)
        self.user_agent = self._ua_generator.random()
        return self.user_agent

    def set_browser_preset(
        self,
        browser: str = "chrome",
        *,
        platform: str = "windows",
        seed: Optional[int] = None,
    ) -> str:
        """Pick a UA matching *browser* / *platform* and apply it.

        Samples :class:`UserAgentGenerator` up to 20 times until a UA matching
        *browser* is found; falls back to the last generated UA otherwise.
        """
        from random import Random

        browser_l = browser.lower()
        method_name = _PLATFORM_METHOD_MAP.get(platform.lower(), "_desktop_windows")

        ua: str = ""
        for attempt in range(20):
            current_seed = (seed + attempt) if seed is not None else attempt
            gen = UserAgentGenerator(seed=current_seed)
            r = Random(current_seed)
            ua = getattr(gen, method_name)(r)
            ua_lower = ua.lower()

            if browser_l == "edge" and "edg/" in ua_lower:
                self.user_agent = ua
                return ua
            if (
                browser_l in ("chrome", "chromium")
                and "chrome/" in ua_lower
                and "edg/" not in ua_lower
            ):
                self.user_agent = ua
                return ua
            if browser_l == "firefox" and "firefox/" in ua_lower:
                self.user_agent = ua
                return ua
            if (
                browser_l == "safari"
                and "safari/" in ua_lower
                and "chrome/" not in ua_lower
                and "firefox/" not in ua_lower
            ):
                self.user_agent = ua
                return ua

        self.user_agent = ua
        return ua

    # ── UA introspection ──────────────────────────────────────────────

    @property
    def is_mobile(self) -> bool:
        return bool(_MOBILE_RE.search(self.user_agent or ""))

    @property
    def platform(self) -> str:
        ua = self.user_agent or ""
        if _ANDROID_RE.search(ua):
            return "Android"
        if _IOS_RE.search(ua):
            return "iOS"
        if _WINDOWS_RE.search(ua):
            return "Windows"
        if _MAC_RE.search(ua):
            return "macOS"
        if _LINUX_RE.search(ua):
            return "Linux"
        return "Unknown"

    @property
    def browser_name(self) -> str:
        ua = self.user_agent or ""
        if _EDGE_RE.search(ua):
            return "Edge"
        if _CHROME_RE.search(ua):
            return "Chrome"
        if _FIREFOX_RE.search(ua):
            return "Firefox"
        if _SAFARI_RE.search(ua):
            return "Safari"
        return "Unknown"

    @property
    def sec_ch_ua(self) -> Optional[str]:
        """Chromium ``sec-ch-ua`` header value, or ``None`` for non-Chromium."""
        ua = self.user_agent or ""
        edge_m = _EDGE_RE.search(ua)
        chrome_m = _CHROME_RE.search(ua)
        if edge_m:
            v = edge_m.group(1)
            return (
                f'"Microsoft Edge";v="{v}", '
                f'"Chromium";v="{v}", '
                '"Not-A.Brand";v="99"'
            )
        if chrome_m:
            v = chrome_m.group(1)
            return (
                f'"Google Chrome";v="{v}", '
                f'"Chromium";v="{v}", '
                '"Not-A.Brand";v="99"'
            )
        return None

    @property
    def sec_ch_ua_platform(self) -> str:
        _map = {
            "Windows": "Windows",
            "macOS": "macOS",
            "Linux": "Linux",
            "Android": "Android",
            "iOS": "iOS",
        }
        return _map.get(self.platform, "Unknown")

    # ------------------------------------------------------------------
    # Header construction
    # ------------------------------------------------------------------

    def _build_request_headers(
        self,
        request: PreparedRequest,
    ) -> Optional[dict[str, str]]:
        """Return the headers dict to merge into *request* before sending.

        Subclasses may override this to inject per-request headers without
        replacing the entire :attr:`send_headers` mapping.  The default
        implementation returns :attr:`send_headers` unchanged.
        """
        return self.send_headers

    def _compute_sec_fetch_site(self, request_url: URL) -> str:
        """Compare :attr:`referrer` against *request_url* for ``Sec-Fetch-Site``.

        Same-site uses a two-label registrable-domain heuristic — sufficient
        for the common single-TLD case, not a public-suffix lookup.
        """
        if not self._referrer:
            return "none"
        try:
            ref = urlsplit(self._referrer)
            req = urlsplit(request_url.to_string())
            if ref.netloc == req.netloc:
                return "same-origin"
            ref_host = ref.hostname or ""
            req_host = req.hostname or ""
            if ref_host and req_host:
                if self._registrable_domain(ref_host) == self._registrable_domain(req_host):
                    return "same-site"
            return "cross-site"
        except Exception:
            return "none"

    @staticmethod
    def _registrable_domain(host: str) -> str:
        parts = host.rsplit(".", 2)
        return ".".join(parts[-2:]) if len(parts) >= 2 else host

    def _build_browser_headers(
        self,
        request_url: URL,
        extra: Optional[Mapping[str, str]] = None,
        *,
        browser_mode: Optional[bool] = None,
    ) -> dict[str, str]:
        """Build the per-request header dict for a verb call.

        ``browser_mode`` (defaulting to :attr:`self.browser_mode` when
        ``None``) controls whether the browser-emulation layer fires:

        * **off (default):** only :attr:`send_headers` and *extra* are
          merged. No User-Agent generator is touched, no Sec-Fetch-*,
          sec-ch-ua-*, Cookie, or Referer is injected.
        * **on:** layered lowest-to-highest as
          (1) browser defaults (``User-Agent``, ``Accept`` family,
          ``Sec-Fetch-*``, sec-ch-ua-*, Cookie, Referer),
          (2) session-level :attr:`send_headers`,
          (3) *extra* (the per-request ``headers=`` argument).
        """
        if browser_mode is None:
            browser_mode = self.browser_mode

        if not browser_mode:
            headers: dict[str, str] = {}
            if self.send_headers:
                headers.update(self.send_headers)
            if extra:
                headers.update(extra)
            return headers

        headers = {
            "User-Agent": self.get_user_agent(),
            "Accept": self.accept,
            "Accept-Language": self.accept_language,
            "Accept-Encoding": self.accept_encoding,
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Cache-Control": "max-age=0",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": self._compute_sec_fetch_site(request_url),
            "Sec-Fetch-User": "?1",
        }

        ch_ua = self.sec_ch_ua
        if ch_ua:
            headers["sec-ch-ua"] = ch_ua
            headers["sec-ch-ua-mobile"] = "?1" if self.is_mobile else "?0"
            headers["sec-ch-ua-platform"] = f'"{self.sec_ch_ua_platform}"'

        if self._cookies:
            cookie_header = self._cookies.to_header()
            if cookie_header:
                headers["Cookie"] = cookie_header

        if self._referrer:
            headers["Referer"] = self._referrer

        if self.send_headers:
            headers.update(self.send_headers)
        if extra:
            headers.update(extra)
        return headers

    # ------------------------------------------------------------------
    # URL / params helpers
    # ------------------------------------------------------------------

    def _resolve_url(self, url: URL | str) -> URL:
        """Resolve *url* to an absolute :class:`~yggdrasil.io.url.URL`.

        Resolution rules: absolute URLs pass through; relative paths
        (``"/x"``, ``"./x"``, ``"../x"``, ``"x"``, ``"x/y"``) join against
        :attr:`base_url`; ``//host`` protocol-relative gets ``https:``;
        a first-segment-with-dot like ``"example.com/x"`` gets ``https://``;
        Windows drive-letter paths raise.
        """
        if isinstance(url, URL):
            if url.is_absolute:
                return url
            if self.base_url:
                return self.base_url.join(url.to_string())
            return url

        s = str(url).strip()
        if not s:
            raise ValueError("URL must not be empty.")

        if s.startswith(("https://", "http://")):
            return URL.from_(s)

        if s.startswith("//"):
            return URL.from_("https:" + s)

        if s.startswith(("/", "./", "../")):
            if self.base_url:
                return self.base_url.join(s)
            raise ValueError(
                f"Cannot resolve path-relative URL {s!r}: no base_url is set on "
                "this session. Set base_url or pass a full URL."
            )

        if len(s) >= 3 and s[0].isalpha() and s[1] == ":" and s[2] in ("/", "\\"):
            raise ValueError(
                f"URL looks like a local Windows path, not an HTTP URL: {s!r}. "
                "Pass a full URL with scheme (e.g. 'https://host/path')."
            )

        first_seg = s.split("/", 1)[0] if "/" in s else s
        if "." in first_seg and not first_seg.startswith("."):
            return URL.from_("https://" + s)

        if self.base_url:
            return self.base_url.join(s)

        raise ValueError(
            f"Cannot resolve relative URL {s!r}: no base_url is set on this "
            "session. Set base_url (e.g. HTTPSession(base_url='https://api.example.com')) "
            "or pass a full URL."
        )

    @staticmethod
    def _apply_params(url: URL, params: Mapping[str, Any]) -> URL:
        for key, value in params.items():
            key_s = str(key)
            if isinstance(value, (list, tuple)):
                for v in value:
                    url = url.add_query_item(key_s, str(v), replace=False)
            else:
                url = url.add_query_item(key_s, str(value), replace=False)
        return url

    # ------------------------------------------------------------------
    # Browser-style HTTP verbs
    # ------------------------------------------------------------------

    def get(
        self,
        url: URL | str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        headers: Optional[Mapping[str, str]] = None,
        browser_mode: Optional[bool] = None,
        config: SendConfig | Mapping[str, Any] | None = None,
        **send_kwargs: Any,
    ) -> HTTPResponse:
        """Issue a ``GET`` to *url*.

        ``browser_mode`` overrides :attr:`self.browser_mode` for this call:
        ``True`` adds the browser-emulation header set, ``False`` sends only
        :attr:`send_headers` + *headers*. ``None`` (the default) follows
        the session flag.
        """
        resolved = self._resolve_url(url)
        if params:
            resolved = self._apply_params(resolved, params)
        merged = self._build_browser_headers(resolved, headers, browser_mode=browser_mode)
        request = PreparedRequest.prepare("GET", resolved, headers=merged)
        return self.send(request, config=config, **send_kwargs)  # type: ignore[return-value]

    def head(
        self,
        url: URL | str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        headers: Optional[Mapping[str, str]] = None,
        browser_mode: Optional[bool] = None,
        config: SendConfig | Mapping[str, Any] | None = None,
        **send_kwargs: Any,
    ) -> HTTPResponse:
        """Issue a ``HEAD`` to *url*. See :meth:`get` for ``browser_mode``."""
        resolved = self._resolve_url(url)
        if params:
            resolved = self._apply_params(resolved, params)
        merged = self._build_browser_headers(resolved, headers, browser_mode=browser_mode)
        request = PreparedRequest.prepare("HEAD", resolved, headers=merged)
        return self.send(request, config=config, **send_kwargs)  # type: ignore[return-value]

    def post(
        self,
        url: URL | str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        body: Optional[Any] = None,
        json: Optional[Any] = None,
        headers: Optional[Mapping[str, str]] = None,
        browser_mode: Optional[bool] = None,
        config: SendConfig | Mapping[str, Any] | None = None,
        **send_kwargs: Any,
    ) -> HTTPResponse:
        """Issue a ``POST`` to *url*. See :meth:`get` for ``browser_mode``."""
        resolved = self._resolve_url(url)
        if params:
            resolved = self._apply_params(resolved, params)
        merged = self._build_browser_headers(resolved, headers, browser_mode=browser_mode)
        request = PreparedRequest.prepare(
            "POST", resolved, headers=merged, body=body, json=json
        )
        return self.send(request, config=config, **send_kwargs)  # type: ignore[return-value]

    def navigate(
        self,
        url: URL | str,
        *,
        method: str = "GET",
        params: Optional[Mapping[str, Any]] = None,
        body: Optional[Any] = None,
        json: Optional[Any] = None,
        headers: Optional[Mapping[str, str]] = None,
        browser_mode: Optional[bool] = None,
        update_referrer: bool = True,
        update_cookies: bool = True,
        raise_error: bool = True,
        config: SendConfig | Mapping[str, Any] | None = None,
        **send_kwargs: Any,
    ) -> HTTPResponse:
        """Navigate to *url*, optionally updating referrer and cookie jar.

        See :meth:`get` for ``browser_mode``. The referrer/cookie-jar
        updates are independent of ``browser_mode`` — they fire whenever
        the corresponding ``update_*`` flag is true.
        """
        resolved = self._resolve_url(url)
        if params:
            resolved = self._apply_params(resolved, params)
        merged = self._build_browser_headers(resolved, headers, browser_mode=browser_mode)
        request = PreparedRequest.prepare(
            method, resolved, headers=merged, body=body, json=json
        )
        result: HTTPResponse = self.send(  # type: ignore[assignment]
            request, config=config, raise_error=False, **send_kwargs
        )

        if update_cookies:
            self.cookies.update_from_response(result)
        if update_referrer and result.status_code < 400:
            self._referrer = request.url.to_string()

        if raise_error:
            result.raise_for_status()

        return result

    def follow_link(
        self,
        url: URL | str,
        *,
        from_url: Optional[str] = None,
        headers: Optional[Mapping[str, str]] = None,
        browser_mode: Optional[bool] = None,
        raise_error: bool = True,
        config: SendConfig | Mapping[str, Any] | None = None,
        **send_kwargs: Any,
    ) -> HTTPResponse:
        """Simulate clicking a hyperlink (sets ``Referer`` then navigates)."""
        if from_url is not None:
            self._referrer = from_url
        return self.navigate(
            url,
            method="GET",
            headers=headers,
            browser_mode=browser_mode,
            raise_error=raise_error,
            config=config,
            **send_kwargs,
        )

    def submit_form(
        self,
        url: URL | str,
        form_data: Mapping[str, Any],
        *,
        method: str = "POST",
        extra_headers: Optional[Mapping[str, str]] = None,
        browser_mode: Optional[bool] = None,
        raise_error: bool = True,
        config: SendConfig | Mapping[str, Any] | None = None,
        **send_kwargs: Any,
    ) -> HTTPResponse:
        """Simulate an HTML form submission (URL-encoded body)."""
        encoded = urlencode(
            {str(k): str(v) for k, v in form_data.items()}
        ).encode("utf-8")

        per_request: dict[str, str] = {
            "Content-Type": "application/x-www-form-urlencoded",
        }
        if extra_headers:
            per_request.update(extra_headers)

        resolved = self._resolve_url(url)
        merged = self._build_browser_headers(resolved, per_request, browser_mode=browser_mode)
        request = PreparedRequest.prepare(
            method, resolved, headers=merged, body=encoded
        )
        result: HTTPResponse = self.send(  # type: ignore[assignment]
            request, config=config, raise_error=False, **send_kwargs
        )

        self.cookies.update_from_response(result)

        if raise_error:
            result.raise_for_status()

        return result

    # ------------------------------------------------------------------
    # Transport
    # ------------------------------------------------------------------

    def _local_send(
        self,
        request: PreparedRequest,
        config: SendConfig,
    ) -> HTTPResponse:
        wait_cfg = self.waiting if config.wait is None else config.wait

        request = request.prepare_to_send(
            sent_at=None,
            headers=self._build_request_headers(request),
        )

        raw_resp = self.http_pool.request(
            method=request.method,
            url=request.url.to_string(),
            body=request.buffer,
            headers=request.headers,
            timeout=wait_cfg.timeout_urllib3,
            preload_content=False,
            decode_content=False,
            redirect=True,
        )

        result = HTTPResponse.from_urllib3(
            request=request,
            response=raw_resp,
            tags=None,
            received_at=dt.datetime.now(dt.timezone.utc),
        )
        result.drain_urllib3(raw_resp, stream=True, release_conn=True)

        x_current_page = raw_resp.headers.get("X-Current-Page")
        x_total_pages = raw_resp.headers.get("X-Last-Page")

        if x_current_page and x_total_pages:
            result = self._combine_paginated_pages(
                result=result,
                request=request,
                current_page=int(x_current_page),
                total_pages=int(x_total_pages),
                wait_cfg=wait_cfg,
                stream=config.stream,
                raise_error=config.raise_error,
            )

        if config.raise_error:
            result.raise_for_status()

        return result

    def _fetch_paginated_page(
        self,
        *,
        request: PreparedRequest,
        page_num: int,
        body_seed: bytes | None,
        wait_cfg: WaitingConfig,
        stream: bool,
        raise_error: bool,
    ) -> tuple[int, HTTPResponse]:
        page_url = request.url.add_query_item("page", str(page_num), replace=True)

        page_request = request.copy(
            url=page_url,
            buffer=BytesIO(body_seed) if body_seed is not None else None,
        )

        raw_resp = self.http_pool.request(
            method=page_request.method,
            url=page_url.to_string(),
            body=page_request.buffer,
            headers=page_request.headers,
            timeout=wait_cfg.timeout_urllib3,
            preload_content=not stream,
            decode_content=False,
            redirect=True,
        )

        page_result = HTTPResponse.from_urllib3(
            request=page_request,
            response=raw_resp,
            tags=None,
            received_at=dt.datetime.now(tz=dt.timezone.utc),
        )
        page_result.drain_urllib3(raw_resp, stream=stream, release_conn=True)

        if raise_error:
            page_result.raise_for_status()

        return page_num, page_result

    def _combine_paginated_pages(
        self,
        *,
        result: HTTPResponse,
        request: PreparedRequest,
        current_page: int,
        total_pages: int,
        wait_cfg: WaitingConfig,
        stream: bool,
        raise_error: bool,
        pool: Optional[JobPoolExecutor | int] = None,
    ) -> HTTPResponse:
        if not isinstance(pool, JobPoolExecutor):
            with JobPoolExecutor.parse(pool) as parsed_pool:
                return self._combine_paginated_pages(
                    result=result,
                    request=request,
                    current_page=current_page,
                    total_pages=total_pages,
                    wait_cfg=wait_cfg,
                    stream=stream,
                    raise_error=raise_error,
                    pool=parsed_pool,
                )

        from yggdrasil.lazy_imports import polars as pl

        init_df = result.to_polars(parse=True, lazy=False)
        if total_pages <= current_page:
            return result

        remaining_pages = list(range(current_page + 1, total_pages + 1))
        body_seed = request.buffer.to_bytes() if request.buffer else None

        def jobs():
            for pn in remaining_pages:
                yield Job.make(
                    self._fetch_paginated_page,
                    request=request,
                    page_num=pn,
                    body_seed=body_seed,
                    wait_cfg=wait_cfg,
                    stream=stream,
                    raise_error=raise_error,
                )

        frames = [init_df]
        for job_result in pool.as_completed(
            jobs(),
            ordered=False,
            max_in_flight=len(remaining_pages),
            cancel_on_exit=False,
            shutdown_on_exit=False,
            raise_error=True,
        ):
            _, page_resp = job_result.result
            frames.append(page_resp.to_polars(parse=True, lazy=False))

        final_df = pl.concat(frames, how="diagonal_relaxed", rechunk=True)

        new_buffer = ArrowIPCIO(media_type=MediaTypes.ARROW_IPC)
        new_buffer.write_arrow_table(
            final_df.to_arrow(compat_level=pl.CompatLevel.newest()),
            compression="zstd",
        )
        new_buffer.seek(0)

        result.buffer.close()
        result.buffer = new_buffer
        result.set_media_type(MediaTypes.ARROW_IPC)

        result.update_tags({
            "page_start": str(current_page),
            "page_total": str(total_pages),
        })

        return result
