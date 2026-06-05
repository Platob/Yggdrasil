from __future__ import annotations

from random import Random
from dataclasses import dataclass


__all__ = [
    "UserAgentGenerator",
    "BrowserProfile",
    "random_user_agent",
    "random_browser_profile",
    "random_browser_headers",
]


@dataclass(frozen=True)
class _Pick:
    """A chosen User-Agent plus the metadata needed to emit *consistent*
    headers (client hints, Accept) for it."""

    user_agent: str
    family: str        # chrome | edge | firefox | safari
    platform: str      # sec-ch-ua-platform value: Windows / macOS / Linux / Android / iOS
    mobile: bool
    chrome_major: int | None  # Chromium families only; drives sec-ch-ua versions


# Literal ``Accept`` strings the real engines send for a top-level navigation,
# so a server's content negotiation (and header-shape heuristics) see a
# believable request rather than a bare client.
_ACCEPT_CHROME = (
    "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,"
    "image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7"
)
_ACCEPT_FIREFOX = (
    "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,"
    "image/webp,*/*;q=0.8"
)
_ACCEPT_SAFARI = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"

_ACCEPT_LANGUAGES = [
    "en-US,en;q=0.9",
    "en-GB,en;q=0.9",
    "en-US,en;q=0.8",
    "en;q=0.9",
    "fr-FR,fr;q=0.9,en;q=0.8",
    "de-DE,de;q=0.9,en;q=0.8",
    "es-ES,es;q=0.9,en;q=0.8",
]

# "Who is making this request" headers (identity / fingerprint), as opposed to
# content negotiation. Rotating only these on a retry presents a fresh,
# internally-consistent client without changing the response shape the caller
# asked for (``Accept`` / ``Accept-Encoding`` stay as set).
_IDENTITY_HEADERS = frozenset({
    "User-Agent",
    "sec-ch-ua",
    "sec-ch-ua-mobile",
    "sec-ch-ua-platform",
    "Sec-Fetch-Dest",
    "Sec-Fetch-Mode",
    "Sec-Fetch-Site",
    "Sec-Fetch-User",
    "Accept-Language",
    "Upgrade-Insecure-Requests",
})


@dataclass(frozen=True)
class BrowserProfile:
    """A coherent browser request fingerprint: a User-Agent and the matching
    headers a real browser sends with it.

    Use :attr:`headers` for a full browser-shaped request, or :attr:`identity`
    to rotate only the who-am-I headers (leaving content negotiation alone).
    """

    user_agent: str
    headers: dict[str, str]

    @property
    def identity(self) -> dict[str, str]:
        return {k: v for k, v in self.headers.items() if k in _IDENTITY_HEADERS}


@dataclass(frozen=True)
class UserAgentGenerator:
    """
    Random (but plausible) User-Agent generator.

    - Pure stdlib
    - Deterministic if seed is provided
    - Biases toward realistic modern UA shapes
    - Avoids obviously fake combos (mostly)
    - :meth:`random_profile` emits a full set of headers consistent with the
      chosen UA (client hints, Accept, Sec-Fetch, language)
    """

    seed: int | None = None

    def _rng(self) -> Random:
        return Random(self.seed)

    def random(self) -> str:
        return self._pick_from(self._rng()).user_agent

    def random_profile(self) -> "BrowserProfile":
        """A :class:`BrowserProfile`: the UA plus headers consistent with it."""
        r = self._rng()
        pick = self._pick_from(r)
        return BrowserProfile(user_agent=pick.user_agent, headers=_profile_headers(r, pick))

    def _pick_from(self, r: Random) -> _Pick:
        # Weighted platforms (desktop-heavy by default)
        platform = r.choices(
            population=["windows", "mac", "linux", "android", "ios"],
            weights=[45, 25, 15, 10, 5],
            k=1,
        )[0]
        if platform == "windows":
            return self._desktop_windows(r)
        if platform == "mac":
            return self._desktop_mac(r)
        if platform == "linux":
            return self._desktop_linux(r)
        if platform == "android":
            return self._android(r)
        return self._ios(r)

    # -------- desktop --------

    def _desktop_windows(self, r: Random) -> _Pick:
        # Browser weights on Windows
        browser = r.choices(["chrome", "edge", "firefox"], weights=[60, 30, 10], k=1)[0]
        if browser in ("chrome", "edge"):
            chrome_major = r.randint(118, 124)
            chrome_build = f"{chrome_major}.0.{r.randint(1000, 6500)}.{r.randint(10, 250)}"
            ua = (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                f"Chrome/{chrome_build} "
            )
            if browser == "edge":
                edge_major = chrome_major  # usually tracks Chromium major closely
                edge_build = f"{edge_major}.0.{r.randint(1000, 2500)}.{r.randint(10, 200)}"
                ua += f"Edg/{edge_build} "
            ua += "Safari/537.36"
            return _Pick(ua, browser, "Windows", False, chrome_major)

        # Firefox
        ff_major = r.randint(115, 123)
        ua = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:"
            f"{ff_major}.0) Gecko/20100101 Firefox/{ff_major}.0"
        )
        return _Pick(ua, "firefox", "Windows", False, None)

    def _desktop_mac(self, r: Random) -> _Pick:
        browser = r.choices(["chrome", "safari", "firefox"], weights=[55, 35, 10], k=1)[0]
        mac_major = r.choice([12, 13, 14])  # Monterey/Ventura/Sonoma-ish
        mac_minor = r.randint(0, 6)

        if browser == "safari":
            # Safari UA is quirky; keep it plausible.
            safari_major = r.randint(16, 17)
            safari_minor = r.randint(0, 6)
            webkit = f"{r.randint(605, 617)}.1.{r.randint(1, 30)}"
            ua = (
                f"Mozilla/5.0 (Macintosh; Intel Mac OS X {mac_major}_{mac_minor}_0) "
                f"AppleWebKit/{webkit} (HTML, like Gecko) "
                f"Version/{safari_major}.{safari_minor} Safari/{webkit}"
            )
            return _Pick(ua, "safari", "macOS", False, None)

        if browser == "chrome":
            chrome_major = r.randint(118, 124)
            chrome_build = f"{chrome_major}.0.{r.randint(1000, 6500)}.{r.randint(10, 250)}"
            ua = (
                f"Mozilla/5.0 (Macintosh; Intel Mac OS X {mac_major}_{mac_minor}_0) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                f"Chrome/{chrome_build} Safari/537.36"
            )
            return _Pick(ua, "chrome", "macOS", False, chrome_major)

        # Firefox
        ff_major = r.randint(115, 123)
        ua = (
            f"Mozilla/5.0 (Macintosh; Intel Mac OS X {mac_major}_{mac_minor}_0; rv:"
            f"{ff_major}.0) Gecko/20100101 Firefox/{ff_major}.0"
        )
        return _Pick(ua, "firefox", "macOS", False, None)

    def _desktop_linux(self, r: Random) -> _Pick:
        browser = r.choices(["chrome", "firefox"], weights=[70, 30], k=1)[0]
        if browser == "chrome":
            chrome_major = r.randint(118, 124)
            chrome_build = f"{chrome_major}.0.{r.randint(1000, 6500)}.{r.randint(10, 250)}"
            ua = (
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                f"Chrome/{chrome_build} Safari/537.36"
            )
            return _Pick(ua, "chrome", "Linux", False, chrome_major)

        ff_major = r.randint(115, 123)
        ua = (
            "Mozilla/5.0 (X11; Linux x86_64; rv:"
            f"{ff_major}.0) Gecko/20100101 Firefox/{ff_major}.0"
        )
        return _Pick(ua, "firefox", "Linux", False, None)

    # -------- mobile --------

    def _android(self, r: Random) -> _Pick:
        # Mobile tends to be Chrome-ish
        android_major = r.randint(10, 14)
        chrome_major = r.randint(118, 124)
        chrome_build = f"{chrome_major}.0.{r.randint(1000, 6500)}.{r.randint(10, 250)}"

        # Simple plausible device tokens
        devices = [
            "Pixel 7", "Pixel 7 Pro", "Pixel 8", "Pixel 8 Pro",
            "SM-G991B", "SM-S911B", "SM-S921B", "SM-A546B",
            "OnePlus 10 Pro", "OnePlus 11", "Mi 11", "Mi 12"
        ]
        device = r.choice(devices)

        ua = (
            f"Mozilla/5.0 (Linux; Android {android_major}; {device}) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            f"Chrome/{chrome_build} Mobile Safari/537.36"
        )
        return _Pick(ua, "chrome", "Android", True, chrome_major)

    def _ios(self, r: Random) -> _Pick:
        # iOS is basically Safari UA shape
        ios_major = r.randint(15, 17)
        ios_minor = r.randint(0, 6)
        webkit = f"{r.randint(605, 617)}.1.{r.randint(1, 30)}"

        # iPhone vs iPad
        device = r.choices(["iPhone", "iPad"], weights=[85, 15], k=1)[0]

        # "Version/x.y Mobile/15E148 Safari/604.1" pattern
        version_major = ios_major
        version_minor = ios_minor
        mobile_token = "15E148"  # common-ish placeholder token in many UAs

        ua = (
            f"Mozilla/5.0 ({device}; CPU {device} OS {ios_major}_{ios_minor} like Mac OS X) "
            f"AppleWebKit/{webkit} (KHTML, like Gecko) "
            f"Version/{version_major}.{version_minor} Mobile/{mobile_token} Safari/{webkit}"
        )
        return _Pick(ua, "safari", "iOS", True, None)


def _profile_headers(r: Random, pick: _Pick) -> dict[str, str]:
    """Build the header set a real browser sends with *pick*'s User-Agent."""
    headers = {
        "User-Agent": pick.user_agent,
        "Accept-Language": r.choice(_ACCEPT_LANGUAGES),
        "Accept-Encoding": "gzip, deflate, br",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Upgrade-Insecure-Requests": "1",
    }
    if pick.family in ("chrome", "edge"):
        headers["Accept"] = _ACCEPT_CHROME
        major = pick.chrome_major or 124
        brand = "Microsoft Edge" if pick.family == "edge" else "Google Chrome"
        # Chromium's GREASE-style brand list: a deliberately varied "Not A Brand"
        # entry plus Chromium and the concrete brand, all at the same major.
        headers["sec-ch-ua"] = (
            f'"Not_A Brand";v="8", "Chromium";v="{major}", "{brand}";v="{major}"'
        )
        headers["sec-ch-ua-mobile"] = "?1" if pick.mobile else "?0"
        headers["sec-ch-ua-platform"] = f'"{pick.platform}"'
    elif pick.family == "firefox":
        headers["Accept"] = _ACCEPT_FIREFOX  # Firefox sends no client hints
    else:  # safari
        headers["Accept"] = _ACCEPT_SAFARI   # Safari sends no client hints
    return headers


# ---- convenience ----

def random_user_agent(seed: int | None = None) -> str:
    """
    Convenience function:
      random_user_agent() -> random UA
      random_user_agent(seed=123) -> deterministic UA
    """
    return UserAgentGenerator(seed=seed).random()


def random_browser_profile(seed: int | None = None) -> "BrowserProfile":
    """A random (or seeded) :class:`BrowserProfile` — UA plus consistent headers."""
    return UserAgentGenerator(seed=seed).random_profile()


def random_browser_headers(seed: int | None = None) -> dict[str, str]:
    """The header dict of a random (or seeded) browser profile."""
    return UserAgentGenerator(seed=seed).random_profile().headers
