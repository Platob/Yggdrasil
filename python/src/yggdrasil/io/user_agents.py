from __future__ import annotations

from random import Random
from dataclasses import dataclass
from typing import Optional


__all__ = [
    "UserAgentGenerator",
    "random_user_agent"
]


@dataclass(frozen=True)
class UserAgentGenerator:
    """
    Random (but plausible) User-Agent generator.

    - Pure stdlib
    - Deterministic if seed is provided
    - Biases toward realistic modern UA shapes
    - Avoids obviously fake combos (mostly)
    """

    seed: Optional[int] = None

    def _rng(self) -> Random:
        return Random(self.seed)

    def random(self) -> str:
        r = self._rng()
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

    def _desktop_windows(self, r: Random) -> str:
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
            return ua

        # Firefox
        ff_major = r.randint(115, 123)
        return (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:"
            f"{ff_major}.0) Gecko/20100101 Firefox/{ff_major}.0"
        )

    def _desktop_mac(self, r: Random) -> str:
        browser = r.choices(["chrome", "safari", "firefox"], weights=[55, 35, 10], k=1)[0]
        mac_major = r.choice([12, 13, 14])  # Monterey/Ventura/Sonoma-ish
        mac_minor = r.randint(0, 6)

        if browser == "safari":
            # Safari UA is quirky; keep it plausible.
            safari_major = r.randint(16, 17)
            safari_minor = r.randint(0, 6)
            webkit = f"{r.randint(605, 617)}.1.{r.randint(1, 30)}"
            return (
                f"Mozilla/5.0 (Macintosh; Intel Mac OS X {mac_major}_{mac_minor}_0) "
                f"AppleWebKit/{webkit} (HTML, like Gecko) "
                f"Version/{safari_major}.{safari_minor} Safari/{webkit}"
            )

        if browser == "chrome":
            chrome_major = r.randint(118, 124)
            chrome_build = f"{chrome_major}.0.{r.randint(1000, 6500)}.{r.randint(10, 250)}"
            return (
                f"Mozilla/5.0 (Macintosh; Intel Mac OS X {mac_major}_{mac_minor}_0) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                f"Chrome/{chrome_build} Safari/537.36"
            )

        # Firefox
        ff_major = r.randint(115, 123)
        return (
            f"Mozilla/5.0 (Macintosh; Intel Mac OS X {mac_major}_{mac_minor}_0; rv:"
            f"{ff_major}.0) Gecko/20100101 Firefox/{ff_major}.0"
        )

    def _desktop_linux(self, r: Random) -> str:
        browser = r.choices(["chrome", "firefox"], weights=[70, 30], k=1)[0]
        if browser == "chrome":
            chrome_major = r.randint(118, 124)
            chrome_build = f"{chrome_major}.0.{r.randint(1000, 6500)}.{r.randint(10, 250)}"
            return (
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                f"Chrome/{chrome_build} Safari/537.36"
            )

        ff_major = r.randint(115, 123)
        return (
            "Mozilla/5.0 (X11; Linux x86_64; rv:"
            f"{ff_major}.0) Gecko/20100101 Firefox/{ff_major}.0"
        )

    # -------- mobile --------

    def _android(self, r: Random) -> str:
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

        return (
            f"Mozilla/5.0 (Linux; Android {android_major}; {device}) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            f"Chrome/{chrome_build} Mobile Safari/537.36"
        )

    def _ios(self, r: Random) -> str:
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

        return (
            f"Mozilla/5.0 ({device}; CPU {device} OS {ios_major}_{ios_minor} like Mac OS X) "
            f"AppleWebKit/{webkit} (KHTML, like Gecko) "
            f"Version/{version_major}.{version_minor} Mobile/{mobile_token} Safari/{webkit}"
        )


# ---- convenience ----

def random_user_agent(seed: Optional[int] = None) -> str:
    """
    Convenience function:
      random_user_agent() -> random UA
      random_user_agent(seed=123) -> deterministic UA
    """
    return UserAgentGenerator(seed=seed).random()
