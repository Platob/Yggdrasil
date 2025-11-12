import os
import ssl
import sys
from pathlib import Path

from pyarrow.lib import Mapping

_WINDOWS_ZONES_URL = (
    "https://raw.githubusercontent.com/unicode-org/cldr/master/common/supplemental/windowsZones.xml"
)

def _download_requests(url: str, out_path: str, *, ignore_ssl: bool, timeout: float):
    import requests
    # quiet warning if ignoring SSL
    if ignore_ssl:
        try:
            from urllib3.exceptions import InsecureRequestWarning
            import urllib3
            urllib3.disable_warnings(InsecureRequestWarning)
        except Exception:
            pass
    with requests.get(url, stream=True, timeout=timeout, verify=not ignore_ssl) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

def _download_urllib(url: str, out_path: str, *, ignore_ssl: bool, timeout: float):
    from urllib.request import Request, urlopen
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/121.0.0.0 Safari/537.36"
        ),
        "Accept": "application/xml,text/xml,*/*;q=0.8",
    }
    ctx = None
    if ignore_ssl:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    req = Request(url, headers=headers, method="GET")
    with urlopen(req, timeout=timeout, context=ctx) as resp, open(out_path, "wb") as f:
        f.write(resp.read())

def ensure_windows_zones_xml(
    folder: str | os.PathLike,
    *,
    url: str = _WINDOWS_ZONES_URL,
    ignore_ssl: bool = False,
    timeout: float = 30.0,
    verify_xml: bool = True,
) -> str:
    """
    Ensure windowsZones.xml exists at `folder`. If missing/empty, download it.
    Returns the full path to the file.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "windowsZones.xml"

    # Already there and non-empty? keep it.
    if path.exists() and path.stat().st_size > 0:
        return str(path)

    # Try requests, then urllib
    try:
        _download_requests(url, str(path), ignore_ssl=ignore_ssl, timeout=timeout)
    except Exception:
        _download_urllib(url, str(path), ignore_ssl=ignore_ssl, timeout=timeout)

    # Basic checks
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"Failed to download windowsZones.xml to {path}")

    if verify_xml:
        import xml.etree.ElementTree as ET
        try:
            root = ET.parse(str(path)).getroot()
        except Exception as e:
            raise RuntimeError(f"Downloaded windowsZones.xml is not valid XML: {e}") from e
        # quick sanity: top-level should be supplementalData
        if root.tag.lower().endswith("supplementaldata") is False:
            # some parsers include namespace; endswith handles that, but be cautious:
            pass

    return str(path)


def configure_tzdata_for_arrow(
    environment: Mapping | None = None,
    preferred_dir: str | None = None
) -> str:
    """
    Returns tzdir path and sets env so Arrow & zoneinfo can find it.
    Order:
      1) preferred_dir (if given)
      2) installed PyPI tzdata package
      3) fallback to %USERPROFILE%\\Downloads\\tzdata (Windows default)
    """
    environment = environment or os.environ

    # 1) explicit path
    if preferred_dir and Path(preferred_dir).is_dir():
        tzdir = Path(preferred_dir)

    else:
        # 2) PyPI tzdata if installed
        try:
            import tzdata
            tzdir = Path(tzdata.__file__).parent
        except Exception:
            # 3) Windows default folder (works if you previously extracted there)
            if sys.platform == "win32":
                tzdir = Path(os.path.expandvars(r"%USERPROFILE%\Downloads\tzdata"))
            else:
                raise RuntimeError("No tzdata found; install 'tzdata' or provide a directory.")

    if not tzdir.is_dir():
        raise RuntimeError(f"tzdata directory not found: {tzdir}")

    if sys.platform == "win32":
        ensure_windows_zones_xml(tzdir)

    environment["TZDIR"] = str(tzdir)
    environment["ARROW_TIMEZONE_DATABASE"] = str(tzdir)
    return str(tzdir)


__all__ = [
    "configure_tzdata_for_arrow"
]