"""``ygg-api`` entry point — uvicorn launcher for :mod:`yggdrasil.fastapi.app`."""

from __future__ import annotations

import uvicorn

from .config import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "yggdrasil.fastapi.app:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
