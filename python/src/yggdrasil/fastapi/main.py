from __future__ import annotations

import uvicorn

from yggdrasil.fastapi.config import get_settings


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
