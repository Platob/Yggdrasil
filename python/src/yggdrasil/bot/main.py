from __future__ import annotations

import uvicorn

from yggdrasil.bot.config import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "yggdrasil.bot.app:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
