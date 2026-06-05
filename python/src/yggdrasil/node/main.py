from __future__ import annotations

import uvicorn

from yggdrasil.node.config import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "yggdrasil.node.app:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
