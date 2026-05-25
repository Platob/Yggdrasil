from __future__ import annotations

import socket

import uvicorn

from yggdrasil.fastapi.config import get_settings


def _find_open_port(host: str, preferred: int) -> int:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, preferred))
            return preferred
    except OSError:
        pass
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def main() -> None:
    settings = get_settings()
    port = _find_open_port(settings.host, settings.port)
    uvicorn.run(
        "yggdrasil.fastapi.app:app",
        host=settings.host,
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    main()
