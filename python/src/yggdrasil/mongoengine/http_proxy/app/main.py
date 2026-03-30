from __future__ import annotations

import os

from yggdrasil.mongoengine.http_proxy import main as proxy_main

__all__ = ["main"]


def main() -> int:
    """Databricks Apps entrypoint for the Mongo HTTP proxy."""
    # Databricks Apps commonly provide PORT and HOST env variables.
    port = os.getenv("PORT")
    host = os.getenv("HOST", "0.0.0.0")
    args: list[str] = []
    if port:
        args.extend(["--listen", f"{host}:{port}"])
    return proxy_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
