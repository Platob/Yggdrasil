from __future__ import annotations

from typing import Optional, TYPE_CHECKING, Any, Callable

from yggdrasil.io.url import URL

if TYPE_CHECKING:
    from pymongo import MongoClient

try:
    import mongoengine
except ImportError:
    from yggdrasil.environ import runtime_import_module

    mongoengine = runtime_import_module(
        module_name="mongoengine", pip_name="mongoengine", install=True
    )

from mongoengine import *  # type: ignore

__all__ = [
    "mongoengine",
    "get_connection_settings",
] + mongoengine.__all__


ALIASES_DB: dict[str, str] = {}

_base_connect = mongoengine.connection.connect
_base_register_connection = mongoengine.connection.register_connection


def _first_host(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def get_connection_settings(
    alias: str = mongoengine.DEFAULT_CONNECTION_NAME,
    db: Optional[str] = None,
    jsonable: bool = False,
    resolver: Optional[Callable[[str], str]] = None,
) -> dict:
    try:
        infos = mongoengine.connection._connection_settings[alias]
    except KeyError as e:
        if resolver is not None:
            if isinstance(resolver, str):
                register_connection(
                    db=db,
                    alias=alias,
                    host=resolver,
                )
            elif callable(resolver):
                resolved = resolver(alias)

                if isinstance(resolved, str):
                    register_connection(
                        db=db,
                        alias=alias,
                        host=resolved,
                    )
                elif isinstance(resolved, dict):
                    db = resolved.pop("db", db)
                    alias = resolved.pop("alias", alias)
                    register_connection(
                        db=db,
                        alias=alias,
                        **resolved,
                    )

            return get_connection_settings(alias=alias, db=db, jsonable=jsonable, resolver=None)

        raise KeyError(
            f"No MongoEngine connection settings found for alias {alias!r}"
        ) from e

    if not jsonable:
        return dict(infos)

    base = {
        "db": db or infos.get("db") or ALIASES_DB.get(alias) or alias,
        "alias": infos.get("alias") or alias,
        "host": _first_host(infos.get("host")),
        "port": infos.get("port"),
        "username": infos.get("username"),
        "password": infos.get("password"),
        "serverSelectionTimeoutMS": infos.get("serverSelectionTimeoutMS"),
        "uuidRepresentation": infos.get("uuidRepresentation"),
    }

    return {
        k: v
        for k, v in base.items()
        if v
    }


def connect(
    db: str = None,
    alias: str = mongoengine.DEFAULT_CONNECTION_NAME,
    host: str | list[str] | None = None,
    resolver: Optional[Callable[[str], str]] = None,
    **kwargs,
) -> "MongoClient":
    existing: Optional["MongoClient"] = mongoengine.connection._connections.get(alias)
    existing_settings = mongoengine.connection._connection_settings.get(alias)

    if existing is not None:
        if getattr(existing, "_closed", False):
            mongoengine.connection._connections.pop(alias, None)
            mongoengine.connection._connection_settings.pop(alias, None)
            existing = None
            existing_settings = None
        else:
            return existing

    new_host = _first_host(host)
    if alias and db:
        ALIASES_DB[alias] = db

    if existing is not None and existing_settings is not None:
        registered_host = _first_host(existing_settings.get("host"))
        registered_db = existing_settings.get("db")

        should_refresh = False

        if new_host and registered_host:
            registered_url = URL.parse_str(registered_host, default_scheme="mongodb")
            new_url = URL.parse_str(new_host, default_scheme="mongodb")
            if str(registered_url) != str(new_url):
                should_refresh = True
        elif new_host != registered_host:
            should_refresh = True

        if db is not None and db != registered_db:
            should_refresh = True

        if should_refresh:
            register_connection(alias=alias, db=db, host=host, **kwargs)

    built: "MongoClient" = _base_connect(
        db=db,
        alias=alias,
        host=host,
        **kwargs,
    )

    return built


def register_connection(
    alias: str,
    db: str = None,
    host: str | list[str] | None = None,
    server_selection_timeout_ms: int = 5000,
    **kwargs,
):
    if server_selection_timeout_ms:
        kwargs.setdefault("serverSelectionTimeoutMS", server_selection_timeout_ms)

    if alias and db:
        ALIASES_DB[alias] = db

    _base_register_connection(
        db=db,
        alias=alias,
        host=host,
        **kwargs,
    )


object.__setattr__(mongoengine.connection, "connect", connect)
object.__setattr__(mongoengine.connection, "register_connection", register_connection)