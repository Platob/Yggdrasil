from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Any, Callable, MutableMapping

from yggdrasil.io import URL
from yggdrasil.io.headers import DEFAULT_HOSTNAME

if TYPE_CHECKING:
    from pymongo import MongoClient
    from pymongo.database import Database

try:
    import mongoengine
except ImportError:
    from yggdrasil.environ import runtime_import_module

    mongoengine = runtime_import_module(
        module_name="mongoengine",
        pip_name="mongoengine",
        install=True,
    )

from mongoengine import *  # type: ignore  # noqa: F401,F403

__all__ = [
    "mongoengine",
    "CONTEXT_KEY_ENVIRON",
    "get_connection_settings",
    "connect",
    "register_connection",
    "get_connection",
    "get_db",
] + mongoengine.__all__

LOGGER = logging.getLogger(__name__)

#: Name of the environment variable used to resolve the current connection context.
CONTEXT_KEY_ENVIRON: str = "MONGOENGINE_CONTEXT_KEY"

#: Prefix used when serializing connection settings into environment variables.
_CONNECTION_ENV_PREFIX = "MONGOENGINE_CONNECTION_"

_base_connect = mongoengine.connection.connect
_base_register_connection = mongoengine.connection.register_connection
_base_get_connection = mongoengine.connection.get_connection
_base_get_db = mongoengine.connection.get_db


def _get_environ(environ: MutableMapping[str, str] | None = None) -> MutableMapping[str, str]:
    """Return the environment mapping to use."""
    return environ if environ is not None else os.environ


def _first_host(value: Any) -> str | None:
    """Return the first host from a scalar or sequence of hosts."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def _normalize_context_key(
    context_key: str | None,
    environ: MutableMapping[str, str] | None = None,
    *,
    default: str | None = DEFAULT_HOSTNAME,
) -> str | None:
    """
    Resolve the effective context key.

    Resolution order:
    1. Explicit ``context_key``
    2. ``MONGOENGINE_CONTEXT_KEY`` from the provided environment
    3. ``default``
    """
    env = _get_environ(environ)
    return context_key or env.get(CONTEXT_KEY_ENVIRON) or default


def _contextual_alias(alias: str, context_key: str | None) -> str:
    """Build a context-prefixed alias."""
    if not context_key:
        return alias
    prefix = context_key.rstrip("_")
    return alias if alias.startswith(prefix + "_") else f"{prefix}_{alias}"


def _iter_alias_candidates(
    alias: str,
    context_key: str | None,
) -> tuple[str, ...]:
    """
    Return alias candidates in lookup order.

    If a context key is present and the alias is not already contextualized,
    the contextual alias is tried first, followed by the original alias.
    """
    contextual = _contextual_alias(alias, context_key)
    if contextual != alias:
        return (contextual, alias)
    return (alias,)


def _settings_env_key(alias: str) -> str:
    """Return the environment variable name for a serialized connection config."""
    return f"{_CONNECTION_ENV_PREFIX}{alias.upper()}"


def _resolve_host(
    alias: str,
    resolver: Callable[[], str] | str | None = None,
    environ: MutableMapping[str, str] | None = None,
) -> str | None:
    """
    Resolve a MongoDB host/URI.

    Rules:
    - If ``resolver`` is a string, it is returned as-is.
    - If ``resolver`` is callable, it is called and must return a string.
    - If ``resolver`` is omitted and alias does not look like a URI, the alias
      name is used as an environment variable key.
    """
    env = _get_environ(environ)

    if resolver is None and "://" not in alias:
        resolver = env.get(alias)

    if resolver is None:
        return None

    if isinstance(resolver, str):
        return resolver

    if callable(resolver):
        resolved = resolver()
        return resolved if isinstance(resolved, str) else None

    return None


def _same_host(left: str | None, right: str | None) -> bool:
    """Return whether two MongoDB hosts/URIs normalize to the same value."""
    if left == right:
        return True
    if not left or not right:
        return False

    left_url = URL.parse_str(left, default_scheme="mongodb")
    right_url = URL.parse_str(right, default_scheme="mongodb")
    return str(left_url) == str(right_url)


def _clear_connection_caches(alias: str, *, clear_settings: bool = False) -> None:
    """Remove cached MongoEngine objects for an alias."""
    mongoengine.connection._dbs.pop(alias, None)
    mongoengine.connection._connections.pop(alias, None)
    if clear_settings:
        mongoengine.connection._connection_settings.pop(alias, None)


def _load_settings_from_environ(
    alias: str,
    environ: MutableMapping[str, str] | None = None,
) -> dict[str, Any] | None:
    """Load serialized connection settings for an alias from the environment."""
    env = _get_environ(environ)
    raw = env.get(_settings_env_key(alias))
    if not raw:
        return None

    try:
        loaded = json.loads(raw)
    except Exception:
        LOGGER.warning("Invalid MongoEngine environment settings for alias %r", alias, exc_info=True)
        return None

    return loaded if isinstance(loaded, dict) else None


def get_connection_settings(
    alias: str = "default",
    db: str | None = None,
    jsonable: bool = False,
    context_key: str | None = None,
    environ: MutableMapping[str, str] | None = None,
) -> dict[str, Any]:
    """
    Return registered MongoEngine connection settings for an alias.

    When a context key is available, the contextual alias is preferred first.
    If ``jsonable`` is true, the result is reduced to a JSON-safe subset that
    can be serialized into the environment and restored later.
    """
    effective_context = _normalize_context_key(context_key, environ, default=None)

    for candidate in _iter_alias_candidates(alias, effective_context):
        info = mongoengine.connection._connection_settings.get(candidate)
        if info is None:
            continue

        if not jsonable:
            return dict(info)

        base = {
            "db": db or info.get("db") or info.get("name") or candidate,
            "alias": info.get("alias") or candidate,
            "host": _first_host(info.get("host")),
            "port": info.get("port"),
            "username": info.get("username"),
            "password": info.get("password"),
            "serverSelectionTimeoutMS": info.get("serverSelectionTimeoutMS"),
            "uuidRepresentation": info.get("uuidRepresentation"),
            "context_key": effective_context,
        }
        return {k: v for k, v in base.items() if v is not None}

    raise KeyError(f"No MongoEngine connection settings found for alias {alias!r}")


def connect(
    db: str | None = None,
    alias: str = "default",
    host: str | list[str] | None = None,
    context_key: str | None = None,
    environ: MutableMapping[str, str] | None = None,
    **kwargs: Any,
) -> MongoClient:
    """
    Return a MongoClient for the given alias.

    Behavior:
    - Prefer a context-prefixed alias when available.
    - Reuse an existing open client if host and database match.
    - Clear stale cached client/database objects when a refresh is required.
    """
    effective_context = _normalize_context_key(context_key, environ, default=None)

    for candidate in _iter_alias_candidates(alias, effective_context):
        existing = mongoengine.connection._connections.get(candidate)
        settings = mongoengine.connection._connection_settings.get(candidate)
        new_host = _first_host(host)

        if existing is not None:
            if getattr(existing, "_closed", False):
                _clear_connection_caches(candidate, clear_settings=True)
                existing = None
                settings = None
            else:
                registered_host = _first_host(settings.get("host")) if settings else None
                registered_db = (settings.get("name") or settings.get("db")) if settings else None

                should_refresh = False
                if settings is not None:
                    if not _same_host(new_host, registered_host):
                        should_refresh = new_host is not None or registered_host is not None
                    if db is not None and db != registered_db:
                        should_refresh = True

                if not should_refresh:
                    return existing

                _clear_connection_caches(candidate, clear_settings=False)

        return _base_connect(
            db=db,
            alias=candidate,
            host=host,
            **kwargs,
        )

    # Realistically unreachable, but keeps type-checkers chill.
    return _base_connect(db=db, alias=alias, host=host, **kwargs)


def register_connection(
    alias: str = "default",
    db: str | None = None,
    host: str | list[str] | None = None,
    server_selection_timeout_ms: int = 5000,
    context_key: str | None = None,
    raise_error: bool = False,
    resolver: Callable[[], str] | str | None = None,
    environ: MutableMapping[str, str] | None = None,
    **kwargs: Any,
) -> None:
    """
    Register a MongoEngine connection and persist a JSON-safe copy of its settings.

    The serialized configuration is written to an environment variable so another
    process or later lookup can restore the registration lazily.
    """
    env = _get_environ(environ)
    effective_context = _normalize_context_key(context_key, env, default=DEFAULT_HOSTNAME)
    target_alias = _contextual_alias(alias, effective_context)

    resolved_host = host if host is not None else _resolve_host(
        alias=target_alias,
        resolver=resolver,
        environ=env,
    )

    if server_selection_timeout_ms:
        kwargs.setdefault("serverSelectionTimeoutMS", server_selection_timeout_ms)

    try:
        _base_register_connection(
            db=db,
            alias=target_alias,
            host=resolved_host,
            **kwargs,
        )

        registered = get_connection_settings(
            alias=target_alias,
            db=db,
            jsonable=True,
            context_key=None,
            environ=env,
        )
        env[_settings_env_key(target_alias)] = json.dumps(registered)
    except Exception:
        LOGGER.exception("Failed to register MongoEngine connection for alias %r", target_alias)
        if raise_error:
            raise


def get_connection(
    alias: str = "default",
    reconnect: bool = False,
    context_key: str | None = None,
    raise_error: bool = True,
    environ: MutableMapping[str, str] | None = None,
) -> MongoClient | None:
    """
    Return a MongoClient for an alias.

    Lookup order:
    1. already-registered contextual alias
    2. already-registered plain alias
    3. env-restored contextual alias
    4. env-restored plain alias
    """
    env = _get_environ(environ)
    effective_context = _normalize_context_key(context_key, env, default=DEFAULT_HOSTNAME)

    last_exception: Exception | None = None

    for candidate in _iter_alias_candidates(alias or "default", effective_context):
        try:
            return _base_get_connection(alias=candidate, reconnect=reconnect)
        except Exception as exc:
            last_exception = exc

        settings = _load_settings_from_environ(candidate, env)
        if not settings:
            continue

        register_connection(
            alias=candidate,
            db=settings.get("db") or settings.get("name") or candidate,
            host=settings.get("host"),
            server_selection_timeout_ms=settings.get("serverSelectionTimeoutMS", 5000),
            context_key=None,
            raise_error=False,
            environ=env,
        )

        try:
            return _base_get_connection(alias=candidate, reconnect=reconnect)
        except Exception as exc:
            last_exception = exc

    if raise_error and last_exception is not None:
        raise last_exception
    return None


def get_db(
    alias: str = "default",
    reconnect: bool = False,
    context_key: str | None = None,
    raise_error: bool = True,
    environ: MutableMapping[str, str] | None = None,
) -> Database | None:
    """
    Return a PyMongo Database for an alias.

    This wrapper mirrors the custom connection resolution logic and refreshes
    MongoEngine's cached database object when reconnecting.
    """
    env = _get_environ(environ)
    effective_context = _normalize_context_key(context_key, env, default=DEFAULT_HOSTNAME)

    last_exception: Exception | None = None

    for candidate in _iter_alias_candidates(alias or "default", effective_context):
        try:
            if reconnect:
                _clear_connection_caches(candidate, clear_settings=False)

            cached_db = mongoengine.connection._dbs.get(candidate)
            if cached_db is not None:
                return cached_db

            conn = get_connection(
                alias=candidate,
                reconnect=reconnect,
                context_key=None,
                raise_error=raise_error,
                environ=env,
            )
            if conn is None:
                continue

            settings = mongoengine.connection._connection_settings.get(candidate)
            if not settings:
                raise KeyError(f"No MongoEngine connection settings found for alias {candidate!r}")

            db_name = settings.get("name") or settings.get("db") or candidate
            db = conn[db_name]
            mongoengine.connection._dbs[candidate] = db
            return db

        except Exception as exc:
            last_exception = exc

    if raise_error and last_exception is not None:
        raise last_exception
    return None


object.__setattr__(mongoengine.connection, "connect", connect)
object.__setattr__(mongoengine.connection, "register_connection", register_connection)
object.__setattr__(mongoengine.connection, "get_connection", get_connection)
object.__setattr__(mongoengine.connection, "get_db", get_db)
