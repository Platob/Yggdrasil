from __future__ import annotations

import functools
import os
from typing import Any, Callable, Mapping, Optional, Sequence, TypeVar, Union, TYPE_CHECKING

from yggdrasil.io.headers import DEFAULT_HOSTNAME
from yggdrasil.io.url import URL
from yggdrasil.mongoengine import register_connection

from .lib import (
    mongoengine,
    get_connection_settings,
    CONTEXT_KEY_ENVIRON
)

if TYPE_CHECKING:
    from yggdrasil.databricks import DatabricksClient
    from yggdrasil.databricks.compute.cluster import Cluster

__all__ = ["with_mongo_connection"]

F = TypeVar("F", bound=Callable[..., Any])
AliasLike = Optional[Union[str, Sequence[str]]]
EnvironLike = Optional[Union[Mapping[str, str], Sequence[str]]]


class MongoConnectError(RuntimeError):
    pass


class MongoHostResolutionError(MongoConnectError):
    pass


class MongoUrlValidationError(MongoConnectError):
    pass


class MongoDatabricksRoutingError(MongoConnectError):
    pass


def _normalize_environ(environ: EnvironLike) -> dict[str, str]:
    if not environ:
        return {}

    elif isinstance(environ, Mapping):
        out = {
            str(k): str(v)
            for k, v in environ.items()
            if v is not None
        }
    else:
        out = {
            key: value
            for key in environ
            if (value := os.getenv(key)) is not None
        }

    return out


def _starts_with(value: str, prefixes: Sequence[str]) -> bool:
    return any(value.startswith(prefix) for prefix in prefixes)


def _normalize_aliases(alias: AliasLike) -> list[str]:
    if alias is None:
        aliases = list(mongoengine.connection._connection_settings.keys())
        checked = aliases or []

    elif isinstance(alias, str):
        checked = alias.split(",") if "," in alias else [alias]

    else:
        aliases = list(dict.fromkeys(alias))
        checked = aliases or []

    return checked


def with_mongo_connection(
    func: Optional[F] = None,
    *,
    aliases: AliasLike = None,
    databricks: Optional["DatabricksClient"] = None,
    cluster: Optional["Cluster"] = None,
    force_local: bool = False,
    environ: EnvironLike = None,
    resolver: Optional[Callable[[], None] | str] = None,
) -> Callable[[F], F]:
    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            import yggdrasil.mongoengine.lib # noqa
            return fn(*args, **kwargs)
        return wrapper  # type: ignore[return-value]

    final_decorator: Callable[[F], F] = decorator

    force_local = force_local or bool(os.getenv("DATABRICKS_RUNTIME_VERSION"))

    if force_local:
        return final_decorator if func is None else final_decorator(func)

    if resolver is not None:
        register_connection(resolver=resolver)

    aliases = _normalize_aliases(aliases)
    context_key = str(DEFAULT_HOSTNAME).upper()
    environ = _normalize_environ(environ)
    configs = [
        c
        for c in (
            get_connection_settings(
                alias=alias, jsonable=True,
                context_key=context_key
            )
            for alias in aliases
        )
        if c and c.get("host") and c["host"] not in ("", "localhost")
    ]

    if databricks is None:
        if cluster is not None:
            from yggdrasil.databricks import DatabricksClient
            from yggdrasil.databricks.compute.cluster import Cluster

            if not isinstance(cluster, Cluster):
                cluster = (
                    DatabricksClient.current()
                    .compute.clusters
                    .find_cluster(cluster_name=str(cluster))
                )
            databricks = cluster.client

    if not force_local and databricks is not None:
        from yggdrasil.databricks import DatabricksClient
        from yggdrasil.databricks.compute.execution_context import exclude_env_key
        from yggdrasil.databricks.compute.cluster import Cluster

        databricks = DatabricksClient.parse(databricks)

        try:
            if isinstance(cluster, Cluster):
                cl = cluster
            elif isinstance(cluster, str):
                cl = databricks.compute.clusters.find_cluster(cluster_name=cluster)
            else:
                if not configs:
                    raise MongoUrlValidationError(
                        "No valid MongoDB connection configurations found for Databricks routing."
                    )

                mongo_url = URL.parse_str(configs[0]["host"], default_scheme="https")
                key = mongo_url.host

                if not key:
                    raise MongoHostResolutionError(
                        f"Failed to extract host from MongoDB connection URL: {configs[0]['host'][0]!r}"
                    )

                cl = databricks.compute.clusters.all_purpose_cluster(
                    key=key,
                    single_user_name=key,
                    libraries=["mongoengine", "sqlalchemy"],
                    permissions=[key],
                    custom_tags={"MongoHost": mongo_url.host},
                )

            if environ:
                environ = {
                    k: v
                    for k, v in environ.items()
                    if not exclude_env_key(k)
                }

            environ[CONTEXT_KEY_ENVIRON] = context_key
            environ.update({
                k: v
                for k, v in os.environ.items()
                if not exclude_env_key(k)
            })

            final_decorator = cl.command(func=decorator, environ=environ)
        except Exception as e:
            raise MongoDatabricksRoutingError(
                "Failed to route MongoDB execution through Databricks.\n"
                f"Root error: {type(e).__name__}: {e}"
            ) from e

    return final_decorator if func is None else final_decorator(func)