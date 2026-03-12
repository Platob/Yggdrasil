from __future__ import annotations

import functools
import os
from typing import Any, Callable, Mapping, Optional, Sequence, TypeVar, Union, TYPE_CHECKING

from yggdrasil.io.url import URL

from .lib import (
    mongoengine,
    get_connection_settings,
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
    if environ is None:
        return {}

    if isinstance(environ, Mapping):
        return {
            str(k): str(v)
            for k, v in environ.items()
            if v is not None
        }

    return {
        key: value
        for key in environ
        if (value := os.getenv(key)) is not None
    }


def _normalize_aliases(alias: AliasLike) -> list[str]:
    if alias is None:
        aliases = list(mongoengine.connection._connection_settings.keys())
        checked = aliases or [mongoengine.DEFAULT_CONNECTION_NAME]

    elif isinstance(alias, str):
        checked = alias.split(",") if "," in alias else [alias]

    else:
        aliases = list(dict.fromkeys(alias))
        checked = aliases or [mongoengine.DEFAULT_CONNECTION_NAME]

    return checked


def with_mongo_connection(
    func: Optional[F] = None,
    *,
    aliases: AliasLike = None,
    databricks: Union["DatabricksClient", str, None] = "DATABRICKS_HOST",
    cluster: Union["Cluster", str, None] = None,
    force_local: bool = False,
    environ: EnvironLike = None,
    resolver: Optional[Callable[[], Any] | str | dict] = None,
) -> Callable[[F], F]:
    force_local = force_local or bool(os.getenv("DATABRICKS_RUNTIME_VERSION"))
    aliases = _normalize_aliases(aliases)
    configs = [
        get_connection_settings(alias=alias, jsonable=True, resolver=resolver)
        for alias in aliases
    ]

    if not configs:
        raise MongoConnectError(
            "No valid MongoDB connection aliases found."
            "Please specify at least one valid alias or ensure that default connection settings are configured."
            "Using mongoengine.register_connection or mongoengine.connect with appropriate aliases can help set up the connections."
        )

    if isinstance(databricks, str):
        databricks = os.getenv(databricks, databricks)

    if databricks is None and cluster is not None:
        if isinstance(cluster, str):
            from yggdrasil.databricks import DatabricksClient

            cluster = (
                DatabricksClient.current()
                .compute.clusters
                .find_cluster(cluster_name=cluster)
            )
        databricks = cluster.client

    check_connections = None if databricks is None else configs

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if check_connections:
                from yggdrasil.mongoengine import connect as mc

                for config in check_connections:
                    mc(**config)

            return fn(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    final_decorator: Callable[[F], F] = decorator

    if not force_local and databricks is not None:
        from yggdrasil.databricks import DatabricksClient
        from yggdrasil.databricks.compute import Cluster

        databricks = DatabricksClient.parse(databricks)

        try:
            mongo_url = URL.parse_str(configs[0]["host"], default_scheme="https")

            if not mongo_url.host:
                raise MongoHostResolutionError(
                    f"Failed to extract host from MongoDB connection URL: {configs[0]['host'][0]!r}"
                )

            key = mongo_url.host

            if cluster is None:
                cl = databricks.compute.clusters.all_purpose_cluster(
                    key=key,
                    single_user_name=key,
                    libraries=["mongoengine", "sqlalchemy"],
                    permissions=[key],
                    custom_tags={"MongoHost": mongo_url.host},
                )
            elif not isinstance(cluster, Cluster):
                cl = databricks.compute.clusters.find_cluster(cluster_name=str(cluster))
            else:
                cl = cluster

            final_decorator = cl.command(func=decorator, environ=environ)
        except Exception as e:
            raise MongoDatabricksRoutingError(
                "Failed to route MongoDB execution through Databricks.\n"
                f"Root error: {type(e).__name__}: {e}"
            ) from e

    return final_decorator if func is None else final_decorator(func)