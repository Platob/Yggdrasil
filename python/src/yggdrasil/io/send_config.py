from __future__ import annotations

import dataclasses
import datetime as dt
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Iterable, Literal, Mapping, MutableMapping, Optional, TYPE_CHECKING

from yggdrasil.data import any_to_datetime, any_to_timedelta
from yggdrasil.dataclasses import DEFAULT_WAITING_CONFIG
from yggdrasil.dataclasses.waiting import WaitingConfig, WaitingConfigArg
from yggdrasil.io import SaveMode
from yggdrasil.io.request import REQUEST_ARROW_SCHEMA, PreparedRequest
from yggdrasil.io.response import RESPONSE_ARROW_SCHEMA

if TYPE_CHECKING:
    from pyspark.sql import SparkSession
    from yggdrasil.databricks.sql.table import Table
    from yggdrasil.io.response import Response


__all__ = ["CacheConfig", "SendConfig", "SendManyConfig"]


_DEFAULT_REQUEST_BY: tuple[str, ...] = (
    "request_method",
    "request_url_scheme",
    "request_url_host",
    "request_url_port",
    "request_url_query",
    "request_content_length",
    "request_body_hash",
)

_CACHE_CONFIG_FIELDS: frozenset[str] = frozenset(
    {
        "path",
        "table",
        "request_by",
        "response_by",
        "mode",
        "anonymize",
        "received_from",
        "received_to",
        "wait",
    }
)

_SEND_CONFIG_FIELDS: frozenset[str] = frozenset(
    {
        "wait",
        "raise_error",
        "stream",
        "remote_cache",
        "local_cache",
        "spark_session",
    }
)

_SEND_MANY_CONFIG_FIELDS: frozenset[str] = _SEND_CONFIG_FIELDS | frozenset(
    {
        "normalize",
        "batch_size",
        "ordered",
        "max_in_flight",
    }
)


def _validate_request_by(arg: list[str] | tuple[str, ...] | None = None) -> list[str]:
    keys = list(_DEFAULT_REQUEST_BY if not arg else arg)
    invalid = [key for key in keys if key not in REQUEST_ARROW_SCHEMA.names]
    if invalid:
        raise ValueError(
            f"Invalid request_by key(s): {invalid!r}. "
            f"Must be within: {REQUEST_ARROW_SCHEMA.names!r}"
        )
    return keys


def _validate_response_by(
    arg: list[str] | tuple[str, ...] | None = None,
) -> list[str] | None:
    if arg is None:
        return None

    keys = list(arg)
    invalid = [key for key in keys if key not in RESPONSE_ARROW_SCHEMA.names]
    if invalid:
        raise ValueError(
            f"Invalid response_by key(s): {invalid!r}. "
            f"Must be within: {RESPONSE_ARROW_SCHEMA.names!r}"
        )
    return keys


def _coerce_optional_datetime(value: Any) -> Optional[dt.datetime]:
    if value in (None, ""):
        return None
    if isinstance(value, dt.datetime):
        return value
    return any_to_datetime(value)


@dataclass(frozen=True, slots=True)
class _ConfigBase:
    _FIELD_NAMES: ClassVar[frozenset[str]]

    @classmethod
    def default(cls):
        return cls()

    @classmethod
    def parse_mapping(cls, options: Mapping[str, Any], **overrides: Any):
        if not isinstance(options, Mapping):
            raise TypeError(
                f"{cls.__name__}.parse_mapping expects a Mapping, "
                f"got {type(options).__name__!r}"
            )
        values = {k: v for k, v in options.items() if k in cls._FIELD_NAMES}
        values.update(overrides)
        return cls(**cls._check_mapping(values))

    @staticmethod
    def _check_mapping(values: MutableMapping[str, Any]):
        spark_session = values.get("spark_session")
        if spark_session is not None and isinstance(spark_session, bool):
            if spark_session:
                from yggdrasil.environ import PyEnv

                values["spark_session"] = PyEnv.spark_session(
                    create=True,
                    install_spark=True,
                    import_error=True,
                )
            else:
                values["spark_session"] = None

        wait = values.get("wait")
        if wait is not None:
            values["wait"] = WaitingConfig.check_arg(wait)

        remote_cache = values.get("remote_cache")
        if remote_cache is not None:
            values["remote_cache"] = CacheConfig.check_arg(remote_cache)

        local_cache = values.get("local_cache")
        if local_cache is not None:
            values["local_cache"] = CacheConfig.check_arg(local_cache)

        return {
            k: v
            for k, v in values.items()
            if v is not None
        }

    def merge(self, **overrides: Any):
        unknown = set(overrides) - self._FIELD_NAMES
        if unknown:
            raise TypeError(
                f"{type(self).__name__}.merge got unexpected field(s): {sorted(unknown)!r}"
            )
        return dataclasses.replace(self, **self._check_mapping(overrides))


@dataclass(frozen=True, slots=True)
class CacheConfig(_ConfigBase):
    _FIELD_NAMES: ClassVar[frozenset[str]] = _CACHE_CONFIG_FIELDS

    path: Optional[Path] = field(default=None, hash=False, compare=False)
    table: Optional["Table"] = field(default=None, hash=False, compare=False)
    request_by: Optional[list[str]] = field(default=None, hash=False, compare=False)
    response_by: Optional[list[str]] = field(default=None, hash=False, compare=False)
    mode: SaveMode = SaveMode.APPEND
    anonymize: Literal["remove", "redact"] = "remove"
    received_from: Optional[dt.datetime] = None
    received_to: Optional[dt.datetime] = None
    received_ttl: Optional[dt.timedelta] = None
    wait: WaitingConfig = False

    @staticmethod
    def _check_mapping(values: MutableMapping[str, Any]):
        wait = values.get("wait")
        if wait is not None:
            values["wait"] = WaitingConfig.check_arg(wait)

        received_ttl = values.get("received_ttl")
        if received_ttl is not None:
            values["received_ttl"] = any_to_timedelta(received_ttl)

        received_from = values.get("received_from")
        if received_from is not None:
            values["received_from"] = _coerce_optional_datetime(received_from)

        received_to = values.get("received_to")
        if received_to is not None:
            values["received_to"] = _coerce_optional_datetime(received_to)

        return values

    def __post_init__(self) -> None:
        object.__setattr__(self, "wait",  WaitingConfig.check_arg(self.wait))

        object.__setattr__(self, "request_by", _validate_request_by(self.request_by))
        object.__setattr__(self, "response_by", _validate_response_by(self.response_by))

        object.__setattr__(self, "received_from", _coerce_optional_datetime(self.received_from))
        object.__setattr__(self, "received_to", _coerce_optional_datetime(self.received_to))

        if self.received_ttl:
            if not self.received_to:
                object.__setattr__(self, "received_to", dt.datetime.now(dt.timezone.utc))

            if not self.received_from:
                object.__setattr__(self, "received_from", self.received_to - self.received_ttl)

    @classmethod
    def check_arg(
        cls,
        arg: "CacheConfig | Mapping[str, Any] | None",
        **overrides: Any,
    ) -> "CacheConfig":
        if arg is None:
            return cls.parse_mapping(overrides) if overrides else cls.default()
        if isinstance(arg, cls):
            return arg.merge(**overrides) if overrides else arg
        if isinstance(arg, Mapping):
            return cls.parse_mapping(arg, **overrides)

        if isinstance(arg, Path):
            overrides["path"] = arg

        elif hasattr(arg, "create") and callable(getattr(arg, "create")):
            overrides["table"] = arg

        elif isinstance(arg, dt.datetime):
            overrides["received_from"] = arg
        elif isinstance(arg, dt.date):
            overrides["received_from"] = dt.datetime.combine(arg, dt.time.min, tzinfo=dt.timezone.utc)

        elif isinstance(arg, dt.timedelta):
            overrides["received_ttl"] = arg

            # fill received_from and received_to if not exists
            received_to = overrides.get("received_to")
            received_to = dt.datetime.now(dt.timezone.utc) if received_to is None else any_to_datetime(received_to)
            overrides["received_to"] = received_to

            received_from = overrides.get("received_from")
            if not received_from:
                overrides["received_from"] = received_to - arg

        return cls.parse_mapping(overrides) if overrides else cls.default()

    @property
    def local_cache_enabled(self):
        return self.received_from is not None or self.received_to is not None

    @property
    def remote_cache_enabled(self):
        return self.table is not None

    @property
    def by(self) -> list[str]:
        return [
            *(self.request_by or ()),
            *(self.response_by or ()),
        ]

    @property
    def defined_received_from(self) -> dt.datetime:
        if self.received_from:
            return self.received_from.timestamp()

        return dt.datetime.fromtimestamp(
            0,
            tz=dt.timezone.utc,
        )

    @property
    def defined_received_to(self) -> dt.datetime:
        if self.received_to:
            return self.received_to.timestamp()

        return dt.datetime.fromtimestamp(
            time.time() + 3600,
            tz=dt.timezone.utc,
        )

    @staticmethod
    def sql_literal(value: Any) -> str:
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, dt.datetime):
            return f"timestamp '{value.isoformat(sep=' ', timespec='microseconds')}'"
        if isinstance(value, bytes):
            import base64
            value = base64.b64encode(value).decode("ascii")
        else:
            value = str(value)
        return f"'{value.replace(chr(39), chr(39) * 2)}'"

    def local_cache_folder(self) -> Path:
        if self.path is None:
            object.__setattr__(self, "path", Path.home() / ".yggdrasil" / "io" / "session")
        return self.path

    def local_cache_file(
        self,
        request: PreparedRequest,
        *,
        suffix: str | None = None,
        force: bool = False
    ) -> Path | None:
        if not force and not self.local_cache_enabled:
            return None

        anonymized = request.anonymize(mode="remove")
        cache_folder = self.local_cache_folder() / "cache"
        url = anonymized.url

        if url.host:
            cache_folder = cache_folder / url.host

        if url.path:
            path_parts = [part for part in url.path.split("/") if part]
            if path_parts:
                cache_folder = cache_folder.joinpath(*path_parts)

        path = cache_folder / f"{anonymized.xxh3_b64(url_safe=True)}{suffix or '.bin'}"

        if force:
            return path

        if not path.exists():
            return None

        mtime = dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.timezone.utc)

        if self.received_from is not None and mtime < self.received_from:
            path.unlink(missing_ok=True)
            return None

        if self.received_to is not None and mtime > self.received_to:
            return None

        return path

    def request_values(
        self,
        request: PreparedRequest,
    ) -> dict[str, Any]:
        return {
            key: request.match_value(key)
            for key in (self.request_by or [])
        }

    def response_values(
        self,
        response: "Response",
    ) -> dict[str, Any]:
        return {key: response.match_value(key) for key in (self.response_by or [])}

    def filter_request(
        self,
        request: PreparedRequest,
    ) -> bool:
        for key in self.request_by or []:
            request.match_value(key)
        return True

    def filter_response(
        self,
        response: "Response",
        request: PreparedRequest | None = None,
    ) -> bool:
        if request is not None:
            for key, expected in self.request_values(request).items():
                actual = response.match_value(key)
                if actual != expected:
                    return False

        for key in self.response_by or []:
            response.match_value(key)

        if self.received_from is not None:
            if response.received_at < self.received_from:
                return False

        if self.received_to is not None:
            if response.received_at >= self.received_to:
                return False

        return True

    def request_tuple(
        self,
        request: PreparedRequest,
    ) -> tuple[Any, ...]:
        values = self.request_values(request)
        return tuple(values[key] for key in (self.request_by or []))

    def response_tuple(
        self,
        response: "Response",
    ) -> tuple[Any, ...]:
        values = self.response_values(response)
        return tuple(values[key] for key in (self.response_by or []))

    def identity_tuple(
        self,
        response: "Response",
        request: PreparedRequest | None = None,
    ) -> tuple[Any, ...]:
        out: list[Any] = []
        if request is not None:
            out.extend(self.request_tuple(request))
        out.extend(self.response_tuple(response))
        return tuple(out)

    def sql_request_clause(
        self,
        request: PreparedRequest | None,
    ) -> str:
        clauses: list[str] = []

        if request is not None:
            for key, value in self.request_values(request).items():
                if value is None:
                    clauses.append(f"{key} IS NULL")
                else:
                    clauses.append(f"{key} = {self.sql_literal(value)}")

        return " AND ".join(clauses) if clauses else "1=1"

    def sql_response_clause(
        self,
        response: "Response | None" = None,
    ) -> str:
        clauses: list[str] = []

        if response is not None:
            for key, value in self.response_values(response).items():
                if value is None:
                    clauses.append(f"{key} IS NULL")
                else:
                    clauses.append(f"{key} = {self.sql_literal(value)}")

        if self.received_from is not None:
            clauses.append(f"response_received_at >= {self.sql_literal(self.received_from)}")

        if self.received_to is not None:
            clauses.append(f"response_received_at < {self.sql_literal(self.received_to)}")

        return " AND ".join(clauses) if clauses else "1=1"

    def sql_clause(
        self,
        request: PreparedRequest | None = None,
        response: "Response | None" = None,
    ) -> str:
        clauses: list[str] = []

        request_clause = self.sql_request_clause(request)
        if request_clause != "1=1":
            clauses.append(f"({request_clause})")

        response_clause = self.sql_response_clause(response)
        if response_clause != "1=1":
            clauses.append(f"({response_clause})")

        return " AND ".join(clauses) if clauses else "1=1"

    def make_lookup_sql(
        self,
        table_name: str,
        request: PreparedRequest | None = None,
        response: "Response | None" = None,
        *,
        identity_by: Optional[Iterable[str]] = None,
    ) -> str:
        where_clause = self.sql_clause(request=request, response=response)
        base_query = f"SELECT * FROM {table_name}"
        if where_clause != "1=1":
            base_query += f" WHERE {where_clause}"

        identity_cols = list(identity_by) if identity_by is not None else self.by
        if identity_cols:
            partition_by = ", ".join(identity_cols)
            return (
                "SELECT * FROM ("
                "  SELECT t.*, row_number() OVER ("
                f"    PARTITION BY {partition_by} "
                "    ORDER BY response_received_at_epoch DESC"
                "  ) AS __rn "
                f"  FROM ({base_query}) t"
                ") ranked WHERE __rn = 1"
            )

        return base_query

    def make_batch_lookup_sql(
        self,
        table_name: str,
        requests: Iterable[PreparedRequest],
        *,
        identity_by: Optional[Iterable[str]] = None,
    ) -> str:
        request_clauses = " OR ".join(
            f"({self.sql_request_clause(req)})"
            for req in requests
        )
        response_clause = self.sql_response_clause(None)

        where_parts: list[str] = []
        if request_clauses:
            where_parts.append(f"({request_clauses})")
        if response_clause != "1=1":
            where_parts.append(f"({response_clause})")

        base_query = f"SELECT * FROM {table_name}"
        if where_parts:
            base_query += " WHERE " + " AND ".join(where_parts)

        identity_cols = list(identity_by) if identity_by is not None else self.by
        if identity_cols:
            partition_by = ", ".join(identity_cols)
            return (
                "SELECT * FROM ("
                "  SELECT t.*, row_number() OVER ("
                f"    PARTITION BY {partition_by} "
                "    ORDER BY response_received_at_epoch DESC"
                "  ) AS __rn "
                f"  FROM ({base_query}) t"
                ") ranked WHERE __rn = 1"
            )

        return base_query


DEFAULT_CACHE_CONFIG = CacheConfig()


@dataclass(frozen=True, slots=True)
class SendConfig(_ConfigBase):
    _FIELD_NAMES: ClassVar[frozenset[str]] = _SEND_CONFIG_FIELDS

    raise_error: bool = True
    stream: bool = True
    wait: WaitingConfig = field(default=DEFAULT_WAITING_CONFIG)
    remote_cache: CacheConfig = field(default=DEFAULT_CACHE_CONFIG)
    local_cache: CacheConfig = field(default=DEFAULT_CACHE_CONFIG)
    spark_session: Optional["SparkSession"] = field(
        default=None,
        hash=False,
        compare=False,
        repr=False,
    )

    def __post_init__(self):
        object.__setattr__(self, "wait", WaitingConfig.check_arg(self.wait))
        object.__setattr__(self, "remote_cache", CacheConfig.check_arg(self.remote_cache))
        object.__setattr__(self, "local_cache", CacheConfig.check_arg(self.local_cache))

    @classmethod
    def check_arg(
        cls,
        arg: "SendConfig | Mapping[str, Any] | None",
        **overrides: Any,
    ) -> "SendConfig":
        if arg is None:
            return cls.parse_mapping(overrides) if overrides else cls.default()
        if isinstance(arg, cls):
            return arg.merge(**overrides) if overrides else arg
        if isinstance(arg, Mapping):
            return cls.parse_mapping(arg, **overrides)
        raise TypeError(
            f"{cls.__name__}.check_arg expects a {cls.__name__}, Mapping, or None; "
            f"got {type(arg).__name__!r}"
        )


@dataclass(frozen=True, slots=True)
class SendManyConfig(_ConfigBase):
    _FIELD_NAMES: ClassVar[frozenset[str]] = _SEND_MANY_CONFIG_FIELDS

    wait: WaitingConfigArg = None
    raise_error: bool = True
    stream: bool = True
    remote_cache: CacheConfig = field(default_factory=CacheConfig)
    local_cache: CacheConfig = field(default_factory=CacheConfig)
    spark_session: Optional["SparkSession"] = field(
        default=None,
        hash=False,
        compare=False,
        repr=False,
    )

    normalize: Optional[bool] = None
    batch_size: Optional[int] = None
    ordered: bool = False
    max_in_flight: Optional[int] = None

    def __post_init__(self):
        object.__setattr__(self, "wait", WaitingConfig.check_arg(self.wait))
        object.__setattr__(self, "remote_cache", CacheConfig.check_arg(self.remote_cache))
        object.__setattr__(self, "local_cache", CacheConfig.check_arg(self.local_cache))

    @classmethod
    def check_arg(
        cls,
        arg: "SendManyConfig | SendConfig | Mapping[str, Any] | None",
        **overrides: Any,
    ) -> "SendManyConfig":
        if arg is None:
            return cls(**overrides) if overrides else cls.default()
        if isinstance(arg, cls):
            return arg.merge(**overrides) if overrides else arg
        if isinstance(arg, SendConfig):
            return cls(
                wait=arg.wait,
                raise_error=arg.raise_error,
                stream=arg.stream,
                remote_cache=arg.remote_cache,
                local_cache=arg.local_cache,
                spark_session=arg.spark_session,
                **overrides,
            )
        if isinstance(arg, Mapping):
            return cls.parse_mapping(arg, **overrides)
        raise TypeError(
            f"{cls.__name__}.check_arg expects a {cls.__name__}, SendConfig, "
            f"Mapping, or None; got {type(arg).__name__!r}"
        )

    def to_send_config(
        self,
        with_remote_cache: bool = True,
        with_local_cache: bool = True,
        with_spark: bool = False,
    ) -> SendConfig:
        return SendConfig(
            wait=self.wait,
            raise_error=self.raise_error,
            stream=self.stream,
            remote_cache=self.remote_cache if with_remote_cache else CacheConfig(),
            local_cache=self.local_cache if with_local_cache else CacheConfig(),
            spark_session=self.spark_session if with_spark else None,
        )
