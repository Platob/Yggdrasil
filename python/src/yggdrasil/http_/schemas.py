from __future__ import annotations

import pyarrow as pa

from yggdrasil.data.data_field import field as schema_field
from yggdrasil.data.schema import schema
from yggdrasil.environ.userinfo import USERINFO_STRUCT
from yggdrasil.url import URL_STRUCT

__all__ = [
    "REQUEST_SCHEMA",
    
    "REQUEST_URL_STRUCT",
    "RESPONSE_SCHEMA",
    
]


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------

_REQUEST_SCHEMA_JSON_TAGS: dict[str, str] = {
    "domain": "http",
    "entity": "request",
    "layer": "bronze",
}


# Nested URL struct — re-exported from :mod:`yggdrasil.url` so the
# request schema and every downstream consumer share a single source
# of truth for column ordering, types, and nullability flags. The
# full string isn't kept here; ``private_url_hash`` covers exact
# identity and ``URL.from_(struct)`` reassembles the URL from its
# parts.
REQUEST_URL_STRUCT = URL_STRUCT


REQUEST_SCHEMA = schema(
    fields=[],
    metadata={
        "comment": "Prepared request flattened into deterministic columns for logging and replay.",
        "time_column": "sent_at",
        # Schema-level identity / partitioning hints — ``autotag`` at
        # the bottom of this block propagates them to the matching
        # children. The primary key is composite ``(hash, body_size)``;
        # ``partition_by`` lists every column that gets a Hive-style
        # partition leaf — ``method`` keeps each verb on its own
        # branch (cheap predicate pushdown for "all GETs"), while
        # ``partition_key`` (the xxh3 of :meth:`PreparedRequest.partition_values`)
        # buckets the rest by endpoint.
        "primary_key": ["hash", "body_size"],
        "partition_by": ["method", "partition_key"],
    },
    tags=_REQUEST_SCHEMA_JSON_TAGS,
)

REQUEST_SCHEMA["hash"] = schema_field(
    "hash",
    pa.int64(),
    nullable=False,
    metadata={
        "comment": "xxh3_64 digest over (method, url, headers, body) — overall request "
                   "identity. Includes sensitive bits (URL userinfo, Authorization / "
                   "API-key headers); use ``public_hash`` for cross-system joins / "
                   "cache lookups that must survive anonymization.",
        "algorithm": "xxh3_64",
    },
).autotag()

REQUEST_SCHEMA["public_hash"] = schema_field(
    "public_hash",
    pa.int64(),
    nullable=False,
    metadata={
        "comment": "xxh3_64 digest over the anonymize='remove' projection of "
                   "(method, url, headers, body). Stable across cache anonymization, so this "
                   "is the right key for dedup / cross-system identity.",
        "algorithm": "xxh3_64",
    },
).autotag()

REQUEST_SCHEMA["method"] = schema_field(
    "method",
    pa.string(),
    nullable=False,
    metadata={
        "comment": "HTTP method (GET, POST, etc.). One of the schema's "
                   "``partition_by`` columns — each verb lands in its own "
                   "Hive partition leaf.",
    },
).autotag()

REQUEST_SCHEMA["url"] = schema_field(
    "url",
    REQUEST_URL_STRUCT,
    nullable=False,
    metadata={"comment": "Parsed URL components with full string for replay"},
).autotag()

REQUEST_SCHEMA["sender"] = schema_field(
    "sender",
    USERINFO_STRUCT,
    nullable=True,
    metadata={
        "comment": "Snapshot of :class:`~yggdrasil.environ.UserInfo` for the sender "
                   "— defaults to ``UserInfo.current()``. Carries identity (key, "
                   "email, hostname, product) plus a stable ``hash``. Per-process "
                   "fields (cwd, url, git_url) are lazy properties on UserInfo and "
                   "are not part of the wire contract.",
    },
).autotag()

REQUEST_SCHEMA["private_url_hash"] = schema_field(
    "private_url_hash",
    pa.int64(),
    nullable=False,
    metadata={
        "comment": "xxh3_64 digest of (method, full URL string) — scheme, userinfo, host, "
                   "port, path, query, fragment exactly as captured. Method is mixed in "
                   "so ``GET /x`` and ``POST /x`` don't collide.",
        "algorithm": "xxh3_64",
    },
).autotag()

REQUEST_SCHEMA["public_url_hash"] = schema_field(
    "public_url_hash",
    pa.int64(),
    nullable=False,
    metadata={
        "comment": "xxh3_64 digest of (method, ``url.anonymize('remove').to_string()``) — "
                   "userinfo and sensitive query params dropped, method mixed in so verbs "
                   "stay distinct. Stable across anonymization and the default cache match "
                   "key.",
        "algorithm": "xxh3_64",
    },
).autotag()

REQUEST_SCHEMA["headers"] = schema_field(
    "headers",
    pa.map_(pa.string(), pa.string()),
    nullable=False,
    metadata={"comment": "All request headers as a name→value map"},
).autotag()

REQUEST_SCHEMA["tags"] = schema_field(
    "tags",
    pa.map_(pa.string(), pa.string()),
    nullable=False,
    metadata={"comment": "Request tags merged with URL query params; explicit tags win on conflict"},
).autotag()

REQUEST_SCHEMA["body"] = schema_field(
    "body",
    pa.large_binary(),
    nullable=True,
    metadata={"comment": "Raw request body bytes"},
).autotag()

REQUEST_SCHEMA["body_size"] = schema_field(
    "body_size",
    pa.int64(),
    nullable=False,
    metadata={
        "comment": "Length of body in bytes; 0 when body is absent",
        "unit": "bytes",
    },
).autotag()

REQUEST_SCHEMA["body_hash"] = schema_field(
    "body_hash",
    pa.int64(),
    nullable=False,
    metadata={
        "comment": "xxh3_64 digest of body bytes; 0 when body is absent",
        "algorithm": "xxh3_64",
    },
).autotag()

REQUEST_SCHEMA["sent_at"] = schema_field(
    "sent_at",
    pa.timestamp("us", "UTC"),
    nullable=False,
    metadata={"comment": "UTC timestamp when request was dispatched"},
).autotag()

REQUEST_SCHEMA["partition_key"] = schema_field(
    "partition_key",
    pa.int64(),
    nullable=False,
    metadata={
        "comment": "xxh3_64 digest of the request's ``partition_values`` — the only "
                   "``partition_by`` column. Override "
                   ":meth:`PreparedRequest.partition_values` to pick a different "
                   "endpoint-grouping strategy; the default groups by URL host+path "
                   "so every call to the same endpoint shares one partition leaf.",
        "algorithm": "xxh3_64",
    },
).autotag()

REQUEST_SCHEMA["_pkl"] = schema_field(
    "_pkl",
    pa.large_binary(),
    nullable=True,
    metadata={
        "comment": "Placeholder for a full ``PreparedRequest`` pickle blob — populated "
                   "by the pickle serializer for lossless round-trips, left null on the "
                   "deterministic-columns-only path.",
    },
).autotag()

# Propagate schema-level ``primary_key`` / ``partition_by`` down to
# the matching children (consumes those metadata keys in place).
REQUEST_SCHEMA = REQUEST_SCHEMA.autotag()



# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------

RESPONSE_SCHEMA = schema(
    fields=[],
    metadata={
        "comment": "Response record (single row), designed for deterministic logging and replay.",
        "time_column": "received_at",
        # Schema-level identity / partitioning hints — ``autotag`` at
        # the bottom of this block propagates them to the matching
        # children. The primary key is composite ``(hash, body_size)``;
        # ``partition_key`` (the only ``partition_by`` column) is
        # derived from :meth:`Response.partition_values` and matches
        # the embedded request's partition_key so they co-locate.
        "primary_key": ["hash", "body_size"],
        "partition_by": ["partition_key"],
    },
    tags={
        "domain": "http",
        "entity": "response",
        "layer": "bronze",
        "namespace": "yggdrasil.io.response",
    },
)

# Unnest the request schema directly into the response schema with a
# ``request_`` prefix. Flattening turns nested struct lookups into
# top-level column accesses, which engines (Delta, Spark, Polars,
# Arrow) can predicate-push and column-prune against without having
# to crack the struct open. ``_pkl`` is intentionally skipped — the
# response carries its own pickle slot and a per-request blob would
# duplicate the same bytes. Schema-level partition_by / primary_key
# flags inherited from REQUEST_SCHEMA.autotag() get cleared on the
# unnested copies — those flags belong to the response's own columns.
for _req_field in REQUEST_SCHEMA.children:
    if _req_field.name == "_pkl":
        continue
    _copied = _req_field.copy(name=f"request_{_req_field.name}")
    _copied.with_partition_by(False, inplace=True)
    _copied.with_primary_key(False, inplace=True)
    if _req_field.comment:
        _copied.metadata[b"comment"] = f"[request] {_req_field.comment}".encode("utf-8")
    RESPONSE_SCHEMA[_copied.name] = _copied.autotag()

RESPONSE_SCHEMA["receiver"] = schema_field(
    "receiver",
    USERINFO_STRUCT,
    nullable=True,
    metadata={
        "comment": "Snapshot of :class:`~yggdrasil.environ.UserInfo` for the receiver "
                   "— defaults to ``UserInfo.current()``. Carries identity (key, "
                   "email, hostname, product) plus a stable ``hash``. Per-process "
                   "fields (cwd, url, git_url) are lazy properties on UserInfo and "
                   "are not part of the wire contract.",
    },
).autotag()

RESPONSE_SCHEMA["hash"] = schema_field(
    "hash",
    pa.int64(),
    nullable=False,
    metadata={
        "comment": "xxh3_64 digest over (request.hash, status_code, headers, body) — "
                   "overall response identity, including sensitive bits.",
        "algorithm": "xxh3_64",
    },
).autotag()

RESPONSE_SCHEMA["public_hash"] = schema_field(
    "public_hash",
    pa.int64(),
    nullable=False,
    metadata={
        "comment": "xxh3_64 digest over (request.public_hash, status_code, anonymized headers, body) — "
                   "stable across cache anonymization and the right key for cross-system identity.",
        "algorithm": "xxh3_64",
    },
).autotag()

RESPONSE_SCHEMA["status_code"] = schema_field(
    "status_code",
    pa.int32(),
    nullable=False,
    metadata={"comment": "HTTP status code returned by the server"},
).autotag()

RESPONSE_SCHEMA["headers"] = schema_field(
    "headers",
    pa.map_(pa.string(), pa.string()),
    nullable=False,
    metadata={"comment": "All response headers as a name→value map"},
).autotag()

RESPONSE_SCHEMA["tags"] = schema_field(
    "tags",
    pa.map_(pa.string(), pa.string()),
    nullable=True,
    metadata={"comment": "Arbitrary string tags attached to this response"},
).autotag()

RESPONSE_SCHEMA["body"] = schema_field(
    "body",
    pa.large_binary(),
    nullable=True,
    metadata={"comment": "Raw binary payload of the response"},
).autotag()

RESPONSE_SCHEMA["body_size"] = schema_field(
    "body_size",
    pa.int64(),
    nullable=False,
    metadata={
        "comment": "Length of body in bytes; 0 when body is absent",
        "unit": "bytes",
    },
).autotag()

RESPONSE_SCHEMA["body_hash"] = schema_field(
    "body_hash",
    pa.int64(),
    nullable=False,
    metadata={
        "comment": "xxh3_64 digest of body bytes; 0 when body is absent",
        "algorithm": "xxh3_64",
    },
).autotag()

RESPONSE_SCHEMA["received_at"] = schema_field(
    "received_at",
    pa.timestamp("us", "UTC"),
    nullable=False,
    metadata={"comment": "UTC timestamp when the response was captured"},
).autotag()

RESPONSE_SCHEMA["partition_key"] = schema_field(
    "partition_key",
    pa.int64(),
    nullable=False,
    metadata={
        "comment": "xxh3_64 digest of the response's ``partition_values`` — the only "
                   "``partition_by`` column. Equal to the embedded request's "
                   "``partition_key`` (default behaviour) so request+response always "
                   "co-locate. Override :meth:`Response.partition_values` to change.",
        "algorithm": "xxh3_64",
    },
).autotag()

RESPONSE_SCHEMA["_pkl"] = schema_field(
    "_pkl",
    pa.large_binary(),
    nullable=True,
    metadata={
        "comment": "Placeholder for a full ``Response`` pickle blob — populated by the "
                   "pickle serializer for lossless round-trips, left null on the "
                   "deterministic-columns-only path.",
    },
).autotag()

# Propagate schema-level ``primary_key`` / ``partition_by`` down to
# the matching children (consumes those metadata keys in place).
RESPONSE_SCHEMA = RESPONSE_SCHEMA.autotag()

