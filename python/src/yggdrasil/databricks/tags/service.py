"""
Collection-level service for Unity Catalog entity tag assignments.

Provides ``entity_tags`` / ``update_entity_tag(s)`` against the Databricks
``entity_tag_assignments`` API.

Caching strategy
----------------
A module-level :class:`ExpiringDict` (``_TAGS_CACHE``) keyed by
``"host|entity_type|entity_name"`` caches the *full list of tags* for an
entity, so repeated reads (and the diff-pass that ``update_entity_tags``
performs before issuing writes) hit local memory instead of the API.

    1. **Local** — check ``_TAGS_CACHE``; return immediately on hit.
    2. **Remote** — call the workspace client; only on miss.
    3. **Update** — store the remote result back.

Writes (``update_entity_tag(s)``) **surgically invalidate** the entry — or,
when we know the new value, patch the cached list in place — so the next
read sees fresh state without forcing every concurrent caller to refetch.

Retry strategy
--------------
Every workspace-client call is funnelled through :class:`RetryPolicy`,
which retries transient errors (``InternalError``, ``TemporarilyUnavailable``,
``DeadlineExceeded``, ``TooManyRequests``, ``ConnectionError``,
``TimeoutError``) with exponential backoff plus full jitter, honoring
server ``Retry-After`` hints when present, capped by an absolute deadline.
Fatal errors (``BadRequest``, ``PermissionDenied``, ``Unauthenticated``)
propagate immediately so caller-visible bugs surface fast.
"""

from __future__ import annotations

import json
import logging
import random
import time
from concurrent.futures.thread import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional, TypeVar

from databricks.sdk.errors import DatabricksError, NotFound
from databricks.sdk.errors.platform import (
    BadRequest,
    DeadlineExceeded,
    InternalError,
    PermissionDenied,
    TemporarilyUnavailable,
    TooManyRequests,
    Unauthenticated,
)
from databricks.sdk.service.catalog import (
    EntityTagAssignment,
    TagAssignmentSourceType,
)

from yggdrasil.dataclasses.expiring import ExpiringDict
from yggdrasil.io.enums.mode import Mode, ModeLike
from ..client import DatabricksService

__all__ = ["EntityTags", "RetryPolicy"]

logger = logging.getLogger(__name__)

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Error classification — drives the retry policy.
# ---------------------------------------------------------------------------
# Retryable: transient platform / network conditions. ``TooManyRequests``
# carries an optional ``retry_after_secs`` attribute that the policy honors
# when present (overriding the exponential schedule for that one attempt).
_RETRYABLE_ERRORS: tuple[type[BaseException], ...] = (
    InternalError,
    TemporarilyUnavailable,
    DeadlineExceeded,
    TooManyRequests,
    ConnectionError,
    TimeoutError,
)
# Fatal: bad input or wrong credentials. Retrying hides the real problem.
_FATAL_ERRORS: tuple[type[BaseException], ...] = (
    BadRequest,
    PermissionDenied,
    Unauthenticated,
)

# ---------------------------------------------------------------------------
# Module-level cache
# Keyed by "host|entity_type|entity_name"; default TTL = 5 minutes.
# Values are the full ``list[EntityTagAssignment]`` for the entity.
# ---------------------------------------------------------------------------
_TAGS_CACHE: ExpiringDict[str, list[EntityTagAssignment]] = ExpiringDict(
    default_ttl=300.0,
)

_DEFAULT_SOURCE_TYPE = TagAssignmentSourceType.TAG_ASSIGNMENT_SOURCE_TYPE_SYSTEM_DATA_CLASSIFICATION

# The Databricks ``entity_tag_assignments.update`` endpoint requires a
# Google-style ``update_mask`` naming the subfields of the tag assignment
# to write. The identity fields (entity_type, entity_name, tag_key) are
# routing — they're never in the mask. ``update_time`` and ``updated_by``
# are server-managed. That leaves ``tag_value`` (and, in principle,
# ``source_type``) as the only writable subfields.
_DEFAULT_UPDATE_MASK: str = "tag_value"


def _build_update_mask(
    user_mask: str | None,
    *,
    source_type: TagAssignmentSourceType | None,
) -> str:
    """Resolve the ``update_mask`` for an ``update`` call.

    Honors a caller-supplied mask; otherwise returns
    :data:`_DEFAULT_UPDATE_MASK`, extended with ``source_type`` when the
    caller has set a non-default one.
    """
    if user_mask:
        return user_mask
    fields = ["tag_value"]
    if source_type is not None and source_type != _DEFAULT_SOURCE_TYPE:
        fields.append("source_type")
    return ",".join(fields)


# ---------------------------------------------------------------------------
# Coercion helpers (unchanged behaviour, kept module-private)
# ---------------------------------------------------------------------------

def _safe_str(obj: Any) -> str:
    if isinstance(obj, str):
        return obj
    if isinstance(obj, bool):
        return "true" if obj else "false"
    if isinstance(obj, (bytes, bytearray)):
        return obj.decode("utf-8")
    if obj is None:
        return ""
    return json.dumps(obj)


def _safe_assignment(
    entity_type: str,
    entity_name: str,
    obj: Any,
) -> EntityTagAssignment:
    if isinstance(obj, EntityTagAssignment):
        return obj
    if isinstance(obj, str):
        return EntityTagAssignment.from_dict(json.loads(obj))
    if isinstance(obj, dict):
        return EntityTagAssignment.from_dict(obj)
    if isinstance(obj, (list, tuple)) and len(obj) == 2:
        return EntityTagAssignment(
            entity_type=entity_type,
            entity_name=entity_name,
            tag_key=_safe_str(obj[0]),
            tag_value=_safe_str(obj[1]),
        )
    raise ValueError(
        f"Cannot parse EntityTagAssignment from object of type {type(obj)}: {obj!r}"
    )


# ---------------------------------------------------------------------------
# Retry policy
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Tunable retry strategy for entity-tag API calls.

    Defaults are tuned for the public Databricks REST API: a handful of
    attempts, sub-second initial backoff with full jitter, capped delay,
    and an absolute deadline so a long stall can't run forever.

    The policy honors ``Retry-After`` hints — when a :class:`TooManyRequests`
    error carries ``retry_after_secs``, that value (clamped to ``max_delay``)
    is used for the next sleep instead of the computed exponential delay.

    Set ``attempts=1`` to disable retries entirely (useful for tests).
    """

    attempts: int = 5
    """Total attempts including the first call. ``1`` = no retries."""

    base_delay: float = 0.25
    """Initial delay in seconds before the second attempt."""

    max_delay: float = 8.0
    """Cap on a single sleep; also caps server ``Retry-After`` hints."""

    multiplier: float = 2.0
    """Per-attempt geometric multiplier on the delay."""

    jitter: float = 0.5
    """Full-jitter factor in ``[0, 1]``: 0 = deterministic, 1 = full random."""

    deadline: float | None = 30.0
    """Absolute wall-clock deadline (seconds). ``None`` disables it."""

    retry_on: tuple[type[BaseException], ...] = field(
        default_factory=lambda: _RETRYABLE_ERRORS,
    )
    """Exception types eligible for retry. Anything else propagates."""

    def compute_delay(self, attempt: int, *, retry_after: float | None = None) -> float:
        """Delay before *attempt* (1-indexed). ``retry_after`` overrides when set."""
        if retry_after is not None and retry_after >= 0:
            return min(float(retry_after), self.max_delay)
        # Geometric: base * multiplier^(attempt-1), capped at max_delay.
        raw = self.base_delay * (self.multiplier ** max(0, attempt - 1))
        capped = min(raw, self.max_delay)
        if self.jitter <= 0:
            return capped
        # "Full jitter" from the AWS Architecture Blog: pick uniformly in
        # [(1-jitter)*capped, capped] — preserves average backoff while
        # spreading concurrent retriers across the window.
        floor = capped * (1.0 - min(self.jitter, 1.0))
        return random.uniform(floor, capped)


# Sensible default — used when callers don't override.
_DEFAULT_RETRY_POLICY = RetryPolicy()


def _retry_after_secs(exc: BaseException) -> float | None:
    """Pull ``Retry-After`` hint from an SDK error, if present."""
    value = getattr(exc, "retry_after_secs", None)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _retry_call(
    fn: Callable[[], T],
    *,
    policy: RetryPolicy,
    op: str = "call",
) -> T:
    """Call *fn* with retry-on-transient-errors per *policy*.

    Logs each retry at WARNING (so operators see flakiness), the final
    exhaustion at ERROR. Raises the last exception when attempts run out
    or the deadline expires; raises immediately on fatal/unknown errors.
    """
    started = time.monotonic()
    last_exc: BaseException | None = None

    for attempt in range(1, max(1, policy.attempts) + 1):
        try:
            return fn()
        except _FATAL_ERRORS:
            # Bad input / auth — never retry; let the caller see the real error.
            raise
        except policy.retry_on as exc:
            last_exc = exc
            if attempt >= policy.attempts:
                break
            # Compute next sleep, honoring server hint when available.
            hint = _retry_after_secs(exc)
            delay = policy.compute_delay(attempt, retry_after=hint)
            # Respect deadline: don't start a sleep that would exceed it.
            if policy.deadline is not None:
                elapsed = time.monotonic() - started
                remaining = policy.deadline - elapsed
                if remaining <= 0:
                    logger.error(
                        "[%s] retry deadline exceeded after %.2fs (%d attempts)",
                        op, elapsed, attempt,
                    )
                    break
                delay = min(delay, remaining)
            logger.warning(
                "[%s] attempt %d/%d failed (%s: %s); sleeping %.2fs%s",
                op, attempt, policy.attempts,
                type(exc).__name__, exc, delay,
                " [Retry-After]" if hint is not None else "",
            )
            time.sleep(delay)

    assert last_exc is not None  # only reached when a retry path failed
    logger.error(
        "[%s] giving up after %d attempt(s): %s: %s",
        op, policy.attempts, type(last_exc).__name__, last_exc,
    )
    raise last_exc


@dataclass
class EntityTags(DatabricksService):
    """Collection-level service for Unity Catalog entity tag assignments."""

    retry_policy: RetryPolicy = field(default_factory=lambda: _DEFAULT_RETRY_POLICY)
    """Retry policy for every workspace-client call. Defaults to
    :data:`_DEFAULT_RETRY_POLICY` (5 attempts, 0.25–8s backoff with jitter,
    30s deadline, ``Retry-After`` honored). Override per-instance:

        >>> client.entity_tags.retry_policy = RetryPolicy(attempts=1)
    """

    def _call(self, fn: Callable[[], T], *, op: str) -> T:
        """Run *fn* under :attr:`retry_policy`. Single funnel for SDK calls."""
        return _retry_call(fn, policy=self.retry_policy, op=op)

    # -------------------------------------------------------------------------
    # Cache helpers
    # -------------------------------------------------------------------------

    def _cache_key(self, entity_type: str, entity_name: str) -> str:
        """Build a stable, host-scoped cache key."""
        host = self.client.base_url.to_string() if self.client else "default"
        return f"{host}|{entity_type}|{entity_name}"

    def invalidate_cached_tags(
        self,
        entity_type: str,
        entity_name: str,
    ) -> None:
        """Evict one entity's tag list from the module-level cache."""
        key = self._cache_key(entity_type, entity_name)
        try:
            del _TAGS_CACHE[key]
        except KeyError:
            pass

    @classmethod
    def invalidate_all(cls) -> None:
        """Clear the entire module-level entity-tag cache."""
        _TAGS_CACHE.clear()

    def _patch_cached_tag(
        self,
        entity_type: str,
        entity_name: str,
        assignment: EntityTagAssignment,
    ) -> None:
        """Replace-or-append a single tag in the cached list, preserving TTL semantics.

        Cheaper than full invalidation when we *know* the new value: keeps the
        next read off the network. Falls back to invalidation if no entry is
        cached (nothing to patch).
        """
        key = self._cache_key(entity_type, entity_name)
        cached = _TAGS_CACHE.get(key)
        if cached is None:
            return  # nothing to patch — next read will populate

        replaced = False
        new_list: list[EntityTagAssignment] = []
        for existing in cached:
            if existing.tag_key == assignment.tag_key:
                new_list.append(assignment)
                replaced = True
            else:
                new_list.append(existing)
        if not replaced:
            new_list.append(assignment)

        _TAGS_CACHE.set(key, new_list)

    def _pop_cached_tag(
        self,
        entity_type: str,
        entity_name: str,
        tag_key: str,
    ) -> None:
        """Drop a single tag from the cached list, mirror of :meth:`_patch_cached_tag`.

        No-op when nothing is cached (next read will populate from server).
        """
        key = self._cache_key(entity_type, entity_name)
        cached = _TAGS_CACHE.get(key)
        if cached is None:
            return

        new_list = [t for t in cached if t.tag_key != tag_key]
        if len(new_list) == len(cached):
            return  # nothing to remove
        _TAGS_CACHE.set(key, new_list)

    # -------------------------------------------------------------------------
    # Read API
    # -------------------------------------------------------------------------

    def entity_tag_key(
        self,
        entity_type: str,
        entity_name: str,
        tag_key: str,
    ) -> Optional[EntityTagAssignment]:
        """Get a single tag by key, served from cache when populated."""
        cached = _TAGS_CACHE.get(self._cache_key(entity_type, entity_name))
        if cached is not None:
            for tag in cached:
                if tag.tag_key == tag_key:
                    return tag
            # Cache is authoritative for the full list — a miss means absent.
            # Fall through to a direct GET only as a defensive courtesy.

        client = self.client.workspace_client().entity_tag_assignments
        return self._call(
            lambda: client.get(
                entity_type=entity_type,
                entity_name=entity_name,
                tag_key=tag_key,
            ),
            op=f"get {entity_type}:{entity_name}#{tag_key}",
        )

    def entity_tags(
        self,
        entity_type: str,
        entity_name: str,
        as_dict: bool = False,
        default: Any = ...,
        *,
        cache_ttl: float | None = 300.0,
    ):
        """List every tag assigned to *entity*.

        Args:
            entity_type:  Unity Catalog entity type.
            entity_name:  Fully qualified entity name.
            as_dict:      Return ``{key: value}`` instead of a list.
            default:      Returned on error if not ``...``; otherwise raise.
            cache_ttl:    Entry TTL in seconds (``None`` → bypass cache).
        """
        cache_key = self._cache_key(entity_type, entity_name)

        # 1. Local cache --------------------------------------------------
        if cache_ttl is not None:
            cached = _TAGS_CACHE.get(cache_key)
            if cached is not None:
                logger.debug(
                    "Cache hit [EntityTags.entity_tags] key=%s n=%d",
                    cache_key, len(cached),
                )
                return (
                    {t.tag_key: t.tag_value for t in cached}
                    if as_dict else list(cached)
                )

        # 2. Remote fetch -------------------------------------------------
        client = self.client.workspace_client().entity_tag_assignments
        try:
            result = self._call(
                lambda: list(client.list(
                    entity_type=entity_type,
                    entity_name=entity_name,
                )),
                op=f"list {entity_type}:{entity_name}",
            )
        except Exception:
            if default is ...:
                raise
            return default

        # 3. Update cache -------------------------------------------------
        if cache_ttl is not None:
            _TAGS_CACHE.set(cache_key, result, ttl=cache_ttl)

        if as_dict:
            return {tag.tag_key: tag.tag_value for tag in result}
        return result

    # -------------------------------------------------------------------------
    # Write API
    # -------------------------------------------------------------------------

    def update_entity_tag(
        self,
        assignment: EntityTagAssignment | None = None,
        *,
        entity_type: str | None = None,
        entity_name: str | None = None,
        key: str | None = None,
        value: str | None = None,
        source_type: TagAssignmentSourceType | None = None,
        update_mask: str | None = None,
        is_new: bool | None = None,
    ) -> Optional[EntityTagAssignment]:
        """Apply a single tag assignment.

        Routes to :meth:`EntityTagAssignmentsAPI.create` for new tags and
        :meth:`update` for existing ones. Falls back automatically when
        the wrong choice is made (``NotFound`` on update → retry with
        create; ``AlreadyExists`` on create → retry with update).

        Args:
            is_new: When known by the caller (e.g. from a diff against the
                cached tag list), short-circuits the create-vs-update
                decision and skips the fallback round-trip. Leave ``None``
                to let the cache decide.
        """
        if assignment is None and (entity_type and entity_name and key is not None):
            assignment = EntityTagAssignment(
                entity_type=entity_type,
                entity_name=entity_name,
                tag_key=_safe_str(key),
                tag_value=_safe_str(value),
                source_type=source_type or _DEFAULT_SOURCE_TYPE,
            )

        if assignment is None:
            return None

        # Resolve the addressing fields from the assignment so the cache
        # patch is correct regardless of how the caller passed them in.
        et = entity_type or assignment.entity_type
        en = entity_name or assignment.entity_name
        tk = key or assignment.tag_key

        client = self.client.workspace_client().entity_tag_assignments

        # Decide create vs. update. ``is_new`` from the caller (the batch
        # path passes this) wins; otherwise consult the cache.
        if is_new is None:
            cached = _TAGS_CACHE.get(self._cache_key(et, en))
            if cached is not None:
                is_new = not any(t.tag_key == tk for t in cached)
            # else: leave as None → try update first (covers warm-up case
            # where we never populated the cache).

        mask = _build_update_mask(update_mask, source_type=assignment.source_type)

        def _do_create() -> EntityTagAssignment:
            return client.create(tag_assignment=assignment)

        def _do_update() -> EntityTagAssignment:
            return client.update(
                entity_type=et,
                entity_name=en,
                tag_key=tk,
                tag_assignment=assignment,
                update_mask=mask,
            )

        # Primary call.
        try:
            if is_new is True:
                result = self._call(_do_create, op=f"create {et}:{en}#{tk}")
            else:
                result = self._call(_do_update, op=f"update {et}:{en}#{tk}")
        except NotFound:
            # We thought it existed, but the server says no — create it.
            result = self._call(_do_create, op=f"create {et}:{en}#{tk}")
        except DatabricksError as exc:
            # Convert "already exists" on create to an update.
            if is_new is True and "already exists" in str(exc).lower():
                result = self._call(_do_update, op=f"update {et}:{en}#{tk}")
            else:
                raise

        # Surgical cache update — patch the one tag rather than invalidating
        # the whole list, when we have one cached.
        self._patch_cached_tag(et, en, assignment)
        return result

    def update_entity_tags(
        self,
        tags: list[EntityTagAssignment] | Mapping | None = None,
        *,
        assignments: list[EntityTagAssignment] | None = None,
        entity_type: str | None = None,
        entity_name: str | None = None,
        mode: ModeLike | None = None,
        parallel: int | bool | None = None,
        cache_ttl: float | None = 300.0,
    ) -> None:
        """Apply a batch of tag assignments, skipping no-op writes.

        Uses the module-level cache for the pre-write diff: when the entity's
        tag list is already cached we skip the ``list()`` call entirely.

        Mode semantics
        --------------
        ``UPSERT`` (the default, also reached via ``AUTO``)
            Add new tags, update existing tags whose value changed, leave
            tags that aren't mentioned in *tags* untouched.
        ``OVERWRITE``
            Reconcile the entity's tags to *exactly* the input set:
            UPSERT-then-DELETE every existing tag whose key isn't in the
            batch. Strict replace.
        ``APPEND``
            Add new tags only — never overwrite an existing value, never
            delete anything. Useful when concurrent writers each own a
            distinct subset of keys.
        ``IGNORE``
            Add only tags whose key isn't already assigned; existing keys
            are left as-is even if the value differs.
        ``ERROR_IF_EXISTS``
            Raise :class:`ValueError` if any input key is already assigned
            on the entity; otherwise behave like APPEND.
        """
        if assignments is None:
            tags_iter: Iterable
            if isinstance(tags, Mapping):
                tags_iter = tags.items()
            else:
                tags_iter = tags or ()
            assignments = [
                _safe_assignment(entity_type, entity_name, t)
                for t in tags_iter
                if t
            ]

        # OVERWRITE with no inputs is meaningful (delete everything);
        # the other modes are no-ops on an empty batch.
        mode = Mode.from_(mode, default=Mode.UPSERT)
        if mode == Mode.AUTO:
            mode = Mode.UPSERT

        if not assignments and mode != Mode.OVERWRITE:
            return None

        # Normalise ``parallel`` (preserving the original bool→4 convention).
        if isinstance(parallel, bool):
            parallel = 4 if parallel else 1
        else:
            parallel = int(parallel) if parallel is not None else 1

        # Resolve addressing — every assignment in a batch belongs to the
        # same entity; take it from the first one if not provided.
        if assignments:
            et = entity_type or assignments[0].entity_type
            en = entity_name or assignments[0].entity_name
        else:
            et, en = entity_type, entity_name
        if not (et and en):
            raise ValueError(
                "update_entity_tags requires entity_type and entity_name"
                " (either as kwargs or carried by the assignments)."
            )

        # Diff against existing tags (cached when possible).
        existing = self.entity_tags(et, en, default=[], cache_ttl=cache_ttl)
        existing_map = {a.tag_key: a for a in existing}
        input_keys = {a.tag_key for a in assignments}

        # Decide which assignments to write, by mode. Each is paired with
        # an ``is_new`` flag so update_entity_tag can route to create vs.
        # update without re-checking the cache (or paying a NotFound
        # round-trip on the wrong branch).
        if mode == Mode.APPEND:
            to_write = [(a, True) for a in assignments if a.tag_key not in existing_map]
        elif mode == Mode.IGNORE:
            to_write = [(a, True) for a in assignments if a.tag_key not in existing_map]
        elif mode == Mode.ERROR_IF_EXISTS:
            collisions = [a.tag_key for a in assignments if a.tag_key in existing_map]
            if collisions:
                raise ValueError(
                    f"Tag key(s) already assigned on {et}:{en}: "
                    f"{sorted(collisions)}"
                )
            to_write = [(a, True) for a in assignments]
        else:
            # UPSERT and OVERWRITE both add-or-update; OVERWRITE then
            # additionally deletes the leftovers below.
            to_write = [
                (a, a.tag_key not in existing_map)
                for a in assignments
                if (existing_map.get(a.tag_key) is None
                    or existing_map[a.tag_key].tag_value != a.tag_value)
            ]

        # Strict-replace: keys present on the server but not in the batch
        # get dropped. Computed here so it stays atomic with the diff above
        # (same ``existing`` snapshot for both halves).
        to_delete: list[str] = []
        if mode == Mode.OVERWRITE:
            to_delete = [k for k in existing_map.keys() if k not in input_keys]

        if not to_write and not to_delete:
            return None

        def _write_one(item: tuple[EntityTagAssignment, bool]) -> None:
            assignment, is_new = item
            self.update_entity_tag(assignment, is_new=is_new)

        # ----- writes ----------------------------------------------------
        if to_write:
            if parallel > 1:
                with ThreadPoolExecutor(max_workers=parallel) as executor:
                    # Drain to surface exceptions deterministically.
                    list(executor.map(_write_one, to_write))
            else:
                for item in to_write:
                    _write_one(item)

        # ----- deletes (OVERWRITE only) ---------------------------------
        if to_delete:
            # Hand off to delete_entity_tags so cache patches and the same
            # parallelism convention apply uniformly.
            self.delete_entity_tags(
                tag_keys=to_delete,
                entity_type=et,
                entity_name=en,
                if_exists=True,
                parallel=parallel,
                cache_ttl=cache_ttl,
            )

        # ``update_entity_tag`` and ``delete_entity_tags`` already patched
        # the cache for each successful write/delete; nothing more to do.
        return None

    def update_entities_tags(
        self,
        tags_by_entity: (
            Mapping[tuple[str, str], list[EntityTagAssignment] | Mapping]
            | Iterable[EntityTagAssignment]
            | None
        ) = None,
        *,
        mode: ModeLike | None = None,
        parallel_entities: int | bool | None = None,
        parallel_per_entity: int | bool | None = None,
        cache_ttl: float | None = 300.0,
        continue_on_error: bool = True,
    ) -> dict[tuple[str, str], BaseException | None]:
        """Apply tag batches to many entities in parallel.

        Multi-entity counterpart of :meth:`update_entity_tags`. Each entity's
        batch is dispatched to ``update_entity_tags`` with the same *mode*
        and *cache_ttl*; entities are processed concurrently up to
        *parallel_entities*.

        Input shapes
        ------------
        Either:
          * ``Mapping[(entity_type, entity_name), tags]`` — explicit grouping;
            ``tags`` may be a list of :class:`EntityTagAssignment`, a
            ``{key: value}`` mapping, or any shape :func:`_safe_assignment`
            accepts.
          * ``Iterable[EntityTagAssignment]`` — flat stream; grouped here by
            ``(entity_type, entity_name)`` carried on each assignment.

        Parallelism
        -----------
        *parallel_entities* sizes the outer pool (entities run concurrently);
        *parallel_per_entity* is forwarded to each ``update_entity_tags``
        call. Defaults: 4 entities at a time, 1 write at a time within an
        entity — keeps the total worker budget bounded and matches the
        original ``parallel: bool → 4`` convention.

        Errors
        ------
        With ``continue_on_error=True`` (default), per-entity failures are
        captured in the returned mapping rather than aborting the whole run;
        callers inspect the result to surface partial success. With
        ``continue_on_error=False``, the first exception propagates after
        cancelling pending entities.

        Returns
        -------
        A ``{(entity_type, entity_name): None | BaseException}`` mapping with
        one entry per input entity. ``None`` denotes success.
        """
        # ---- normalize input into a {(et, en): [assignments]} dict --------
        grouped: dict[tuple[str, str], list[EntityTagAssignment]] = {}

        if tags_by_entity is None:
            return {}

        if isinstance(tags_by_entity, Mapping):
            for (et, en), batch in tags_by_entity.items():
                if not (et and en):
                    raise ValueError(
                        "update_entities_tags: mapping keys must be "
                        "(entity_type, entity_name) tuples with both set."
                    )
                if isinstance(batch, Mapping):
                    items = batch.items()
                else:
                    items = batch or ()
                grouped[(et, en)] = [
                    _safe_assignment(et, en, t) for t in items if t
                ]
        else:
            # Flat iterable of EntityTagAssignment — group by (et, en).
            for raw in tags_by_entity:
                if raw is None:
                    continue
                assignment = (
                    raw if isinstance(raw, EntityTagAssignment)
                    else _safe_assignment("", "", raw)
                )
                et, en = assignment.entity_type, assignment.entity_name
                if not (et and en):
                    raise ValueError(
                        "update_entities_tags: flat-iterable input requires "
                        "entity_type and entity_name on every assignment; "
                        f"got {assignment!r}."
                    )
                grouped.setdefault((et, en), []).append(assignment)

        # OVERWRITE with an empty batch is meaningful (clear all tags); other
        # modes drop empty entries to avoid a wasted list() round-trip.
        resolved_mode = Mode.from_(mode, default=Mode.UPSERT)
        if resolved_mode == Mode.AUTO:
            resolved_mode = Mode.UPSERT
        if resolved_mode != Mode.OVERWRITE:
            grouped = {k: v for k, v in grouped.items() if v}

        if not grouped:
            return {}

        # ---- normalize parallelism (same convention as the single-entity API)
        if isinstance(parallel_entities, bool):
            parallel_entities = 4 if parallel_entities else 1
        else:
            parallel_entities = int(parallel_entities) if parallel_entities is not None else 4

        # parallel_per_entity is passed through; bool→4 handled by update_entity_tags.

        # ---- dispatch -----------------------------------------------------
        results: dict[tuple[str, str], BaseException | None] = {}

        def _run_one(key: tuple[str, str]) -> tuple[tuple[str, str], BaseException | None]:
            et, en = key
            try:
                self.update_entity_tags(
                    assignments=grouped[key],
                    entity_type=et,
                    entity_name=en,
                    mode=resolved_mode,
                    parallel=parallel_per_entity,
                    cache_ttl=cache_ttl,
                )
                return key, None
            except BaseException as exc:  # noqa: BLE001 — captured for caller
                if not continue_on_error:
                    raise
                logger.exception(
                    "[update_entities_tags] %s:%s failed: %s",
                    et, en, exc,
                )
                return key, exc

        keys = list(grouped.keys())

        if parallel_entities > 1 and len(keys) > 1:
            with ThreadPoolExecutor(max_workers=parallel_entities) as executor:
                for key, err in executor.map(_run_one, keys):
                    results[key] = err
        else:
            for key in keys:
                k, err = _run_one(key)
                results[k] = err

        return results

    # -------------------------------------------------------------------------
    # Delete API
    # -------------------------------------------------------------------------

    def delete_entity_tag(
        self,
        entity_type: str,
        entity_name: str,
        tag_key: str,
        *,
        if_exists: bool = True,
    ) -> bool:
        """Delete a single tag assignment.

        Returns ``True`` when the server confirmed the deletion (or the tag
        already didn't exist with ``if_exists=True``), ``False`` when the
        call was suppressed by ``if_exists`` after a :class:`NotFound`.
        """
        client = self.client.workspace_client().entity_tag_assignments
        try:
            self._call(
                lambda: client.delete(
                    entity_type=entity_type,
                    entity_name=entity_name,
                    tag_key=tag_key,
                ),
                op=f"delete {entity_type}:{entity_name}#{tag_key}",
            )
        except NotFound:
            if not if_exists:
                raise
            # Still drop from cache — if we had a stale entry claiming the
            # tag exists, the server-side reality is "absent".
            self._pop_cached_tag(entity_type, entity_name, tag_key)
            return False

        self._pop_cached_tag(entity_type, entity_name, tag_key)
        return True

    def delete_entity_tags(
        self,
        tag_keys: Iterable[str] | None = None,
        *,
        entity_type: str | None = None,
        entity_name: str | None = None,
        if_exists: bool = True,
        parallel: int | bool | None = None,
        cache_ttl: float | None = 300.0,
    ) -> int:
        """Delete a batch of tag assignments by key.

        Skips no-op deletes by checking the (cached) tag list first: keys
        that aren't currently assigned never reach the API. Returns the
        number of tags actually deleted server-side.
        """
        if not entity_type or not entity_name:
            raise ValueError(
                "delete_entity_tags requires entity_type and entity_name"
            )

        keys = [k for k in (tag_keys or ()) if k]
        if not keys:
            return 0

        # Normalise parallelism (same convention as update_entity_tags).
        if isinstance(parallel, bool):
            parallel = 4 if parallel else 1
        else:
            parallel = int(parallel) if parallel is not None else 1

        # Diff against existing tags (cached when possible) so we don't
        # round-trip for keys that aren't there.
        existing = self.entity_tags(
            entity_type, entity_name, default=[], cache_ttl=cache_ttl,
        )
        existing_keys = {a.tag_key for a in existing}
        # When the entity has any cached/known tags, restrict to those that
        # actually exist; otherwise fall through and let the server decide
        # (with ``if_exists`` swallowing 404s).
        if existing_keys:
            keys = [k for k in keys if k in existing_keys]
            if not keys:
                return 0

        def _delete_one(k: str) -> bool:
            return self.delete_entity_tag(
                entity_type=entity_type,
                entity_name=entity_name,
                tag_key=k,
                if_exists=if_exists,
            )

        if parallel > 1:
            with ThreadPoolExecutor(max_workers=parallel) as executor:
                results = list(executor.map(_delete_one, keys))
        else:
            results = [_delete_one(k) for k in keys]

        return sum(1 for r in results if r)