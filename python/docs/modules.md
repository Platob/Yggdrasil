# Module reference overview

This guide summarizes the Python submodules shipped with the Yggdrasil package
and how they fit together. Use it as a quick map before diving into code, the
source docstrings, or the detailed module pages under `docs/modules/` (see
`modules/README.md` for a directory-style index).

## `yggdrasil.dataclasses`

Utilities that extend the standard library `dataclasses` with runtime safety and
Arrow interoperability. See the [dataclasses module doc](modules/dataclasses/README.md)
for usage examples and helper behavior.

## `yggdrasil.libs`

Dependency-guard utilities and engine-specific type conversions. Browse the
[libs module doc](modules/libs/README.md) for runtime installation flows and
conversion helpers.

## `yggdrasil.requests`

HTTP utilities with retry support and optional Microsoft identity integration.
See the [requests module doc](modules/requests/README.md) for session defaults
and MSAL configuration tips.

## `yggdrasil.types`

Type-hint aware helpers for cross-engine schema handling. The
[types module doc](modules/types/README.md) covers hint normalization, Arrow
inference, and backend-specific casting helpers.

## `yggdrasil.databricks`

Databricks-focused helpers grouped by service area. Explore the
[Databricks module doc](modules/databricks/README.md) for remote execution
options, job config helpers, SQL utilities, and workspace setup patterns.

