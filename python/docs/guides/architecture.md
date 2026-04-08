# Architecture Guide

Yggdrasil revolves around a converter registry in `yggdrasil.data.cast.registry`.

## Dispatch flow

Converter lookup follows this strategy:

1. Exact source/target type match.
2. Identity conversion.
3. Any-wildcard converters.
4. MRO fallback.
5. One-hop converter composition.

## Cast options

`CastOptions` is threaded through conversion paths and is the canonical way to control:

- target schema/field,
- strictness,
- coercion behavior.

Use `CastOptions.check_arg()` in custom helpers to normalize user input.

## Optional dependencies

All optional libraries should be imported through each module's `lib.py` guard pattern to keep base installs functional.
