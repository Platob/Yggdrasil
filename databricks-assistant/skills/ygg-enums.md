# Skill: normalize fixed-vocabulary tokens via `yggdrasil.data.enums`

## When to use

The user is parsing / validating / formatting a value that belongs
to a fixed token set — a currency code, byte size (`"32 MiB"`),
MIME type, codec name, file mode, geozone, timezone, time unit,
join type — or asks "what's the right way to parse this size /
currency / timezone string?". Also use when reviewing code that
hand-rolls `int(x) * 1024 * 1024`, a `{"usd": "USD", ...}` dict, or
a regex for one of these vocabularies.

## Primary surface

```python
from yggdrasil.data.enums import (
    ByteUnit, Codec, Currency, GeoZone, JoinType, MediaType, MimeType,
    Mode, NodeType, Scheme, State, TimeUnit, TimeZone,
)
```

Every enum follows the same coercion shape:

| Method | Returns | Use for |
| --- | --- | --- |
| `Enum.from_(value)` | member or raises | "I expect this to be a member" |
| `Enum.parse(text)` | member or raises | string with aliases / case / spaces |
| `Enum.parse_size(text)` (sizes only) | bytes (int) | `"32 MiB"`, `"1.5 GB"`, `"512K"` |
| `Enum.is_valid(value)` | bool | gate logic without raising |

## Typical call sites

```python
ByteUnit.parse_size("32 MiB")           # 33554432
ByteUnit.parse_size("1.5GB")            # 1610612736

Currency.from_("usd")                   # Currency.USD
Currency.from_("United States Dollar")  # Currency.USD

TimeUnit.from_("ms")                    # TimeUnit.MILLISECOND
TimeZone.from_("UTC+02:00")             # TimeZone.from_("Europe/Athens") etc.

Mode.from_("rb")                        # Mode.READ_BINARY
Scheme.from_("https://...")             # Scheme.HTTPS

MimeType.from_("application/json")      # MimeType.APPLICATION_JSON
MediaType.from_("text/csv; charset=utf-8")
```

Routes every alias through one normalisation path so downstream
code sees one canonical token, not the dozen variants users type.

## Cross-engine mapping comes for free

Enums project to the right type per engine (`MimeType` → HTTP header
value; `Codec` → pyarrow / polars / Spark codec name; `TimeUnit` →
Arrow `TimestampType` unit; `Currency` → ISO 4217 numeric / decimals;
`TimeZone` → `pytz` / Arrow tz string). Use the projection method
rather than a parallel dict at the call site.

## Add an alias, don't fork

Missing alias or member? **Edit the enum in
`yggdrasil/data/enums/<file>.py`** and route the call site through
`Enum.from_(...)`. Don't write a sibling lookup at the call site —
the next caller wants the same alias to work.

## New fixed vocabulary?

Add a new enum module under `yggdrasil/data/enums/` matching the
`from_` / `parse*` / `is_valid` shape of the existing ones (see
`byteunit.py` for the size-parsing variant, `currency/` for the
package-style one with ISO data). Don't scatter string constants.

## Don'ts

- Don't write `int(x) * 1024 * 1024` to parse `"32 MiB"` — call
  `ByteUnit.parse_size("32 MiB")`.
- Don't build a `{"usd": "USD", "eur": "EUR", …}` dict — call
  `Currency.from_(text)`.
- Don't `.upper()` / `.lower()` / `.strip()` a token to "normalise"
  it; `Enum.from_(...)` handles case + whitespace + aliases.
- Don't branch on raw strings (`if mime == "application/json": …`)
  — branch on the enum member (`if mt is MimeType.APPLICATION_JSON`).
