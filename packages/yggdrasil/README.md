# yggdrasil (JS/TS)

A client-side replication of the Python **`yggdrasil`** package so the frontend
(and any JS consumer) can run the same contracts without a server round-trip.

> **⚠️ Parity rule.** This is a *port*, not independent code. The reference
> implementation is Python (`python/src/yggdrasil/`). Every concept here must
> match its Python counterpart in **name, structure, and behavior**. Change one
> side → change the other in the same commit. See the global rule in the repo
> `CLAUDE.md` ("Cross-language parity").

## Design

Object-oriented, mirroring the Python classes:

- Python `MimeType` dataclass + `MimeType.from_` classmethods ↔ TS `MimeType`
  class + `MimeType.fromName` / `fromExtension` / `fromMagic` / `get` statics.
- Python properties (`mt.is_tabular`, `mt.extension`) ↔ TS getters
  (`mt.isTabular`, `mt.extension`). Python `snake_case` ↔ TS `camelCase`.

## File map (TS ↔ Python)

| JS/TS (`packages/yggdrasil/`) | Python (`python/src/yggdrasil/`) |
|---|---|
| `enums/mimeType.ts` · `mediaType.ts` · `state.ts` · `byteUnit.ts` | `enums/mime_type.py` · `media_type.py` · `state.py` · `byteunit.py` |
| `url/url.ts` | `url/url.py` |
| `path/path.ts` | `path/path.py` (value identity) |
| `http_/request.ts` | `http_/request.py` + session |
| `data/types.ts` · `field.ts` · `schema.ts` · `options.ts` | `data/types/` · `data_field.py` · `schema.py` · `options.py` |
| `arrow/cast.ts` | `arrow/cast.py` (any-source → Arrow Table) |
| `io/tabular/base.ts` · `io/arrowIpc.ts` | `io/tabular/` · `io/arrow_ipc_file.py` (leaves live directly under `io/`) |
| `index.ts` | `__init__.py` |

Implemented: **`enums`** (MIME/media-type, State, ByteUnit), **`url`** (URL
value type — parity-tested against the Python reference), **`path`** (URL-backed
pathlib identity), **`http_`** (HTTPRequest/HTTPSession over `fetch`), **`data`**
(the `DataType`/`Field`/`Schema` type system + `CastOptions`, with `toDict`
round-tripping the *exact* Python `to_dict` wire form and `DataTypeId` integer
codes shared across languages), **`arrow`** (any-source → Arrow Table coercion),
**`io`** (the Arrow-IPC `Tabular` core under `io/tabular/` + the format leaves
directly under `io/`, e.g. `io/arrowIpc.ts`;
`apache-arrow` is a peer dependency). Tested with **vitest** (`npm test`) +
benchmarks (`npm run bench`).

> **Note — structural vs. value casting.** Apache Arrow's JS build ships no
> compute/cast kernels (unlike pyarrow), so `CastOptions` does the *structural*
> cast (project/reorder to the target schema + row limit). Value-level type
> coercion (int→float, string→date, timezone normalisation) routes through
> **polars** — the documented compute follow-on, alongside `path` IO holders
> (Local/S3/Node), `io` parquet/csv/ndjson readers, and the `http_` cache layer.

## Usage

```ts
import { MimeType, MimeTypes, MediaType, Tabular, Schema, field, Int64Type } from "@platob/yggdrasil";

MimeType.fromName("trades.csv.gz");   // { mime: CSV, codec: GZIP }
MimeTypes.PARQUET.isTabular;          // true
MediaType.from("a.csv.gz").value;     // "text/csv+application/gzip"

// One typing contract with Python: the canonical dict is byte-for-byte identical.
const s = new Schema([field("id", Int64Type()), field("name", "string")]);
s.toDict();                           // { name: "", nullable: false, dtype: { id: 102, name: "STRUCT", fields: [...] } }

// Arrow IPC stream is the cross-language wire format.
const ipc = Tabular.from([{ id: 1, name: "a" }]).toArrowIPC();
Tabular.fromArrowIPC(ipc).numRows;    // 1
```

## Publishing to npm

The package is published publicly as **`@platob/yggdrasil`**, so any JS consumer
(including the in-repo Next.js app) installs it with `npm install
@platob/yggdrasil`. The build is run by **tsup** (`npm run build` → ESM bundle +
`.d.ts` in `dist/`, with `apache-arrow` left external as a peer dependency);
`prepublishOnly` runs the tests then the build, and only `dist/` is shipped
(`files` in `package.json`).

Releases are automated by `.github/workflows/publish-yggdrasil-npm.yml` (an
`NPM_TOKEN` repo secret): push a `yggdrasil-js-v*` tag, or run the workflow
manually, and CI runs `npm publish --access public` (`--access public` is
required for the scoped name).

To cut a release: bump `version` here (`npm version patch|minor|major`), commit,
then push the matching tag.
