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

| JS/TS (`nextjs/src/lib/yggdrasil/`) | Python (`python/src/yggdrasil/`) |
|---|---|
| `enums/mimeType.ts` | `enums/mime_type.py` |
| `enums/mediaType.ts` | `enums/media_type.py` |
| `enums/index.ts` | `enums/__init__.py` |
| `index.ts` | `__init__.py` |

Implemented so far: **`enums`** (MIME / media-type registry). More modules
follow the same one-to-one mapping.

## Usage

```ts
import { MimeType, MimeTypes, MediaType } from "@/lib/yggdrasil/enums";

MimeType.fromName("trades.csv.gz");   // { mime: CSV, codec: GZIP }
MimeTypes.PARQUET.isTabular;          // true
MediaType.from("a.csv.gz").value;     // "text/csv+application/gzip"
```

## Publishing to npm (making it public)

Today this module is imported in-repo by the Next.js app via the `@/lib/...`
path alias — it is **not** an npm package yet. npm is the public registry for
JS/TS; "publishing" means uploading a built, versioned package so anyone can
`npm install` it.

To ship it publicly, extract it into its own package directory (e.g.
`packages/yggdrasil/` at the repo root so it doesn't sit inside the Next app)
with this shape:

```
packages/yggdrasil/
  src/            # the .ts sources (this folder's contents)
  package.json
  tsconfig.json   # compiles src -> dist with .d.ts type declarations
  README.md
```

`package.json`:

```jsonc
{
  "name": "@platob/yggdrasil",        // scoped to your npm org/user; pick a free name
  "version": "0.1.0",                  // semver; bump on every publish
  "type": "module",
  "main": "./dist/index.js",
  "types": "./dist/index.d.ts",
  "exports": { ".": "./dist/index.js", "./enums": "./dist/enums/index.js" },
  "files": ["dist"],                   // only ship the build, not the sources
  "sideEffects": false,
  "scripts": { "build": "tsc -p tsconfig.json", "prepublishOnly": "npm run build" },
  "license": "MIT",
  "repository": "github:platob/yggdrasil"
}
```

`tsconfig.json`: `{"compilerOptions":{"target":"ES2020","module":"ESNext","moduleResolution":"bundler","declaration":true,"outDir":"dist","strict":true},"include":["src"]}`

Then, one-time:

```bash
npm login                       # create a free account at npmjs.com first
cd packages/yggdrasil
npm publish --access public     # --access public is required for scoped names
```

Each release after that: bump `version` (`npm version patch|minor|major`), then
`npm publish`. Consumers install with `npm install @platob/yggdrasil`.

Notes:
- The package name must be globally unique on npm; a **scoped** name
  (`@your-org/yggdrasil`) avoids collisions and is free for public packages.
- Ship `dist/` (compiled JS + `.d.ts`), not the `.ts` sources — `files` and
  `.npmignore` control what's included.
- For automated releases, run `npm publish` from CI (GitHub Actions) with an
  `NPM_TOKEN` secret instead of publishing from a laptop.
- After extraction, point the Next app at it (workspace dep or
  `npm install @platob/yggdrasil`) and keep the **parity rule** above.
