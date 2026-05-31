// yggdrasil — a JS/TS replication of the Python ``yggdrasil`` package for
// client-side operations. Mirrors the Python module layout so the two stay a
// single, uniform contract (see ./README.md and the repo CLAUDE.md parity
// rule). Implemented incrementally; start with ``enums``.

export * as enums from "./enums";
export { MimeType, MimeTypes, MediaType, type Magic, type MimeOpts } from "./enums";
