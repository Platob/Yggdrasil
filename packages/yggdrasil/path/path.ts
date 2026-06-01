// Client-side port of ``yggdrasil.path.Path`` — the identity / value layer.
//
// PARITY: python/src/yggdrasil/path/path.py. The Python ``Path`` is a
// URL-backed IO holder (LocalPath / S3Path / NodePath …); this port carries the
// pathlib-style value identity (URL-backed accessors + navigation). The IO
// contract (read/write arrow batches) lands in ``../io`` and, for remote nodes,
// the node HTTP client in ``../http_`` — see those modules' headers.
//
// PARITY GAP: the Python ``Path`` also carries a stat layer the IO contract
// rides on — backend stat probing plus a *contextual* stat cache (a probe
// held for the lifetime of an open context, unpersisted on release and
// dropped on write) that collapses the burst of size/exists/is_* checks one
// read or write makes. That lives with the unported ``../io`` holder
// lifecycle; this value-only port has no acquire/release window to cache
// against, so there is nothing to mirror here yet.

import { URL } from "../url";
import type { MediaType } from "../enums";

export class Path {
  readonly url: URL;

  constructor(url: URL | string) {
    this.url = URL.from(url);
  }

  static from(obj: unknown): Path {
    return obj instanceof Path ? obj : new Path(URL.from(obj));
  }

  get parts(): string[] { return this.url.parts; }
  get name(): string { return this.url.name; }
  get stem(): string { return this.url.stem; }

  /** Final suffix with its dot (``.csv``), or ``""`` (pathlib semantics). */
  get suffix(): string {
    const e = this.url.extensions;
    return e.length ? "." + e[e.length - 1] : "";
  }

  /** All suffixes with dots (``a.csv.zst`` -> ``[".csv", ".zst"]``). */
  get suffixes(): string[] {
    return this.url.extensions.map((e) => "." + e);
  }

  get parent(): Path { return new Path(this.url.parent); }
  get mediaType(): MediaType | null { return this.url.mediaType; }
  get isAbsolute(): boolean { return this.url.isAbsolute; }

  /** Append path segments (``p.joinpath("a", "b/c")``). */
  joinpath(...segments: (string | number)[]): Path {
    return new Path(this.url.joinpath(...segments));
  }

  withName(name: string): Path {
    if (!name || name.includes("/")) throw new Error(`Invalid name ${JSON.stringify(name)}`);
    return this.parent.joinpath(name);
  }

  withSuffix(suffix: string): Path {
    if (suffix && !suffix.startsWith(".")) throw new Error(`Invalid suffix ${JSON.stringify(suffix)}: must start with '.'`);
    return this.withName(this.stem + suffix);
  }

  withStem(stem: string): Path {
    return this.withName(stem + this.suffix);
  }

  toString(): string { return this.url.toString(); }
}
