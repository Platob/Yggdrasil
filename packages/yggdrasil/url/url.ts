// Client-side port of ``yggdrasil.url.URL``.
//
// PARITY: python/src/yggdrasil/url/url.py. An immutable URL value type — the
// identity layer ``path`` and ``io`` build on. Parsing mirrors Python's
// ``urlsplit`` + the from_str fixups (windows drives, authority-less schemes,
// schemeless relative paths). Cross-checked against the Python reference
// (see url.test.ts parity fixtures). Keep the two in sync.

import { MediaType, MimeTypes } from "../enums";

const SCHEME_RE = /^([a-zA-Z][a-zA-Z0-9+.-]*):/;

export interface URLParts {
  scheme: string;
  userinfo: string | null;
  host: string;
  port: number | null;
  path: string;
  query: string | null;
  fragment: string | null;
}

export class URL {
  readonly scheme: string;
  readonly userinfo: string | null;
  readonly host: string;
  readonly port: number | null;
  readonly path: string;
  readonly query: string | null;
  readonly fragment: string | null;

  constructor(p: Partial<URLParts> = {}) {
    this.scheme = p.scheme ?? "";
    this.userinfo = p.userinfo ?? null;
    this.host = p.host ?? "";
    this.port = p.port === 0 ? null : (p.port ?? null);
    this.path = p.path ?? "/";
    this.query = p.query ?? null;
    this.fragment = p.fragment ?? null;
  }

  // -- factories --------------------------------------------------------------

  /** Coerce a URL / string / {url} into a URL (mirrors ``URL.from_``). */
  static from(obj: unknown): URL {
    if (obj instanceof URL) return obj;
    if (typeof obj === "string") return URL.fromStr(obj);
    if (obj && typeof (obj as { url?: unknown }).url !== "undefined") return URL.from((obj as { url: unknown }).url);
    return URL.fromStr(String(obj));
  }

  /** Parse a raw URL/path string (mirrors ``URL.from_str`` + fixups). */
  static fromStr(raw: string): URL {
    // fragment, then query
    let fragment: string | null = null;
    const hashIdx = raw.indexOf("#");
    if (hashIdx >= 0) { fragment = raw.slice(hashIdx + 1).replace(/&amp;/g, "&"); raw = raw.slice(0, hashIdx); }
    let query: string | null = null;
    const qIdx = raw.indexOf("?");
    if (qIdx >= 0) { query = raw.slice(qIdx + 1).replace(/&amp;/g, "&"); raw = raw.slice(0, qIdx); }

    // scheme
    let scheme = "";
    const m = SCHEME_RE.exec(raw);
    if (m) { scheme = m[1]; raw = raw.slice(m[0].length); }

    // single-letter scheme == windows drive (C:\path / C:/path)
    if (scheme.length === 1 && /[a-zA-Z]/.test(scheme)) {
      const drive = scheme.toUpperCase();
      let rest = raw.replace(/\\/g, "/");
      if (!rest.startsWith("/")) rest = "/" + rest;
      return new URL({ scheme: "file", path: `/${drive}:${rest}`, query, fragment });
    }

    let userinfo: string | null = null;
    let host = "";
    let port: number | null = null;
    let path: string;

    if (raw.startsWith("//")) {
      const rest = raw.slice(2);
      const slash = rest.indexOf("/");
      let netloc = slash >= 0 ? rest.slice(0, slash) : rest;
      path = slash >= 0 ? rest.slice(slash) : "";
      const at = netloc.lastIndexOf("@");
      if (at >= 0) { userinfo = netloc.slice(0, at); netloc = netloc.slice(at + 1); }
      const colon = netloc.lastIndexOf(":");
      if (colon >= 0 && /^\d+$/.test(netloc.slice(colon + 1))) {
        host = netloc.slice(0, colon);
        port = parseInt(netloc.slice(colon + 1), 10);
      } else {
        host = netloc;
      }
    } else {
      path = raw;
    }

    // authority-less scheme (``http:example.com/path``) — pull host off the path.
    if (scheme && scheme !== "file" && !host && path) {
      if (path.includes("/")) {
        const i = path.indexOf("/");
        host = path.slice(0, i);
        path = "/" + path.slice(i + 1).replace(/^\/+/, "");
      } else {
        host = path;
        path = "/";
      }
    }

    if (!path) path = "/";
    return new URL({ scheme, userinfo: userinfo || null, host, port: port || null, path, query, fragment });
  }

  static isPathish(obj: unknown): boolean {
    if (obj == null) return false;
    if (obj instanceof URL || typeof obj === "string") return true;
    return typeof (obj as { url?: unknown }).url !== "undefined";
  }

  // -- derived path views -----------------------------------------------------

  get parts(): string[] {
    if (!this.path || this.path === "/") return [];
    return this.path.replace(/^\/+/, "").split("/");
  }

  /** Path extensions, outer-to-inner, lowercased (``a.csv.zst`` -> [csv, zst]). */
  get extensions(): string[] {
    let path = this.path;
    if (!path || path === "/") return [];
    if (path.endsWith("/")) path = path.slice(0, -1);
    if (!path) return [];
    const idx = path.lastIndexOf("/");
    const name = idx !== -1 ? path.slice(idx + 1) : path;
    if (!name) return [];
    if (name[0] === "." && !name.slice(1).includes(".")) return []; // dotfile, no ext
    if (name[0] === ".") return name.slice(1).split(".").slice(1).map((s) => s.toLowerCase());
    if (!name.includes(".")) return [];
    return name.split(".").slice(1).map((s) => s.toLowerCase());
  }

  /** Last path segment, trailing slash stripped. */
  get name(): string {
    let path = this.path;
    if (!path || path === "/") return "";
    if (path.endsWith("/")) { path = path.slice(0, -1); if (!path) return ""; }
    const idx = path.lastIndexOf("/");
    return idx !== -1 ? path.slice(idx + 1) : path;
  }

  /** :attr:`name` with only its final extension removed (pathlib stem). */
  get stem(): string {
    const n = this.name;
    if (!n) return "";
    if (n[0] === "." && !n.slice(1).includes(".")) return n; // dotfile
    const idx = n.lastIndexOf(".");
    return idx > 0 ? n.slice(0, idx) : n;
  }

  /** The URL one segment up; scheme/authority/query/fragment preserved. */
  get parent(): URL {
    let p = this.path;
    if (p === "/" || p === "") return this;
    if (p.endsWith("/")) p = p.slice(0, -1);
    const idx = p.lastIndexOf("/");
    const parentPath = idx < 0 ? "" : idx === 0 ? "/" : p.slice(0, idx);
    return this.with({ path: parentPath || "/" });
  }

  /** Codec-aware media type of the URL (directory for ``/``). */
  get mediaType(): MediaType | null {
    if (this.path === "/" || this.path === "") return new MediaType(MimeTypes.DIRECTORY);
    return MediaType.from(this.name);
  }

  inferMediaType(dflt: MediaType | null = null): MediaType | null {
    return this.mediaType ?? dflt;
  }

  get isAbsolute(): boolean {
    return this.scheme !== "" || this.host !== "" || this.path.startsWith("/");
  }

  // -- transforms (immutable) -------------------------------------------------

  with(parts: Partial<URLParts>): URL {
    return new URL({
      scheme: parts.scheme ?? this.scheme,
      userinfo: parts.userinfo !== undefined ? parts.userinfo : this.userinfo,
      host: parts.host ?? this.host,
      port: parts.port !== undefined ? parts.port : this.port,
      path: parts.path ?? this.path,
      query: parts.query !== undefined ? parts.query : this.query,
      fragment: parts.fragment !== undefined ? parts.fragment : this.fragment,
    });
  }

  withScheme(scheme: string | null): URL { return this.with({ scheme: scheme ?? "" }); }
  withHost(host: string | null): URL { return this.with({ host: host ?? "" }); }
  withPath(path: string): URL { return this.with({ path: path || "/" }); }
  withQuery(query: string | null): URL { return this.with({ query }); }
  withFragment(fragment: string | null): URL { return this.with({ fragment }); }

  /** Append path segments (``url.joinpath("a", "b/c")``). */
  joinpath(...segments: (string | number)[]): URL {
    const segs = segments.flatMap((s) => String(s).split("/")).filter(Boolean);
    const base = this.path === "/" ? "" : this.path.replace(/\/+$/, "");
    return this.withPath(`${base}/${segs.join("/")}`);
  }

  // -- serialization ----------------------------------------------------------

  toString(): string {
    let s = "";
    if (this.scheme) s += this.scheme + ":";
    if (this.host !== "" || this.userinfo != null || this.scheme === "file") {
      s += "//";
      if (this.userinfo != null) s += this.userinfo + "@";
      s += this.host;
      if (this.port != null) s += ":" + this.port;
    }
    s += this.path;
    if (this.query != null) s += "?" + this.query;
    if (this.fragment != null) s += "#" + this.fragment;
    return s;
  }

  equals(other: unknown): boolean {
    if (other === this) return true;
    if (other instanceof URL) {
      return this.scheme === other.scheme && this.path === other.path && this.host === other.host
        && this.port === other.port && this.query === other.query && this.fragment === other.fragment
        && this.userinfo === other.userinfo;
    }
    if (typeof other === "string") return this.toString() === other;
    return false;
  }
}
