/**
 * Immutable URL value object with convenient parsing and manipulation.
 */
export class URL_ {
  readonly scheme: string;
  readonly host: string;
  readonly port: number | null;
  readonly path: string;
  readonly query: ReadonlyMap<string, string>;
  readonly fragment: string | null;

  private constructor(opts: { scheme: string; host: string; port?: number | null; path?: string; query?: Map<string, string>; fragment?: string | null }) {
    this.scheme = opts.scheme;
    this.host = opts.host;
    this.port = opts.port ?? null;
    this.path = opts.path || "/";
    this.query = opts.query ?? new Map();
    this.fragment = opts.fragment ?? null;
  }

  // ── Factory methods ──
  static from_(value: string | URL | URL_ | null | undefined): URL_ | null {
    if (!value) return null;
    if (value instanceof URL_) return value;
    try {
      return URL_.parse(String(value));
    } catch {
      return null;
    }
  }

  static parse(raw: string): URL_ {
    const u = new globalThis.URL(raw);
    const query = new Map<string, string>();
    u.searchParams.forEach((v, k) => query.set(k, v));
    return new URL_({
      scheme: u.protocol.replace(":", ""),
      host: u.hostname,
      port: u.port ? parseInt(u.port) : null,
      path: u.pathname,
      query,
      fragment: u.hash ? u.hash.slice(1) : null,
    });
  }

  static of(scheme: string, host: string, port?: number, path?: string): URL_ {
    return new URL_({ scheme, host, port, path });
  }

  // ── Query helpers ──
  getParam(key: string): string | null { return this.query.get(key) ?? null; }
  hasParam(key: string): boolean { return this.query.has(key); }

  withParam(key: string, value: string): URL_ {
    const q = new Map(this.query);
    q.set(key, value);
    return new URL_({ ...this, query: q });
  }

  withoutParam(key: string): URL_ {
    const q = new Map(this.query);
    q.delete(key);
    return new URL_({ ...this, query: q });
  }

  withPath(path: string): URL_ {
    return new URL_({ scheme: this.scheme, host: this.host, port: this.port, path, query: new Map(this.query), fragment: this.fragment });
  }

  withFragment(fragment: string | null): URL_ {
    return new URL_({ scheme: this.scheme, host: this.host, port: this.port, path: this.path, query: new Map(this.query), fragment });
  }

  // ── Computed ──
  get origin(): string {
    const p = this.port ? `:${this.port}` : "";
    return `${this.scheme}://${this.host}${p}`;
  }

  get authority(): string {
    return this.port ? `${this.host}:${this.port}` : this.host;
  }

  // ── Serialization ──
  toString(): string {
    let s = `${this.scheme}://${this.host}`;
    if (this.port) s += `:${this.port}`;
    s += this.path;
    if (this.query.size > 0) {
      const params = Array.from(this.query.entries())
        .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(v)}`)
        .join("&");
      s += `?${params}`;
    }
    if (this.fragment) s += `#${this.fragment}`;
    return s;
  }

  toJSON(): string { return this.toString(); }
  equals(other: URL_): boolean { return this.toString() === other.toString(); }
}

/** Normalize a path: resolve .., remove double slashes, ensure leading / */
export function normalizePath(path: string): string {
  const parts = path.split("/").filter(Boolean);
  const resolved: string[] = [];
  for (const part of parts) {
    if (part === "..") resolved.pop();
    else if (part !== ".") resolved.push(part);
  }
  return "/" + resolved.join("/");
}

/** Join path segments safely */
export function joinPath(...segments: string[]): string {
  return normalizePath(segments.join("/"));
}
