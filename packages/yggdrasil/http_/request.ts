// Client-side port of ``yggdrasil.http_`` (request/session core).
//
// PARITY: python/src/yggdrasil/http_/request.py + session. The Python class
// carries cache/hash/anonymize machinery; this port mirrors the *value-type*
// core (method + URL + headers + body) and a fetch-based ``send`` / ``sendMany``
// — the browser/node-native transport. Caching, identity hashes and the
// Response Arrow projection are PARITY follow-ons.

import { URL } from "../url";

export type HTTPHeaders = Record<string, string>;

export interface HTTPResponse {
  status: number;
  headers: HTTPHeaders;
  body: Uint8Array;
  url: string;
  ok: boolean;
}

export interface RequestInitLike {
  headers?: HTTPHeaders;
  body?: Uint8Array | string | null;
}

export class HTTPRequest {
  readonly method: string;
  readonly url: URL;
  readonly headers: HTTPHeaders;
  readonly body: Uint8Array | string | null;

  constructor(opts: { method?: string; url: URL | string } & RequestInitLike) {
    this.method = (opts.method ?? "GET").toUpperCase();
    this.url = URL.from(opts.url);
    this.headers = opts.headers ?? {};
    this.body = opts.body ?? null;
  }

  static prepare(method: string, url: URL | string, init: RequestInitLike = {}): HTTPRequest {
    return new HTTPRequest({ method, url, ...init });
  }

  get isHttp(): boolean {
    return this.url.scheme === "http" || this.url.scheme === "https";
  }

  /** Execute via the platform ``fetch``. Returns the raw bytes + headers. */
  async send(signal?: AbortSignal): Promise<HTTPResponse> {
    const res = await fetch(this.url.toString(), {
      method: this.method,
      headers: this.headers,
      body: this.body as BodyInit | null,
      signal,
    });
    const headers: HTTPHeaders = {};
    res.headers.forEach((v, k) => { headers[k] = v; });
    return { status: res.status, headers, body: new Uint8Array(await res.arrayBuffer()), url: res.url, ok: res.ok };
  }

  toString(): string { return `${this.method} ${this.url.toString()}`; }
}

/** A lightweight session: a base URL + shared headers + parallel ``sendMany``. */
export class HTTPSession {
  readonly base: URL | null;
  readonly headers: HTTPHeaders;

  constructor(opts: { base?: URL | string | null; headers?: HTTPHeaders } = {}) {
    this.base = opts.base != null ? URL.from(opts.base) : null;
    this.headers = opts.headers ?? {};
  }

  private resolve(req: HTTPRequest): HTTPRequest {
    // Join the base when the request carries no authority of its own (a
    // path-only URL like ``/a/b.json``), regardless of a leading slash.
    const needsBase = this.base && !req.url.host && !req.url.scheme;
    const url = needsBase ? this.base!.joinpath(req.url.path) : req.url;
    return new HTTPRequest({ method: req.method, url, headers: { ...this.headers, ...req.headers }, body: req.body });
  }

  send(req: HTTPRequest, signal?: AbortSignal): Promise<HTTPResponse> {
    return this.resolve(req).send(signal);
  }

  /** Fire all requests concurrently, preserving order (like ``send_many``). */
  sendMany(reqs: HTTPRequest[], signal?: AbortSignal): Promise<HTTPResponse[]> {
    return Promise.all(reqs.map((r) => this.send(r, signal)));
  }
}
