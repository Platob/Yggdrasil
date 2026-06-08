import { describe, it, expect } from "vitest";
import { URL } from "./url";

// PARITY fixtures captured from the Python reference (yggdrasil.url.URL).
// Regenerate with the snippet in url.ts' commit message if the contract changes.
const FIXTURES = [
  { in: "https://example.com/a/b/c.csv?x=1#frag", scheme: "https", host: "example.com", port: null, path: "/a/b/c.csv", query: "x=1", fragment: "frag", userinfo: null, name: "c.csv", stem: "c", extensions: ["csv"], parts: ["a", "b", "c.csv"], parent: "https://example.com/a/b?x=1#frag", str: "https://example.com/a/b/c.csv?x=1#frag", media_type: "text/csv" },
  { in: "https://user:pw@host.com:8443/p/q/", scheme: "https", host: "host.com", port: 8443, path: "/p/q/", query: null, fragment: null, userinfo: "user:pw", name: "q", stem: "q", extensions: [], parts: ["p", "q", ""], parent: "https://user:pw@host.com:8443/p", str: "https://user:pw@host.com:8443/p/q/", media_type: null },
  { in: "s3://bucket/key/data.parquet", scheme: "s3", host: "bucket", port: null, path: "/key/data.parquet", query: null, fragment: null, userinfo: null, name: "data.parquet", stem: "data", extensions: ["parquet"], parts: ["key", "data.parquet"], parent: "s3://bucket/key", str: "s3://bucket/key/data.parquet", media_type: "application/vnd.apache.parquet" },
  { in: "file:///home/u/data/file.tar.gz", scheme: "file", host: "", port: null, path: "/home/u/data/file.tar.gz", query: null, fragment: null, userinfo: null, name: "file.tar.gz", stem: "file.tar", extensions: ["tar", "gz"], parts: ["home", "u", "data", "file.tar.gz"], parent: "file:///home/u/data", str: "file:///home/u/data/file.tar.gz", media_type: "application/x-tar" },
  { in: "/data/archive.csv.zst", scheme: "", host: "", port: null, path: "/data/archive.csv.zst", query: null, fragment: null, userinfo: null, name: "archive.csv.zst", stem: "archive.csv", extensions: ["csv", "zst"], parts: ["data", "archive.csv.zst"], parent: "/data", str: "/data/archive.csv.zst", media_type: "text/csv" },
  { in: "/a/b/c", scheme: "", host: "", port: null, path: "/a/b/c", query: null, fragment: null, userinfo: null, name: "c", stem: "c", extensions: [], parts: ["a", "b", "c"], parent: "/a/b", str: "/a/b/c", media_type: null },
  { in: "/a/b/c/", scheme: "", host: "", port: null, path: "/a/b/c/", query: null, fragment: null, userinfo: null, name: "c", stem: "c", extensions: [], parts: ["a", "b", "c", ""], parent: "/a/b", str: "/a/b/c/", media_type: null },
  { in: "/a/b/", scheme: "", host: "", port: null, path: "/a/b/", query: null, fragment: null, userinfo: null, name: "b", stem: "b", extensions: [], parts: ["a", "b", ""], parent: "/a", str: "/a/b/", media_type: null },
  { in: "/", scheme: "", host: "", port: null, path: "/", query: null, fragment: null, userinfo: null, name: "", stem: "", extensions: [], parts: [], parent: "/", str: "/", media_type: "inode/directory" },
  { in: "", scheme: "", host: "", port: null, path: "/", query: null, fragment: null, userinfo: null, name: "", stem: "", extensions: [], parts: [], parent: "/", str: "/", media_type: "inode/directory" },
  { in: "/data/README", scheme: "", host: "", port: null, path: "/data/README", query: null, fragment: null, userinfo: null, name: "README", stem: "README", extensions: [], parts: ["data", "README"], parent: "/data", str: "/data/README", media_type: null },
  { in: "/data/.hidden", scheme: "", host: "", port: null, path: "/data/.hidden", query: null, fragment: null, userinfo: null, name: ".hidden", stem: ".hidden", extensions: [], parts: ["data", ".hidden"], parent: "/data", str: "/data/.hidden", media_type: null },
  { in: "/data/.env.local", scheme: "", host: "", port: null, path: "/data/.env.local", query: null, fragment: null, userinfo: null, name: ".env.local", stem: ".env", extensions: ["local"], parts: ["data", ".env.local"], parent: "/data", str: "/data/.env.local", media_type: null },
  { in: "relative/path/x.json", scheme: "", host: "", port: null, path: "relative/path/x.json", query: null, fragment: null, userinfo: null, name: "x.json", stem: "x", extensions: ["json"], parts: ["relative", "path", "x.json"], parent: "relative/path", str: "relative/path/x.json", media_type: "application/json" },
  { in: "postgres://u@db:5432/mydb", scheme: "postgres", host: "db", port: 5432, path: "/mydb", query: null, fragment: null, userinfo: "u", name: "mydb", stem: "mydb", extensions: [], parts: ["mydb"], parent: "postgres://u@db:5432/", str: "postgres://u@db:5432/mydb", media_type: null },
  { in: "npfs://node1/folder/sub.arrow", scheme: "npfs", host: "node1", port: null, path: "/folder/sub.arrow", query: null, fragment: null, userinfo: null, name: "sub.arrow", stem: "sub", extensions: ["arrow"], parts: ["folder", "sub.arrow"], parent: "npfs://node1/folder", str: "npfs://node1/folder/sub.arrow", media_type: "application/vnd.apache.arrow.file" },
  { in: "C:\\Users\\x\\data.csv", scheme: "file", host: "", port: null, path: "/C:/Users/x/data.csv", query: null, fragment: null, userinfo: null, name: "data.csv", stem: "data", extensions: ["csv"], parts: ["C:", "Users", "x", "data.csv"], parent: "file:///C:/Users/x", str: "file:///C:/Users/x/data.csv", media_type: "text/csv" },
  { in: "http:example.com/path", scheme: "http", host: "example.com", port: null, path: "/path", query: null, fragment: null, userinfo: null, name: "path", stem: "path", extensions: [], parts: ["path"], parent: "http://example.com/", str: "http://example.com/path", media_type: null },
] as const;

describe("URL parity with the Python reference", () => {
  for (const f of FIXTURES) {
    it(`parses ${JSON.stringify(f.in)}`, () => {
      const u = URL.from(f.in);
      expect(u.scheme).toBe(f.scheme);
      expect(u.host).toBe(f.host);
      expect(u.port).toBe(f.port);
      expect(u.path).toBe(f.path);
      expect(u.query).toBe(f.query);
      expect(u.fragment).toBe(f.fragment);
      expect(u.userinfo).toBe(f.userinfo);
      expect(u.name).toBe(f.name);
      expect(u.stem).toBe(f.stem);
      expect(u.extensions).toEqual([...f.extensions]);
      expect(u.parts).toEqual([...f.parts]);
      expect(u.parent.toString()).toBe(f.parent);
      expect(u.toString()).toBe(f.str);
      expect(u.mediaType?.mimeType.value ?? null).toBe(f.media_type);
    });
  }
});

describe("URL transforms", () => {
  it("joinpath appends segments", () => {
    expect(URL.from("s3://b/key").joinpath("a", "b/c.csv").toString()).toBe("s3://b/key/a/b/c.csv");
    expect(URL.from("/").joinpath("x", "y").toString()).toBe("/x/y");
  });
  it("with* replaces components immutably", () => {
    const u = URL.from("https://h/a/b.csv");
    expect(u.withScheme("http").toString()).toBe("http://h/a/b.csv");
    expect(u.withQuery("z=1").toString()).toBe("https://h/a/b.csv?z=1");
    expect(u.toString()).toBe("https://h/a/b.csv"); // original unchanged
  });
  it("round-trips fromStr(toString())", () => {
    for (const f of FIXTURES) {
      const u = URL.from(f.in);
      expect(URL.fromStr(u.toString()).toString()).toBe(u.toString());
    }
  });
  it("isPathish guards", () => {
    expect(URL.isPathish("/a/b")).toBe(true);
    expect(URL.isPathish(URL.from("/"))).toBe(true);
    expect(URL.isPathish(null)).toBe(false);
    expect(URL.isPathish(42)).toBe(false);
  });
});
