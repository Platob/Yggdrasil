import { describe, it, expect, vi } from "vitest";
import { HTTPRequest, HTTPSession } from "./request";
import { URL } from "../url";

describe("HTTPRequest value type", () => {
  it("normalises method + URL", () => {
    const r = HTTPRequest.prepare("get", "https://api.example.com/v1/data.json", { headers: { accept: "application/json" } });
    expect(r.method).toBe("GET");
    expect(r.url).toBeInstanceOf(URL);
    expect(r.url.host).toBe("api.example.com");
    expect(r.isHttp).toBe(true);
    expect(r.toString()).toBe("GET https://api.example.com/v1/data.json");
  });

  it("session merges base + headers", async () => {
    const fetchMock = vi.fn(async (u: string, init: RequestInit) => ({
      status: 200, ok: true, url: u,
      headers: { forEach: (cb: (v: string, k: string) => void) => cb("application/json", "content-type") },
      arrayBuffer: async () => new TextEncoder().encode(JSON.stringify({ u, method: init.method })).buffer,
    }));
    vi.stubGlobal("fetch", fetchMock);

    const s = new HTTPSession({ base: "https://api.example.com", headers: { "x-token": "abc" } });
    const res = await s.sendMany([
      HTTPRequest.prepare("GET", "/a/b.json"),
      HTTPRequest.prepare("GET", "/c/d.json", { headers: { accept: "*/*" } }),
    ]);
    expect(res).toHaveLength(2);
    expect(res[0].status).toBe(200);
    // base joined with the request path
    expect(fetchMock.mock.calls[0][0]).toBe("https://api.example.com/a/b.json");
    expect(fetchMock.mock.calls[1][0]).toBe("https://api.example.com/c/d.json");
    // shared header merged
    expect((fetchMock.mock.calls[0][1].headers as Record<string, string>)["x-token"]).toBe("abc");
    vi.unstubAllGlobals();
  });
});
