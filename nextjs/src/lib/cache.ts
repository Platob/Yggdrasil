// Client-side response cache + in-flight de-duplication for GET requests.
//
// Two wins, both about render speed:
//   1. De-dupe — when several components mount at once and ask for the same
//      URL, they share a single network round-trip instead of N.
//   2. Short TTL reuse — navigating back to a page within the window paints
//      from cache instead of re-fetching.
//
// TTL is tuned per *asset criticality* (the brain-node "vital signs" idea):
// fast-changing vital signs get a sub-second window so live pollers still see
// fresh data, structural cluster views get a few seconds, and definitions /
// identity that only change on explicit user action get tens of seconds.
//
// The windows are deliberately kept *below* each endpoint's polling cadence
// (run list polls at 1s, stats at 3s, topology at 5s) so caching coalesces
// burst reads without ever stalling a live poll.

export const TTL = {
  // Vital signs — resource snapshots, run state, health. Must look live.
  VITAL: 800,
  // Structural cluster shape — topology, peers, node card. Shifts slowly.
  STRUCTURAL: 4_000,
  // Definitions & identity — envs, functions, DAGs, users. Change only when
  // the user creates/edits/deletes something (and we invalidate on those).
  DEFINITION: 30_000,
} as const;

interface Entry {
  expires: number;
  value: unknown;
}

const store = new Map<string, Entry>();
const inflight = new Map<string, Promise<unknown>>();

// Fetch ``url`` through the cache. ``fresh`` skips a warm hit (for explicit
// "Refresh" actions) while still repopulating the cache for later reads.
export async function cachedGet<T>(
  url: string,
  ttlMs: number,
  fetcher: (u: string) => Promise<T>,
  fresh = false,
): Promise<T> {
  if (!fresh) {
    const hit = store.get(url);
    if (hit && hit.expires > Date.now()) return hit.value as T;
    const pending = inflight.get(url);
    if (pending) return pending as Promise<T>;
  }
  const p = fetcher(url)
    .then((value) => {
      store.set(url, { expires: Date.now() + ttlMs, value });
      return value;
    })
    .finally(() => {
      inflight.delete(url);
    });
  inflight.set(url, p);
  return p as Promise<T>;
}

// Same coalesce+TTL reuse for POST analytics (keyed by url + body), so
// re-running the same query or zooming back to a window is served from the
// client instead of hitting the node again.
export async function cachedPost<T>(url: string, body: unknown, ttlMs: number, fetcher: () => Promise<T>): Promise<T> {
  const key = `${url}|${JSON.stringify(body)}`;
  const hit = store.get(key);
  if (hit && hit.expires > Date.now()) return hit.value as T;
  const pending = inflight.get(key);
  if (pending) return pending as Promise<T>;
  const p = fetcher()
    .then((value) => { store.set(key, { expires: Date.now() + ttlMs, value }); return value; })
    .finally(() => { inflight.delete(key); });
  inflight.set(key, p);
  return p as Promise<T>;
}

// Drop cached entries whose URL contains any of the given fragments. Call
// after a mutation so the next read reflects the change immediately, e.g.
// ``invalidate("pyfunc", "stats")`` after creating a function.
export function invalidate(...fragments: string[]): void {
  for (const key of store.keys()) {
    if (fragments.some((f) => key.includes(f))) store.delete(key);
  }
}
