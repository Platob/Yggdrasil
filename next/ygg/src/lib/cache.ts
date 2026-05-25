/**
 * Simple TTL cache for API responses.
 * Entries expire after `ttl` milliseconds.
 */
export class Cache<T> {
  private entries = new Map<string, { value: T; expiresAt: number }>();

  constructor(private defaultTtl: number = 30_000) {} // 30s default

  get(key: string): T | null {
    const entry = this.entries.get(key);
    if (!entry) return null;
    if (Date.now() > entry.expiresAt) {
      this.entries.delete(key);
      return null;
    }
    return entry.value;
  }

  set(key: string, value: T, ttl?: number): void {
    this.entries.set(key, {
      value,
      expiresAt: Date.now() + (ttl ?? this.defaultTtl),
    });
  }

  has(key: string): boolean { return this.get(key) !== null; }

  delete(key: string): void { this.entries.delete(key); }

  clear(): void { this.entries.clear(); }

  /** Get or compute: returns cached value or calls fn and caches result */
  async getOrFetch(key: string, fn: () => Promise<T>, ttl?: number): Promise<T> {
    const cached = this.get(key);
    if (cached !== null) return cached;
    const value = await fn();
    this.set(key, value, ttl);
    return value;
  }

  get size(): number {
    // Purge expired while counting
    const now = Date.now();
    for (const [k, v] of this.entries) {
      if (now > v.expiresAt) this.entries.delete(k);
    }
    return this.entries.size;
  }
}

// Shared cache instances for common data
export const nodeCache = new Cache<unknown>(30_000);   // 30s for node info
export const listCache = new Cache<unknown>(10_000);   // 10s for lists
