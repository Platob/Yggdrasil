import type {
  AggregateRequest,
  AggregateResponse,
  AssetInfo,
  MarketDataResponse,
  Portfolio,
  PortfolioSummary,
  Tick,
  Trade,
} from "./types";

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8765";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
      cache: "no-store",
    });
  } catch (err) {
    throw new Error(
      `Backend offline: cannot reach ${BASE} (${(err as Error).message})`,
    );
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      if (body?.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* non-JSON error body — keep statusText */
    }
    throw new Error(`${res.status} ${detail}`);
  }
  return res.json() as Promise<T>;
}

export async function ping(): Promise<{ ok: boolean; ts: number; node_id: string }> {
  return request("/api/ping");
}

export async function getHealth(): Promise<unknown> {
  return request("/api/v2/health");
}

export async function getAssets(): Promise<AssetInfo[]> {
  return request("/api/v2/market/assets");
}

export async function getCandles(
  symbol: string,
  interval: string,
  limit: number,
): Promise<MarketDataResponse> {
  const qs = new URLSearchParams({ symbol, interval, limit: String(limit) });
  return request(`/api/v2/market/candles?${qs.toString()}`);
}

export async function getTick(symbol: string): Promise<Tick> {
  const qs = new URLSearchParams({ symbol });
  return request(`/api/v2/market/tick?${qs.toString()}`);
}

export async function getPortfolio(id: number): Promise<Portfolio> {
  return request(`/api/v2/portfolio/${id}`);
}

export async function getPortfolioSummary(id: number): Promise<PortfolioSummary> {
  return request(`/api/v2/portfolio/${id}/summary`);
}

export async function getTrades(id: number, limit = 50): Promise<Trade[]> {
  const qs = new URLSearchParams({ limit: String(limit) });
  return request(`/api/v2/portfolio/${id}/trades?${qs.toString()}`);
}

export async function cancelOrder(portfolioId: number, orderId: number): Promise<void> {
  await request(`/api/v2/portfolio/${portfolioId}/orders/${orderId}`, {
    method: "DELETE",
  });
}

export async function runAggregate(req: AggregateRequest): Promise<AggregateResponse> {
  return request("/api/v2/analysis/aggregate", {
    method: "POST",
    body: JSON.stringify(req),
  });
}
