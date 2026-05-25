import { botAPI } from "@/lib/bot-client";
import type { NodeInfo, ChannelInfo } from "@/lib/api";

export const dynamic = "force-dynamic";

interface DashboardCache {
  nodeInfo: NodeInfo | null;
  registry: Record<string, string>;
  channels: ChannelInfo[];
  cachedAt: string;
}

let cached: { data: DashboardCache; expiresAt: number } | null = null;
const CACHE_TTL_MS = 10_000;

export async function GET(request: Request) {
  const url = new URL(request.url);
  const forceRefresh = url.searchParams.get("refresh") === "1";
  const now = Date.now();

  if (!forceRefresh && cached && now < cached.expiresAt) {
    return Response.json(cached.data, {
      headers: {
        "Cache-Control": "private, max-age=10, stale-while-revalidate=5",
        "X-Cache": "HIT",
        "X-Cache-Age": String(Math.round((now - new Date(cached.data.cachedAt).getTime()) / 1000)),
      },
    });
  }

  let nodeInfo: NodeInfo | null = null;
  let registry: Record<string, string> = {};
  let channels: ChannelInfo[] = [];

  const results = await Promise.allSettled([
    botAPI.getNodeInfo(),
    botAPI.getRegistry(),
    botAPI.getChannels(),
  ]);

  if (results[0].status === "fulfilled") nodeInfo = results[0].value;
  if (results[1].status === "fulfilled") registry = results[1].value;
  if (results[2].status === "fulfilled") channels = results[2].value;

  const data: DashboardCache = {
    nodeInfo,
    registry,
    channels,
    cachedAt: new Date().toISOString(),
  };

  cached = { data, expiresAt: now + CACHE_TTL_MS };

  return Response.json(data, {
    headers: {
      "Cache-Control": "private, max-age=10, stale-while-revalidate=5",
      "X-Cache": "MISS",
    },
  });
}
