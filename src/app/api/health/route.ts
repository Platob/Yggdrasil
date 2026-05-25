import { botFetch, BotAPIError } from "@/lib/bot-client";
import type { NodeInfo } from "@/lib/api";

export const dynamic = "force-dynamic";

let cachedStatus: { healthy: boolean; checkedAt: number } | null = null;
const HEALTH_TTL_MS = 5_000;

export async function GET() {
  const now = Date.now();

  if (cachedStatus && now - cachedStatus.checkedAt < HEALTH_TTL_MS) {
    return healthResponse(cachedStatus.healthy, cachedStatus.checkedAt);
  }

  let botHealthy = false;

  try {
    await botFetch<NodeInfo>("/api/hello");
    botHealthy = true;
  } catch (e) {
    if (e instanceof BotAPIError) {
      console.error("[health] Bot API error:", e.status, e.body);
    }
  }

  cachedStatus = { healthy: botHealthy, checkedAt: now };
  return healthResponse(botHealthy, now);
}

function healthResponse(botHealthy: boolean, checkedAt: number) {
  const status = botHealthy ? "healthy" : "degraded";
  return Response.json(
    {
      status,
      nextjs: true,
      bot: botHealthy,
      timestamp: new Date(checkedAt).toISOString(),
    },
    {
      status: botHealthy ? 200 : 503,
      headers: {
        "Cache-Control": "no-store, max-age=0",
        "X-Health-TTL": String(HEALTH_TTL_MS),
      },
    },
  );
}
