import { botFetch, BotAPIError } from "@/lib/bot-client";
import type { NodeInfo } from "@/lib/api";

export const dynamic = "force-dynamic";

export async function GET() {
  let botHealthy = false;

  try {
    await botFetch<NodeInfo>("/api/hello");
    botHealthy = true;
  } catch (e) {
    if (e instanceof BotAPIError) {
      console.error("[health] Bot API error:", e.status, e.body);
    } else {
      console.error("[health] Bot unreachable:", e);
    }
  }

  const status = botHealthy ? "healthy" : "degraded";

  return Response.json(
    {
      status,
      nextjs: true,
      bot: botHealthy,
      timestamp: new Date().toISOString(),
    },
    { status: botHealthy ? 200 : 503 }
  );
}
