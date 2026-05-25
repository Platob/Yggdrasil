import { nodeAPI, NodeAPIError } from "@/lib/node-client";
import type { NodeInfo, ChannelInfo } from "@/lib/api";

export const dynamic = "force-dynamic";

/**
 * Aggregates node info, registry, and channels into a single response.
 * Useful for initial dashboard load - reduces 3 requests to 1.
 */
export async function GET() {
  let nodeInfo: NodeInfo | null = null;
  let registry: Record<string, string> = {};
  let channels: ChannelInfo[] = [];

  const results = await Promise.allSettled([
    nodeAPI.getNodeInfo(),
    nodeAPI.getRegistry(),
    nodeAPI.getChannels(),
  ]);

  if (results[0].status === "fulfilled") {
    nodeInfo = results[0].value;
  } else {
    console.error("[cache/dashboard] Failed to fetch node info:", results[0].reason);
  }

  if (results[1].status === "fulfilled") {
    registry = results[1].value;
  } else {
    console.error("[cache/dashboard] Failed to fetch registry:", results[1].reason);
  }

  if (results[2].status === "fulfilled") {
    channels = results[2].value;
  } else {
    console.error("[cache/dashboard] Failed to fetch channels:", results[2].reason);
  }

  return Response.json({
    nodeInfo,
    registry,
    channels,
    cachedAt: new Date().toISOString(),
  });
}
