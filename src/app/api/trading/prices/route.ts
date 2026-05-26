import { botFetch, BotAPIError } from "@/lib/bot-client";

export const dynamic = "force-dynamic";

interface PriceQuote {
  symbol: string;
  price: number;
  currency: string;
  source: string;
  timestamp: string;
  stale: boolean;
}

interface PricesResponse {
  prices: Record<string, PriceQuote>;
  timestamp: string;
}

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const symbols = searchParams.get("symbols");
  const qs = symbols ? `?symbols=${encodeURIComponent(symbols)}` : "";

  try {
    const data = await botFetch<PricesResponse>(`/api/trading/prices${qs}`);
    return Response.json(data);
  } catch (e) {
    if (e instanceof BotAPIError) {
      return Response.json({ error: e.body }, { status: e.status });
    }
    return Response.json({ error: "Bot unavailable" }, { status: 503 });
  }
}
