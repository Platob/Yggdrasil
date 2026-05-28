// High-level typed API for the Yggdrasil node.
// All cross-cutting transport details live in ./bot-client.
import { botFetch, botStream, BotAPIError } from "./bot-client";

export { BotAPIError };

// ── Discovery / node identity ──────────────────────────────────

export interface NodeInfo {
  node_id: string;
  host: string;
  port: number;
  version: string;
  uptime: number;
  channels: string[];
  functions: string[];
  lat?: number | null;
  lon?: number | null;
}

// ── Messenger ──────────────────────────────────────────────────

export interface Message {
  id: string;
  sender: string;
  text: string;
  channel: string;
  timestamp: string;
  node_id: string;
}

export interface ChannelInfo {
  name: string;
  created_at: string;
  last_active: string;
  message_count: number;
  members: string[];
}

// ── Trading ────────────────────────────────────────────────────

export interface PriceQuote {
  symbol: string;
  price: number;
  change_pct: number;
  volume: number;
  timestamp_ms: number;
}

export interface Position {
  id: number;
  symbol: string;
  qty: number;
  avg_price: number;
  current_price: number;
  pnl: number;
  pnl_pct: number;
}

export type OrderSide = "buy" | "sell";
export type OrderType = "market" | "limit";
export type OrderStatus = "pending" | "filled" | "cancelled" | "rejected";

export interface Order {
  id: number;
  symbol: string;
  side: OrderSide;
  qty: number;
  filled_qty: number;
  order_type: OrderType;
  limit_price: number | null;
  status: OrderStatus;
  avg_fill_price: number | null;
  created_at: number;
  filled_at: number | null;
}

export interface PortfolioSummary {
  cash: number;
  equity: number;
  total_value: number;
  total_pnl: number;
  total_pnl_pct: number;
  positions: Position[];
}

export type SignalAction = "strong_buy" | "buy" | "hold" | "sell" | "strong_sell";

export interface TradingSignal {
  symbol: string;
  signal: SignalAction;
  confidence: number;
  reason: string;
  indicators: Record<string, number>;
  timestamp_ms: number;
}

export interface WatchlistEntry {
  symbol: string;
  added_at: number;
}

export interface PriceAlert {
  id: number;
  symbol: string;
  condition: "above" | "below";
  threshold: number;
  triggered: boolean;
  created_at: number;
  triggered_at: number | null;
}

export interface TradeHistoryEntry {
  order_id: number;
  symbol: string;
  side: OrderSide;
  qty: number;
  price: number;
  realized_pnl: number | null;
  timestamp_ms: number;
}

// ── AI ─────────────────────────────────────────────────────────

export interface AIChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface AIChatResponse {
  content: string;
  model: string;
  backend: "anthropic" | "mock";
}

export interface AIAnalyzeResponse {
  symbol: string;
  analysis: string;
  signal: SignalAction;
  confidence: number;
  key_levels: Record<string, number>;
  timestamp_ms: number;
}

export interface AIConversation {
  id: number;
  title: string;
  messages: AIChatMessage[];
  created_at: number;
  updated_at: number;
}

export interface AISuggestion {
  symbol: string;
  title: string;
  detail: string;
  action: "buy" | "sell" | "hold" | "watch";
  confidence: number;
}

export interface AIPortfolioAnalysis {
  summary: string;
  risk_score: number;
  diversification: string;
  recommendations: string[];
  timestamp_ms: number;
}

// ── Execute ────────────────────────────────────────────────────

export interface PythonResponse {
  id: string;
  node_id: string;
  returncode: number | null;
  stdout: string | null;
  stderr: string | null;
  result: unknown;
  duration: number | null;
  status: string;
}

export interface CmdResponse {
  id: string;
  node_id: string;
  command: string[];
  returncode: number | null;
  stdout: string | null;
  stderr: string | null;
  duration: number | null;
  status: string;
}

// ── Discovery functions ────────────────────────────────────────

export async function getNodeInfo(): Promise<NodeInfo> {
  return botFetch<NodeInfo>("/api/hello");
}

// ── Messenger functions ────────────────────────────────────────

export async function getChannels(): Promise<ChannelInfo[]> {
  const resp = await botFetch<{ channels: ChannelInfo[] }>("/api/messenger/channels");
  return resp.channels;
}

export async function getMessages(channel: string, limit = 50): Promise<Message[]> {
  const resp = await botFetch<{ messages: Message[] }>(
    `/api/messenger/channels/${encodeURIComponent(channel)}/messages?limit=${limit}`
  );
  return resp.messages;
}

export async function pollMessages(
  channel: string,
  afterId: string,
  timeout = 30
): Promise<Message[]> {
  const q = new URLSearchParams({ timeout: String(timeout) });
  if (afterId) q.set("after_id", afterId);
  const resp = await botFetch<{ messages: Message[] }>(
    `/api/messenger/channels/${encodeURIComponent(channel)}/poll?${q.toString()}`
  );
  return resp.messages;
}

export async function sendMessage(
  text: string,
  sender?: string,
  channel = "general"
): Promise<Message> {
  return botFetch<Message>("/api/messenger", {
    method: "POST",
    body: JSON.stringify({ text, sender, channel }),
  });
}

export async function createChannel(name: string): Promise<ChannelInfo> {
  const resp = await botFetch<{ channel: ChannelInfo }>(
    `/api/messenger/channels?name=${encodeURIComponent(name)}`,
    { method: "POST" }
  );
  return resp.channel;
}

export function streamChannel(channel: string): EventSource {
  return botStream(`/api/messenger/channels/${encodeURIComponent(channel)}/stream`);
}

// ── Trading functions ──────────────────────────────────────────

export async function getPrices(): Promise<PriceQuote[]> {
  const resp = await botFetch<{ prices: PriceQuote[] }>("/api/trading/prices");
  return resp.prices;
}

export async function getPrice(symbol: string): Promise<PriceQuote> {
  return botFetch<PriceQuote>(`/api/trading/prices/${encodeURIComponent(symbol)}`);
}

export function streamPrices(intervalSeconds = 1): EventSource {
  return botStream(`/api/trading/prices/stream?interval=${intervalSeconds}`);
}

export async function getPortfolio(): Promise<PortfolioSummary> {
  return botFetch<PortfolioSummary>("/api/trading/portfolio");
}

export interface PlaceOrderInput {
  symbol: string;
  side: OrderSide;
  qty: number;
  order_type?: OrderType;
  limit_price?: number | null;
}

export async function placeOrder(input: PlaceOrderInput): Promise<Order> {
  const resp = await botFetch<{ order: Order }>("/api/trading/orders", {
    method: "POST",
    body: JSON.stringify({
      order_type: "market",
      limit_price: null,
      ...input,
    }),
  });
  return resp.order;
}

export async function getOrders(): Promise<Order[]> {
  const resp = await botFetch<{ orders: Order[] }>("/api/trading/orders");
  return resp.orders;
}

export async function cancelOrder(id: number): Promise<Order> {
  const resp = await botFetch<{ order: Order }>(`/api/trading/orders/${id}`, {
    method: "DELETE",
  });
  return resp.order;
}

export async function getWatchlist(): Promise<WatchlistEntry[]> {
  const resp = await botFetch<{ entries: WatchlistEntry[] }>("/api/trading/watchlist");
  return resp.entries;
}

export async function addWatchlist(symbol: string): Promise<WatchlistEntry> {
  const resp = await botFetch<{ entry: WatchlistEntry }>("/api/trading/watchlist", {
    method: "POST",
    body: JSON.stringify({ symbol }),
  });
  return resp.entry;
}

export async function removeWatchlist(symbol: string): Promise<WatchlistEntry> {
  const resp = await botFetch<{ entry: WatchlistEntry }>(
    `/api/trading/watchlist/${encodeURIComponent(symbol)}`,
    { method: "DELETE" }
  );
  return resp.entry;
}

export async function getSignals(): Promise<TradingSignal[]> {
  const resp = await botFetch<{ signals: TradingSignal[] }>("/api/trading/signals");
  return resp.signals;
}

export async function getSignal(symbol: string): Promise<TradingSignal> {
  return botFetch<TradingSignal>(`/api/trading/signals/${encodeURIComponent(symbol)}`);
}

export async function getAlerts(): Promise<PriceAlert[]> {
  const resp = await botFetch<{ alerts: PriceAlert[] }>("/api/trading/alerts");
  return resp.alerts;
}

export async function createAlert(
  symbol: string,
  condition: "above" | "below",
  threshold: number,
): Promise<PriceAlert> {
  const resp = await botFetch<{ alert: PriceAlert }>("/api/trading/alerts", {
    method: "POST",
    body: JSON.stringify({ symbol, condition, threshold }),
  });
  return resp.alert;
}

export async function deleteAlert(id: number): Promise<PriceAlert> {
  const resp = await botFetch<{ alert: PriceAlert }>(`/api/trading/alerts/${id}`, {
    method: "DELETE",
  });
  return resp.alert;
}

export async function getTradeHistory(): Promise<TradeHistoryEntry[]> {
  const resp = await botFetch<{ trades: TradeHistoryEntry[] }>("/api/trading/history");
  return resp.trades;
}

// ── AI functions ───────────────────────────────────────────────

export async function aiChat(
  messages: AIChatMessage[],
  opts?: { system?: string; context?: Record<string, unknown> }
): Promise<AIChatResponse> {
  return botFetch<AIChatResponse>("/api/ai/chat", {
    method: "POST",
    body: JSON.stringify({
      messages,
      system: opts?.system,
      context: opts?.context,
      stream: false,
    }),
  });
}

export async function aiAnalyze(
  symbol: string,
  opts?: { timeframe?: string; include_portfolio?: boolean }
): Promise<AIAnalyzeResponse> {
  const q = new URLSearchParams();
  if (opts?.timeframe) q.set("timeframe", opts.timeframe);
  if (opts?.include_portfolio) q.set("include_portfolio", "true");
  return botFetch<AIAnalyzeResponse>(
    `/api/ai/analyze/${encodeURIComponent(symbol)}?${q.toString()}`,
    { method: "POST" }
  );
}

export async function aiAnalyzePortfolio(): Promise<AIPortfolioAnalysis> {
  return botFetch<AIPortfolioAnalysis>("/api/ai/analyze/portfolio", { method: "POST" });
}

export async function aiSuggestions(): Promise<AISuggestion[]> {
  const resp = await botFetch<{ suggestions: AISuggestion[] }>("/api/ai/suggestions");
  return resp.suggestions;
}

export async function aiListConversations(): Promise<AIConversation[]> {
  const resp = await botFetch<{ conversations: AIConversation[] }>("/api/ai/conversations");
  return resp.conversations;
}

export async function aiCreateConversation(title = "New Conversation"): Promise<AIConversation> {
  const resp = await botFetch<{ conversation: AIConversation }>("/api/ai/conversations", {
    method: "POST",
    body: JSON.stringify({ title }),
  });
  return resp.conversation;
}

export async function aiDeleteConversation(id: number): Promise<void> {
  await botFetch(`/api/ai/conversations/${id}`, { method: "DELETE" });
}

// ── Execute (Python / Cmd) ─────────────────────────────────────

export async function executePython(code: string, timeout?: number): Promise<PythonResponse> {
  return botFetch<PythonResponse>("/api/python", {
    method: "POST",
    body: JSON.stringify({ code, timeout, env: {} }),
  });
}

export async function executeCmd(command: string[], timeout?: number): Promise<CmdResponse> {
  return botFetch<CmdResponse>("/api/cmd", {
    method: "POST",
    body: JSON.stringify({ command, timeout, env: {} }),
  });
}

// ── Legacy ergonomic alias used by /bot/page.tsx ───────────────

export const bot = {
  getNodeInfo,
  getChannels,
  getMessages,
  sendMessage,
  createChannel,
  getPortfolio,
  getPrices,
  placeOrder,
  aiChat,
};
