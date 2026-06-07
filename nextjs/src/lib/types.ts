export interface Candle {
  ts: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  symbol: string;
  interval: string;
}

export interface Tick {
  ts: number;
  symbol: string;
  price: number;
  volume: number;
  side: string;
}

export interface AssetInfo {
  symbol: string;
  name: string;
  type: string;
  currency: string;
  exchange: string | null;
}

export interface Position {
  id: number;
  symbol: string;
  side: string;
  qty: number;
  avg_entry: number;
  current_price: number;
  unrealized_pnl: number;
  realized_pnl: number;
  opened_at: number;
}

export interface Order {
  id: number;
  symbol: string;
  side: string;
  type: string;
  qty: number;
  price: number | null;
  status: string;
  created_at: number;
  filled_at: number | null;
}

export interface Trade {
  id: number;
  symbol: string;
  side: string;
  qty: number;
  price: number;
  fee: number;
  pnl: number;
  ts: number;
}

export interface Portfolio {
  id: number;
  name: string;
  equity: number;
  cash: number;
  margin_used: number;
  total_pnl: number;
  daily_pnl: number;
  positions: Position[];
  open_orders: Order[];
  updated_at: number;
}

export interface PortfolioSummary {
  equity: number;
  cash: number;
  total_pnl: number;
  daily_pnl: number;
  position_count: number;
  open_order_count: number;
  win_rate: number;
}

export interface MarketDataResponse {
  symbol: string;
  candles: Candle[];
  count: number;
}

export interface AggregateRequest {
  path: string;
  column: string;
  agg: string;
  group_by?: string | null;
}

export interface AggregateRow {
  group: string | number;
  value: number;
}

export interface AggregateResponse {
  path: string;
  column: string;
  agg: string;
  group_by: string | null;
  rows: AggregateRow[];
  columns: string[];
}
