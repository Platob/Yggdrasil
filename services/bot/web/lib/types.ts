export interface Quote {
  symbol: string;
  price: number;
  change: number;
  change_pct: number;
  volume: number;
  market_cap?: number;
  bid?: number;
  ask?: number;
  high_52w?: number;
  low_52w?: number;
  timestamp: string;
}

export interface OHLCV {
  symbol: string;
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  vwap?: number;
}

export type SignalDirection =
  | "strong_buy"
  | "buy"
  | "neutral"
  | "sell"
  | "strong_sell";

export interface Indicator {
  name: string;
  value: number;
  signal: SignalDirection;
  description: string;
}

export interface Signal {
  symbol: string;
  direction: SignalDirection;
  confidence: number;
  price_target?: number;
  stop_loss?: number;
  indicators: Indicator[];
  timestamp: string;
  timeframe: string;
}

export interface AIAnalysis {
  symbol: string;
  summary: string;
  sentiment: string;
  key_factors: string[];
  risks: string[];
  recommendation: SignalDirection;
  confidence: number;
  model: string;
  timestamp: string;
}

export interface Position {
  symbol: string;
  quantity: number;
  avg_cost: number;
  current_price: number;
  market_value: number;
  unrealized_pnl: number;
  unrealized_pnl_pct: number;
  realized_pnl: number;
  weight: number;
}

export interface Trade {
  id: number;
  symbol: string;
  side: "buy" | "sell";
  quantity: number;
  price: number;
  fee: number;
  status: string;
  timestamp: string;
  notes?: string;
}

export interface Portfolio {
  id: number;
  name: string;
  cash: number;
  positions: Record<string, Position>;
  trades: Trade[];
  created_at: string;
  updated_at: string;
}

export interface PnL {
  total_value: number;
  cash: number;
  invested: number;
  unrealized_pnl: number;
  unrealized_pnl_pct: number;
  realized_pnl: number;
  total_pnl: number;
  total_pnl_pct: number;
  day_pnl: number;
  day_pnl_pct: number;
}

export interface Ticker {
  symbol: string;
  name: string;
  exchange: string;
  asset_type: string;
}
