export interface NodeStats { uptime_s: number; requests: number }
export interface FsEntry { name: string; is_dir: boolean; size: number; modified_at: string }
export interface FsListResult { entries: FsEntry[]; total: number }
export interface IndicatorsResult {
  price: number[]; ts: (string | number)[]; ema_9: (number | null)[];
  ema_21: (number | null)[]; ema_50: (number | null)[]; rsi_14: (number | null)[];
  macd_line: (number | null)[]; macd_signal: (number | null)[]; macd_hist: (number | null)[];
  bb_upper: (number | null)[]; bb_lower: (number | null)[];
}
export interface BacktestResult {
  strategy: string; initial_cash: number; final_value: number;
  total_return: number; ann_return: number; max_drawdown: number;
  sharpe: number; sortino: number; n_trades: number; win_rate: number;
  equity_curve: number[]; benchmark_return: number; benchmark_equity: number[];
}
export interface FinanceResult {
  total_return: number; cagr: number; ann_return: number; ann_volatility: number;
  sharpe: number; sortino: number; max_drawdown: number; calmar: number;
  ema: number[]; drawdown: number[];
}
export interface SagaCatalog { name: string }
export interface SqlResult { columns: string[]; rows: unknown[][] }
export interface Message { id: string; text: string; sender: string; channel: string; timestamp: string }
