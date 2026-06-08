'use client';

import { useState } from 'react';
import useSWR from 'swr';
import { api } from '@/lib/api';
import SignalBadge from '@/components/SignalBadge';
import TradeForm from '@/components/TradeForm';
import PriceChart from '@/components/PriceChart';

interface Props { params: { symbol: string } }

function fmt(v: number | null | undefined, digits = 2): string {
  if (v == null) return '—';
  return v.toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

export default function SymbolPage({ params }: Props) {
  const sym = params.symbol.toUpperCase();
  const [aiQ, setAiQ] = useState('');
  const [aiResult, setAiResult] = useState<string | null>(null);
  const [aiLoading, setAiLoading] = useState(false);

  const { data: signals, isLoading: sigLoading } = useSWR(
    `signals-${sym}`, () => api.signals(sym), { refreshInterval: 60_000 }
  );
  const { data: ohlcv } = useSWR(
    `ohlcv-${sym}`, () => api.ohlcv(sym, '1d', '3mo'), { refreshInterval: 300_000 }
  );

  async function askAi(e: React.FormEvent) {
    e.preventDefault();
    setAiLoading(true);
    try {
      const res = await api.analyze(sym, aiQ || undefined);
      setAiResult(res.analysis);
    } catch (err: unknown) {
      setAiResult(`Error: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setAiLoading(false);
    }
  }

  if (sigLoading) return <p style={{ color: 'var(--muted)', padding: 16 }}>Loading {sym}…</p>;

  const q = signals?.quote;
  const changePos = (q?.change_pct ?? 0) >= 0;
  const inds = signals?.indicators;

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 280px', gap: 16, alignItems: 'start' }}>
      {/* Left column */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
        {/* Header */}
        <div className="card" style={{ display: 'flex', alignItems: 'center', gap: 20, flexWrap: 'wrap' }}>
          <div>
            <h1 style={{ fontSize: 22, fontWeight: 700 }}>{sym}</h1>
            <p style={{ color: 'var(--muted)', fontSize: 11 }}>{q?.exchange}</p>
          </div>
          <div>
            <span style={{ fontSize: 24, fontWeight: 700 }}>{fmt(q?.price)}</span>
            <span style={{ marginLeft: 8, fontSize: 14 }} className={changePos ? 'pos' : 'neg'}>
              {changePos ? '+' : ''}{fmt(q?.change)} ({changePos ? '+' : ''}{fmt(q?.change_pct)}%)
            </span>
          </div>
          {signals && <SignalBadge signal={signals.signal} strength={signals.strength} />}
        </div>

        {/* Chart */}
        {ohlcv?.data && (
          <div className="card">
            <h2 style={{ marginBottom: 12, fontSize: 11, color: 'var(--text-muted)', letterSpacing: 1, textTransform: 'uppercase' }}>
              Price (3mo)
            </h2>
            <PriceChart data={ohlcv.data} height={260} />
          </div>
        )}

        {/* Indicators */}
        {inds && (
          <div className="card">
            <h2 style={{ marginBottom: 12, fontSize: 11, color: 'var(--text-muted)', letterSpacing: 1, textTransform: 'uppercase' }}>
              Indicators
            </h2>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12 }}>
              {[
                { label: 'SMA 20', value: fmt(inds.sma20) },
                { label: 'SMA 50', value: fmt(inds.sma50) },
                { label: 'RSI 14', value: fmt(inds.rsi, 1), color: rsiColor(inds.rsi) },
                { label: 'MACD', value: fmt(inds.macd, 4), color: (inds.macd ?? 0) >= 0 ? 'var(--green)' : 'var(--red)' },
                { label: 'MACD Signal', value: fmt(inds.macd_signal, 4) },
                { label: 'BB Upper', value: fmt(inds.bb_upper) },
                { label: 'BB Lower', value: fmt(inds.bb_lower) },
              ].map(({ label, value, color }) => (
                <div key={label} style={{ background: '#0a0e1a', borderRadius: 4, padding: '8px 10px' }}>
                  <div style={{ color: 'var(--muted)', fontSize: 10, marginBottom: 2 }}>{label}</div>
                  <div style={{ fontWeight: 700, color: color ?? 'var(--text)' }}>{value}</div>
                </div>
              ))}
            </div>
            {signals?.reasons && signals.reasons.length > 0 && (
              <div style={{ marginTop: 12, display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                {signals.reasons.map((r) => (
                  <span key={r} style={{ background: '#0f1929', borderRadius: 3, padding: '2px 8px', fontSize: 11, color: 'var(--text-muted)' }}>
                    {r}
                  </span>
                ))}
              </div>
            )}
          </div>
        )}

        {/* AI Analysis */}
        <div className="card">
          <h2 style={{ marginBottom: 12, fontSize: 11, color: 'var(--text-muted)', letterSpacing: 1, textTransform: 'uppercase' }}>
            AI Analysis
          </h2>
          <form onSubmit={askAi} style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
            <input
              value={aiQ} onChange={(e) => setAiQ(e.target.value)}
              placeholder="Ask about this stock… (or leave blank for auto-analysis)"
              style={{ flex: 1 }}
            />
            <button type="submit" disabled={aiLoading} className="btn-blue">
              {aiLoading ? '…' : 'Analyze'}
            </button>
          </form>
          {aiResult && (
            <p style={{ fontSize: 12, lineHeight: 1.6, color: 'var(--text-muted)' }}>{aiResult}</p>
          )}
        </div>
      </div>

      {/* Right column */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
        <TradeForm symbol={sym} currentPrice={q?.price ?? undefined} />
      </div>
    </div>
  );
}

function rsiColor(rsi: number | null | undefined): string {
  if (rsi == null) return 'var(--text)';
  if (rsi < 30) return 'var(--green)';
  if (rsi > 70) return 'var(--red)';
  return 'var(--text)';
}
