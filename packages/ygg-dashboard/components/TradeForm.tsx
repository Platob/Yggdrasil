'use client';

import { useState } from 'react';
import { api } from '@/lib/api';

interface Props {
  symbol: string;
  currentPrice?: number;
  onTrade?: () => void;
}

export default function TradeForm({ symbol, currentPrice, onTrade }: Props) {
  const [action, setAction] = useState<'BUY' | 'SELL'>('BUY');
  const [shares, setShares] = useState('');
  const [price, setPrice] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<string | null>(null);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setResult(null);
    try {
      await api.trade({
        symbol,
        action,
        shares: parseFloat(shares),
        price: price ? parseFloat(price) : undefined,
      });
      setResult(`${action} ${shares} ${symbol} executed.`);
      setShares('');
      onTrade?.();
    } catch (err: unknown) {
      setResult(`Error: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setLoading(false);
    }
  }

  const estValue = shares && (price || currentPrice)
    ? (parseFloat(shares) * (price ? parseFloat(price) : currentPrice!)).toFixed(2)
    : null;

  return (
    <div className="card">
      <h3 style={{ marginBottom: 12, fontSize: 12, color: 'var(--text-muted)', letterSpacing: 1, textTransform: 'uppercase' }}>
        Quick Trade — {symbol}
      </h3>
      <form onSubmit={submit}>
        <div style={{ display: 'flex', gap: 8, marginBottom: 8 }}>
          <button type="button" className="btn-buy" onClick={() => setAction('BUY')}
            style={{ flex: 1, opacity: action === 'BUY' ? 1 : 0.4 }}>BUY</button>
          <button type="button" className="btn-sell" onClick={() => setAction('SELL')}
            style={{ flex: 1, opacity: action === 'SELL' ? 1 : 0.4 }}>SELL</button>
        </div>
        <div style={{ display: 'flex', gap: 8, marginBottom: 8 }}>
          <input
            type="number" placeholder="Shares" value={shares} min="0" step="any"
            onChange={(e) => setShares(e.target.value)} style={{ flex: 1 }} required
          />
          <input
            type="number" placeholder={currentPrice ? `${currentPrice} (live)` : 'Price'} value={price} min="0" step="any"
            onChange={(e) => setPrice(e.target.value)} style={{ flex: 1 }}
          />
        </div>
        {estValue && (
          <p style={{ color: 'var(--muted)', fontSize: 11, marginBottom: 8 }}>
            Est. value: ${estValue}
          </p>
        )}
        <button type="submit" disabled={loading || !shares} className={action === 'BUY' ? 'btn-buy' : 'btn-sell'} style={{ width: '100%' }}>
          {loading ? 'Executing…' : `${action} ${shares || '—'} ${symbol}`}
        </button>
      </form>
      {result && (
        <p style={{ marginTop: 8, fontSize: 11, color: result.startsWith('Error') ? 'var(--red)' : 'var(--green)' }}>
          {result}
        </p>
      )}
    </div>
  );
}
