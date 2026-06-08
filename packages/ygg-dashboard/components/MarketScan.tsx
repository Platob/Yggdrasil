'use client';

import Link from 'next/link';
import useSWR from 'swr';
import { api } from '@/lib/api';
import type { ScanRow } from '@/lib/types';
import SignalBadge from './SignalBadge';

function fmt(v: number | null, digits = 2): string {
  if (v == null) return '—';
  return v.toFixed(digits);
}

function pctClass(v: number | null): string {
  if (v == null) return 'neu';
  return v >= 0 ? 'pos' : 'neg';
}

export default function MarketScan() {
  const { data, error, isLoading } = useSWR('scan', () => api.scan(), {
    refreshInterval: 60_000,
  });

  if (isLoading) return <p style={{ color: 'var(--muted)', padding: 16 }}>Loading market scan…</p>;
  if (error) return <p style={{ color: 'var(--red)', padding: 16 }}>Failed: {error.message}</p>;

  const rows: ScanRow[] = data?.scan ?? [];

  return (
    <div className="card">
      <h2 style={{ marginBottom: 12, fontSize: 12, color: 'var(--text-muted)', letterSpacing: 1, textTransform: 'uppercase' }}>
        Market Scan
      </h2>
      <table>
        <thead>
          <tr>
            <th>Symbol</th>
            <th style={{ textAlign: 'right' }}>Price</th>
            <th style={{ textAlign: 'right' }}>Change %</th>
            <th style={{ textAlign: 'right' }}>RSI</th>
            <th>Signal</th>
            <th style={{ textAlign: 'right' }}>Score</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.symbol}>
              <td>
                <Link href={`/market/${r.symbol}`} style={{ color: 'var(--accent)', fontWeight: 700 }}>
                  {r.symbol}
                </Link>
              </td>
              <td style={{ textAlign: 'right' }}>{fmt(r.price, 2)}</td>
              <td style={{ textAlign: 'right' }} className={pctClass(r.change_pct)}>
                {r.change_pct != null ? (r.change_pct >= 0 ? '+' : '') + fmt(r.change_pct) + '%' : '—'}
              </td>
              <td style={{ textAlign: 'right', color: rsiColor(r.indicators?.rsi) }}>
                {fmt(r.indicators?.rsi, 1)}
              </td>
              <td>
                <SignalBadge signal={r.signal} strength={r.strength} />
              </td>
              <td style={{ textAlign: 'right', color: r.score > 0 ? 'var(--green)' : r.score < 0 ? 'var(--red)' : 'var(--muted)' }}>
                {fmt(r.score, 3)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function rsiColor(rsi: number | null | undefined): string {
  if (rsi == null) return 'var(--muted)';
  if (rsi < 30) return 'var(--green)';
  if (rsi > 70) return 'var(--red)';
  return 'var(--text)';
}
