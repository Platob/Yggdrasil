'use client';

import useSWR from 'swr';
import Link from 'next/link';
import { api } from '@/lib/api';
import type { Position } from '@/lib/types';

function fmt(v: number, digits = 2): string {
  return v.toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

export default function PortfolioTable() {
  const { data, error, isLoading, mutate } = useSWR('portfolio', api.portfolio, {
    refreshInterval: 30_000,
  });

  if (isLoading) return <p style={{ color: 'var(--muted)', padding: 16 }}>Loading portfolio…</p>;
  if (error) return <p style={{ color: 'var(--red)', padding: 16 }}>Failed: {error.message}</p>;

  const positions: Position[] = data?.positions ?? [];

  if (positions.length === 0) {
    return (
      <div className="card" style={{ textAlign: 'center', padding: 32, color: 'var(--muted)' }}>
        No positions yet. Click a symbol to trade.
      </div>
    );
  }

  return (
    <div className="card">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
        <h2 style={{ fontSize: 12, color: 'var(--text-muted)', letterSpacing: 1, textTransform: 'uppercase' }}>Positions</h2>
        <div style={{ fontSize: 12 }}>
          <span style={{ color: 'var(--muted)' }}>Total P&L </span>
          <span className={data!.total_pnl >= 0 ? 'pos' : 'neg'} style={{ fontWeight: 700 }}>
            {data!.total_pnl >= 0 ? '+' : ''}{fmt(data!.total_pnl)} ({fmt(data!.total_pnl_pct)}%)
          </span>
        </div>
      </div>
      <table>
        <thead>
          <tr>
            <th>Symbol</th>
            <th style={{ textAlign: 'right' }}>Shares</th>
            <th style={{ textAlign: 'right' }}>Avg Cost</th>
            <th style={{ textAlign: 'right' }}>Price</th>
            <th style={{ textAlign: 'right' }}>Value</th>
            <th style={{ textAlign: 'right' }}>P&L</th>
            <th style={{ textAlign: 'right' }}>P&L %</th>
          </tr>
        </thead>
        <tbody>
          {positions.map((p) => (
            <tr key={p.symbol}>
              <td>
                <Link href={`/market/${p.symbol}`} style={{ color: 'var(--accent)', fontWeight: 700 }}>
                  {p.symbol}
                </Link>
              </td>
              <td style={{ textAlign: 'right' }}>{fmt(p.shares, 4)}</td>
              <td style={{ textAlign: 'right' }}>{fmt(p.avg_cost)}</td>
              <td style={{ textAlign: 'right' }}>{fmt(p.current_price)}</td>
              <td style={{ textAlign: 'right' }}>{fmt(p.market_value)}</td>
              <td style={{ textAlign: 'right' }} className={p.pnl >= 0 ? 'pos' : 'neg'}>
                {p.pnl >= 0 ? '+' : ''}{fmt(p.pnl)}
              </td>
              <td style={{ textAlign: 'right' }} className={p.pnl_pct >= 0 ? 'pos' : 'neg'}>
                {p.pnl_pct >= 0 ? '+' : ''}{fmt(p.pnl_pct)}%
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
