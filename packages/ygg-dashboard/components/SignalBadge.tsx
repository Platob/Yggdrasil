'use client';

interface Props {
  signal: 'BUY' | 'SELL' | 'HOLD';
  strength?: number;
}

export default function SignalBadge({ signal, strength }: Props) {
  const cls = signal === 'BUY' ? 'badge-buy' : signal === 'SELL' ? 'badge-sell' : 'badge-hold';
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
      <span className={cls}>{signal}</span>
      {strength !== undefined && (
        <span style={{ color: 'var(--muted)', fontSize: 10 }}>{(strength * 100).toFixed(0)}%</span>
      )}
    </span>
  );
}
