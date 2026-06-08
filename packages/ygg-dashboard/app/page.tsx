import MarketScan from '@/components/MarketScan';

export default function DashboardPage() {
  return (
    <div>
      <div style={{ marginBottom: 20, display: 'flex', alignItems: 'baseline', gap: 12 }}>
        <h1 style={{ fontSize: 16, fontWeight: 700, letterSpacing: 1 }}>Markets</h1>
        <span style={{ fontSize: 11, color: 'var(--muted)' }}>Signals auto-refresh every 60s</span>
      </div>
      <MarketScan />
    </div>
  );
}
