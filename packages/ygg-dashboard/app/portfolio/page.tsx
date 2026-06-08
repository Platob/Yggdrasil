'use client';

import { useCallback } from 'react';
import PortfolioTable from '@/components/PortfolioTable';

export default function PortfolioPage() {
  return (
    <div>
      <div style={{ marginBottom: 20, display: 'flex', alignItems: 'baseline', gap: 12 }}>
        <h1 style={{ fontSize: 16, fontWeight: 700, letterSpacing: 1 }}>Portfolio</h1>
        <span style={{ fontSize: 11, color: 'var(--muted)' }}>Simulated positions — no real money</span>
      </div>
      <PortfolioTable />
    </div>
  );
}
