import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'YGG Dashboard',
  description: 'Yggdrasil trading terminal',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <nav style={{
          background: '#060a14',
          borderBottom: '1px solid var(--border)',
          padding: '0 20px',
          height: 44,
          display: 'flex',
          alignItems: 'center',
          gap: 24,
          position: 'sticky',
          top: 0,
          zIndex: 100,
        }}>
          <span style={{ color: 'var(--accent)', fontWeight: 700, fontSize: 14, letterSpacing: 2 }}>
            ⬡ YGG
          </span>
          <a href="/" style={{ color: 'var(--text-muted)', fontSize: 12 }}>Markets</a>
          <a href="/portfolio" style={{ color: 'var(--text-muted)', fontSize: 12 }}>Portfolio</a>
        </nav>
        <main style={{ padding: '20px', maxWidth: 1400, margin: '0 auto' }}>
          {children}
        </main>
      </body>
    </html>
  );
}
