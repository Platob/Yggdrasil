import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "YGG Bot",
  description: "Trading + AI dashboard — ENTSOE energy prices, FX rates, Loki AI",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-bg text-text antialiased">
        {children}
      </body>
    </html>
  );
}
