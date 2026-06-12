import { MarketTicker } from "@/components/dashboard/MarketTicker";
import { PortfolioCard } from "@/components/dashboard/PortfolioCard";
import { SignalsPanel } from "@/components/dashboard/SignalsPanel";
import { QuoteCard } from "@/components/dashboard/QuoteCard";

export default function Dashboard() {
  return (
    <div>
      <MarketTicker />
      <div className="mx-auto max-w-7xl px-4 py-6 space-y-6">
        {/* Hero row */}
        <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
          <div className="md:col-span-1">
            <PortfolioCard />
          </div>
          <div className="md:col-span-2">
            <QuoteCard symbol="SPY" />
          </div>
        </div>

        {/* Mid row */}
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
          <QuoteCard symbol="AAPL" />
          <QuoteCard symbol="NVDA" />
          <QuoteCard symbol="BTC-USD" />
        </div>

        {/* Signals */}
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          <SignalsPanel />
          <QuoteCard symbol="ETH-USD" />
        </div>
      </div>
    </div>
  );
}
