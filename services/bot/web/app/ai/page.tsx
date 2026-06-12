"use client";
import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { signalBadge, signalLabel, cn } from "@/lib/utils";
import type { AIAnalysis } from "@/lib/types";
import { BrainCircuit, Loader2, ChevronRight } from "lucide-react";

const QUICK_SYMBOLS = ["AAPL", "NVDA", "TSLA", "BTC-USD", "SPY", "META", "MSFT"];

export default function AIPage() {
  const [symbol, setSymbol] = useState("");
  const [context, setContext] = useState("");
  const [result, setResult] = useState<AIAnalysis | null>(null);

  const analyze = useMutation({
    mutationFn: () => api.analyze(symbol || "AAPL", context || undefined),
    onSuccess: (data) => setResult(data),
  });

  return (
    <div className="mx-auto max-w-4xl px-4 py-6 space-y-6">
      <div className="flex items-center gap-3">
        <BrainCircuit className="h-6 w-6 text-indigo-400" />
        <h1 className="text-xl font-bold text-slate-100">AI Analysis</h1>
      </div>
      <p className="text-sm text-slate-400">
        Claude-powered fundamental + technical analysis. Falls back to rule-based signals when AI is unavailable.
      </p>

      {/* Input form */}
      <div className="rounded-xl border border-slate-800/60 bg-slate-900/50 p-6 space-y-4">
        <div className="flex flex-wrap gap-2">
          {QUICK_SYMBOLS.map((s) => (
            <button
              key={s}
              onClick={() => setSymbol(s)}
              className={cn(
                "rounded-full border px-3 py-1 text-xs font-medium transition-colors",
                symbol === s
                  ? "border-indigo-500/60 bg-indigo-500/20 text-indigo-300"
                  : "border-slate-700 text-slate-400 hover:border-slate-600 hover:text-slate-300"
              )}
            >
              {s}
            </button>
          ))}
        </div>

        <div className="flex gap-3">
          <input
            value={symbol}
            onChange={(e) => setSymbol(e.target.value.toUpperCase())}
            placeholder="Symbol"
            className="w-32 rounded-lg border border-slate-700 bg-slate-800/60 px-3 py-2 text-sm text-slate-200 placeholder:text-slate-600 focus:border-indigo-500 focus:outline-none"
          />
          <input
            value={context}
            onChange={(e) => setContext(e.target.value)}
            placeholder="Optional context (e.g. 'earnings next week')"
            className="flex-1 rounded-lg border border-slate-700 bg-slate-800/60 px-3 py-2 text-sm text-slate-200 placeholder:text-slate-600 focus:border-indigo-500 focus:outline-none"
          />
          <button
            onClick={() => analyze.mutate()}
            disabled={(!symbol && !result) || analyze.isPending}
            className="flex items-center gap-2 rounded-lg bg-indigo-600 px-5 py-2 text-sm font-medium text-white transition-colors hover:bg-indigo-500 disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {analyze.isPending ? (
              <><Loader2 className="h-4 w-4 animate-spin" /> Analyzing…</>
            ) : (
              <><ChevronRight className="h-4 w-4" /> Analyze</>
            )}
          </button>
        </div>
      </div>

      {/* Result */}
      {result && <AnalysisResult analysis={result} />}
    </div>
  );
}

function AnalysisResult({ analysis: a }: { analysis: AIAnalysis }) {
  const sentimentColor =
    a.sentiment === "bullish" ? "text-green-400" :
    a.sentiment === "bearish" ? "text-red-400" : "text-slate-400";

  return (
    <div className="rounded-xl border border-slate-800/60 bg-slate-900/50 p-6 space-y-5">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-slate-100">{a.symbol}</h2>
          <p className={`text-sm font-medium capitalize ${sentimentColor}`}>{a.sentiment}</p>
        </div>
        <div className="text-right">
          <span className={cn("rounded-lg border px-3 py-1.5 text-sm font-semibold", signalBadge(a.recommendation))}>
            {signalLabel(a.recommendation)}
          </span>
          <p className="mt-1 text-xs text-slate-500">{(a.confidence * 100).toFixed(0)}% confidence</p>
        </div>
      </div>

      <div className="rounded-lg bg-slate-800/40 p-4">
        <p className="text-sm leading-relaxed text-slate-300">{a.summary}</p>
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <div>
          <h3 className="mb-2 text-xs font-semibold text-slate-400 uppercase tracking-wider">Key Factors</h3>
          <ul className="space-y-1">
            {a.key_factors.map((f, i) => (
              <li key={i} className="flex items-start gap-2 text-sm text-slate-300">
                <span className="mt-0.5 text-green-400">+</span> {f}
              </li>
            ))}
          </ul>
        </div>
        <div>
          <h3 className="mb-2 text-xs font-semibold text-slate-400 uppercase tracking-wider">Risks</h3>
          <ul className="space-y-1">
            {a.risks.map((r, i) => (
              <li key={i} className="flex items-start gap-2 text-sm text-slate-300">
                <span className="mt-0.5 text-red-400">−</span> {r}
              </li>
            ))}
          </ul>
        </div>
      </div>

      <p className="text-xs text-slate-600">Powered by {a.model} · {new Date(a.timestamp).toLocaleString()}</p>
    </div>
  );
}
