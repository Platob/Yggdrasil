"use client";

import { useRef, useState } from "react";

interface Message {
  id: number;
  role: "user" | "assistant";
  content: string;
}

let _id = 0;
const uid = () => ++_id;

export function AiChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  async function send() {
    const text = input.trim();
    if (!text || loading) return;
    setInput("");
    setMessages((m) => [...m, { id: uid(), role: "user", content: text }]);
    setLoading(true);

    const assistantId = uid();
    setMessages((m) => [...m, { id: assistantId, role: "assistant", content: "" }]);

    try {
      const r = await fetch("/bot-api/api/v2/ai/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: text }),
      });

      if (!r.ok || !r.body) throw new Error(`${r.status}`);

      const reader = r.body.getReader();
      const dec = new TextDecoder();
      let buf = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        const lines = buf.split("\n");
        buf = lines.pop() ?? "";
        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          try {
            const evt = JSON.parse(line.slice(6)) as {
              kind: string;
              text?: string;
              detail?: string;
            };
            if (evt.kind === "chunk" && evt.text) {
              setMessages((m) =>
                m.map((msg) =>
                  msg.id === assistantId
                    ? { ...msg, content: msg.content + evt.text }
                    : msg
                )
              );
              bottomRef.current?.scrollIntoView({ behavior: "smooth" });
            }
          } catch {
            // skip malformed SSE line
          }
        }
      }
    } catch (err) {
      setMessages((m) =>
        m.map((msg) =>
          msg.id === assistantId
            ? { ...msg, content: `error: ${err}` }
            : msg
        )
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="bg-surface border border-border rounded-lg p-4 flex flex-col h-80">
      <div className="text-muted text-xs mb-3 shrink-0">loki ai</div>
      <div className="flex-1 overflow-y-auto space-y-3 text-xs mb-3">
        {messages.length === 0 && (
          <p className="text-muted">ask loki about market conditions, signals, or energy prices…</p>
        )}
        {messages.map((m) => (
          <div key={m.id} className={m.role === "user" ? "text-right" : "text-left"}>
            <span
              className={`inline-block px-3 py-1.5 rounded max-w-[85%] text-left leading-relaxed ${
                m.role === "user"
                  ? "bg-accent text-white"
                  : "bg-border text-text"
              }`}
            >
              {m.content || (loading ? "▌" : "")}
            </span>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>
      <div className="flex gap-2 shrink-0">
        <input
          className="flex-1 bg-bg border border-border rounded px-3 py-1.5 text-xs text-text placeholder-muted outline-none focus:border-accent transition-colors"
          placeholder="ask loki…"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && send()}
          disabled={loading}
        />
        <button
          onClick={send}
          disabled={loading || !input.trim()}
          className="px-3 py-1.5 text-xs bg-accent text-white rounded disabled:opacity-40 hover:bg-indigo-500 transition-colors"
        >
          {loading ? "…" : "send"}
        </button>
      </div>
    </div>
  );
}
