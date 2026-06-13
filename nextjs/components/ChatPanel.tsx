"use client";
import { useState, useEffect, useRef } from "react";
import { getChannels, getMessages, sendMessage } from "@/lib/api";
import type { Channel, Message } from "@/lib/types";

export default function ChatPanel() {
  const [channels, setChannels] = useState<Channel[]>([]);
  const [active, setActive] = useState("general");
  const [messages, setMessages] = useState<Message[]>([]);
  const [text, setText] = useState("");
  const [sender] = useState("user");
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    getChannels().then(setChannels).catch(() => setChannels([]));
  }, []);

  useEffect(() => {
    let cancelled = false;
    function poll() {
      getMessages(active, 50)
        .then((msgs) => { if (!cancelled) setMessages(msgs); })
        .catch(() => {});
    }
    poll();
    const id = setInterval(poll, 2000);
    return () => { cancelled = true; clearInterval(id); };
  }, [active]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function handleSend(e: React.FormEvent) {
    e.preventDefault();
    if (!text.trim()) return;
    try {
      await sendMessage(text.trim(), sender, active);
      setText("");
    } catch {}
  }

  return (
    <div className="flex h-full gap-4">
      {/* channel list */}
      <div className="w-44 shrink-0 bg-zinc-900 rounded-lg border border-zinc-800 p-2 space-y-0.5">
        <p className="text-zinc-500 text-xs px-2 py-1 font-medium">Channels</p>
        {channels.length === 0 && (
          <p className="text-zinc-600 text-xs px-2">—</p>
        )}
        {channels.map((c) => (
          <button
            key={c.name}
            onClick={() => setActive(c.name)}
            className={`w-full text-left px-2 py-1.5 rounded text-sm transition-colors
              ${active === c.name ? "bg-zinc-800 text-zinc-100" : "text-zinc-400 hover:bg-zinc-800/60"}`}
          >
            <span className="text-zinc-600">#</span> {c.name}
            <span className="float-right text-zinc-600 text-xs">{c.message_count}</span>
          </button>
        ))}
      </div>

      {/* message thread */}
      <div className="flex-1 flex flex-col bg-zinc-900 rounded-lg border border-zinc-800 overflow-hidden">
        <div className="px-4 py-2 border-b border-zinc-800 text-sm text-zinc-300 font-medium">
          #{active}
        </div>
        <div className="flex-1 overflow-y-auto p-4 space-y-2">
          {messages.map((m) => (
            <div key={m.id} className="flex gap-3">
              <span className="text-xs font-medium text-emerald-500 shrink-0 mt-0.5">
                {m.sender}
              </span>
              <span className="text-sm text-zinc-300 break-words">{m.text}</span>
              <span className="ml-auto text-zinc-700 text-xs shrink-0">
                {new Date(m.timestamp * 1000).toLocaleTimeString()}
              </span>
            </div>
          ))}
          <div ref={bottomRef} />
        </div>
        <form onSubmit={handleSend} className="border-t border-zinc-800 p-3 flex gap-2">
          <input
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder={`Message #${active}`}
            className="flex-1 bg-zinc-800 border border-zinc-700 rounded px-3 py-2 text-sm text-zinc-100 placeholder:text-zinc-600 focus:outline-none focus:border-emerald-600"
          />
          <button
            type="submit"
            className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white text-sm rounded transition-colors"
          >
            Send
          </button>
        </form>
      </div>
    </div>
  );
}
