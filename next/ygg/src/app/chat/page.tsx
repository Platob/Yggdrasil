"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  getChannels, getMessages, sendMessage, pollMessages, createChannel,
  type Message, type ChannelInfo,
} from "@/lib/api";

const COLORS = ["text-red-400", "text-green-400", "text-yellow-400", "text-blue-400", "text-purple-400", "text-cyan-400"];

function senderColor(name: string): string {
  let h = 0;
  for (let i = 0; i < name.length; i++) h = (h * 31 + name.charCodeAt(i)) | 0;
  return COLORS[Math.abs(h) % COLORS.length];
}

function formatTime(ts: string): string {
  try {
    return new Date(ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  } catch {
    return ts.slice(11, 16);
  }
}

export default function ChatPage() {
  const [channels, setChannels] = useState<ChannelInfo[]>([]);
  const [channel, setChannel] = useState("general");
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [sender, setSender] = useState("web-user");
  const [newCh, setNewCh] = useState("");
  const [showNewCh, setShowNewCh] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const lastIdRef = useRef<string>("");
  const pollRef = useRef(true);

  const loadChannels = useCallback(async () => {
    try { setChannels(await getChannels()); } catch {}
  }, []);

  const loadMessages = useCallback(async () => {
    try {
      const msgs = await getMessages(channel, 200);
      setMessages(msgs);
      if (msgs.length > 0) lastIdRef.current = msgs[msgs.length - 1].id;
    } catch {}
  }, [channel]);

  useEffect(() => {
    loadChannels();
    loadMessages();
    pollRef.current = true;

    const poll = async () => {
      while (pollRef.current) {
        try {
          const newMsgs = await pollMessages(channel, lastIdRef.current, 25);
          if (newMsgs.length > 0) {
            setMessages((prev) => [...prev, ...newMsgs]);
            lastIdRef.current = newMsgs[newMsgs.length - 1].id;
          }
        } catch {
          await new Promise((r) => setTimeout(r, 2000));
        }
      }
    };
    poll();

    return () => { pollRef.current = false; };
  }, [channel, loadChannels, loadMessages]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function handleSend(e: React.FormEvent) {
    e.preventDefault();
    if (!input.trim()) return;
    const text = input;
    setInput("");
    try {
      const msg = await sendMessage(text, sender, channel);
      setMessages((prev) => [...prev, msg]);
      lastIdRef.current = msg.id;
    } catch {}
  }

  async function handleCreateChannel() {
    if (!newCh.trim()) return;
    try {
      await createChannel(newCh.trim());
      setShowNewCh(false);
      setNewCh("");
      loadChannels();
      setChannel(newCh.trim());
    } catch {}
  }

  return (
    <div className="flex h-[calc(100vh-3rem)] gap-4 animate-in">
      {/* Channel sidebar */}
      <div className="w-48 bg-card border border-border rounded-xl flex flex-col">
        <div className="p-3 border-b border-border flex items-center justify-between">
          <h3 className="text-xs font-semibold text-gold uppercase tracking-widest">Channels</h3>
          <button onClick={() => setShowNewCh(!showNewCh)} className="text-gold text-lg leading-none hover:text-gold-dim">+</button>
        </div>

        {showNewCh && (
          <div className="p-2 border-b border-border flex gap-1">
            <input
              value={newCh}
              onChange={(e) => setNewCh(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleCreateChannel()}
              placeholder="name"
              className="flex-1 bg-background border border-border rounded px-2 py-1 text-xs font-mono focus:outline-none focus:border-gold/40"
            />
          </div>
        )}

        <div className="flex-1 overflow-auto p-2 space-y-0.5">
          {channels.map((ch) => (
            <button
              key={ch.name}
              onClick={() => setChannel(ch.name)}
              className={`w-full text-left px-2.5 py-2 rounded-lg text-sm transition-all ${
                channel === ch.name
                  ? "bg-gold/10 text-gold"
                  : "text-muted hover:text-foreground hover:bg-card-hover"
              }`}
            >
              <span className="font-mono text-xs">#</span>{ch.name}
              <span className="block text-[10px] text-muted">{ch.message_count} msgs</span>
            </button>
          ))}
        </div>

        <div className="p-3 border-t border-border">
          <input
            value={sender}
            onChange={(e) => setSender(e.target.value)}
            className="w-full bg-background border border-border rounded px-2 py-1.5 text-xs font-mono focus:outline-none focus:border-gold/40"
            placeholder="your name"
          />
        </div>
      </div>

      {/* Chat area */}
      <div className="flex-1 bg-card border border-border rounded-xl flex flex-col">
        <div className="px-5 py-3 border-b border-border flex items-center gap-2">
          <span className="text-gold font-mono text-sm">#{channel}</span>
          <span className="text-muted text-xs">&middot; {messages.length} messages</span>
        </div>

        <div className="flex-1 overflow-auto p-4 space-y-1">
          {messages.map((msg, i) => {
            const showHeader = i === 0 || messages[i - 1].sender !== msg.sender;
            return (
              <div key={msg.id} className={`${showHeader ? "mt-3" : ""} animate-in`}>
                {showHeader && (
                  <div className="flex items-center gap-2 mb-0.5">
                    <span className={`text-sm font-semibold ${senderColor(msg.sender)}`}>{msg.sender}</span>
                    <span className="text-[10px] text-muted">{formatTime(msg.timestamp)}</span>
                  </div>
                )}
                <p className="text-sm text-foreground pl-0">{msg.text}</p>
              </div>
            );
          })}
          <div ref={bottomRef} />
        </div>

        <form onSubmit={handleSend} className="p-3 border-t border-border flex gap-2">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={`Message #${channel}...`}
            className="flex-1 bg-background border border-border rounded-lg px-3 py-2.5 text-sm font-mono focus:outline-none focus:border-gold/40 transition-colors"
            autoFocus
          />
          <button
            type="submit"
            disabled={!input.trim()}
            className="px-4 py-2.5 rounded-lg text-sm font-medium bg-gold/10 text-gold border border-gold/20 hover:bg-gold/20 disabled:opacity-30 transition-all"
          >
            Send
          </button>
        </form>
      </div>
    </div>
  );
}
