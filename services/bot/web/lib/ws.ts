type QuoteUpdate = {
  symbol: string;
  price: number;
  change: number;
  change_pct: number;
  ts: string;
};

type WsMessage =
  | { type: "quotes"; data: QuoteUpdate[] }
  | { type: "subscribed"; symbols: string[] }
  | { type: "pong"; ts: number };

type QuoteHandler = (updates: QuoteUpdate[]) => void;

export class MarketWs {
  private ws: WebSocket | null = null;
  private handlers: Set<QuoteHandler> = new Set();
  private subscribed: Set<string> = new Set();
  private reconnectDelay = 2000;
  private url: string;

  constructor(url = `ws://${typeof window !== "undefined" ? window.location.host : "localhost:8000"}/api/ws/market`) {
    this.url = url;
  }

  connect(): void {
    if (this.ws?.readyState === WebSocket.OPEN) return;
    this.ws = new WebSocket(this.url);

    this.ws.onopen = () => {
      this.reconnectDelay = 2000;
      if (this.subscribed.size > 0) {
        this.send({ action: "subscribe", symbols: [...this.subscribed] });
      }
    };

    this.ws.onmessage = (evt) => {
      const msg: WsMessage = JSON.parse(evt.data);
      if (msg.type === "quotes") {
        this.handlers.forEach((h) => h(msg.data));
      }
    };

    this.ws.onclose = () => {
      setTimeout(() => this.connect(), this.reconnectDelay);
      this.reconnectDelay = Math.min(this.reconnectDelay * 2, 30000);
    };

    this.ws.onerror = () => this.ws?.close();
  }

  subscribe(symbols: string[]): void {
    symbols.forEach((s) => this.subscribed.add(s));
    this.send({ action: "subscribe", symbols });
  }

  unsubscribe(symbols: string[]): void {
    symbols.forEach((s) => this.subscribed.delete(s));
    this.send({ action: "unsubscribe", symbols });
  }

  onQuotes(handler: QuoteHandler): () => void {
    this.handlers.add(handler);
    return () => this.handlers.delete(handler);
  }

  disconnect(): void {
    this.ws?.close();
    this.ws = null;
  }

  private send(data: object): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    }
  }
}

export const marketWs = typeof window !== "undefined" ? new MarketWs() : null;
