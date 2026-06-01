// Client-side port of ``yggdrasil.enums.state``.
//
// PARITY: python/src/yggdrasil/enums/state.py. One shared execution / order
// lifecycle vocabulary (IDLE -> … -> terminal) with forgiving parsing. Member
// codes and the alias table mirror the Python source — keep them in sync.

const ALIASES: Record<string, string> = {
  "": "IDLE", idle: "IDLE", new: "IDLE", not_started: "IDLE", draft: "IDLE", created: "IDLE",
  queued: "QUEUED", waiting: "QUEUED", scheduled: "QUEUED",
  pending: "PENDING", submitted: "PENDING", pending_new: "PENDING", sent: "PENDING",
  accepted: "ACCEPTED", ack: "ACCEPTED", acknowledged: "ACCEPTED", working: "ACCEPTED", open: "ACCEPTED", ready: "ACCEPTED",
  running: "RUNNING", in_progress: "RUNNING", started: "RUNNING", active: "RUNNING", executing: "RUNNING",
  partial: "PARTIAL", partially_filled: "PARTIAL", partial_fill: "PARTIAL", partially_complete: "PARTIAL", streaming: "PARTIAL",
  succeeded: "SUCCEEDED", success: "SUCCEEDED", completed: "SUCCEEDED", complete: "SUCCEEDED", done: "SUCCEEDED",
  ok: "SUCCEEDED", finished: "SUCCEEDED", filled: "SUCCEEDED", settled: "SUCCEEDED", closed: "SUCCEEDED",
  rejected: "REJECTED", reject: "REJECTED", refused: "REJECTED", denied: "REJECTED",
  failed: "FAILED", fail: "FAILED", error: "FAILED", errored: "FAILED", broken: "FAILED",
  canceled: "CANCELED", cancelled: "CANCELED", aborted: "CANCELED", abort: "CANCELED", killed: "CANCELED", stopped: "CANCELED",
  expired: "EXPIRED", expire: "EXPIRED", timed_out: "EXPIRED", timeout: "EXPIRED", ttl_elapsed: "EXPIRED", done_for_day: "EXPIRED",
};

export type StateLike = State | string | number | null | undefined;

export class State {
  private constructor(readonly name: string, readonly code: number) {}

  static readonly IDLE = new State("IDLE", 0);
  static readonly QUEUED = new State("QUEUED", 1);
  static readonly PENDING = new State("PENDING", 2);
  static readonly ACCEPTED = new State("ACCEPTED", 3);
  static readonly RUNNING = new State("RUNNING", 4);
  static readonly PARTIAL = new State("PARTIAL", 5);
  static readonly SUCCEEDED = new State("SUCCEEDED", 6);
  static readonly REJECTED = new State("REJECTED", 7);
  static readonly FAILED = new State("FAILED", 8);
  static readonly CANCELED = new State("CANCELED", 9);
  static readonly EXPIRED = new State("EXPIRED", 10);

  private static readonly _members = [
    State.IDLE, State.QUEUED, State.PENDING, State.ACCEPTED, State.RUNNING, State.PARTIAL,
    State.SUCCEEDED, State.REJECTED, State.FAILED, State.CANCELED, State.EXPIRED,
  ];

  valueOf(): number { return this.code; }
  toString(): string { return this.name; }

  /** Terminal — no more transitions expected (SUCCEEDED…EXPIRED). */
  get isDone(): boolean { return this.code >= 6; }
  get isTerminal(): boolean { return this.code >= 6; }
  /** Non-success terminal (REJECTED / FAILED / CANCELED / EXPIRED). */
  get isFailed(): boolean { return this.code >= 7; }
  get isSucceeded(): boolean { return this === State.SUCCEEDED; }
  /** Non-terminal and submitted — something a caller can wait on. */
  get isActive(): boolean { return this.code >= 1 && this.code <= 5; }
  get isStarted(): boolean { return this.code >= 3 && this.code <= 5; }
  get isIdle(): boolean { return this === State.IDLE; }

  static members(): State[] { return [...State._members]; }

  /** Coerce a State / alias string / int code into a State (mirrors ``from_``). */
  static from(value: StateLike, dflt?: State): State {
    if (value instanceof State) return value;
    if (value == null) return dflt ?? State.IDLE;
    if (typeof value === "number") {
      const m = State._members.find((s) => s.code === value);
      if (m) return m;
      if (dflt) return dflt;
      throw new Error(`Cannot parse ${value} as a State (codes 0..10).`);
    }
    const key = String(value).trim().toLowerCase().replace(/[\s-]+/g, "_");
    const name = ALIASES[key] ?? (key.toUpperCase() in State ? key.toUpperCase() : undefined);
    const m = name ? State._members.find((s) => s.name === name) : undefined;
    if (m) return m;
    if (dflt) return dflt;
    throw new Error(`Cannot parse ${JSON.stringify(value)} as a State.`);
  }
}
