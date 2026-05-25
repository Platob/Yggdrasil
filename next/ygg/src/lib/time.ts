/**
 * Java-style time utilities for correct datetime/timezone handling.
 * All types are immutable value objects.
 */

// ─── ZoneId ─────────────────────────────────────────────────
export class ZoneId {
  private constructor(public readonly id: string) {}

  static of(zoneId: string): ZoneId {
    return new ZoneId(zoneId);
  }

  static from_(value: string | ZoneId | null | undefined): ZoneId | null {
    if (!value) return null;
    if (value instanceof ZoneId) return value;
    return ZoneId.of(value);
  }
  static UTC = new ZoneId("UTC");
  static SYSTEM = new ZoneId(
    Intl.DateTimeFormat().resolvedOptions().timeZone,
  );

  toString(): string {
    return this.id;
  }
  equals(other: ZoneId): boolean {
    return this.id === other.id;
  }
}

// ─── Instant ────────────────────────────────────────────────
/** A point on the timeline (epoch milliseconds). Like java.time.Instant. */
export class Instant {
  private constructor(public readonly epochMillis: number) {}

  static now(): Instant {
    return new Instant(Date.now());
  }
  static ofEpochMillis(ms: number): Instant {
    return new Instant(ms);
  }
  static ofEpochSeconds(s: number): Instant {
    return new Instant(s * 1000);
  }
  static parse(iso: string): Instant {
    return new Instant(new Date(iso).getTime());
  }

  static from_(value: string | number | Date | Instant | null | undefined): Instant | null {
    if (!value) return null;
    if (value instanceof Instant) return value;
    if (value instanceof Date) return Instant.ofEpochMillis(value.getTime());
    if (typeof value === "number") return Instant.ofEpochMillis(value);
    try { return Instant.parse(String(value)); } catch { return null; }
  }

  get epochSeconds(): number {
    return Math.floor(this.epochMillis / 1000);
  }

  toDate(): Date {
    return new Date(this.epochMillis);
  }
  toISO(): string {
    return this.toDate().toISOString();
  }

  atZone(zone: ZoneId): ZonedDateTime {
    return ZonedDateTime.ofInstant(this, zone);
  }

  plus(duration: Duration): Instant {
    return new Instant(this.epochMillis + duration.toMillis());
  }
  minus(duration: Duration): Instant {
    return new Instant(this.epochMillis - duration.toMillis());
  }

  isAfter(other: Instant): boolean {
    return this.epochMillis > other.epochMillis;
  }
  isBefore(other: Instant): boolean {
    return this.epochMillis < other.epochMillis;
  }

  durationSince(other: Instant): Duration {
    return Duration.ofMillis(this.epochMillis - other.epochMillis);
  }

  /** Human-readable relative time: "2m ago", "3h ago", "just now" */
  toRelative(): string {
    const diff = Date.now() - this.epochMillis;
    if (diff < 0) return "in the future";
    if (diff < 1000) return "just now";
    if (diff < 60_000) return `${Math.floor(diff / 1000)}s ago`;
    if (diff < 3_600_000) return `${Math.floor(diff / 60_000)}m ago`;
    if (diff < 86_400_000) return `${Math.floor(diff / 3_600_000)}h ago`;
    return `${Math.floor(diff / 86_400_000)}d ago`;
  }

  toString(): string {
    return this.toISO();
  }
  equals(other: Instant): boolean {
    return this.epochMillis === other.epochMillis;
  }
}

// ─── Duration ───────────────────────────────────────────────
/** A time-based amount. Like java.time.Duration. */
export class Duration {
  private constructor(private readonly millis: number) {}

  static ofMillis(ms: number): Duration {
    return new Duration(ms);
  }
  static ofSeconds(s: number): Duration {
    return new Duration(s * 1000);
  }
  static ofMinutes(m: number): Duration {
    return new Duration(m * 60_000);
  }
  static ofHours(h: number): Duration {
    return new Duration(h * 3_600_000);
  }
  static ofDays(d: number): Duration {
    return new Duration(d * 86_400_000);
  }
  static ZERO = new Duration(0);

  /** Parse from seconds (API returns duration as float seconds) */
  static fromSeconds(s: number): Duration {
    return new Duration(s * 1000);
  }

  static from_(value: number | Duration | null | undefined): Duration | null {
    if (!value && value !== 0) return null;
    if (value instanceof Duration) return value;
    return Duration.fromSeconds(value as number);
  }

  /** Between two instants */
  static between(start: Instant, end: Instant): Duration {
    return new Duration(end.epochMillis - start.epochMillis);
  }

  toMillis(): number {
    return this.millis;
  }
  toSeconds(): number {
    return this.millis / 1000;
  }
  toMinutes(): number {
    return this.millis / 60_000;
  }
  toHours(): number {
    return this.millis / 3_600_000;
  }

  plus(other: Duration): Duration {
    return new Duration(this.millis + other.millis);
  }
  minus(other: Duration): Duration {
    return new Duration(this.millis - other.millis);
  }

  isZero(): boolean {
    return this.millis === 0;
  }
  isNegative(): boolean {
    return this.millis < 0;
  }
  abs(): Duration {
    return new Duration(Math.abs(this.millis));
  }

  /** Human-readable: "1h 23m", "45s", "123ms" */
  toHuman(): string {
    const abs = Math.abs(this.millis);
    if (abs < 1000) return `${abs}ms`;
    if (abs < 60_000) return `${(abs / 1000).toFixed(1)}s`;
    if (abs < 3_600_000) {
      const m = Math.floor(abs / 60_000);
      const s = Math.floor((abs % 60_000) / 1000);
      return s > 0 ? `${m}m ${s}s` : `${m}m`;
    }
    const h = Math.floor(abs / 3_600_000);
    const m = Math.floor((abs % 3_600_000) / 60_000);
    return m > 0 ? `${h}h ${m}m` : `${h}h`;
  }

  toString(): string {
    return this.toHuman();
  }
}

// ─── LocalDate ──────────────────────────────────────────────
/** A date without time or timezone. Like java.time.LocalDate. */
export class LocalDate {
  constructor(
    public readonly year: number,
    public readonly month: number, // 1-12
    public readonly day: number, // 1-31
  ) {}

  static now(zone?: ZoneId): LocalDate {
    const d = zone
      ? new Date(new Date().toLocaleString("en-US", { timeZone: zone.id }))
      : new Date();
    return new LocalDate(d.getFullYear(), d.getMonth() + 1, d.getDate());
  }

  static of(year: number, month: number, day: number): LocalDate {
    return new LocalDate(year, month, day);
  }

  static parse(iso: string): LocalDate {
    const [y, m, d] = iso.split("-").map(Number);
    return new LocalDate(y, m, d);
  }

  static from_(value: string | LocalDate | null | undefined): LocalDate | null {
    if (!value) return null;
    if (value instanceof LocalDate) return value;
    try { return LocalDate.parse(String(value)); } catch { return null; }
  }

  atTime(time: LocalTime): LocalDateTime {
    return new LocalDateTime(this, time);
  }

  toISO(): string {
    return `${this.year}-${String(this.month).padStart(2, "0")}-${String(this.day).padStart(2, "0")}`;
  }

  toString(): string {
    return this.toISO();
  }
  equals(other: LocalDate): boolean {
    return (
      this.year === other.year &&
      this.month === other.month &&
      this.day === other.day
    );
  }
}

// ─── LocalTime ──────────────────────────────────────────────
/** A time without date or timezone. Like java.time.LocalTime. */
export class LocalTime {
  constructor(
    public readonly hour: number, // 0-23
    public readonly minute: number, // 0-59
    public readonly second: number = 0,
    public readonly millis: number = 0,
  ) {}

  static now(zone?: ZoneId): LocalTime {
    const d = zone
      ? new Date(new Date().toLocaleString("en-US", { timeZone: zone.id }))
      : new Date();
    return new LocalTime(
      d.getHours(),
      d.getMinutes(),
      d.getSeconds(),
      d.getMilliseconds(),
    );
  }

  static of(h: number, m: number, s = 0, ms = 0): LocalTime {
    return new LocalTime(h, m, s, ms);
  }

  static parse(iso: string): LocalTime {
    const parts = iso.split(/[:.]/).map(Number);
    return new LocalTime(
      parts[0] || 0,
      parts[1] || 0,
      parts[2] || 0,
      parts[3] || 0,
    );
  }

  static MIDNIGHT = new LocalTime(0, 0, 0, 0);
  static NOON = new LocalTime(12, 0, 0, 0);

  toISO(): string {
    const h = String(this.hour).padStart(2, "0");
    const m = String(this.minute).padStart(2, "0");
    const s = String(this.second).padStart(2, "0");
    return this.millis
      ? `${h}:${m}:${s}.${String(this.millis).padStart(3, "0")}`
      : `${h}:${m}:${s}`;
  }

  toString(): string {
    return this.toISO();
  }
}

// ─── LocalDateTime ──────────────────────────────────────────
/** Date + time without timezone. Like java.time.LocalDateTime. */
export class LocalDateTime {
  constructor(
    public readonly date: LocalDate,
    public readonly time: LocalTime,
  ) {}

  static now(zone?: ZoneId): LocalDateTime {
    return new LocalDateTime(LocalDate.now(zone), LocalTime.now(zone));
  }

  static of(
    year: number,
    month: number,
    day: number,
    hour = 0,
    minute = 0,
    second = 0,
  ): LocalDateTime {
    return new LocalDateTime(
      LocalDate.of(year, month, day),
      LocalTime.of(hour, minute, second),
    );
  }

  static parse(iso: string): LocalDateTime {
    const [datePart, timePart] = iso.split("T");
    return new LocalDateTime(
      LocalDate.parse(datePart),
      LocalTime.parse(timePart || "00:00:00"),
    );
  }

  atZone(zone: ZoneId): ZonedDateTime {
    // Create a Date in the target zone and get the instant
    const iso = `${this.date.toISO()}T${this.time.toISO()}`;
    const d = new Date(iso);
    // This is approximate — proper tz handling would need a library
    return new ZonedDateTime(Instant.ofEpochMillis(d.getTime()), zone);
  }

  get year(): number {
    return this.date.year;
  }
  get month(): number {
    return this.date.month;
  }
  get day(): number {
    return this.date.day;
  }
  get hour(): number {
    return this.time.hour;
  }
  get minute(): number {
    return this.time.minute;
  }
  get second(): number {
    return this.time.second;
  }

  toISO(): string {
    return `${this.date.toISO()}T${this.time.toISO()}`;
  }
  toString(): string {
    return this.toISO();
  }
}

// ─── ZonedDateTime ──────────────────────────────────────────
/** Instant + timezone. Like java.time.ZonedDateTime. */
export class ZonedDateTime {
  constructor(
    public readonly instant: Instant,
    public readonly zone: ZoneId,
  ) {}

  static now(zone?: ZoneId): ZonedDateTime {
    return new ZonedDateTime(Instant.now(), zone || ZoneId.SYSTEM);
  }

  static ofInstant(instant: Instant, zone: ZoneId): ZonedDateTime {
    return new ZonedDateTime(instant, zone);
  }

  static parse(iso: string): ZonedDateTime {
    const instant = Instant.parse(iso);
    // Extract timezone from ISO string if present
    const match = iso.match(/([+-]\d{2}:\d{2}|Z)$/);
    const zone =
      match && match[1] !== "Z" ? ZoneId.of(`UTC${match[1]}`) : ZoneId.UTC;
    return new ZonedDateTime(instant, zone);
  }

  static from_(value: string | ZonedDateTime | null | undefined): ZonedDateTime | null {
    if (!value) return null;
    if (value instanceof ZonedDateTime) return value;
    try { return ZonedDateTime.parse(String(value)); } catch { return null; }
  }

  /** Format in the target timezone */
  format(options?: Intl.DateTimeFormatOptions): string {
    return this.instant.toDate().toLocaleString("en-US", {
      timeZone: this.zone.id,
      ...options,
    });
  }

  toLocalDateTime(): LocalDateTime {
    const formatted = this.instant.toDate().toLocaleString("en-CA", {
      timeZone: this.zone.id,
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
    });
    // Parse "YYYY-MM-DD, HH:MM:SS" format
    const [datePart, timePart] = formatted.split(", ");
    return new LocalDateTime(
      LocalDate.parse(datePart),
      LocalTime.parse(timePart),
    );
  }

  withZoneSameInstant(zone: ZoneId): ZonedDateTime {
    return new ZonedDateTime(this.instant, zone);
  }

  get epochMillis(): number {
    return this.instant.epochMillis;
  }

  toISO(): string {
    return this.instant.toISO();
  }
  toString(): string {
    return `${this.format()} [${this.zone.id}]`;
  }
}

// ─── Convenience Helpers ────────────────────────────────────

/** Parse any API timestamp (ISO 8601 string or null) into an Instant */
export function parseTimestamp(
  iso: string | null | undefined,
): Instant | null {
  if (!iso) return null;
  return Instant.parse(iso);
}

/** Format a duration in seconds (from API) to human-readable */
export function formatDuration(
  seconds: number | null | undefined,
): string {
  if (seconds == null) return "—";
  return Duration.fromSeconds(seconds).toHuman();
}

/** Format an ISO timestamp as relative ("2m ago") */
export function formatRelative(
  iso: string | null | undefined,
): string {
  if (!iso) return "—";
  return Instant.parse(iso).toRelative();
}
