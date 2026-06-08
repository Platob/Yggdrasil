// Client-side port of ``yggdrasil.enums.byteunit``.
//
// PARITY: python/src/yggdrasil/enums/byteunit.py. Base-1024 (IEC binary) byte
// units with forgiving parsing + human formatting. ``KB`` etc. mean the binary
// form (KB == KiB == 1024) — no SI base-1000, same as the Python source.

const ALIASES: Record<string, string> = {
  "": "B", b: "B", byte: "B", bytes: "B",
  k: "KIB", kb: "KIB", kib: "KIB", kibibyte: "KIB", kibibytes: "KIB", kilobyte: "KIB", kilobytes: "KIB",
  m: "MIB", mb: "MIB", mib: "MIB", mebibyte: "MIB", mebibytes: "MIB", megabyte: "MIB", megabytes: "MIB",
  g: "GIB", gb: "GIB", gib: "GIB", gibibyte: "GIB", gibibytes: "GIB", gigabyte: "GIB", gigabytes: "GIB",
  t: "TIB", tb: "TIB", tib: "TIB", tebibyte: "TIB", tebibytes: "TIB", terabyte: "TIB", terabytes: "TIB",
  p: "PIB", pb: "PIB", pib: "PIB", pebibyte: "PIB", pebibytes: "PIB", petabyte: "PIB", petabytes: "PIB",
};

const QUANTITY = /^([\d.]+)\s*([a-zA-Z]*)$/;

export class ByteUnit {
  private constructor(readonly name: string, readonly bytes: number, readonly iec: string, readonly short: string) {}

  static readonly B = new ByteUnit("B", 1, "B", "B");
  static readonly KIB = new ByteUnit("KIB", 1024, "KiB", "KB");
  static readonly MIB = new ByteUnit("MIB", 1024 ** 2, "MiB", "MB");
  static readonly GIB = new ByteUnit("GIB", 1024 ** 3, "GiB", "GB");
  static readonly TIB = new ByteUnit("TIB", 1024 ** 4, "TiB", "TB");
  static readonly PIB = new ByteUnit("PIB", 1024 ** 5, "PiB", "PB");
  // colloquial short aliases (binary forms)
  static readonly KB = ByteUnit.KIB;
  static readonly MB = ByteUnit.MIB;
  static readonly GB = ByteUnit.GIB;
  static readonly TB = ByteUnit.TIB;
  static readonly PB = ByteUnit.PIB;

  private static readonly _members = [ByteUnit.B, ByteUnit.KIB, ByteUnit.MIB, ByteUnit.GIB, ByteUnit.TIB, ByteUnit.PIB];

  valueOf(): number { return this.bytes; }
  toString(): string { return this.iec; }

  /** Resolve a unit token (``"MiB"`` / ``"mb"`` / ByteUnit) to a ByteUnit. */
  static from(value: string | ByteUnit, dflt?: ByteUnit): ByteUnit {
    if (value instanceof ByteUnit) return value;
    const name = ALIASES[String(value).trim().toLowerCase()];
    const m = name ? ByteUnit._members.find((u) => u.name === name) : undefined;
    if (m) return m;
    if (dflt) return dflt;
    throw new Error(`Cannot parse ${JSON.stringify(value)} as a ByteUnit.`);
  }

  /** Coerce a size-like value to an integer byte count (``"128 MB"`` -> 134217728). */
  static parseSize(value: number | string | ByteUnit, dflt?: number): number {
    if (value instanceof ByteUnit) return value.bytes;
    if (typeof value === "number") {
      if (value < 0) throw new Error("parse_size: negative size");
      return Math.trunc(value);
    }
    const s = value.trim();
    const m = QUANTITY.exec(s);
    try {
      if (m) {
        const unit = m[2] ? ByteUnit.from(m[2]) : ByteUnit.B;
        const bytes = Math.round(parseFloat(m[1]) * unit.bytes);
        if (bytes < 0) throw new Error("parse_size: negative size");
        return bytes;
      }
      return ByteUnit.from(s).bytes; // bare unit string ("MiB") -> one unit
    } catch (e) {
      if (dflt != null) return dflt;
      throw e;
    }
  }

  /** Human-readable byte count: largest unit at which n >= 1 (IEC by default). */
  static format(n: number, opts: { iec?: boolean; precision?: number } = {}): string {
    const iec = opts.iec ?? true;
    const precision = opts.precision ?? 1;
    if (n < 0) return "-" + ByteUnit.format(-n, opts);
    for (const unit of [ByteUnit.PIB, ByteUnit.TIB, ByteUnit.GIB, ByteUnit.MIB, ByteUnit.KIB]) {
      if (n >= unit.bytes) return `${(n / unit.bytes).toFixed(precision)} ${iec ? unit.iec : unit.short}`;
    }
    return `${n} B`;
  }

  /**
   * Human-readable rendering of a quantity ``v`` expressed in ``unit``.
   *
   * ``v`` is a scalar count of ``unit`` (bytes by default); it's scaled to a
   * byte count and handed to {@link format}. The companion to ``format`` for
   * the "I have N MiB, show it nicely" case:
   *
   *     ByteUnit.pretty(1536)             // "1.5 KiB"
   *     ByteUnit.pretty(8, ByteUnit.MIB)  // "8.0 MiB"
   */
  static pretty(v: number, unit: ByteUnit = ByteUnit.B, opts: { iec?: boolean; precision?: number } = {}): string {
    return ByteUnit.format(Math.round(v * unit.bytes), opts);
  }
}
