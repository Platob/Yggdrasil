// Client-side port of ``yggdrasil.enums.mime_type`` — object-oriented, mirroring
// the Python ``MimeType`` dataclass + registry. Keep in sync with
// python/src/yggdrasil/enums/mime_type.py; the two are one contract.

export interface Magic {
  prefix: number[]; // byte values
  offset?: number;  // default 0 (prefix); tar's ``ustar`` sits at 257
}

export interface MimeOpts {
  ext?: string[];
  codec?: boolean;
  tabular?: boolean;
  blob?: boolean;
  magic?: Magic[];
}

// ascii string -> byte values
const bytes = (s: string): number[] => Array.from(s, (c) => c.charCodeAt(0));

/** A MIME descriptor + the global registry (matches Python's ``MimeType``). */
export class MimeType {
  readonly extensions: readonly string[];
  readonly isCodec: boolean;
  readonly isTabular: boolean;
  readonly isBlob: boolean;
  readonly magics: readonly Magic[];

  constructor(readonly name: string, readonly value: string, opts: MimeOpts = {}) {
    this.extensions = opts.ext ?? [];
    this.isCodec = !!opts.codec;
    this.isTabular = !!opts.tabular;
    this.isBlob = !!opts.blob;
    this.magics = opts.magic ?? [];
  }

  /** Primary (canonical) extension, if any. */
  get extension(): string | undefined {
    return this.extensions[0];
  }

  get isOctet(): boolean {
    return this.value === "application/octet-stream";
  }

  toString(): string {
    return this.value;
  }

  // -- registry ---------------------------------------------------------------

  private static _byName = new Map<string, MimeType>();
  private static _byValue = new Map<string, MimeType>();
  private static _byExt = new Map<string, MimeType>();
  private static _withMagic: MimeType[] = [];
  private static _codecExts = new Set<string>();

  /** Register and return a MimeType (mirrors Python ``MimeType.define``). */
  static define(mt: MimeType): MimeType {
    MimeType._byName.set(mt.name.toLowerCase(), mt);
    MimeType._byValue.set(mt.value.toLowerCase(), mt);
    for (const e of mt.extensions) {
      const k = e.toLowerCase().replace(/^\.+/, "");
      MimeType._byExt.set(k, mt);
      if (mt.isCodec) MimeType._codecExts.add(k);
    }
    if (mt.magics.length) MimeType._withMagic.push(mt);
    return mt;
  }

  /** Pure lookup by mime value or registered name. ``null`` on miss. */
  static get(s: string): MimeType | null {
    const k = s.trim().toLowerCase();
    return MimeType._byValue.get(k) ?? MimeType._byName.get(k) ?? null;
  }

  /** Resolve a dotless extension. */
  static fromExtension(ext: string): MimeType | null {
    return MimeType._byExt.get(ext.toLowerCase().replace(/^\.+/, "")) ?? null;
  }

  /**
   * Resolve a filename/path to its *format* MimeType, honoring a trailing
   * compression wrapper (``trades.csv.gz`` -> CSV). ``codec`` is the wrapper.
   */
  static fromName(name: string): { mime: MimeType | null; codec: MimeType | null } {
    const parts = name.toLowerCase().split(".");
    if (parts.length < 2) return { mime: null, codec: null };
    let ext = parts[parts.length - 1];
    let codec: MimeType | null = null;
    if (MimeType._codecExts.has(ext) && parts.length >= 3) {
      codec = MimeType.fromExtension(ext);
      ext = parts[parts.length - 2];
    }
    return { mime: MimeType.fromExtension(ext), codec };
  }

  /** Best-effort magic-byte sniff over a head buffer. */
  static fromMagic(head: Uint8Array): MimeType | null {
    if (!head.length) return null;
    for (const mt of MimeType._withMagic) {
      for (const mg of mt.magics) {
        const off = mg.offset ?? 0;
        if (off + mg.prefix.length > head.length) continue;
        let ok = true;
        for (let i = 0; i < mg.prefix.length; i++) {
          if (head[off + i] !== mg.prefix[i]) { ok = false; break; }
        }
        if (ok) return mt;
      }
    }
    const c0 = head[0];
    if (c0 === 0x7b || c0 === 0x5b) return MimeTypes.JSON; // { or [
    if (c0 === 0x3c) return MimeTypes.XML;                  // <
    return null;
  }

  /** All registered MimeTypes (deduped, definition order). */
  static all(): MimeType[] {
    return Array.from(new Set(MimeType._byName.values()));
  }
}

const def = (name: string, value: string, opts?: MimeOpts) =>
  MimeType.define(new MimeType(name, value, opts));

export const MimeTypes = {
  // --- Compression / codecs ---
  GZIP: def("GZIP", "application/gzip", { ext: ["gz", "gzip", "tgz"], codec: true, magic: [{ prefix: [0x1f, 0x8b] }] }),
  ZSTD: def("ZSTD", "application/zstd", { ext: ["zst", "zstd"], codec: true, magic: [{ prefix: [0x28, 0xb5, 0x2f, 0xfd] }] }),
  BROTLI: def("BROTLI", "application/x-brotli", { ext: ["br", "brotli"], codec: true }),
  LZ4: def("LZ4", "application/x-lz4", { ext: ["lz4"], codec: true, magic: [{ prefix: [0x04, 0x22, 0x4d, 0x18] }] }),
  SNAPPY: def("SNAPPY", "application/x-snappy", { ext: ["snappy", "sz"], codec: true }),
  BZ2: def("BZ2", "application/x-bzip2", { ext: ["bz2", "bzip2", "tbz2"], codec: true, magic: [{ prefix: [0x42, 0x5a, 0x68] }] }),
  XZ: def("XZ", "application/x-xz", { ext: ["xz", "txz"], codec: true, magic: [{ prefix: [0xfd, 0x37, 0x7a, 0x58, 0x5a, 0x00] }] }),
  ZLIB: def("ZLIB", "application/zlib", { ext: ["zlib"], codec: true, magic: [{ prefix: [0x78, 0x01] }, { prefix: [0x78, 0x9c] }, { prefix: [0x78, 0xda] }] }),
  LZMA: def("LZMA", "application/x-lzma", { ext: ["lzma"], codec: true }),
  ZZIP: def("ZZIP", "application/x-compress", { ext: ["z"], codec: true }),

  // --- Archives & multi-file containers (blobs) ---
  ZIP: def("ZIP", "application/zip", { ext: ["zip"], blob: true, magic: [{ prefix: bytes("PK\x03\x04") }] }),
  ZIP_ENTRY: def("ZIP_ENTRY", "application/zip-entry", { ext: ["zipentry"], blob: true, magic: [{ prefix: bytes("PK\x01\x02") }] }),
  TAR: def("TAR", "application/x-tar", { ext: ["tar"], blob: true, magic: [{ prefix: bytes("ustar"), offset: 257 }] }),
  SEVEN_ZIP: def("SEVEN_ZIP", "application/x-7z-compressed", { ext: ["7z"], blob: true, magic: [{ prefix: [0x37, 0x7a, 0xbc, 0xaf, 0x27, 0x1c] }] }),
  RAR: def("RAR", "application/vnd.rar", { ext: ["rar"], blob: true, magic: [{ prefix: bytes("Rar!\x1a\x07") }] }),

  // --- Documents / office ---
  PDF: def("PDF", "application/pdf", { ext: ["pdf"], blob: true, magic: [{ prefix: bytes("%PDF-") }] }),
  XLSX: def("XLSX", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", { ext: ["xlsx", "xls"], tabular: true }),
  DOCX: def("DOCX", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", { ext: ["docx"], blob: true }),

  // --- Embedded stores (single-file databases) ---
  SQLITE: def("SQLITE", "application/vnd.sqlite3", { ext: ["db", "sqlite", "sqlite3"], blob: true, magic: [{ prefix: bytes("SQLite format 3\x00") }] }),
  HDF5: def("HDF5", "application/x-hdf5", { ext: ["h5", "hdf5", "he5"], blob: true, magic: [{ prefix: [0x89, 0x48, 0x44, 0x46, 0x0d, 0x0a, 0x1a, 0x0a] }] }),

  // --- Columnar / analytics (tabular) ---
  PARQUET: def("PARQUET", "application/vnd.apache.parquet", { ext: ["parquet", "pq"], tabular: true, magic: [{ prefix: bytes("PAR1") }] }),
  PARQUET_DELTA: def("PARQUET_DELTA", "application/vnd.apache.parquet+delta", { tabular: true }),
  ARROW_IPC: def("ARROW_IPC", "application/vnd.apache.arrow.file", { ext: ["ipc", "feather", "arrow", "arrows"], tabular: true, magic: [{ prefix: bytes("ARROW1") }] }),
  ARROW_STREAM: def("ARROW_STREAM", "application/vnd.apache.arrow.stream", { tabular: true }),
  ORC: def("ORC", "application/vnd.apache.orc", { ext: ["orc"], tabular: true, magic: [{ prefix: bytes("ORC") }] }),
  AVRO: def("AVRO", "application/vnd.apache.avro", { ext: ["avro"], tabular: true, magic: [{ prefix: bytes("Obj\x01") }] }),
  ICEBERG: def("ICEBERG", "application/vnd.apache.iceberg", { ext: ["iceberg"], tabular: true }),
  DELTA: def("DELTA", "application/vnd.delta", { ext: ["delta", "deltatable"], tabular: true }),

  // --- Tabular text ---
  JSON: def("JSON", "application/json", { ext: ["json"], tabular: true }),
  NDJSON: def("NDJSON", "application/ld+json", { ext: ["ndjson"], tabular: true }),
  CSV: def("CSV", "text/csv", { ext: ["csv"], tabular: true }),
  TSV: def("TSV", "text/tab-separated-values", { ext: ["tsv"], tabular: true }),

  // --- Text / markup (blobs) ---
  XML: def("XML", "application/xml", { ext: ["xml"], blob: true }),
  HTML: def("HTML", "text/html", { ext: ["html", "htm"], blob: true }),
  PLAIN: def("PLAIN", "text/plain", { ext: ["txt", "text"], blob: true }),
  MARKDOWN: def("MARKDOWN", "text/markdown", { ext: ["md", "markdown"], blob: true }),
  YAML: def("YAML", "application/yaml", { ext: ["yaml", "yml"], blob: true }),
  TOML: def("TOML", "application/toml", { ext: ["toml"], blob: true }),

  // --- Binary serialisation (blobs) ---
  MSGPACK: def("MSGPACK", "application/msgpack", { ext: ["msgpack", "mpk"], blob: true }),
  PROTOBUF: def("PROTOBUF", "application/x-protobuf", { ext: ["pb", "proto", "protobuf"], blob: true }),
  FLATBUFFERS: def("FLATBUFFERS", "application/x-flatbuffers", { ext: ["fbs"], blob: true }),
  CBOR: def("CBOR", "application/cbor", { ext: ["cbor"], blob: true }),
  BSON: def("BSON", "application/bson", { ext: ["bson"], blob: true }),
  PICKLE: def("PICKLE", "application/x-python-pickle", { ext: ["pkl", "pickle"], blob: true }),
  NUMPY: def("NUMPY", "application/x-npy", { ext: ["npy"], blob: true, magic: [{ prefix: bytes("\x93NUMPY") }] }),
  NUMPY_ARCHIVE: def("NUMPY_ARCHIVE", "application/x-npz", { ext: ["npz"], blob: true }),

  // --- Images (blobs) ---
  PNG: def("PNG", "image/png", { ext: ["png"], blob: true, magic: [{ prefix: [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a] }] }),
  JPEG: def("JPEG", "image/jpeg", { ext: ["jpg", "jpeg"], blob: true, magic: [{ prefix: [0xff, 0xd8, 0xff] }] }),
  GIF: def("GIF", "image/gif", { ext: ["gif"], blob: true, magic: [{ prefix: bytes("GIF87a") }, { prefix: bytes("GIF89a") }] }),
  WEBP: def("WEBP", "image/webp", { ext: ["webp"], blob: true }),
  TIFF: def("TIFF", "image/tiff", { ext: ["tif", "tiff"], blob: true, magic: [{ prefix: [0x49, 0x49, 0x2a, 0x00] }, { prefix: [0x4d, 0x4d, 0x00, 0x2a] }] }),
  BMP: def("BMP", "image/bmp", { ext: ["bmp"], blob: true, magic: [{ prefix: bytes("BM") }] }),
  SVG: def("SVG", "image/svg+xml", { ext: ["svg"], blob: true }),
  ICO: def("ICO", "image/x-icon", { ext: ["ico"], blob: true, magic: [{ prefix: [0x00, 0x00, 0x01, 0x00] }] }),
  AVIF: def("AVIF", "image/avif", { ext: ["avif"], blob: true }),
  HEIC: def("HEIC", "image/heic", { ext: ["heic", "heif"], blob: true }),

  // --- Audio (blobs) ---
  MP3: def("MP3", "audio/mpeg", { ext: ["mp3"], blob: true, magic: [{ prefix: bytes("ID3") }] }),
  WAV: def("WAV", "audio/wav", { ext: ["wav"], blob: true }),
  FLAC: def("FLAC", "audio/flac", { ext: ["flac"], blob: true, magic: [{ prefix: bytes("fLaC") }] }),
  OGG: def("OGG", "audio/ogg", { ext: ["ogg", "oga"], blob: true, magic: [{ prefix: bytes("OggS") }] }),
  AAC: def("AAC", "audio/aac", { ext: ["aac"], blob: true }),

  // --- Video (blobs) ---
  MP4: def("MP4", "video/mp4", { ext: ["mp4", "m4v"], blob: true }),
  WEBM: def("WEBM", "video/webm", { ext: ["webm"], blob: true }),
  MKV: def("MKV", "video/x-matroska", { ext: ["mkv"], blob: true }),
  MOV: def("MOV", "video/quicktime", { ext: ["mov"], blob: true }),
  AVI: def("AVI", "video/x-msvideo", { ext: ["avi"], blob: true }),

  // --- Filesystem containers (directories) ---
  DIRECTORY: def("FOLDER", "inode/directory"),
  PARTITIONED_FOLDER: def("PARTITIONED_FOLDER", "inode/directory+partitioned"),
  DELTA_FOLDER: def("DELTA_FOLDER", "inode/directory+delta"),

  // --- Fallback (generic opaque bytes) ---
  OCTET_STREAM: def("OCTET_STREAM", "application/octet-stream", { blob: true }),
} as const;
