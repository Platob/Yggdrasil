// Client-side port of ``yggdrasil.enums.media_type`` — object-oriented.
//
// A ``MediaType`` couples a format ``MimeType`` with an optional outer
// compression codec: ``data.csv.gz`` -> ``MediaType(CSV, GZIP)``. Mirrors the
// resolution conventions of python/src/yggdrasil/enums/media_type.py, scoped to
// the inputs a browser has (a filename / path, a mime value, or a head buffer).

import { MimeType, MimeTypes } from "./mimeType";

export class MediaType {
  constructor(readonly mimeType: MimeType, readonly codec: MimeType | null = null) {}

  /** ``text/csv`` or ``text/csv+application/gzip`` for the codec'd form. */
  get value(): string {
    return this.codec ? `${this.mimeType.value}+${this.codec.value}` : this.mimeType.value;
  }

  get isTabular(): boolean { return this.mimeType.isTabular; }
  get isBlob(): boolean { return this.mimeType.isBlob; }
  get isCodec(): boolean { return this.mimeType.isCodec; }
  get compressed(): boolean { return this.codec !== null; }

  toString(): string { return this.value; }

  /**
   * Resolve a MediaType from a filename/path (codec-suffix aware), a bare
   * extension, or a mime value/name. ``null`` on a clean miss.
   */
  static from(input: string): MediaType | null {
    const s = input.trim();
    if (s.includes("/") || s.includes("\\") || s.includes(".")) {
      const { mime, codec } = MimeType.fromName(s);
      if (mime) return new MediaType(mime, codec);
      const ext = MimeType.fromExtension(s.replace(/^.*\./, ""));
      if (ext) return new MediaType(ext);
    } else {
      const ext = MimeType.fromExtension(s);
      if (ext) return new MediaType(ext);
    }
    const mv = MimeType.get(s);
    if (mv) return mv.isCodec ? new MediaType(MimeTypes.OCTET_STREAM, mv) : new MediaType(mv);
    return null;
  }

  /** Resolve by sniffing a head buffer. */
  static fromBytes(head: Uint8Array): MediaType | null {
    const mt = MimeType.fromMagic(head);
    return mt ? new MediaType(mt) : null;
  }
}
