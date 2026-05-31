// Client-side port of ``yggdrasil.enums.media_type``.
//
// A MediaType couples a format ``MimeType`` with an optional outer compression
// codec — ``data.csv.gz`` -> MediaType(CSV, codec=GZIP). Mirrors the resolution
// conventions of python/src/yggdrasil/enums/media_type.py (``from_``), scoped to
// the inputs a browser has: a filename / path, a mime value, or a head buffer.

import {
  MimeTypes,
  type MimeType,
  fromExtension,
  fromName,
  fromMagic,
  get as getMime,
} from "./mimeType";

export interface MediaType {
  mimeType: MimeType;
  codec: MimeType | null; // outer compression wrapper, when present
}

/** ``text/csv`` or ``text/csv + application/gzip`` for the codec'd form. */
export function value(mt: MediaType): string {
  return mt.codec ? `${mt.mimeType.value}+${mt.codec.value}` : mt.mimeType.value;
}

export const isTabular = (mt: MediaType): boolean => mt.mimeType.isTabular;
export const isBlob = (mt: MediaType): boolean => mt.mimeType.isBlob;
export const isCodec = (mt: MediaType): boolean => mt.mimeType.isCodec;

/**
 * Resolve a MediaType from a filename/path (codec-suffix aware), a bare
 * extension, or a mime value. Returns ``null`` on a clean miss.
 */
export function from(input: string): MediaType | null {
  const s = input.trim();
  // Path / filename with a dotted suffix chain.
  if (s.includes("/") || s.includes("\\") || s.includes(".")) {
    const { mime, codec } = fromName(s);
    if (mime) return { mimeType: mime, codec };
    // Bare extension (``"csv"``, ``"gz"``).
    const ext = fromExtension(s.replace(/^.*\./, ""));
    if (ext) return { mimeType: ext, codec: null };
  } else {
    const ext = fromExtension(s);
    if (ext) return { mimeType: ext, codec: null };
  }
  // Mime value or registry name.
  const mv = getMime(s);
  if (mv) return mv.isCodec ? { mimeType: MimeTypes.OCTET_STREAM, codec: mv } : { mimeType: mv, codec: null };
  return null;
}

/** Resolve a MediaType by sniffing a head buffer. */
export function fromBytes(head: Uint8Array): MediaType | null {
  const mt = fromMagic(head);
  return mt ? { mimeType: mt, codec: null } : null;
}
