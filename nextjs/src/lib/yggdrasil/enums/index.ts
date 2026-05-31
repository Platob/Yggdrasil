// yggdrasil.enums — client-side port. Start with the MIME / media-type
// registry (the most-used cross-language contract); codec lives inside the
// MimeType ``isCodec`` flag and the MediaType wrapper.

export * as mimeType from "./mimeType";
export * as mediaType from "./mediaType";
export { MimeTypes, type MimeType } from "./mimeType";
export { type MediaType } from "./mediaType";
