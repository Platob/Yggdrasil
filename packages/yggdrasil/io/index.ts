// yggdrasil.io — PARITY: python/src/yggdrasil/io/
// Arrow-IPC tabular core + format leaves (primitive / nested). apache-arrow is
// a peer dep; a polars adapter (compute/casting) is a documented follow-on.
export { Tabular, type TabularSource } from "./tabular/base";
export * as primitive from "./primitive";
export * as nested from "./nested";
export { ArrowIPCFile } from "./primitive";
