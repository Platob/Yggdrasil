// yggdrasil.io — PARITY: python/src/yggdrasil/io/
// Arrow-IPC tabular core (``tabular/``) + the format leaves, which live
// directly under ``io`` (the ``primitive``/``nested`` grouping layer was
// flattened away on both sides). apache-arrow is a peer dep; a polars adapter
// (compute/casting) and the parquet/csv/ndjson leaves are documented follow-ons.
export { Tabular, type TabularSource } from "./tabular/base";
export { ArrowIPCFile } from "./arrowIpc";
