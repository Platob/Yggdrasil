// Client-side port of ``yggdrasil.io.arrow_ipc_file``.
//
// PARITY: python/src/yggdrasil/io/arrow_ipc_file.py. The Arrow IPC (feather v2 /
// stream) leaf — read a buffer into a Tabular (applying CastOptions
// projection/limit) and write a Tabular back to IPC bytes. The format leaves
// live directly under ``io`` (no ``primitive``/``nested`` grouping layer).

import * as arrow from "apache-arrow";
import { Tabular, type TabularSource } from "./tabular/base";
import { CastOptions, type CastOptionsArg } from "../data/options";

export class ArrowIPCFile implements TabularSource {
  constructor(readonly bytes: Uint8Array) {}

  readArrowTable(options?: CastOptionsArg): Tabular {
    return new Tabular(arrow.tableFromIPC(this.bytes)).cast(CastOptions.check(options));
  }

  static read(bytes: Uint8Array, options?: CastOptionsArg): Tabular {
    return new ArrowIPCFile(bytes).readArrowTable(options);
  }

  /** Serialize a Tabular to Arrow IPC (``stream`` default, or ``file`` framing). */
  static write(t: Tabular, opts: { file?: boolean } = {}): Uint8Array {
    return opts.file ? t.toArrowFile() : t.toArrowIPC();
  }
}
