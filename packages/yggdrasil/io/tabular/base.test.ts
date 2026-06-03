// Tabular core + Arrow IPC leaf tests. Arrow IPC stream framing
// (application/vnd.apache.arrow.stream) is the cross-language wire format, so an
// IPC buffer written here decodes identically in the Python reference.

import { describe, it, expect } from "vitest";
import * as arrow from "apache-arrow";
import { Tabular } from "./base";
import { ArrowIPCFile } from "../arrowIpc";
import { MimeTypes } from "../../enums";
import { DataTypeId } from "../../data/types";

function sample(): Tabular {
  return Tabular.fromArrays({ id: Int32Array.from([1, 2, 3]), name: ["a", "b", "c"] });
}

describe("Tabular core", () => {
  it("reports shape, columns, schema and media type", () => {
    const t = sample();
    expect(t.numRows).toBe(3);
    expect(t.numCols).toBe(2);
    expect(t.columnNames).toEqual(["id", "name"]);
    expect(t.mediaType).toBe(MimeTypes.ARROW_STREAM);
    expect(t.schema.names).toEqual(["id", "name"]);
    expect(t.schema.field("name")?.dtype.id).toBe(DataTypeId.STRING);
  });

  it("from() coerces rows / arrays / Arrow tables", () => {
    expect(Tabular.from([{ id: 1 }, { id: 2 }]).numRows).toBe(2);
    expect(Tabular.from({ id: Int32Array.from([1]) }).numRows).toBe(1);
    expect(Tabular.from(sample()).numRows).toBe(3);
  });

  it("select / slice / column", () => {
    const t = sample();
    expect(t.select(["name"]).columnNames).toEqual(["name"]);
    expect(t.slice(0, 2).numRows).toBe(2);
    expect(t.column("id")?.get(2)).toBe(3);
  });

  it("cast projects + limits", () => {
    const t = sample().cast({ columns: ["name"], rowLimit: 2 });
    expect(t.columnNames).toEqual(["name"]);
    expect(t.numRows).toBe(2);
  });

  it("toArray materialises plain rows", () => {
    expect(sample().toArray()).toEqual([
      { id: 1, name: "a" }, { id: 2, name: "b" }, { id: 3, name: "c" },
    ]);
  });

  it("scanArrowTable / scanArrowBatches expose zero-copy views", () => {
    const t = sample();
    // The held Table is returned verbatim — same reference, no copy/cast.
    expect(t.scanArrowTable()).toBe(t.toArrow());
    const batches = t.scanArrowBatches();
    expect(batches).toBe(t.toArrow().batches);
    expect(batches.reduce((n, b) => n + b.numRows, 0)).toBe(3);
  });
});

describe("Arrow IPC round-trip (the wire contract)", () => {
  it("stream framing survives a Tabular round-trip", () => {
    const ipc = sample().toArrowIPC();
    const back = Tabular.fromArrowIPC(ipc);
    expect(back.numRows).toBe(3);
    expect(back.columnNames).toEqual(["id", "name"]);
  });

  it("file framing carries the ARROW1 magic header", () => {
    const file = sample().toArrowFile();
    expect(new TextDecoder().decode(file.slice(0, 6))).toBe("ARROW1");
  });

  it("ArrowIPCFile.read applies CastOptions; write mirrors Tabular", () => {
    const ipc = ArrowIPCFile.write(sample());
    const t = ArrowIPCFile.read(ipc, { columns: ["id"] });
    expect(t.columnNames).toEqual(["id"]);
    expect(arrow.tableFromIPC(ArrowIPCFile.write(sample())).numRows).toBe(3);
  });
});
