// "any source -> Arrow" coercion tests.

import { describe, it, expect } from "vitest";
import * as arrow from "apache-arrow";
import { anyToArrowTable, anyToArrowBatches, castArrowTable, tableFromRows } from "./cast";

describe("anyToArrowTable", () => {
  it("passes an Arrow Table through (with projection)", () => {
    const t = arrow.tableFromArrays({ a: Int32Array.from([1, 2]), b: ["x", "y"] });
    const out = anyToArrowTable(t, { columns: ["b"] });
    expect(out.schema.fields.map((f) => f.name)).toEqual(["b"]);
  });

  it("builds from column arrays", () => {
    const out = anyToArrowTable({ id: Int32Array.from([1, 2, 3]), name: ["a", "b", "c"] });
    expect(out.numRows).toBe(3);
    expect(out.schema.fields.map((f) => f.name)).toEqual(["id", "name"]);
  });

  it("builds from row objects", () => {
    const out = anyToArrowTable([{ id: 1, name: "a" }, { id: 2, name: "b" }]);
    expect(out.numRows).toBe(2);
    expect(out.getChild("id")?.get(1)).toBe(2);
  });

  it("accepts a RecordBatch", () => {
    const t = arrow.tableFromArrays({ a: Int32Array.from([1, 2]) });
    const out = anyToArrowTable(t.batches[0]);
    expect(out.numRows).toBe(2);
  });

  it("honors a toArrow()-bearing source (polars / Tabular shape)", () => {
    const t = arrow.tableFromArrays({ a: Int32Array.from([7]) });
    const out = anyToArrowTable({ toArrow: () => t });
    expect(out.getChild("a")?.get(0)).toBe(7);
  });

  it("applies the row limit", () => {
    const out = anyToArrowTable({ a: Int32Array.from([1, 2, 3, 4]) }, { rowLimit: 2 });
    expect(out.numRows).toBe(2);
  });
});

describe("anyToArrowBatches", () => {
  it("streams rechunked batches", () => {
    const batches = [...anyToArrowBatches({ a: Int32Array.from([1, 2, 3, 4, 5]) }, { rowSize: 2 })];
    expect(batches.reduce((n, b) => n + b.numRows, 0)).toBe(5);
  });
});

describe("tableFromRows / castArrowTable", () => {
  it("empty rows -> empty table", () => {
    expect(tableFromRows([]).numRows).toBe(0);
  });
  it("castArrowTable projects", () => {
    const t = arrow.tableFromArrays({ a: Int32Array.from([1]), b: Int32Array.from([2]) });
    expect(castArrowTable(t, { columns: ["b"] }).schema.fields.map((f) => f.name)).toEqual(["b"]);
  });
});
