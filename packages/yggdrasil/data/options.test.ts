// CastOptions tests — structural projection/reorder + row limit + rechunk.
// (Apache Arrow JS has no compute kernels; value coercion is a polars follow-on.)

import { describe, it, expect } from "vitest";
import * as arrow from "apache-arrow";
import { CastOptions } from "./options";
import { schema } from "./schema";
import { field } from "./field";
import { Int64Type, StringType } from "./types";

function sampleTable(): arrow.Table {
  return arrow.tableFromArrays({
    id: Int32Array.from([1, 2, 3, 4]),
    name: ["a", "b", "c", "d"],
    score: Float64Array.from([1.5, 2.5, 3.5, 4.5]),
  });
}

describe("CastOptions.check", () => {
  it("normalizes args / instances / null", () => {
    expect(CastOptions.check(null)).toBeInstanceOf(CastOptions);
    const o = new CastOptions({ rowLimit: 2 });
    expect(CastOptions.check(o)).toBe(o);
    expect(CastOptions.check({ rowLimit: 5 }).rowLimit).toBe(5);
  });
});

describe("structural castArrowTable", () => {
  it("projects + reorders to the target columns", () => {
    const target = schema([field("score", Int64Type()), field("id", Int64Type())]);
    const out = new CastOptions({ target }).castArrowTable(sampleTable());
    expect(out.schema.fields.map((f) => f.name)).toEqual(["score", "id"]);
  });

  it("explicit columns override the target projection", () => {
    const out = new CastOptions({ columns: ["name"] }).castArrowTable(sampleTable());
    expect(out.schema.fields.map((f) => f.name)).toEqual(["name"]);
  });

  it("ignores target columns absent from the source", () => {
    const target = schema([field("id", Int64Type()), field("missing", new StringType())]);
    const out = new CastOptions({ target }).castArrowTable(sampleTable());
    expect(out.schema.fields.map((f) => f.name)).toEqual(["id"]);
  });

  it("applies the row limit", () => {
    const out = new CastOptions({ rowLimit: 2 }).castArrowTable(sampleTable());
    expect(out.numRows).toBe(2);
  });
});

describe("rechunk", () => {
  it("splits a table into row-bounded batches", () => {
    const batches = [...new CastOptions({ rowSize: 2 }).rechunk(sampleTable())];
    expect(batches.reduce((n, b) => n + b.numRows, 0)).toBe(4);
    expect(batches.every((b) => b.numRows <= 2)).toBe(true);
  });
});

describe("copy / withTarget / readColumns", () => {
  it("copy overrides selected fields only", () => {
    const o = new CastOptions({ rowLimit: 3, safe: true });
    const c = o.copy({ rowLimit: 1 });
    expect(c.rowLimit).toBe(1);
    expect(c.safe).toBe(true);
  });
  it("readColumns prefers explicit columns then target names", () => {
    expect(new CastOptions({ columns: ["x"] }).readColumns()).toEqual(["x"]);
    expect(new CastOptions({ target: schema([field("y", Int64Type())]) }).readColumns()).toEqual(["y"]);
    expect(new CastOptions().readColumns()).toBeNull();
  });
});
