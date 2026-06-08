// Schema parity + round-trip tests. The canonical Schema dict is a struct Field
// with an empty name (captured from Python ``Schema.to_dict``).

import { describe, it, expect } from "vitest";
import * as arrow from "apache-arrow";
import { Schema, schema } from "./schema";
import { field } from "./field";
import { Int64Type, StringType, DataTypeId } from "./types";

const CANONICAL = {
  name: "",
  nullable: false,
  dtype: {
    id: 102, name: "STRUCT",
    fields: [
      { name: "id", dtype: { id: 24, name: "INT64", byte_size: 8, signed: true }, nullable: true },
      { name: "name", dtype: { id: 71, name: "STRING" }, nullable: true },
    ],
  },
};

describe("Schema ↔ Python to_dict parity", () => {
  it("toDict is a struct Field with empty name", () => {
    const s = new Schema([field("id", Int64Type()), field("name", new StringType())]);
    expect(s.toDict()).toEqual(CANONICAL);
  });

  it("fromDict round-trips", () => {
    expect(Schema.fromDict(CANONICAL).toDict()).toEqual(CANONICAL);
  });
});

describe("Schema construction + helpers", () => {
  it("accepts Field / string / arrow.Field members", () => {
    const s = new Schema([field("id", Int64Type()), "name"]);
    expect(s.names).toEqual(["id", "name"]);
    expect(s.length).toBe(2);
    expect(s.field("id")?.dtype.id).toBe(DataTypeId.INT64);
    expect(s.field(1)?.name).toBe("name");
  });

  it("select projects + reorders", () => {
    const s = schema([field("a", Int64Type()), field("b", new StringType()), field("c", Int64Type())]);
    expect(s.select(["c", "a"]).names).toEqual(["c", "a"]);
  });

  it("round-trips an Arrow Schema", () => {
    const as_ = new arrow.Schema([
      new arrow.Field("id", new arrow.Int(true, 64), true),
      new arrow.Field("name", new arrow.Utf8(), true),
    ]);
    const back = Schema.from(as_).toArrow();
    expect(back.fields.map((f) => f.name)).toEqual(["id", "name"]);
    expect(back.fields[0].typeId).toBe(as_.fields[0].typeId);
  });
});
