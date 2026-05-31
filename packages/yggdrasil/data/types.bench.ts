// Micro-benchmarks for the DataType wire path (the cross-language hot loops:
// serialize a schema to its canonical dict, rebuild it, and Arrow round-trip).

import { bench, describe } from "vitest";
import * as arrow from "apache-arrow";
import { DataType } from "./types";
import { Schema } from "./schema";
import { field } from "./field";
import { Int64Type, StringType, Float64Type, TimestampType } from "./types";

const schema = new Schema([
  field("id", Int64Type()),
  field("name", new StringType()),
  field("score", Float64Type()),
  field("ts", new TimestampType("us", "UTC")),
]);
const dict = schema.toDict();
const arrowType = new arrow.Timestamp(arrow.TimeUnit.MICROSECOND, "UTC");

describe("data", () => {
  bench("schema.toDict()", () => { schema.toDict(); });
  bench("Schema.fromDict()", () => { Schema.fromDict(dict); });
  bench("DataType.fromArrow().toArrow()", () => { DataType.fromArrow(arrowType).toArrow(); });
});
