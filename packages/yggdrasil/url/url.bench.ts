import { bench, describe } from "vitest";
import { URL } from "./url";

const SAMPLES = [
  "https://user:pw@host.com:8443/a/b/c.csv?x=1#frag",
  "s3://bucket/key/part-0001.snappy.parquet",
  "file:///home/u/data/file.tar.gz",
  "/data/archive.csv.zst",
  "relative/path/x.json",
  "npfs://node1/folder/sub.arrow",
];

describe("URL", () => {
  bench("fromStr (parse) x6 samples", () => {
    for (const s of SAMPLES) URL.fromStr(s);
  });

  bench("parse + name + extensions + mediaType", () => {
    for (const s of SAMPLES) {
      const u = URL.fromStr(s);
      void u.name;
      void u.extensions;
      void u.mediaType;
    }
  });

  bench("parse + parent + toString", () => {
    for (const s of SAMPLES) {
      const u = URL.fromStr(s);
      void u.parent.toString();
    }
  });
});
