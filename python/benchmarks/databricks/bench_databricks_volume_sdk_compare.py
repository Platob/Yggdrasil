"""Head-to-head: Databricks SDK Files client vs yggdrasil VolumePath.

Both clients drive the *same* localhost server emulating the Databricks
Files REST API (``GET/PUT/HEAD /api/2.0/fs/files{path}``, with ``Range``
support). The SDK side uses the real
:class:`databricks.sdk.service.files.FilesAPI`
(``disable_experimental_files_api_client=True`` so it takes the simple
GET/PUT path — the apples-to-apples transport, not the multipart /
presigned-URL machinery). The yggdrasil side uses a real
:class:`~yggdrasil.http_.HTTPSession` through :class:`VolumePath`.

The interesting axis is *random-access* reads. The SDK's ``download()``
always streams the whole object from offset 0 — there is no public
partial-read API — so a 4 KiB read off a 16 MiB object pulls all 16 MiB.
VolumePath issues an HTTP ``Range`` and pulls only the slice. The same
gap shows up wherever a reader seeks: Parquet column/row-group
projection in particular reads a file's footer plus a few column chunks,
so projecting one column of a many-column file transfers a fraction of
the bytes.

Each scenario reports wall time (best of N) and bytes the server sent,
for both clients, plus the speedup / byte-reduction factor.

Usage::

    python benchmarks/databricks/bench_databricks_volume_sdk_compare.py
    python benchmarks/databricks/bench_databricks_volume_sdk_compare.py \\
        --size-mib 32 --reads 200 --repeat 3
"""
from __future__ import annotations

import argparse
import io
import pickle
import statistics
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import pyarrow as pa
import pyarrow.parquet as pq

from yggdrasil.databricks.fs.volume_path import VolumePath
from yggdrasil.http_ import HTTPSession
from yggdrasil.io.parquet_file import ParquetFile
from yggdrasil.url import URL


# ---------------------------------------------------------------------------
# Localhost Files-API server (Range-aware, byte-counting)
# ---------------------------------------------------------------------------


class _Store:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.bytes_served = 0
        self.calls = 0


def _make_handler(store: _Store):
    class _Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        disable_nagle_algorithm = True  # avoid 40ms delayed-ACK on small bodies

        def log_message(self, *a):
            pass

        def _key(self) -> str:
            return urlsplit(self.path).path[len("/api/2.0/fs/files"):]

        def do_PUT(self):  # noqa: N802
            store.calls += 1
            # Drain the body in small chunks and discard it. The server
            # shares this process, so materialising the whole uploaded
            # body here would land in the same tracemalloc trace as the
            # client and mask the client-side memory we're measuring.
            # (Read scenarios are served from a pre-seeded store.)
            remaining = int(self.headers.get("Content-Length", "0"))
            while remaining > 0:
                chunk = self.rfile.read(min(remaining, 256 * 1024))
                if not chunk:
                    break
                remaining -= len(chunk)
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_HEAD(self):  # noqa: N802
            store.calls += 1
            blob = store.files.get(self._key())
            if blob is None:
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(blob)))
            self.end_headers()

        def do_GET(self):  # noqa: N802
            store.calls += 1
            blob = store.files.get(self._key())
            if blob is None:
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            rng = self.headers.get("Range")
            if rng and rng.startswith("bytes="):
                spec = rng[len("bytes="):].split("-", 1)
                start = int(spec[0]) if spec[0].strip().isdigit() else 0
                end = spec[1].strip() if len(spec) > 1 else ""
                stop = int(end) + 1 if end.isdigit() else len(blob)
                body = blob[start:stop]
                store.bytes_served += len(body)
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{start + len(body) - 1}/{len(blob)}")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            store.bytes_served += len(blob)
            self.send_response(200)
            self.send_header("Content-Length", str(len(blob)))
            self.end_headers()
            self.wfile.write(blob)

    return _Handler


# ---------------------------------------------------------------------------
# Client shims
# ---------------------------------------------------------------------------


class _Client:
    def __init__(self, host: str) -> None:
        self.base_url = URL.from_(host)
        self._session = HTTPSession(base_url=host, verify=False)

    def files_session(self):
        return self._session

    def files_authorization(self):
        return "Bearer bench"


class _Service:
    def __init__(self, client):
        self.client = client


def _volume(host: str, path: str) -> VolumePath:
    return VolumePath(path, service=_Service(_Client(host)))


def _sdk_files(host: str):
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.config import Config

    cfg = Config(host=host, token="dummy", disable_experimental_files_api_client=True)
    return WorkspaceClient(config=cfg).files


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def _best(fn, repeat: int, store: _Store):
    import tracemalloc

    times, served, calls, peak = [], 0, 0, 0
    for _ in range(repeat):
        store.bytes_served = 0
        store.calls = 0
        tracemalloc.start()
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
        peak = max(peak, tracemalloc.get_traced_memory()[1])
        tracemalloc.stop()
        served = store.bytes_served
        calls = store.calls
    return min(times), served, calls, peak


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size-mib", type=int, default=16)
    ap.add_argument("--reads", type=int, default=200)
    ap.add_argument("--block", type=int, default=4096)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    import random

    rnd = random.Random(args.seed)
    blob = rnd.randbytes(args.size_mib * 1024 * 1024)

    # A many-column, many-row-group Parquet table so column projection
    # has to seek to scattered column chunks (the random-access win).
    ncols, nrows = 16, 200_000
    table = pa.table({f"c{i}": pa.array(range(nrows), type=pa.int64()) for i in range(ncols)})
    pbuf = io.BytesIO()
    pq.write_table(table, pbuf, row_group_size=20_000)
    parquet_bytes = pbuf.getvalue()

    obj = {"ids": list(range(50_000)), "labels": [f"row-{i}" for i in range(50_000)]}
    pickle_bytes = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)

    store = _Store()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(store))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host = f"http://127.0.0.1:{server.server_address[1]}"

    try:
        sdk = _sdk_files(host)
        bin_path = "/Volumes/b/m/v/blob.bin"
        pq_path = "/Volumes/b/m/v/data.parquet"
        pk_path = "/Volumes/b/m/v/obj.pkl"
        vp_bin = _volume(host, bin_path)
        vp_pq = _volume(host, pq_path)
        vp_pk = _volume(host, pk_path)

        # Seed the read store directly (PUT now drains-and-discards, so
        # the upload scenario's server side doesn't pollute client memory).
        store.files[bin_path] = blob
        store.files[pq_path] = parquet_bytes
        store.files[pk_path] = pickle_bytes

        offsets = [rnd.randint(0, len(blob) - args.block) for _ in range(args.reads)]

        def sdk_read_whole():
            assert len(sdk.download(bin_path).contents.read()) == len(blob)

        def ygg_read_whole():
            assert len(vp_bin.read_bytes()) == len(blob)

        def sdk_random():
            for off in offsets:
                data = sdk.download(bin_path).contents.read()  # whole object each seek
                _ = data[off:off + args.block]

        def ygg_random():
            for off in offsets:
                _ = vp_bin._read_mv(args.block, off)  # Range → slice only

        def sdk_upload():
            sdk.upload(bin_path, io.BytesIO(blob), overwrite=True)

        def ygg_upload():
            vp_bin.write_bytes(blob, overwrite=True)

        def sdk_parquet_full():
            data = sdk.download(pq_path).contents.read()
            pq.read_table(pa.BufferReader(data))

        def ygg_parquet_full():
            # Full read (no projection): both snapshot the whole object —
            # one big GET beats many ranged round trips. Parity with SDK.
            ParquetFile(holder=vp_pq, owns_holder=False).read_arrow_table()

        proj_target = pa.schema([("c7", pa.int64())])

        def sdk_parquet_proj():
            # SDK has no partial read — pull the whole file, then project.
            data = sdk.download(pq_path).contents.read()
            pq.read_table(pa.BufferReader(data), columns=["c7"])

        def ygg_parquet_proj():
            # Bound target → ranged random-access: footer + the c7 column
            # chunks only, over HTTP Range.
            ParquetFile(holder=vp_pq, owns_holder=False).read_arrow_table(
                target=proj_target,
            )

        def sdk_pickle():
            assert pickle.loads(sdk.download(pk_path).contents.read()) == obj

        def ygg_pickle():
            assert pickle.loads(vp_pk.read_bytes()) == obj

        scenarios = [
            ("read whole 16MiB", sdk_read_whole, ygg_read_whole),
            (f"random {args.reads}x{args.block}B", sdk_random, ygg_random),
            ("upload 16MiB", sdk_upload, ygg_upload),
            ("parquet full read", sdk_parquet_full, ygg_parquet_full),
            ("parquet 1-col project", sdk_parquet_proj, ygg_parquet_proj),
            ("pickle roundtrip", sdk_pickle, ygg_pickle),
        ]

        print(
            f"\nSDK FilesAPI vs yggdrasil VolumePath — payload={args.size_mib} MiB, "
            f"parquet={len(parquet_bytes) // 1024} KiB ({ncols} cols), repeat={args.repeat}\n"
        )
        # Lead with the network-relevant axes: bytes on the wire, calls
        # (round-trips, each a latency hit), and peak resident memory.
        # ``sdk``/``ygg`` columns are paired so the ratio is read at a glance.
        hdr = (
            f"{'scenario':<24} {'MiB sdk':>8} {'MiB ygg':>8} "
            f"{'calls s':>7} {'calls y':>7} {'memMiB s':>9} {'memMiB y':>9}"
        )
        print(hdr)
        print("-" * len(hdr))
        for name, sdk_fn, ygg_fn in scenarios:
            _st, sb, sc, sm = _best(sdk_fn, args.repeat, store)
            _yt, yb, yc, ym = _best(ygg_fn, args.repeat, store)
            print(
                f"{name:<24} {sb / 1048576:>8.2f} {yb / 1048576:>8.2f} "
                f"{sc:>7} {yc:>7} {sm / 1048576:>9.2f} {ym / 1048576:>9.2f}"
            )
        print(
            "\nbytes/calls/memory are what cost on a real network. "
            "Random + projection: VolumePath ranges only what's needed "
            "(far fewer bytes/memory) at one call per read; the SDK pulls "
            "the whole object each time. Whole-file reads match.\n"
            "Memory caveat: tracemalloc traces Python allocations only. "
            "ygg buffers (bytearray) are visible; the SDK's io.BytesIO + "
            "requests/urllib3 buffers are C-level and NOT traced, so the "
            "SDK memory column reads far lower than its true footprint "
            "(its upload really holds the ~16 MiB BytesIO). Read the ygg "
            "column as an absolute (our copies), not a 1:1 ratio to the SDK."
        )
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
