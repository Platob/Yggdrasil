"""Tests for Loki on the internet — fetch / browse / table / image.

Runs against a localhost HTTP server (no external network) so the full chain
— HTTPSession → HTTPResponse → io tabular handlers — is exercised for real.
"""
from __future__ import annotations

import functools
import http.server
import tempfile
import threading
import unittest
from pathlib import Path

try:
    import polars  # noqa: F401  (the io tabular stack)

    from yggdrasil.loki import web

    _HAVE_STACK = True
except Exception:  # pragma: no cover - environment without the data stack
    _HAVE_STACK = False


@unittest.skipUnless(_HAVE_STACK, "requires the polars/io data stack")
class TestWeb(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp(prefix="ygg-web-")
        d = Path(cls.dir)
        (d / "data.csv").write_text("city,pop\nParis,2161\nTokyo,13960\nLagos,15388\n")
        (d / "page.html").write_text(
            "<html><head><style>x{}</style></head><body>"
            "<h1>Hello</h1><p>Some text.</p>"
            "<a href='https://example.com/next'>Next page</a>"
            "<script>ignore()</script></body></html>"
        )
        # A 1x1 PNG.
        png = (b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
               b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00"
               b"\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82")
        (d / "dot.png").write_bytes(png)
        (d / "app.html").write_text(
            '<html><head><title>Shop</title>'
            '<meta name="description" content="A demo shop">'
            '<script type="application/ld+json">{"@type":"Product","name":"Widget"}</script>'
            '</head><body><h1>Items</h1>'
            '<script>fetch("/api/products.json").then(r=>r.json());'
            'const u="https://cdn.example.com/data/prices.csv";</script>'
            '<a href="/v1/catalog">catalog</a></body></html>'
        )

        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=cls.dir)
        cls.srv = http.server.HTTPServer(("127.0.0.1", 0), handler)
        cls.base = f"http://127.0.0.1:{cls.srv.server_address[1]}"
        cls.thread = threading.Thread(target=cls.srv.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def test_read_table_parses_csv_via_io(self):
        df = web.read_table(f"{self.base}/data.csv")
        self.assertEqual(df.shape, (3, 2))
        self.assertEqual(list(df.columns), ["city", "pop"])
        self.assertEqual(df["pop"].sum(), 2161 + 13960 + 15388)

    def test_read_text_strips_html_and_collects_links(self):
        page = web.read_text(f"{self.base}/page.html")
        self.assertEqual(page["status"], 200)
        self.assertIn("Hello", page["text"])
        self.assertIn("Some text.", page["text"])
        self.assertNotIn("ignore()", page["text"])     # script dropped
        self.assertEqual(page["links"][0]["href"], "https://example.com/next")
        self.assertEqual(page["links"][0]["text"], "Next page")

    def test_read_image_reports_type_size_dims(self):
        info = web.read_image(f"{self.base}/dot.png")
        self.assertEqual(info["status"], 200)
        self.assertIn("png", info["content_type"])
        self.assertGreater(info["bytes"], 0)
        self.assertEqual((info.get("width"), info.get("height")), (1, 1))

    def test_read_image_saves_to_disk(self):
        out = Path(self.dir) / "saved.png"
        info = web.read_image(f"{self.base}/dot.png", save_to=str(out))
        self.assertTrue(out.is_file())
        self.assertEqual(info["saved_to"], str(out))

    def test_scrape_extracts_title_meta_jsonld(self):
        s = web.scrape(f"{self.base}/app.html")
        self.assertEqual(s["title"], "Shop")
        self.assertEqual(s["description"], "A demo shop")
        self.assertTrue(any(b.get("@type") == "Product" for b in s["json_ld"]))
        self.assertIn("Items", s["text"])

    def test_discover_apis_finds_endpoints(self):
        d = web.discover_apis(f"{self.base}/app.html")
        eps = set(d["endpoints"])
        self.assertIn("/api/products.json", eps)            # fetch(...) target
        self.assertIn("https://cdn.example.com/data/prices.csv", eps)  # *.csv
        self.assertIn("/v1/catalog", eps)                   # /v1/ path
        self.assertTrue(any(b.get("@type") == "Product" for b in d["json_ld"]))

    def test_default_user_agent_is_browserish(self):
        self.assertIn("Mozilla/5.0", web.DEFAULT_USER_AGENT)

    def test_web_behavior_auto_routes_by_extension(self):
        from yggdrasil.loki import Loki
        from yggdrasil.loki.capability import Backend

        loki = Loki(); loki._backends = [Backend("local", True)]
        res = loki.run("web", url=f"{self.base}/data.csv")
        self.assertEqual(res["action"], "table")
        self.assertEqual(res["shape"], [3, 2])
        res = loki.run("web", url=f"{self.base}/dot.png")
        self.assertEqual(res["action"], "image")


if __name__ == "__main__":
    unittest.main()
