"""Tests for SSL verification bypass and custom CA support.

Covers:
- ``_make_ssl_context`` with verify=True / False / CA-path
- ``InsecureRequestWarning`` emission on ``verify=False``
- ``insecure()`` factory method
- ``verify=False`` live HTTPS request to example.com
- ``verify="/path/to/ca"`` custom CA bundle loading
- singleton key isolation (different verify → different instance)
"""
from __future__ import annotations

import socket
import ssl
import warnings

import pytest

from yggdrasil.http_.exceptions import InsecureRequestWarning
from yggdrasil.http_.session import HTTPSession, _make_ssl_context


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _fresh_singletons():
    HTTPSession._INSTANCES.clear()
    yield
    HTTPSession._INSTANCES.clear()


@pytest.fixture(scope="module")
def _check_network():
    try:
        socket.create_connection(("example.com", 443), timeout=5).close()
    except OSError:
        pytest.skip("Network unreachable")


# ---------------------------------------------------------------------------
# _make_ssl_context unit tests
# ---------------------------------------------------------------------------


class TestMakeSSLContext:

    def test_verify_true_returns_default_context(self):
        ctx = _make_ssl_context(True)
        assert isinstance(ctx, ssl.SSLContext)
        assert ctx.check_hostname is True
        assert ctx.verify_mode == ssl.CERT_REQUIRED

    def test_verify_false_disables_verification(self):
        ctx = _make_ssl_context(False)
        assert isinstance(ctx, ssl.SSLContext)
        assert ctx.check_hostname is False
        assert ctx.verify_mode == ssl.CERT_NONE

    def test_verify_ca_file(self, tmp_path):
        ca_file = tmp_path / "ca-bundle.crt"
        ca_file.write_text(ssl.get_default_verify_paths().cafile and open(ssl.get_default_verify_paths().cafile).read() or "")
        if not ca_file.stat().st_size:
            pytest.skip("No system CA bundle to copy")
        ctx = _make_ssl_context(str(ca_file))
        assert ctx.check_hostname is True
        assert ctx.verify_mode == ssl.CERT_REQUIRED

    def test_verify_ca_directory(self, tmp_path):
        ca_dir = tmp_path / "certs"
        ca_dir.mkdir()
        ctx = _make_ssl_context(str(ca_dir))
        assert isinstance(ctx, ssl.SSLContext)

    def test_verify_nonexistent_path_raises(self):
        with pytest.raises((ssl.SSLError, OSError)):
            _make_ssl_context("/nonexistent/ca-bundle.crt")

    def test_verify_pathlib_path(self, tmp_path):
        ca_dir = tmp_path / "certs"
        ca_dir.mkdir()
        ctx = _make_ssl_context(str(ca_dir))
        assert isinstance(ctx, ssl.SSLContext)


# ---------------------------------------------------------------------------
# InsecureRequestWarning
# ---------------------------------------------------------------------------


class TestInsecureWarning:

    def test_verify_false_emits_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            HTTPSession(base_url="https://example.com", verify=False)
            insecure_warnings = [x for x in w if issubclass(x.category, InsecureRequestWarning)]
            assert len(insecure_warnings) >= 1
            assert "SSL certificate verification is disabled" in str(insecure_warnings[0].message)

    def test_verify_true_no_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            HTTPSession(base_url="https://example.com", verify=True)
            insecure_warnings = [x for x in w if issubclass(x.category, InsecureRequestWarning)]
            assert len(insecure_warnings) == 0

    def test_verify_ca_path_no_warning(self, tmp_path):
        ca_dir = tmp_path / "certs"
        ca_dir.mkdir()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            HTTPSession(base_url="https://example.com", verify=str(ca_dir))
            insecure_warnings = [x for x in w if issubclass(x.category, InsecureRequestWarning)]
            assert len(insecure_warnings) == 0


# ---------------------------------------------------------------------------
# insecure() factory
# ---------------------------------------------------------------------------


class TestInsecureFactory:

    def test_insecure_returns_verify_false_session(self):
        session = HTTPSession(base_url="https://example.com")
        assert session.verify is True
        insecure = session.insecure()
        assert insecure.verify is False

    def test_insecure_preserves_base_url(self):
        session = HTTPSession(base_url="https://example.com")
        insecure = session.insecure()
        assert insecure.base_url == session.base_url

    def test_insecure_is_different_instance(self):
        session = HTTPSession(base_url="https://example.com")
        insecure = session.insecure()
        assert insecure is not session

    def test_insecure_on_already_insecure_returns_self(self):
        session = HTTPSession(base_url="https://example.com", verify=False)
        assert session.insecure() is session

    def test_insecure_preserves_proxy(self):
        session = HTTPSession(base_url="https://example.com", proxy="http://proxy:8080")
        insecure = session.insecure()
        assert insecure.proxy is not None
        assert insecure.proxy.host == "proxy"


# ---------------------------------------------------------------------------
# Singleton isolation by verify value
# ---------------------------------------------------------------------------


class TestSingletonIsolation:

    def test_different_verify_different_instance(self):
        a = HTTPSession(base_url="https://example.com", verify=True)
        b = HTTPSession(base_url="https://example.com", verify=False)
        assert a is not b

    def test_same_verify_same_instance(self):
        a = HTTPSession(base_url="https://example.com", verify=True)
        b = HTTPSession(base_url="https://example.com", verify=True)
        assert a is b

    def test_same_verify_false_same_instance(self):
        a = HTTPSession(base_url="https://example.com", verify=False)
        b = HTTPSession(base_url="https://example.com", verify=False)
        assert a is b

    def test_ca_path_in_singleton_key(self, tmp_path):
        ca_dir = tmp_path / "certs"
        ca_dir.mkdir()
        a = HTTPSession(base_url="https://example.com", verify=str(ca_dir))
        b = HTTPSession(base_url="https://example.com", verify=True)
        assert a is not b

    def test_pathlib_normalised_to_str(self, tmp_path):
        ca_dir = tmp_path / "certs"
        ca_dir.mkdir()
        a = HTTPSession(base_url="https://example.com", verify=ca_dir)
        assert isinstance(a.verify, str)


# ---------------------------------------------------------------------------
# Live HTTPS with verify=False
# ---------------------------------------------------------------------------


class TestLiveInsecure:

    @pytest.fixture(autouse=True)
    def _need_network(self, _check_network):
        pass

    def test_verify_false_https_succeeds(self):
        session = HTTPSession(base_url="https://example.com", verify=False)
        resp = session.get("/")
        assert resp.status_code == 200
        assert "example" in resp.text.lower()

    def test_insecure_factory_https_succeeds(self):
        session = HTTPSession(base_url="https://example.com").insecure()
        resp = session.get("/")
        assert resp.status_code == 200

    def test_verify_true_https_also_succeeds(self):
        session = HTTPSession(base_url="https://example.com", verify=True)
        resp = session.get("/")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# verify on session properties
# ---------------------------------------------------------------------------


class TestVerifyProperty:

    def test_verify_true_stored(self):
        session = HTTPSession(base_url="https://example.com", verify=True)
        assert session.verify is True

    def test_verify_false_stored(self):
        session = HTTPSession(base_url="https://example.com", verify=False)
        assert session.verify is False

    def test_verify_string_stored(self, tmp_path):
        ca_dir = tmp_path / "certs"
        ca_dir.mkdir()
        session = HTTPSession(base_url="https://example.com", verify=str(ca_dir))
        assert session.verify == str(ca_dir)

    def test_verify_pathlib_normalised(self, tmp_path):
        ca_dir = tmp_path / "certs"
        ca_dir.mkdir()
        session = HTTPSession(base_url="https://example.com", verify=ca_dir)
        assert isinstance(session.verify, str)
        assert session.verify == str(ca_dir)


# ---------------------------------------------------------------------------
# Invalid-cert auto-disable — warn once, flip verify=False, retry
# ---------------------------------------------------------------------------


def _self_signed_cert(tmp_path):
    """Write a throwaway self-signed cert + key; return (cert_path, key_path).

    Uses the ``openssl`` CLI — the ``cryptography`` wheel in this env has no
    working ``_cffi_backend``. Skips the test if ``openssl`` isn't available.
    """
    import shutil
    import subprocess

    if shutil.which("openssl") is None:
        pytest.skip("openssl CLI not available")
    cert_path = tmp_path / "cert.pem"
    key_path = tmp_path / "key.pem"
    subprocess.run(
        [
            "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
            "-keyout", str(key_path), "-out", str(cert_path),
            "-days", "1", "-subj", "/CN=127.0.0.1",
            "-addext", "subjectAltName=IP:127.0.0.1",
        ],
        check=True, capture_output=True,
    )
    return str(cert_path), str(key_path)


@pytest.fixture
def tls_server(tmp_path):
    import http.server
    import threading

    cert_path, key_path = _self_signed_cert(tmp_path)

    class _H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = b'{"ok": true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _H)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(certfile=cert_path, keyfile=key_path)
    srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
    # Swallow the client-aborted handshake (verify=True attempt) so it
    # doesn't spew tracebacks from the server thread.
    srv.handle_error = lambda *a, **k: None
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"https://127.0.0.1:{port}"
    srv.shutdown()


class TestInvalidCertAutoDisable:

    def test_cert_failure_disables_verify_and_succeeds(self, tls_server):
        session = HTTPSession(base_url=tls_server, verify=True)
        resp = session.get("/json")
        assert resp.status_code == 200
        assert resp.json()["ok"] is True
        # Verification was flipped off for the session (the "disable once").
        assert session.verify is False

    def test_cert_failure_keeps_proxy(self, tls_server):
        # The proxy must survive a target-cert failure (it's the target's
        # cert, not the proxy's). No real proxy reachable here, so point at a
        # dead one and confirm it's still set after the verify flip — the
        # request itself goes direct (no_proxy covers the loopback host).
        session = HTTPSession(
            base_url=tls_server, verify=True,
            proxy="http://127.0.0.1:9", no_proxy="127.0.0.1",
        )
        resp = session.get("/json")
        assert resp.status_code == 200
        assert session.verify is False
        assert session.proxy is not None          # not dropped
        assert session.proxy.host == "127.0.0.1"

    def test_cert_failure_warns_once(self, tls_server):
        import logging
        logger = logging.getLogger("yggdrasil.http_.session")
        records: list[logging.LogRecord] = []

        class _Cap(logging.Handler):
            def emit(self, record):
                records.append(record)

        handler = _Cap(level=logging.WARNING)
        logger.addHandler(handler)
        try:
            HTTPSession(base_url=tls_server, verify=True).get("/json")
        finally:
            logger.removeHandler(handler)
        hits = [r for r in records
                if "certificate verification failed" in r.getMessage().lower()]
        assert len(hits) == 1
