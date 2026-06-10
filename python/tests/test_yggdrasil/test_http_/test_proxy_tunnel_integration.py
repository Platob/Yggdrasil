"""End-to-end CONNECT-tunnel integration for HTTPS-through-proxy.

``test_proxy.py`` covers proxy *resolution*, ``no_proxy`` matching, plain-HTTP
forwarding, and the dead-proxy → direct fallback against an HTTP origin. What it
can't cover with its stub (whose ``do_CONNECT`` only acknowledges the verb) is
the real HTTPS path: a CONNECT tunnel spliced through to a TLS origin. These
tests stand up a socket-splicing forward proxy and a self-signed TLS origin so
:meth:`HTTPSession._build_connect_tunnel` runs for real, plus the HTTPS variant
of proxy-down → direct fallback.
"""
from __future__ import annotations

import datetime
import http.server
import ipaddress
import socket
import socketserver
import ssl
import threading
import warnings

import pytest

from yggdrasil.http_.session import HTTPSession


# ---------------------------------------------------------------------------
# Self-signed TLS origin
# ---------------------------------------------------------------------------

def _self_signed(tmp_path):
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "127.0.0.1")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name).issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(
            x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    certfile = tmp_path / "cert.pem"
    keyfile = tmp_path / "key.pem"
    certfile.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    keyfile.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
    )
    return str(certfile), str(keyfile)


class _TLSOriginHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        body = b'{"tls": true, "path": "%s"}' % self.path.encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


class _TLSOrigin(socketserver.ThreadingMixIn, http.server.HTTPServer):
    # Threaded so a pooled keep-alive tunnel held open by the client doesn't
    # wedge serve_forever — otherwise shutdown() blocks in teardown.
    daemon_threads = True


@pytest.fixture
def tls_origin(tmp_path):
    cert, key = _self_signed(tmp_path)
    srv = _TLSOrigin(("127.0.0.1", 0), _TLSOriginHandler)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(cert, key)
    srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]
    yield f"https://127.0.0.1:{port}", port
    srv.shutdown()


# ---------------------------------------------------------------------------
# Socket-splicing forward proxy (real CONNECT tunnel)
# ---------------------------------------------------------------------------

class _ConnectProxy(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.connects: list[str] = []  # CONNECT targets seen ("host:port")


class _ConnectHandler(socketserver.BaseRequestHandler):
    def handle(self):
        sock = self.request
        buf = b""
        while b"\r\n\r\n" not in buf:
            chunk = sock.recv(4096)
            if not chunk:
                return
            buf += chunk
        header, _, rest = buf.partition(b"\r\n\r\n")
        line = header.split(b"\r\n", 1)[0].decode(errors="replace")
        parts = line.split(None, 2)
        if len(parts) < 2 or parts[0] != "CONNECT":
            sock.sendall(b"HTTP/1.1 405 Method Not Allowed\r\n\r\n")
            return
        target = parts[1]
        self.server.connects.append(target)
        host, _, port = target.rpartition(":")
        try:
            upstream = socket.create_connection((host, int(port)), timeout=5)
        except OSError:
            sock.sendall(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
            return
        sock.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
        if rest:  # any client bytes that arrived with the CONNECT
            upstream.sendall(rest)
        self._splice(sock, upstream)

    @staticmethod
    def _splice(a, b):
        def pipe(src, dst):
            try:
                while True:
                    data = src.recv(65536)
                    if not data:
                        break
                    dst.sendall(data)
            except OSError:
                pass
            finally:
                for s in (src, dst):
                    try:
                        s.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
        threads = [
            threading.Thread(target=pipe, args=(a, b), daemon=True),
            threading.Thread(target=pipe, args=(b, a), daemon=True),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()


@pytest.fixture
def connect_proxy():
    srv = _ConnectProxy(("127.0.0.1", 0), _ConnectHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]
    yield f"http://127.0.0.1:{port}", srv
    srv.shutdown()


@pytest.fixture(autouse=True)
def _fresh():
    HTTPSession._INSTANCES.clear()
    yield
    HTTPSession._INSTANCES.clear()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_https_request_tunnels_through_connect_proxy(tls_origin, connect_proxy):
    origin_url, origin_port = tls_origin
    proxy_url, proxy = connect_proxy
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # verify=False InsecureRequestWarning
        session = HTTPSession(proxy=proxy_url, verify=False)
        resp = session.get(f"{origin_url}/secret")
    assert resp.status_code == 200
    assert resp.json()["tls"] is True
    assert resp.json()["path"] == "/secret"
    # The proxy actually tunneled a CONNECT to the TLS origin.
    assert f"127.0.0.1:{origin_port}" in proxy.connects


def test_tunnel_reused_for_second_request(tls_origin, connect_proxy):
    # The pooled tunnel socket serves a follow-up request without a second
    # CONNECT (keep-alive through the proxy).
    origin_url, origin_port = tls_origin
    proxy_url, proxy = connect_proxy
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        session = HTTPSession(proxy=proxy_url, verify=False)
        r1 = session.get(f"{origin_url}/one")
        r2 = session.get(f"{origin_url}/two")
    assert r1.json()["path"] == "/one"
    assert r2.json()["path"] == "/two"
    assert proxy.connects.count(f"127.0.0.1:{origin_port}") == 1


def test_https_falls_back_to_direct_when_proxy_unreachable(tls_origin):
    # An unreachable proxy for an HTTPS target must mark the proxy dead and
    # connect directly to the origin — the HTTPS mirror of the plain-HTTP
    # dead-proxy fallback covered in test_proxy.py.
    origin_url, _ = tls_origin
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        session = HTTPSession(proxy="http://127.0.0.1:19897", verify=False)
        assert session._dead_proxies == set()
        resp = session.get(f"{origin_url}/direct")
    assert resp.status_code == 200
    assert resp.json()["tls"] is True
    assert len(session._dead_proxies) == 1
