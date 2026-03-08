# yggdrasil.web.http_proxy.py
import asyncio
import logging
from urllib.parse import urlsplit

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


HOP_BY_HOP_HEADERS = {
    "connection",
    "proxy-connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}


async def pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, *, label: str = ""):
    try:
        while True:
            chunk = await reader.read(64 * 1024)
            if not chunk:
                break
            writer.write(chunk)
            await writer.drain()
    except Exception as e:
        logging.debug("pipe(%s) error: %r", label, e)
    finally:
        try:
            writer.close()
        except Exception:
            pass


async def read_headers(reader: asyncio.StreamReader, max_bytes: int = 64 * 1024) -> bytes:
    """
    Read until CRLFCRLF (end of headers). Raises if too large.
    """
    data = b""
    while b"\r\n\r\n" not in data:
        chunk = await reader.read(4096)
        if not chunk:
            break
        data += chunk
        if len(data) > max_bytes:
            raise ValueError("Headers too large")
    return data


def parse_request(head: bytes):
    """
    Returns: method, target, version, headers(list of (k,v)), rest_after_headers(bytes)
    """
    header_end = head.find(b"\r\n\r\n")
    if header_end == -1:
        raise ValueError("Malformed request (no header end)")

    header_block = head[:header_end].decode("iso-8859-1")
    rest = head[header_end + 4 :]

    lines = header_block.split("\r\n")
    req_line = lines[0]
    parts = req_line.split(" ", 2)
    if len(parts) != 3:
        raise ValueError(f"Bad request line: {req_line!r}")

    method, target, version = parts[0], parts[1], parts[2]
    headers = []
    for line in lines[1:]:
        if not line:
            continue
        k, _, v = line.partition(":")
        headers.append((k.strip(), v.lstrip()))
    return method, target, version, headers, rest


def headers_to_dict(headers):
    d = {}
    for k, v in headers:
        lk = k.lower()
        # combine repeated headers minimally (good enough for a proxy)
        if lk in d:
            d[lk] = d[lk] + ", " + v
        else:
            d[lk] = v
    return d


def build_forward_request(method: str, path: str, version: str, headers, host: str):
    """
    Build a proxied request for origin server (absolute-form -> origin-form).
    Removes hop-by-hop headers, sets Host, forces Connection: close for simplicity.
    """
    filtered = []
    for k, v in headers:
        if k.lower() in HOP_BY_HOP_HEADERS:
            continue
        # Drop Proxy-Authorization etc by hop-by-hop list already
        filtered.append((k, v))

    # Ensure Host exists and is correct
    # Remove any existing Host then add one
    filtered = [(k, v) for (k, v) in filtered if k.lower() != "host"]
    filtered.append(("Host", host))

    # Keep it simple & predictable
    filtered = [(k, v) for (k, v) in filtered if k.lower() != "connection"]
    filtered.append(("Connection", "close"))

    req = f"{method} {path} {version}\r\n"
    for k, v in filtered:
        req += f"{k}: {v}\r\n"
    req += "\r\n"
    return req.encode("iso-8859-1")


async def handle_connect(client_reader, client_writer, target: str):
    """
    CONNECT host:port -> open tunnel and blindly pipe bytes both ways.
    """
    host, _, port_s = target.partition(":")
    port = int(port_s) if port_s else 443
    logging.info("CONNECT %s:%s", host, port)

    try:
        upstream_reader, upstream_writer = await asyncio.open_connection(host, port)
    except Exception as e:
        resp = b"HTTP/1.1 502 Bad Gateway\r\nConnection: close\r\n\r\n"
        client_writer.write(resp)
        await client_writer.drain()
        client_writer.close()
        logging.info("CONNECT failed: %r", e)
        return

    # Tell client tunnel is established
    client_writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
    await client_writer.drain()

    # Bidirectional tunnel
    await asyncio.gather(
        pipe(client_reader, upstream_writer, label="c->u"),
        pipe(upstream_reader, client_writer, label="u->c"),
    )


def normalize_target_to_host_path(target: str, headers_dict):
    """
    Handle both:
      - absolute-form:  GET http://example.com/path HTTP/1.1
      - origin-form:    GET /path HTTP/1.1  with Host header
    Returns: host, port, path
    """
    if target.startswith("http://") or target.startswith("https://"):
        u = urlsplit(target)
        host = u.hostname
        scheme = u.scheme
        port = u.port or (443 if scheme == "https" else 80)
        path = u.path or "/"
        if u.query:
            path += "?" + u.query
        return host, port, path
    else:
        # origin-form, must use Host header
        host_hdr = headers_dict.get("host")
        if not host_hdr:
            raise ValueError("No Host header for origin-form request")
        # host may include port
        h, _, p = host_hdr.partition(":")
        host = h
        port = int(p) if p else 80
        path = target or "/"
        return host, port, path


async def handle_http(client_reader, client_writer, method, target, version, headers, body_prefetch: bytes):
    hdict = headers_to_dict(headers)
    host, port, path = normalize_target_to_host_path(target, hdict)
    logging.info("%s %s -> %s:%s%s", method, target, host, port, path)

    try:
        upstream_reader, upstream_writer = await asyncio.open_connection(host, port)
    except Exception as e:
        resp = b"HTTP/1.1 502 Bad Gateway\r\nConnection: close\r\n\r\n"
        client_writer.write(resp)
        await client_writer.drain()
        client_writer.close()
        logging.info("Upstream connect failed: %r", e)
        return

    # Build and send request line + headers to origin
    req_head = build_forward_request(method, path, version, headers, host=host if port in (80, 443) else f"{host}:{port}")
    upstream_writer.write(req_head)

    # Send any already-read body bytes
    if body_prefetch:
        upstream_writer.write(body_prefetch)

    await upstream_writer.drain()

    # If there is a request body not yet read, stream it.
    # We try to stream until EOF if client closes; or if Content-Length exists, read exactly.
    content_length = None
    if "content-length" in hdict:
        try:
            content_length = int(hdict["content-length"])
        except ValueError:
            content_length = None

    if content_length is not None:
        remaining = max(0, content_length - len(body_prefetch))
        while remaining > 0:
            chunk = await client_reader.read(min(64 * 1024, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            upstream_writer.write(chunk)
            await upstream_writer.drain()
    else:
        # No length: could be chunked from client; we removed Transfer-Encoding header
        # For simplicity, just stream what comes until client stops sending.
        # Many clients will use Content-Length anyway.
        # NOTE: this is "best effort" behavior.
        try:
            while True:
                chunk = await asyncio.wait_for(client_reader.read(64 * 1024), timeout=0.01)
                if not chunk:
                    break
                upstream_writer.write(chunk)
                await upstream_writer.drain()
        except asyncio.TimeoutError:
            pass

    # Now relay the upstream response back to client
    await asyncio.gather(
        pipe(upstream_reader, client_writer, label="resp u->c"),
    )
    try:
        upstream_writer.close()
    except Exception:
        pass


async def handle_client(client_reader: asyncio.StreamReader, client_writer: asyncio.StreamWriter):
    peer = client_writer.get_extra_info("peername")
    try:
        head = await read_headers(client_reader)
        if not head:
            client_writer.close()
            return

        method, target, version, headers, rest = parse_request(head)

        if method.upper() == "CONNECT":
            await handle_connect(client_reader, client_writer, target)
        else:
            await handle_http(client_reader, client_writer, method, target, version, headers, rest)
    except Exception as e:
        logging.info("Client %s error: %r", peer, e)
        try:
            client_writer.write(b"HTTP/1.1 400 Bad Request\r\nConnection: close\r\n\r\n")
            await client_writer.drain()
        except Exception:
            pass
        try:
            client_writer.close()
        except Exception:
            pass


async def main(host="127.0.0.1", port=8888):
    server = await asyncio.start_server(handle_client, host, port)
    addrs = ", ".join(str(sock.getsockname()) for sock in server.sockets or [])
    logging.info("Proxy listening on %s", addrs)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass