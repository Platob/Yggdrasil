# cython: language_level=3
"""URL parsing, normalization, and percent-encoding kernels (Cython).

Byte-for-byte equivalent to the helpers in
``yggdrasil/io/url.py`` and the deprecated ``rust/src/io/url.rs``:

* :func:`parse_url`        — ``urlsplit`` + ``_parse_netloc`` + entity
                             fix-up + Windows drive-letter fix-up +
                             "no //" authority recovery, returning a
                             tuple
                             ``(scheme, userinfo, host, port, path,
                             query, fragment)``.
* :func:`normalize_components` — lowercase scheme/host, strip
                                 trailing dot, drop default port,
                                 sorted-query, fragment lstrip.
* :func:`encode_path` / :func:`encode_query` /
  :func:`encode_fragment` / :func:`encode_userinfo` — percent-encoders
  with the project's ``safe`` sets.
* :func:`parse_netloc`     — standalone netloc splitter.
* :func:`normalize_query`  — sorted ``urlencode(parse_qsl(...))``
                             output, matching the Python normalizer.

Keep the unit tests in
``python/tests/test_yggdrasil/test_io/test_url.py`` green when changing
anything here.
"""

from cpython.bytes cimport PyBytes_AsStringAndSize, PyBytes_FromStringAndSize
from libc.stdint cimport uint8_t, uint32_t
from libc.string cimport memset


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

cdef str SAFE_PATH = "/:@-._~!$&'()*+,;="
cdef str SAFE_QUERY = "-._~!$'()*+,;=:@/?&="
cdef str SAFE_FRAGMENT = "-._~!$&'()*+,;=:@/?"
cdef str SAFE_USERINFO = ":!$&'()*+,;="


cdef inline uint32_t _default_port(str scheme):
    if scheme == "http" or scheme == "ws":
        return 80
    if scheme == "https" or scheme == "wss":
        return 443
    return 0


# ---------------------------------------------------------------------------
# Byte-level helpers
# ---------------------------------------------------------------------------

cdef inline bint _is_unreserved(uint8_t b) nogil:
    # alnum + "-._~"
    if b >= 48 and b <= 57:        # 0-9
        return True
    if b >= 65 and b <= 90:        # A-Z
        return True
    if b >= 97 and b <= 122:       # a-z
        return True
    if b == 45 or b == 46 or b == 95 or b == 126:  # -, ., _, ~
        return True
    return False


cdef inline char _hex_upper(uint8_t nibble) nogil:
    if nibble <= 9:
        return <char>(48 + nibble)   # '0' + nibble
    return <char>(65 + nibble - 10)  # 'A' + (nibble - 10)


cdef inline int _decode_hex(uint8_t b) nogil:
    """Return 0-15 for an ASCII hex digit, -1 otherwise."""
    if b >= 48 and b <= 57:
        return b - 48
    if b >= 97 and b <= 102:
        return 10 + b - 97
    if b >= 65 and b <= 70:
        return 10 + b - 65
    return -1


cdef void _build_safe_table(uint8_t *table, str safe):
    """Populate a 256-byte lookup table marking ``safe`` chars as 1."""
    memset(table, 0, 256)
    cdef bytes safe_b = safe.encode("ascii")
    cdef int i, n = len(safe_b)
    for i in range(n):
        table[<uint8_t>safe_b[i]] = 1


cdef bytes _percent_encode_bytes(bytes value, str safe):
    """Percent-encode *value* keeping unreserved chars and *safe* chars."""
    cdef uint8_t safe_table[256]
    _build_safe_table(safe_table, safe)

    cdef const char *src
    cdef Py_ssize_t n
    PyBytes_AsStringAndSize(value, <char**>&src, &n)

    # Worst case: every byte expands 1 -> 3.
    cdef bytearray out = bytearray()
    cdef Py_ssize_t i
    cdef uint8_t b
    for i in range(n):
        b = <uint8_t>src[i]
        if _is_unreserved(b) or safe_table[b]:
            out.append(b)
        else:
            out.append(37)  # '%'
            out.append(_hex_upper(b >> 4))
            out.append(_hex_upper(b & 0x0f))
    return bytes(out)


cdef str _percent_encode(str value, str safe):
    """Encode *value* using ``safe`` — Python `urllib.parse.quote` shape.

    Empty input → empty output.
    """
    if not value:
        return ""
    cdef bytes encoded = _percent_encode_bytes(value.encode("utf-8"), safe)
    return encoded.decode("ascii")


cdef str _percent_decode(str value):
    """Decode percent triplets; malformed triplets pass through verbatim.

    Mirrors :func:`urllib.parse.unquote` (UTF-8, ``errors='replace'``).
    """
    if not value:
        return ""
    cdef bytes src_b = value.encode("utf-8")
    cdef const char *src
    cdef Py_ssize_t n
    PyBytes_AsStringAndSize(src_b, <char**>&src, &n)

    cdef bytearray out = bytearray()
    cdef Py_ssize_t i = 0
    cdef int hi, lo
    while i < n:
        if src[i] == b'%' and i + 2 < n:
            hi = _decode_hex(<uint8_t>src[i + 1])
            lo = _decode_hex(<uint8_t>src[i + 2])
            if hi >= 0 and lo >= 0:
                out.append((hi << 4) | lo)
                i += 3
                continue
        out.append(<uint8_t>src[i])
        i += 1
    return bytes(out).decode("utf-8", "replace")


cdef inline str _maybe_decode(str value, bint decode):
    if decode and value:
        return _percent_decode(value)
    return value


cdef inline str _lower_if(str value):
    if not value:
        return ""
    return value.lower()


cdef inline str _strip_trailing_dot(str host):
    if host.endswith("."):
        return host[:-1]
    return host


cdef int _parse_port_text(str text):
    """Return parsed port (>0) or 0 when *text* isn't a positive integer."""
    if not text:
        return 0
    cdef Py_ssize_t i, n = len(text)
    cdef Py_UCS4 c
    for i in range(n):
        c = text[i]
        if c < u'0' or c > u'9':
            return 0
    cdef int port
    try:
        port = int(text)
    except ValueError:
        return 0
    return port if port > 0 else 0


# ---------------------------------------------------------------------------
# Splitters mirroring urlsplit + _parse_netloc
# ---------------------------------------------------------------------------

cdef tuple _parse_netloc_internal(str netloc, bint decode):
    """Split *netloc* into ``(userinfo, host, port)``.

    ``port = 0`` means "no port" — kept as the project's _NO_PORT sentinel.
    """
    if not netloc:
        return ("", "", 0)

    cdef str userinfo = ""
    cdef str hostport = netloc
    cdef Py_ssize_t at_idx = netloc.rfind("@")
    if at_idx >= 0:
        userinfo = _maybe_decode(netloc[:at_idx], decode)
        hostport = netloc[at_idx + 1:]

    cdef Py_ssize_t rb
    cdef str rest, after
    cdef int port
    if hostport.startswith("["):
        rest = hostport[1:]
        rb = rest.find("]")
        if rb >= 0:
            host = rest[:rb]
            after = rest[rb + 1:]
            if after.startswith(":"):
                port = _parse_port_text(after[1:])
            else:
                port = 0
            return (userinfo, host, port)
        return (userinfo, hostport, 0)

    cdef Py_ssize_t colon_idx = hostport.rfind(":")
    if colon_idx >= 0:
        host = hostport[:colon_idx]
        port = _parse_port_text(hostport[colon_idx + 1:])
        return (userinfo, host, port)

    return (userinfo, hostport, 0)


cdef tuple _split_authority(str after_scheme):
    """Return ``(netloc_or_None, rest_after_authority)``."""
    if not after_scheme.startswith("//"):
        return (None, after_scheme)
    cdef str rest = after_scheme[2:]
    cdef Py_ssize_t i, n = len(rest)
    cdef Py_UCS4 c
    for i in range(n):
        c = rest[i]
        if c == u'/' or c == u'?' or c == u'#':
            return (rest[:i], rest[i:])
    return (rest, "")


cdef tuple _split_query_fragment(str rest):
    """Split ``rest`` into ``(path, query, fragment)`` — fragment wins."""
    cdef Py_ssize_t hash_idx = rest.find("#")
    cdef str head, fragment
    if hash_idx >= 0:
        head = rest[:hash_idx]
        fragment = rest[hash_idx + 1:]
    else:
        head = rest
        fragment = ""
    cdef Py_ssize_t q_idx = head.find("?")
    cdef str path, query
    if q_idx >= 0:
        path = head[:q_idx]
        query = head[q_idx + 1:]
    else:
        path = head
        query = ""
    return (path, query, fragment)


cdef tuple _split_scheme(str raw):
    """Return ``(scheme_or_None, after_scheme)`` — urlsplit's scheme rule."""
    if not raw:
        return (None, raw)
    cdef Py_UCS4 first = raw[0]
    if not (
        (first >= u'A' and first <= u'Z')
        or (first >= u'a' and first <= u'z')
    ):
        return (None, raw)
    cdef Py_ssize_t i, n = len(raw)
    cdef Py_UCS4 c
    for i in range(1, n):
        c = raw[i]
        if c == u':':
            return (raw[:i], raw[i + 1:])
        if not (
            (c >= u'0' and c <= u'9')
            or (c >= u'A' and c <= u'Z')
            or (c >= u'a' and c <= u'z')
            or c == u'+'
            or c == u'-'
            or c == u'.'
        ):
            return (None, raw)
    return (None, raw)


cdef tuple _split_url(str raw, object default_scheme, bint decode):
    cdef object scheme_opt
    cdef str after_scheme, scheme
    scheme_opt, after_scheme = _split_scheme(raw)
    scheme = scheme_opt if scheme_opt is not None else ""

    cdef object netloc_opt
    cdef str rest
    netloc_opt, rest = _split_authority(after_scheme)

    cdef str path_raw, query_raw, fragment_raw
    path_raw, query_raw, fragment_raw = _split_query_fragment(rest)

    cdef str userinfo, host
    cdef int port
    if netloc_opt is not None:
        userinfo, host, port = _parse_netloc_internal(netloc_opt, decode)
    else:
        userinfo = ""
        host = ""
        port = 0

    cdef str path = _maybe_decode(path_raw, decode)

    # Single-letter "scheme" → Windows drive letter. Reattach drive,
    # normalize backslash separators, force the file scheme.
    cdef str drive, rest_path
    cdef Py_UCS4 sc0
    if len(scheme) == 1:
        sc0 = scheme[0]
        if (sc0 >= u'A' and sc0 <= u'Z') or (sc0 >= u'a' and sc0 <= u'z'):
            drive = scheme.upper()
            rest_path = path.replace("\\", "/")
            if not rest_path.startswith("/"):
                rest_path = "/" + rest_path
            path = "/" + drive + ":" + rest_path
            scheme = "file"

    if default_scheme is not None and default_scheme != "":
        scheme = <str>default_scheme

    # XML/HTML entity fix-up: ``&amp;`` → ``&``.
    cdef str query = _maybe_decode(query_raw, decode).replace("&amp;", "&")
    cdef str fragment = _maybe_decode(fragment_raw, decode).replace("&amp;", "&")

    # "http:example.com/path" — recover host from the first path segment
    # when the authority "//" is missing. Skipped for ``file`` and for
    # schemaless input (which would mis-extract a real path).
    cdef Py_ssize_t slash_idx
    if scheme and scheme != "file" and not host and path:
        slash_idx = path.find("/")
        if slash_idx >= 0:
            host = path[:slash_idx]
            path = "/" + path[slash_idx:].lstrip("/")
        else:
            host = path
            path = "/"

    return (scheme, userinfo, host, port, path, query, fragment)


# ---------------------------------------------------------------------------
# Query normalization
# ---------------------------------------------------------------------------

cdef bytes _urlencode_pair_bytes(bytes value):
    """quote_plus semantics — space → '+', else urllib.quote default safe."""
    cdef const char *src
    cdef Py_ssize_t n
    PyBytes_AsStringAndSize(value, <char**>&src, &n)
    cdef bytearray out = bytearray()
    cdef Py_ssize_t i
    cdef uint8_t b
    for i in range(n):
        b = <uint8_t>src[i]
        if _is_unreserved(b):
            out.append(b)
        elif b == 32:  # ' '
            out.append(43)  # '+'
        else:
            out.append(37)  # '%'
            out.append(_hex_upper(b >> 4))
            out.append(_hex_upper(b & 0x0f))
    return bytes(out)


cdef inline str _urlencode_pair(str value):
    return _urlencode_pair_bytes(value.encode("utf-8")).decode("ascii")


cdef str _normalize_query_internal(str query):
    if not query:
        return ""
    cdef str q = query.lstrip("?")
    if not q:
        return ""

    cdef list items = []
    cdef Py_ssize_t eq_idx
    cdef str key, val
    cdef str pair
    for pair in q.split("&"):
        if not pair:
            continue
        eq_idx = pair.find("=")
        if eq_idx >= 0:
            key = pair[:eq_idx]
            val = pair[eq_idx + 1:]
        else:
            key = pair
            val = ""
        key = _percent_decode(key.replace("+", " "))
        val = _percent_decode(val.replace("+", " "))
        items.append((key, val))

    items.sort()
    cdef list parts = []
    for key, val in items:
        parts.append(_urlencode_pair(key) + "=" + _urlencode_pair(val))
    return "&".join(parts)


cdef inline str _normalize_path_internal(str path):
    if not path:
        return "/"
    return path


cdef inline int _remove_default_port(str scheme, str host, int port):
    if not scheme or not host or port == 0:
        return 0
    if <int>_default_port(scheme) == port:
        return 0
    return port


cdef tuple _normalize_components_internal(
    str scheme, str userinfo, str host, int port,
    str path, str query, str fragment,
):
    cdef str scheme_n = _lower_if(scheme)
    cdef str host_lower = _lower_if(host)
    cdef str host_n = _strip_trailing_dot(host_lower)
    cdef int port_n = _remove_default_port(scheme_n, host_n, port)
    cdef str path_n = _normalize_path_internal(path)
    cdef str query_n = _normalize_query_internal(query)
    cdef str fragment_n = fragment.lstrip("#") if fragment else ""
    return (scheme_n, userinfo, host_n, port_n, path_n, query_n, fragment_n)


# ---------------------------------------------------------------------------
# Public surface — these are the names ``yggdrasil/cy.py`` re-exports.
# ---------------------------------------------------------------------------


def parse_url(str raw, *, default_scheme=None, bint decode=False, bint normalize=True):
    """Parse a URL into ``(scheme, userinfo, host, port, path, query, fragment)``.

    ``userinfo`` / ``query`` / ``fragment`` may be empty strings — the
    Python caller maps those back to ``None`` to match the
    :class:`URL` dataclass shape. ``port = 0`` means "no port".
    """
    cdef str scheme, userinfo, host, path, query, fragment
    cdef int port
    scheme, userinfo, host, port, path, query, fragment = _split_url(
        raw, default_scheme, decode,
    )

    if not normalize:
        return (
            scheme,
            userinfo,
            host,
            port,
            "/" if not path else path,
            query.lstrip("?") if query else "",
            fragment.lstrip("#") if fragment else "",
        )

    return _normalize_components_internal(
        scheme, userinfo, host, port, path, query, fragment,
    )


def normalize_components(
    str scheme, str userinfo, str host, int port,
    str path, str query, str fragment, bint os_find_file=False,
):
    """Apply the cross-component normalization pipeline.

    ``os_find_file`` is accepted for API parity with the Rust kernel
    but unused here — the Python wrapper handles ``file://`` realpath
    resolution because it's platform-sensitive.
    """
    return _normalize_components_internal(
        scheme, userinfo, host, port, path, query, fragment,
    )


def encode_path(str value):
    """Percent-encode *value* using the URL-path safe set."""
    if not value:
        return ""
    # When the input has no space we extend the safe set with '%' so
    # already-encoded triplets don't get double-encoded.
    cdef str safe
    if " " in value:
        safe = SAFE_PATH
    else:
        safe = SAFE_PATH + "%"
    return _percent_encode(value, safe)


def encode_query(str value):
    """Percent-encode *value* using the URL-query safe set."""
    return _percent_encode(value, SAFE_QUERY + "%")


def encode_fragment(str value):
    """Percent-encode *value* using the URL-fragment safe set."""
    return _percent_encode(value, SAFE_FRAGMENT + "%")


def encode_userinfo(str value):
    """Percent-encode *value* using the userinfo safe set."""
    if not value:
        return ""
    return _percent_encode(value, SAFE_USERINFO)


def parse_netloc(str netloc, *, bint decode=False):
    """Split *netloc* into ``(userinfo, host, port)`` — port=0 means absent."""
    return _parse_netloc_internal(netloc, decode)


def normalize_query(str query):
    """Sorted ``urlencode(parse_qsl(...))`` — stable canonical form."""
    return _normalize_query_internal(query)
