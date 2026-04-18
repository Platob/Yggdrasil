"""`Response._to_asgi_payload`, `apply`, `__repr__`, `update_*`."""

from __future__ import annotations

from yggdrasil.io.response import Response

from .._helpers import make_request, make_response


def test_to_asgi_payload_strips_hop_by_hop_and_sets_content_length() -> None:
    resp = make_response(
        headers={
            "Content-Type": "application/json",
            "Transfer-Encoding": "chunked",
            "Connection": "keep-alive",
            "X-Test": "1",
        },
        body=b'{"ok":true}',
    )

    body, headers, media_type = resp._to_asgi_payload()

    assert body == b'{"ok":true}'
    assert headers["Content-Length"] == str(len(body))
    assert headers["X-Test"] == "1"
    # hop-by-hop headers are dropped in ASGI output
    assert "Connection" not in headers
    assert "Transfer-Encoding" not in headers
    assert media_type == "application/json"


def test_apply_returns_transformed_response() -> None:
    resp = make_response()

    def transform(r: Response) -> Response:
        r.status_code = 201
        return r

    out = resp.apply(transform)
    assert out.status_code == 201


def test_repr_contains_status_code_and_url() -> None:
    resp = make_response(body=b"hello")

    rep = repr(resp)

    assert "200" in rep
    assert "https://example.com/a" in rep


def test_update_headers_and_tags_mutate_in_place() -> None:
    resp = Response(
        request=make_request(),
        status_code=200,
        headers={"Content-Type": "text/plain"},
        tags={"a": "1"},
        buffer=b"hello",  # type: ignore[arg-type]
        received_at=1,
    )

    resp.update_headers({"X-Test": "2"})
    resp.update_tags({"b": "2"})

    assert resp.headers["X-Test"] == "2"
    assert resp.tags == {"a": "1", "b": "2"}
