# ygg (Python)

Python bindings over the Rust [`ygg`](../rust/ygg) engine. All logic lives
in Rust; this package only exposes a Pythonic surface.

## Install (from source)

```bash
pip install maturin
cd python
maturin develop        # build + install into the active venv
```

## Use

```python
from ygg import Uri, Url

u = Url.parse("https://user:pw@example.com:8443/a/b?q=1#f")
u.scheme    # 'https'
u.host      # 'example.com'
u.port      # 8443
u.path      # '/a/b'
str(u)      # round-trips back to the original

Uri.parse("urn:isbn:0451450523").path   # 'isbn:0451450523'
```

Built and published to PyPI by `.github/workflows/publish-python.yml`.
