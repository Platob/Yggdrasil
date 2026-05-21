# yggdrasil.jwt

Parsing-only JWT primitives (RFC 7519). Signature verification is out of scope — that belongs in the caller with an algorithm plug-in (PyJWT, `cryptography`) and a key-resolution policy.

## One-liner

```python
from yggdrasil.jwt import JWTToken

tok = JWTToken.from_("eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjMifQ.sig")
print(tok.alg, tok.sub)   # HS256  123
```

## Parse a token

```python
from yggdrasil.jwt import JWTToken, JWTParseError

raw = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyQGV4YW1wbGUuY29tIiwiZXhwIjoxNzk5OTk5OTk5fQ.sig"

try:
    tok = JWTToken.from_(raw)
except JWTParseError as exc:
    print("Invalid token:", exc)
```

## Strip the Bearer prefix

```python
from yggdrasil.jwt import JWTToken

authorization_header = "Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjMifQ.sig"
tok = JWTToken.from_authorization(authorization_header)
print(tok.sub)
```

## Inspect claims

```python
tok = JWTToken.from_(raw)

# Header
print(tok.alg)           # "HS256"
print(tok.typ)           # "JWT"
print(tok.kid)           # key ID (if present)

# Registered claims
print(tok.sub)           # subject
print(tok.iss)           # issuer
print(tok.aud)           # audience (str or list)
print(tok.exp)           # expiry (datetime)
print(tok.iat)           # issued-at (datetime)
print(tok.jti)           # JWT ID

# Full raw payload dict
print(tok.payload)

# Check expiry
import datetime
print(tok.is_expired())
print(tok.is_expired(at=datetime.datetime.now(tz=datetime.timezone.utc)))

# Raw token string
print(tok.raw)
```

## Extract from HTTP requests (FastAPI / Starlette)

```python
from yggdrasil.jwt import JWTToken

def get_current_user(authorization: str) -> str:
    tok = JWTToken.from_authorization(authorization)
    # Pass tok.raw to PyJWT / python-jose for signature verification
    return tok.sub
```

## Databricks PAT / OAuth token introspection

```python
from yggdrasil.jwt import JWTToken

# Databricks OAuth access tokens are JWTs
access_token = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9..."
tok = JWTToken.from_(access_token)
print(tok.sub)    # service principal application ID
print(tok.exp)    # expiry time
print(tok.iss)    # Databricks issuer URL
```
