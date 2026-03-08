from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass, field, is_dataclass, fields
from enum import Enum
from typing import Optional, Tuple, Any, Iterator, Union

from databricks.sdk.service.workspace import AclPermission

from ..client import DatabricksService

__all__ = ["Secrets"]


BytesLike = Union[bytes, bytearray, memoryview]

# Signature bytes are exactly 4 bytes and serve as a content prefix
# embedded in the base64-encoded bytes_value stored in Databricks.
#
# Wire format (unencrypted):
#   [ 4-byte signature ] [ payload bytes ]
#
# Wire format (encrypted, Signature.ENCR):
#   [ b"ENCR" ] [ 16-byte PBKDF2 salt ] [ Fernet ciphertext ]
#
# BYTE  -> payload is raw binary
# JSON  -> payload is UTF-8 JSON text
# DILL  -> payload is dill-serialized Python object (caller manages dill import)
# ENCR  -> payload is AES-128-CBC + HMAC-SHA256 (Fernet) encrypted bytes;
#          the inner plaintext carries its own nested signature so the
#          round-trip is: encrypt(sig_prefix + plaintext) -> ENCR-prefixed blob

_PBKDF2_ITERATIONS = 480_000  # OWASP 2023 recommendation for PBKDF2-HMAC-SHA256
_SALT_LEN = 16


def _derive_fernet_key(secret_key: str, salt: bytes) -> bytes:
    """Derive a 32-byte key from *secret_key* + *salt* via PBKDF2-HMAC-SHA256,
    then base64url-encode it so Fernet can consume it directly."""
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.backends import default_backend

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=_PBKDF2_ITERATIONS,
        backend=default_backend(),
    )
    raw_key = kdf.derive(secret_key.encode("utf-8"))
    return base64.urlsafe_b64encode(raw_key)


def _encrypt(plaintext: bytes, secret_key: str) -> bytes:
    """Return ``salt (16 B) + fernet_ciphertext``."""
    from cryptography.fernet import Fernet

    salt = os.urandom(_SALT_LEN)
    fernet = Fernet(_derive_fernet_key(secret_key, salt))
    return salt + fernet.encrypt(plaintext)


def _decrypt(ciphertext: bytes, secret_key: str) -> bytes:
    """Expect ``salt (16 B) + fernet_ciphertext``, return plaintext bytes."""
    from cryptography.fernet import Fernet, InvalidToken

    if len(ciphertext) < _SALT_LEN:
        raise ValueError("Encrypted payload is too short to contain a valid salt.")
    salt, token = ciphertext[:_SALT_LEN], ciphertext[_SALT_LEN:]
    fernet = Fernet(_derive_fernet_key(secret_key, salt))
    try:
        return fernet.decrypt(token)
    except InvalidToken as exc:
        raise ValueError("Decryption failed – wrong secret_key or corrupted payload.") from exc


class Signature(bytes, Enum):
    RAW = b"BYTE"
    JSON = b"JSON"
    DILL = b"DILL"
    ENCR = b"ENCR"

    # ---------------------------------------------------------------
    # Encoding helpers
    # ---------------------------------------------------------------

    def encode_payload(self, payload: bytes, *, secret_key: Optional[str] = None) -> str:
        """Prepend this signature to *payload* and return a base64 string.

        When ``self`` is ``Signature.ENCR`` a *secret_key* **must** be supplied;
        the payload is encrypted before the prefix is attached.
        """
        if self is Signature.ENCR:
            if not secret_key:
                raise ValueError("secret_key is required for Signature.ENCR encoding.")
            payload = _encrypt(payload, secret_key)
        return base64.b64encode(self.value + payload).decode("ascii")

    @staticmethod
    def decode_payload(
        b64: str,
        *,
        secret_key: Optional[str] = None,
    ) -> Tuple["Signature | None", bytes]:
        """Decode a base64 string and strip the leading signature.

        Returns ``(signature, payload_bytes)``.

        For ``Signature.ENCR`` blobs the payload is decrypted automatically when
        *secret_key* is provided.  If *secret_key* is omitted the raw encrypted
        bytes are returned so callers can decide what to do.

        If no recognised signature is found the full decoded bytes are returned
        with ``signature=None`` so callers can fall back to heuristics.
        """
        try:
            raw = base64.b64decode(b64, validate=True)
        except Exception:
            return None, b64.encode("utf-8") if isinstance(b64, str) else bytes(b64)

        if len(raw) < 4:
            return None, raw

        prefix = raw[:4]
        payload = raw[4:]

        for sig in Signature:
            if sig.value == prefix:
                if sig is Signature.ENCR and secret_key:
                    payload = _decrypt(payload, secret_key)
                return sig, payload

        # No known prefix – treat the whole buffer as the payload
        return None, raw


@dataclass(frozen=True)
class Secrets(DatabricksService):
    scope: Optional[str] = None
    key: Optional[str] = None
    update_timestamp: Optional[float] = None

    # When set, all reads and writes transparently encrypt / decrypt the payload
    # using AES-128-CBC + HMAC-SHA256 (Fernet) with a PBKDF2-derived key.
    # The wire format is: base64( b"ENCR" + 16-byte-salt + fernet-ciphertext ).
    # Set too None to disable encryption (plain signature-prefixed storage).
    secret_key: Optional[str] = field(default=None, repr=False)

    _value: Optional[Any] = field(default=None, repr=False)

    @classmethod
    def service_name(cls):
        return "secrets"

    # -------------------------
    # Ergonomics
    # -------------------------
    def __getitem__(self, item):
        return self.find_secret(full_key=item)

    def __setitem__(self, key, value):
        return self.update(value, full_key=key)

    def secrets_client(self):
        return self.client.workspace_client().secrets

    # -------------------------
    # Parsing / resolving
    # -------------------------
    @staticmethod
    def _parse_full_key(full_key: str) -> Tuple[str, str]:
        fk = full_key.strip()
        if ":" in fk:
            scope, key = fk.split(":", 1)
        elif "/" in fk:
            scope, key = fk.split("/", 1)
        else:
            raise ValueError("full_key must be in the format 'scope:key' (or 'scope/key').")

        scope = scope.strip()
        key = key.strip()
        if not scope or not key:
            raise ValueError("full_key must contain both non-empty scope and key.")
        return scope, key

    def _resolve_scope_key(
        self,
        *,
        full_key: Optional[str],
        scope: Optional[str],
        key: Optional[str],
    ) -> Tuple[str, str]:
        if full_key:
            scope, key = self._parse_full_key(full_key)
        scope = scope or self.scope
        key = key or self.key
        if not scope or not key:
            raise ValueError("Both 'scope' and 'key' must be provided (or set on self).")
        return scope, key

    # -------------------------
    # Value coercion (write)
    # -------------------------
    @staticmethod
    def coerce_to_put_payload(
        value: Any,
        *,
        signature: Signature | None = None,
        secret_key: Optional[str] = None,
        bytes_value: Optional[str] = None,
        string_value: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Convert *value* to either ``(bytes_value, None)`` or ``(None, string_value)``
        as expected by the Databricks ``put_secret`` API.

        When a *secret_key* is provided, the payload is **always** written as an
        encrypted ``bytes_value`` regardless of the input type, and the signature
        is forced to ``Signature.ENCR``:

            bytes_value = base64( b"ENCR" + 16-byte-salt + Fernet(plaintext) )

        where ``plaintext = <4-byte-inner-sig> + <serialised-payload>`` so the
        type information survives the encryption round-trip.

        Without a *secret_key* the inference rules are:
          - dataclass / dict / list / other JSON-able types  -> ``Signature.JSON``
          - ``bytes`` / ``bytearray`` / ``memoryview``       -> ``Signature.RAW``
          - ``str``                                          -> plain ``string_value``
                                                               (no prefix, backwards-compatible)

        Explicit ``bytes_value`` / ``string_value`` kwargs bypass all coercion.
        """
        # Caller-supplied raw values bypass all coercion
        if bytes_value is not None or string_value is not None:
            return bytes_value, string_value

        if value is None:
            return None, None

        # ── serialise to (inner_sig, payload_bytes) ────────────────
        if is_dataclass(value):
            metadata: dict[str, Any] = {
                f.name: getattr(value, f.name)
                for f in fields(value)
                if not f.name.startswith("_")
            }
            inner_sig = signature or Signature.JSON
            payload = json.dumps(metadata).encode("utf-8")

        elif isinstance(value, str):
            if secret_key or signature is not None:
                inner_sig = signature or Signature.JSON
                payload = value.encode("utf-8")
            else:
                # Default: keep as plain string_value (no prefix) for readability
                return None, value

        elif isinstance(value, (bytes, bytearray, memoryview)):
            inner_sig = signature or Signature.RAW
            payload = bytes(value)

        else:
            inner_sig = signature or Signature.JSON
            try:
                payload = json.dumps(value).encode("utf-8")
            except (TypeError, ValueError):
                payload = str(value).encode("utf-8")

        # ── wrap with encryption when secret_key is present ────────
        if secret_key:
            # Embed the inner signature inside the plaintext so type info
            # is preserved after decryption: plaintext = inner_sig + payload
            plaintext = inner_sig.value + payload
            return Signature.ENCR.encode_payload(plaintext, secret_key=secret_key), None

        return inner_sig.encode_payload(payload), None

    # -------------------------
    # Value parsing (read)
    # -------------------------
    @staticmethod
    def _looks_like_json(s: str) -> bool:
        s = s.strip()
        return (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]"))

    @staticmethod
    def _try_parse_value(v: Any, *, secret_key: Optional[str] = None) -> Any:
        """
        Best-effort parsing when reading back a secret.

        1. If the stored value is base64 with a known ``Signature`` prefix:
             - ``Signature.ENCR`` -> decrypt with *secret_key*, then recurse on
               the plaintext which carries its own inner 4-byte signature.
               Raises ``ValueError`` if *secret_key* is missing or wrong.
             - ``Signature.JSON`` -> ``json.loads(payload)``
             - ``Signature.RAW``  -> raw ``bytes``
             - ``Signature.DILL`` -> raw ``bytes`` (caller must ``dill.loads``)
        2. No signature found: fall back to heuristics (direct JSON, plain string).
        """
        if v is None:
            return None

        # ── bytes input ────────────────────────────────────────────
        if isinstance(v, (bytes, bytearray, memoryview)):
            b = bytes(v)
            try:
                s = b.decode("utf-8")
            except Exception:
                return b
            return Secrets._try_parse_value(s, secret_key=secret_key)

        if not isinstance(v, str):
            return v

        s = v.strip()
        if not s:
            return v

        # ── try signature-prefixed base64 first ────────────────────
        sig, payload = Signature.decode_payload(s, secret_key=secret_key)

        if sig is Signature.ENCR:
            # decode_payload already decrypted; plaintext = inner_sig (4B) + data
            if len(payload) < 4:
                raise ValueError("Decrypted ENCR payload is too short to contain an inner signature.")
            inner_sig_bytes = payload[:4]
            inner_payload = payload[4:]
            # Resolve inner signature and recurse as if we read a plain prefixed blob
            for inner_sig in Signature:
                if inner_sig.value == inner_sig_bytes:
                    # Reconstruct a standard prefixed b64 string and recurse
                    reconstructed = base64.b64encode(inner_sig.value + inner_payload).decode("ascii")
                    return Secrets._try_parse_value(reconstructed)
            # Unknown inner sig – return raw bytes
            return inner_payload

        if sig is Signature.JSON:
            try:
                return json.loads(payload.decode("utf-8"))
            except Exception:
                return payload.decode("utf-8", errors="replace")

        if sig is Signature.RAW:
            return payload  # raw bytes; caller decides

        if sig is Signature.DILL:
            return payload  # raw bytes; caller does dill.loads(...)

        # sig is None -> no recognized prefix, fall through to heuristics

        # ── heuristic: direct JSON string ─────────────────────────
        if Secrets._looks_like_json(s):
            try:
                return json.loads(s)
            except Exception:
                return v

        # ── heuristic: base64 containing utf-8 / json ─────────────
        # (legacy values written before signatures were introduced)
        try:
            decoded = base64.b64decode(s, validate=True)
        except Exception:
            return v

        try:
            decoded_text = decoded.decode("utf-8")
        except Exception:
            return decoded  # opaque binary

        dt = decoded_text.strip()
        if Secrets._looks_like_json(dt):
            try:
                return json.loads(dt)
            except Exception:
                return decoded_text

        return decoded_text

    def decode_value(
        self,
        *,
        as_type: type | None = None,
        encoding: str = "utf-8",
        secret_key: str | None = None,
    ) -> Any:
        """Decode the secret's stored value into a typed Python object.

        Builds on :meth:`value` (which handles signature stripping and optional
        decryption) and adds an explicit *as_type* coercion layer on top, so
        callers can request a concrete type without writing their own
        ``isinstance`` ladder.

        Supported *as_type* targets
        ---------------------------
        ``str``
            Decode bytes with *encoding*; JSON-serialise non-string objects.
        ``bytes``
            Encode strings with *encoding*; JSON-serialise other objects.
        ``dict`` / ``list``
            Parse via ``json.loads`` when the value is a string or bytes;
            raise ``TypeError`` if the result is the wrong container type.
        ``int`` / ``float``
            Cast via the target type's constructor.
        ``None`` *(default)*
            Return the value as-is after signature parsing (same as
            :attr:`value`).

        Parameters
        ----------
        as_type:
            Target Python type.  ``None`` returns the parsed value unchanged.
        encoding:
            Charset used for ``str`` ↔ ``bytes`` coercions.  Defaults to
            ``"utf-8"``.
        secret_key:
            Override the instance-level ``self.secret_key`` for this single
            call.  Useful when the caller holds the key but does not want it
            stored on the object.

        Returns
        -------
        Any
            Parsed and optionally coerced secret value.

        Raises
        ------
        ValueError
            When decryption fails (wrong key or corrupted payload).
        TypeError
            When the parsed value cannot be coerced to *as_type*.
        json.JSONDecodeError
            When ``dict`` / ``list`` coercion is requested but the value is
            not valid JSON.

        Examples
        --------
        >>> secret["myapp:db_password"].decode_value(as_type=str)
        'hunter2'

        >>> secret["myapp:config"].decode_value(as_type=dict)
        {'host': 'localhost', 'port': 5432}

        >>> secret["myapp:cert"].decode_value(as_type=bytes)
        b'\\x30\\x82...'
        """
        effective_key = secret_key or self.secret_key

        # Re-parse from raw storage when a different secret_key is supplied,
        # otherwise use the already-cached value to avoid a network round-trip.
        if effective_key != self.secret_key and self.scope and self.key:
            raw_secret = self.find_secret(scope=self.scope, key=self.key, secret_key=effective_key)
            v = raw_secret._value
        else:
            v = self.value  # triggers lazy fetch if needed

        if v is None or as_type is None:
            return v

        return _coerce_value(v, as_type=as_type, encoding=encoding)

    # -------------------------
    # Errors
    # -------------------------
    @staticmethod
    def _is_scope_not_found_error(e: Exception) -> bool:
        msg = (getattr(e, "message", None) or str(e) or "").lower()
        code = (getattr(e, "error_code", None) or "").lower()

        if "secret does not exist" in msg or "key does not exist" in msg:
            return False

        if (
            "scope does not exist" in msg
            or ("secret scope" in msg and "does not exist" in msg)
            or "resource_does_not_exist" in msg
            or "not_found" in msg
        ):
            return True

        if code in {"resource_does_not_exist", "not_found"}:
            return True

        return False

    # -------------------------
    # Secret value operations
    # -------------------------
    def find_secret(
        self,
        full_key: Optional[str] = None,
        *,
        scope: Optional[str] = None,
        key: Optional[str] = None,
        secret_key: Optional[str] = None,
    ) -> "Secrets":
        scope, key = self._resolve_scope_key(full_key=full_key, scope=scope, key=key)
        client = self.secrets_client()

        response = client.get_secret(scope=scope, key=key)

        raw = response.value
        effective_key = secret_key or self.secret_key
        parsed = self._try_parse_value(raw, secret_key=effective_key)

        return Secrets(
            client=self.client,
            scope=scope,
            key=response.key,
            secret_key=effective_key,
            _value=parsed,
        )

    @property
    def value(self) -> Optional[Any]:
        """
        Lazy read + best-effort parse.
        Values written with a ``Signature`` prefix are decoded accordingly.
        When ``self.secret_key`` is set, ``ENCR``-prefixed blobs are decrypted
        transparently.  Legacy values (no prefix) fall back to heuristic detection.
        """
        if self._value is None and self.scope and self.key:
            fetched = self.find_secret(scope=self.scope, key=self.key)

            object.__setattr__(self, "_value", fetched._value)  # cache the parsed value for future accesses
        return self._value

    def value_text(self, encoding: str = "utf-8", errors: str = "strict") -> Optional[str]:
        v = self.value
        if v is None:
            return None
        if isinstance(v, str):
            return v
        if isinstance(v, bytes):
            return v.decode(encoding, errors=errors)
        try:
            return json.dumps(v)
        except Exception:
            return str(v)

    def value_bytes(self) -> Optional[bytes]:
        v = self.value
        if v is None:
            return None
        if isinstance(v, bytes):
            return v
        if isinstance(v, str):
            return v.encode("utf-8")
        try:
            return json.dumps(v).encode("utf-8")
        except Exception:
            return str(v).encode("utf-8")

    def update(
        self,
        value: Optional[Any] = None,
        *,
        full_key: Optional[str] = None,
        scope: Optional[str] = None,
        key: Optional[str] = None,
        signature: Optional[Signature] = None,
        secret_key: Optional[str] = None,
        bytes_value: Optional[str] = None,
        string_value: Optional[str] = None,
        create_scope_if_missing: bool = True,
        initial_manage_principal: Optional[str] = None,
        permissions: Optional[list[tuple[str, AclPermission]]] = None,
    ) -> "Secrets":
        effective_key = secret_key or self.secret_key
        bytes_value, string_value = self.coerce_to_put_payload(
            value,
            signature=signature,
            secret_key=effective_key,
            bytes_value=bytes_value,
            string_value=string_value,
        )

        if bytes_value is None and string_value is None:
            return self

        target_scope, target_key = self._resolve_scope_key(full_key=full_key, scope=scope, key=key)

        client = self.secrets_client()
        try:
            client.put_secret(
                scope=target_scope,
                key=target_key,
                bytes_value=bytes_value,
                string_value=string_value,
            )
        except Exception as e:
            if create_scope_if_missing and self._is_scope_not_found_error(e):
                self.create_scope(scope=target_scope, initial_manage_principal=initial_manage_principal)
                client.put_secret(
                    scope=target_scope,
                    key=target_key,
                    bytes_value=bytes_value,
                    string_value=string_value,
                )
            else:
                raise

        same_identity = (self.scope == target_scope) and (self.key == target_key)
        out = self if same_identity else Secrets(
            client=self.client,
            scope=target_scope,
            key=target_key,
            secret_key=effective_key,
        )

        # Update local cache via the same parse path so .value is consistent.
        # For encrypted writes we already have the plaintext in *value*; cache
        # it directly to avoid a redundant decrypt round-trip.
        if effective_key and value is not None:
            # Re-parse the original (unencrypted) value for the cache
            if bytes_value is not None:
                object.__setattr__(self, "_value", self._try_parse_value(bytes_value, secret_key=effective_key))
            else:
                object.__setattr__(self, "_value", value)
        elif bytes_value is not None:
            object.__setattr__(self, "_value", self._try_parse_value(bytes_value))
        elif string_value is not None:
            object.__setattr__(self, "_value", self._try_parse_value(string_value))

        if permissions:
            out.put_acl(permissions=permissions, scope=target_scope)

        return out

    def delete_secret(
        self,
        *,
        full_key: Optional[str] = None,
        scope: Optional[str] = None,
        key: Optional[str] = None,
    ) -> "Secrets":
        scope, key = self._resolve_scope_key(full_key=full_key, scope=scope, key=key)
        client = self.secrets_client()
        client.delete_secret(scope=scope, key=key)
        if self.scope == scope and self.key == key:
            object.__setattr__(self, "_value", None)
        return self

    # -------------------------
    # Metadata / listing
    # -------------------------
    def list_secrets(self, *, scope: Optional[str] = None) -> Iterator["Secrets"]:
        scope = scope or self.scope
        if not scope:
            raise ValueError("'scope' must be provided (or set on self).")

        client = self.secrets_client()

        return (
            Secrets(
                client=self.client,
                scope=scope,
                key=info.key,
                secret_key=self.secret_key,
                update_timestamp=float(info.last_updated_timestamp) / 1000.0,
            )
            for info in client.list_secrets(scope=scope)
        )

    def list_scopes(self) -> Iterator[Any]:
        client = self.secrets_client()
        return client.list_scopes()

    # -------------------------
    # Scope management
    # -------------------------
    def create_scope(
        self,
        *,
        scope: Optional[str] = None,
        initial_manage_principal: Optional[str] = None,
        scope_backend_type: Optional[Any] = None,
        backend_azure_keyvault: Optional[Any] = None,
    ) -> "Secrets":
        scope = scope or self.scope
        if not scope:
            raise ValueError("'scope' must be provided (or set on self).")
        client = self.secrets_client()
        client.create_scope(
            scope=scope,
            initial_manage_principal=initial_manage_principal,
            scope_backend_type=scope_backend_type,
            backend_azure_keyvault=backend_azure_keyvault,
        )
        object.__setattr__(self, "scope", scope)
        return self

    def delete_scope(self, *, scope: Optional[str] = None) -> "Secrets":
        scope = scope or self.scope
        if not scope:
            raise ValueError("'scope' must be provided (or set on self).")
        client = self.secrets_client()
        client.delete_scope(scope=scope)
        if self.scope == scope:
            object.__setattr__(self, "_value", None)
        return self

    # -------------------------
    # ACL management
    # -------------------------

    @staticmethod
    def check_permission(
        principal: str,
        permission: AclPermission = None,
    ):
        if permission is None:
            if principal == "users":
                return AclPermission.READ
            return AclPermission.MANAGE
        return permission

    def put_acl(
        self,
        permissions: Optional[list[tuple[str, AclPermission]]] = None,
        *,
        principal: Optional[str] = None,
        permission: Optional[AclPermission] = None,
        scope: Optional[str] = None,
        strict: bool = True,
    ) -> "Secrets":
        """
        Put one ACL (principal+permission) or many (acls=[(principal, permission), ...]).

        strict=True  -> fail fast on first bad entry
        strict=False -> skip invalid entries (still raises if nothing valid is left)
        """
        permissions = [self.check_permission(_) for _ in permissions]

        scope = scope or self.scope
        if not scope:
            raise ValueError("'scope' must be provided (or set on self).")

        batch: list[tuple[str, AclPermission]] = []
        if permissions:
            batch.extend(permissions)

        if principal is not None or permission is not None:
            if not principal:
                raise ValueError("Both 'principal' and 'permission' must be provided for single ACL.")
            if not permission:
                permission = self.check_permission(principal)
            batch.append((principal, permission))

        if not batch:
            return self

        client = self.secrets_client()

        valid = 0
        for p, perm in batch:
            if not p or perm is None:
                if strict:
                    raise ValueError(f"Invalid ACL entry: {(p, perm)}")
                continue
            client.put_acl(scope=scope, principal=p, permission=perm)
            valid += 1

        if valid == 0:
            raise ValueError("No valid ACL entries to apply.")

        return self

    def get_acl(
        self,
        *,
        principal: str,
        scope: Optional[str] = None,
    ) -> Any:
        scope = scope or self.scope
        if not scope:
            raise ValueError("'scope' must be provided (or set on self).")
        if not principal:
            raise ValueError("'principal' must be provided.")
        client = self.secrets_client()
        return client.get_acl(scope=scope, principal=principal)

    def list_acls(self, *, scope: Optional[str] = None) -> Iterator[Any]:
        scope = scope or self.scope
        if not scope:
            raise ValueError("'scope' must be provided (or set on self).")
        client = self.secrets_client()
        return client.list_acls(scope=scope)

    def delete_acl(
        self,
        *,
        principal: str,
        scope: Optional[str] = None,
    ) -> "Secrets":
        scope = scope or self.scope
        if not scope:
            raise ValueError("'scope' must be provided (or set on self).")
        if not principal:
            raise ValueError("'principal' must be provided.")
        client = self.secrets_client()
        client.delete_acl(scope=scope, principal=principal)
        return self


def _coerce_value(v: Any, *, as_type: type, encoding: str = "utf-8") -> Any:
    """Coerce *v* to *as_type*.

    Kept separate from :meth:`Secret.decode_value` so it can be unit-tested
    without a Databricks workspace.

    Raises
    ------
    TypeError
        When the coercion path does not exist or the result is the wrong type.
    json.JSONDecodeError
        When JSON parsing is required and the input is not valid JSON.
    """
    # ── str ────────────────────────────────────────────────────────────────
    if as_type is str:
        if isinstance(v, str):
            return v
        if isinstance(v, (bytes, bytearray, memoryview)):
            return bytes(v).decode(encoding)
        try:
            return json.dumps(v)
        except (TypeError, ValueError):
            return str(v)

    # ── bytes ──────────────────────────────────────────────────────────────
    if as_type is bytes:
        if isinstance(v, (bytes, bytearray, memoryview)):
            return bytes(v)
        if isinstance(v, str):
            return v.encode(encoding)
        try:
            return json.dumps(v).encode(encoding)
        except (TypeError, ValueError):
            return str(v).encode(encoding)

    # ── dict ───────────────────────────────────────────────────────────────
    if as_type is dict:
        if isinstance(v, dict):
            return v
        if isinstance(v, (str, bytes, bytearray)):
            s = v.decode(encoding) if isinstance(v, (bytes, bytearray)) else v
            result = json.loads(s)
            if not isinstance(result, dict):
                raise TypeError(f"JSON decoded to {type(result).__name__!r}, expected dict.")
            return result
        raise TypeError(f"Cannot coerce {type(v).__name__!r} to dict.")

    # ── list ───────────────────────────────────────────────────────────────
    if as_type is list:
        if isinstance(v, list):
            return v
        if isinstance(v, (str, bytes, bytearray)):
            s = v.decode(encoding) if isinstance(v, (bytes, bytearray)) else v
            result = json.loads(s)
            if not isinstance(result, list):
                raise TypeError(f"JSON decoded to {type(result).__name__!r}, expected list.")
            return result
        raise TypeError(f"Cannot coerce {type(v).__name__!r} to list.")

    # ── numeric ────────────────────────────────────────────────────────────
    if as_type in (int, float):
        if isinstance(v, as_type):
            return v
        try:
            return as_type(v)
        except (ValueError, TypeError) as exc:
            raise TypeError(f"Cannot coerce {type(v).__name__!r} to {as_type.__name__}.") from exc

    # ── fallback: attempt direct constructor ───────────────────────────────
    try:
        return as_type(v)
    except Exception as exc:
        raise TypeError(
            f"No coercion path from {type(v).__name__!r} to {as_type.__name__!r}."
        ) from exc