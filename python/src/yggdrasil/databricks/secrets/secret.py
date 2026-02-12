from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field, is_dataclass, fields
from typing import Optional, Tuple, Any, Iterator, Union

from databricks.sdk.service.workspace import AclPermission

from ..workspaces import WorkspaceService

__all__ = ["Secret"]

BytesLike = Union[bytes, bytearray, memoryview]


@dataclass
class Secret(WorkspaceService):
    scope: Optional[str] = None
    key: Optional[str] = None
    update_timestamp: Optional[float] = None

    _value: Optional[Any] = field(default=None, repr=False)

    # -------------------------
    # Ergonomics
    # -------------------------
    def __getitem__(self, item):
        return self.find_secret(full_key=item)

    def __setitem__(self, key, value):
        return self.update(value, full_key=key)

    def secrets_client(self):
        return self.workspace.sdk().secrets

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
        bytes_value: Optional[str],
        string_value: Optional[str],
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Databricks put_secret supports either:
          - string_value: str
          - bytes_value: base64-encoded string

        Enhancements:
          - if `value` is a dataclass, serialize via asdict() then json.dumps()
        """
        if bytes_value is not None or string_value is not None:
            return bytes_value, string_value

        if value is None:
            return None, None

        # dataclass -> JSON
        if is_dataclass(value):
            metadata: dict[str, Any] = {}
            for f in fields(value):
                if f.name.startswith("_"):
                    continue
                metadata[f.name] = getattr(value, f.name)

            return None, json.dumps(metadata)

        # Already a string: use string_value
        if isinstance(value, str):
            return None, value

        # Raw bytes: encode as base64 string
        if isinstance(value, (bytes, bytearray, memoryview)):
            b = bytes(value)
            return base64.b64encode(b).decode("utf-8"), None

        # Anything else: try JSON, fall back to str()
        return None, json.dumps(value)

    # -------------------------
    # Value parsing (read)
    # -------------------------
    @staticmethod
    def _looks_like_json(s: str) -> bool:
        s = s.strip()
        return (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]"))

    @staticmethod
    def _try_parse_value(v: Any) -> Any:
        """
        Best-effort parsing when reading.
        - If it's JSON (object/array), parse into Python types.
        - If it's base64 that decodes cleanly into UTF-8 JSON/object/array, parse that too.
        - Otherwise return as-is.
        """
        if v is None:
            return None

        # bytes -> maybe utf-8 -> maybe json
        if isinstance(v, (bytes, bytearray, memoryview)):
            b = bytes(v)
            try:
                s = b.decode("utf-8")
            except Exception:
                return b
            # recurse through string path
            return Secret._try_parse_value(s)

        if not isinstance(v, str):
            return v

        s = v.strip()
        if not s:
            return v

        # direct JSON
        if Secret._looks_like_json(s):
            try:
                return json.loads(s)
            except Exception:
                return v

        # maybe base64 containing utf-8 (possibly json)
        # IMPORTANT: don't aggressively "decode everything"; only accept if decode is clean and "printable-ish".
        try:
            decoded = base64.b64decode(s, validate=True)
        except Exception:
            return v

        # if it decodes, see if it's utf-8; if not, return raw bytes
        try:
            decoded_text = decoded.decode("utf-8")
        except Exception:
            return decoded

        dt = decoded_text.strip()
        if Secret._looks_like_json(dt):
            try:
                return json.loads(dt)
            except Exception:
                return decoded_text

        # not json; keep as text (not bytes) because caller probably wants a string
        return decoded_text

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
    ) -> "Secret":
        scope, key = self._resolve_scope_key(full_key=full_key, scope=scope, key=key)
        client = self.secrets_client()

        response = client.get_secret(scope=scope, key=key)

        raw = response.value
        parsed = self._try_parse_value(raw)

        return Secret(
            workspace=self.workspace,
            scope=scope,
            key=response.key,
            _value=parsed,
        )

    @property
    def value(self) -> Optional[Any]:
        """
        Lazy read + best-effort parse.
        If the secret looks like JSON, returns Python objects (dict/list/etc).
        If it looks like base64-encoded UTF-8, returns decoded string (or parsed JSON).
        Otherwise returns original string/bytes.
        """
        if self._value is None and self.scope and self.key:
            # fetch raw value, parse, cache
            fetched = self.find_secret(scope=self.scope, key=self.key)
            self._value = fetched._value
        return self._value

    def value_text(self, encoding: str = "utf-8", errors: str = "strict") -> Optional[str]:
        v = self.value
        if v is None:
            return None
        if isinstance(v, str):
            return v
        if isinstance(v, bytes):
            return v.decode(encoding, errors=errors)
        # for dict/list/etc
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
        bytes_value: Optional[str] = None,
        string_value: Optional[str] = None,
        create_scope_if_missing: bool = True,
        initial_manage_principal: Optional[str] = None,
        acls: Optional[list[tuple[str, AclPermission]] | bool] = True,
    ) -> "Secret":
        bytes_value, string_value = self.coerce_to_put_payload(
            value, bytes_value=bytes_value, string_value=string_value
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

        # decide whether to mutate self or return a new instance
        same_identity = (self.scope == target_scope) and (self.key == target_key)
        out = self if same_identity else Secret(workspace=self.workspace, scope=target_scope, key=target_key)

        # update local cache in a parsed form (matches .value behavior)
        if string_value is not None:
            out._value = out._try_parse_value(string_value)
        elif bytes_value is not None:
            try:
                decoded = base64.b64decode(bytes_value)
            except Exception:
                decoded = bytes_value
            out._value = out._try_parse_value(decoded)

        # apply ACLs (if requested) against the *target scope*
        if acls:
            if acls is True:
                # "True" means: apply whatever your put_acl() defaults to (usually self principal/permission)
                # If your put_acl requires explicit tuples, keep this as no-op or define sensible defaults.
                out.put_acl(scope=target_scope)
            else:
                out.put_acl(acls=acls, scope=target_scope)

        return out

    def delete_secret(
        self,
        *,
        full_key: Optional[str] = None,
        scope: Optional[str] = None,
        key: Optional[str] = None,
    ) -> "Secret":
        scope, key = self._resolve_scope_key(full_key=full_key, scope=scope, key=key)
        client = self.secrets_client()
        client.delete_secret(scope=scope, key=key)
        if self.scope == scope and self.key == key:
            self._value = None
        return self

    # -------------------------
    # Metadata / listing
    # -------------------------
    def list_secrets(self, *, scope: Optional[str] = None) -> Iterator["Secret"]:
        scope = scope or self.scope
        if not scope:
            raise ValueError("'scope' must be provided (or set on self).")

        client = self.secrets_client()

        return (
            Secret(
                workspace=self.workspace,
                scope=scope,
                key=info.key,
                update_timestamp=float(info.last_updated_timestamp) / 1000.0
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
    ) -> "Secret":
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
        self.scope = scope
        return self

    def delete_scope(self, *, scope: Optional[str] = None) -> "Secret":
        scope = scope or self.scope
        if not scope:
            raise ValueError("'scope' must be provided (or set on self).")
        client = self.secrets_client()
        client.delete_scope(scope=scope)
        if self.scope == scope:
            self.key = None
            self._value = None
        return self

    # -------------------------
    # ACL management
    # -------------------------

    def put_acl(
        self,
        acls: Optional[list[tuple[str, AclPermission]] | bool] = True,
        *,
        principal: Optional[str] = None,
        permission: Optional[AclPermission] = None,
        scope: Optional[str] = None,
        strict: bool = True,
    ) -> "Secret":
        """
        Put one ACL (principal+permission) or many (acls=[(principal, permission), ...]).

        strict=True  -> fail fast on first bad entry
        strict=False -> skip invalid entries (still raises if nothing valid is left)
        """
        if isinstance(acls, bool):
            current_groups = self.current_user_groups(
                with_public=False,
                raise_error=False
            )

            acls = [
                (group.display, AclPermission.MANAGE)
                for group in current_groups
                if group.display not in {"users"}
            ]

        scope = scope or self.scope
        if not scope:
            raise ValueError("'scope' must be provided (or set on self).")

        # Build batch from either (principal, permission) or `acls`
        batch: list[tuple[str, AclPermission]] = []
        if acls:
            batch.extend(acls)

        if principal is not None or permission is not None:
            if not principal or permission is None:
                raise ValueError("Both 'principal' and 'permission' must be provided for single ACL.")
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
    ) -> "Secret":
        scope = scope or self.scope
        if not scope:
            raise ValueError("'scope' must be provided (or set on self).")
        if not principal:
            raise ValueError("'principal' must be provided.")
        client = self.secrets_client()
        client.delete_acl(scope=scope, principal=principal)
        return self
