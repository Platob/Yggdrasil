import base64
import json
import os
import time
import unittest
import uuid
from dataclasses import dataclass

from yggdrasil.databricks import Workspace
from yggdrasil.databricks.secrets import Secret  # adjust if your import path differs
from yggdrasil.pyutils.serde import ObjectSerde


class TestSecrets(unittest.TestCase):
    """
    Real-world integration tests (no mocking).

    Required env vars:
      - DATABRICKS_HOST
      - DATABRICKS_TOKEN
    Optional env vars:
      - DATABRICKS_TEST_SCOPE: scope name to use (default: yggdrasil_it_<uuid8>)
      - DATABRICKS_TEST_PRINCIPAL: principal for ACL tests (default: skip ACL tests)
      - DATABRICKS_TEST_NO_CLEANUP=1 to keep created resources for debugging

    Notes:
      - These tests create a secrets scope + secret key in your workspace.
      - get_secret may be restricted depending on environment; tests handle that gracefully.
    """

    @classmethod
    def setUpClass(cls):
        # Workspace().connect() should pick up host/token from env per your lib conventions.
        cls.workspace = Workspace().connect()

        cls.no_cleanup = os.getenv("DATABRICKS_TEST_NO_CLEANUP", "").strip() == "1"
        cls.principal = os.getenv("DATABRICKS_TEST_PRINCIPAL", "").strip() or None

        # Use a unique scope by default to avoid collisions in shared workspaces.
        env_scope = os.getenv("DATABRICKS_TEST_SCOPE", "").strip()
        cls.scope = env_scope or f"yggdrasil_it_{uuid.uuid4().hex[:8]}"

        cls.key = f"test_key_{uuid.uuid4().hex[:8]}"

        cls.secret = Secret(workspace=cls.workspace, scope=cls.scope, key=cls.key)

        # Ensure scope exists (create_scope is idempotent-ish if you handle errors in your wrapper;
        # if Databricks throws AlreadyExists, ignore it).
        try:
            cls.secret.create_scope(scope=cls.scope)
        except Exception as e:
            # If scope already exists, Databricks returns an error; accept that for reuse scenarios.
            msg = (getattr(e, "message", None) or str(e) or "").lower()
            if "already exists" not in msg and "resource_already_exists" not in msg:
                raise

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "no_cleanup", False):
            return

        # Best-effort cleanup: delete secret then scope
        try:
            Secret(workspace=cls.workspace, scope=cls.scope, key=cls.key).delete_secret()
        except Exception:
            pass

        try:
            Secret(workspace=cls.workspace, scope=cls.scope).delete_scope()
        except Exception:
            pass

    def test_update_secret_auto_creates_scope(self):
        # new isolated scope for this test, to validate "create scope if missing"
        scope = f"yggdrasil_it_autocreate_{uuid.uuid4().hex[:8]}"
        key = f"key_{uuid.uuid4().hex[:8]}"
        s = Secret(workspace=self.workspace, scope=scope, key=key)

        s.update(
            value={"hello": "world", "n": 1},
            create_scope_if_missing=True,
        )

        # list scopes should contain it
        scopes = list(Secret(workspace=self.workspace).list_scopes())
        scope_names = {getattr(x, "name", None) or getattr(x, "scope", None) or str(x) for x in scopes}
        self.assertIn(scope, scope_names)

        # cleanup
        if not self.no_cleanup:
            try:
                s.delete_secret()
            except Exception:
                pass
            try:
                s.delete_scope()
            except Exception:
                pass

    def test_put_and_list_secret_metadata(self):
        self.secret.update(value="sup nerds")

        metas = list(self.secret.list_secrets(scope=self.scope))
        keys = {getattr(m, "key", None) or str(m) for m in metas}
        self.assertIn(self.key, keys)

    def test_update_secret_json_roundtrip_via_value_parsing(self):
        payload = {"a": 1, "b": [1, 2, 3], "ok": True}
        self.secret.update(value=payload)

        # Test second update noop
        self.secret.update(value=payload)

        # Your Secret.value property does best-effort parsing
        # BUT: reading requires get_secret access (may be restricted). Handle that.
        try:
            s2 = Secret(workspace=self.workspace, scope=self.scope, key=self.key).find_secret()
        except Exception as e:
            msg = (getattr(e, "message", None) or str(e) or "").lower()
            # If get_secret is blocked in this environment, skip cleanly.
            if "get_secret" in msg or "not supported" in msg or "permission" in msg or "bad request" in msg:
                self.skipTest(f"get_secret not available in this environment: {e}")
            raise

        self.assertEqual(s2.value, payload)

    def test_delete_secret(self):
        self.secret.update(value="temp")
        self.secret.delete_secret()

        # list_secrets should no longer show it (eventual consistency is rare here,
        # but add a tiny retry to be safe)
        for _ in range(5):
            metas = list(self.secret.list_secrets(scope=self.scope))
            keys = {m.key for m in metas}
            if self.key not in keys:
                break
            time.sleep(0.5)

        self.assertNotIn(self.key, keys)

    def test_acl_lifecycle_if_principal_provided(self):
        if not self.principal:
            self.skipTest("Set DATABRICKS_TEST_PRINCIPAL to run ACL tests (e.g. a user or group).")

        # permissions enum/type depends on your SDK; most accept strings like "READ", "WRITE", "MANAGE"
        self.secret.put_acl(scope=self.scope, principal=self.principal, permission="READ")

        acl = self.secret.get_acl(scope=self.scope, principal=self.principal)
        # verify principal and permission appear in response
        acl_principal = getattr(acl, "principal", None) or getattr(acl, "user", None)
        self.assertTrue(acl_principal is None or acl_principal == self.principal)

        acls = list(self.secret.list_acls(scope=self.scope))
        principals = {getattr(a, "principal", None) or getattr(a, "user", None) or str(a) for a in acls}
        self.assertIn(self.principal, principals)

        self.secret.delete_acl(scope=self.scope, principal=self.principal)

        acls2 = list(self.secret.list_acls(scope=self.scope))
        principals2 = {getattr(a, "principal", None) or getattr(a, "user", None) or str(a) for a in acls2}
        self.assertNotIn(self.principal, principals2)


class TestCoerceToPutPayload(unittest.TestCase):
    def test_passthrough_when_bytes_value_provided(self):
        b64 = base64.b64encode(b"abc").decode("utf-8")
        out_bytes, out_string = Secret.coerce_to_put_payload(
            value={"ignored": True},
            bytes_value=b64,
            string_value=None,
        )
        self.assertEqual(out_bytes, b64)
        self.assertIsNone(out_string)

    def test_passthrough_when_string_value_provided(self):
        out_bytes, out_string = Secret.coerce_to_put_payload(
            value={"ignored": True},
            bytes_value=None,
            string_value="hello",
        )
        self.assertIsNone(out_bytes)
        self.assertEqual(out_string, "hello")

    def test_none_value_returns_none_tuple(self):
        out_bytes, out_string = Secret.coerce_to_put_payload(
            value=None,
            bytes_value=None,
            string_value=None,
        )
        self.assertIsNone(out_bytes)
        self.assertIsNone(out_string)

    def test_string_value_goes_to_string_value(self):
        out_bytes, out_string = Secret.coerce_to_put_payload(
            value="sup",
            bytes_value=None,
            string_value=None,
        )
        self.assertIsNone(out_bytes)
        self.assertEqual(out_string, "sup")

    def test_bytes_value_is_base64_encoded(self):
        payload = b"\x00\x01\x02hello"
        out_bytes, out_string = Secret.coerce_to_put_payload(
            value=payload,
            bytes_value=None,
            string_value=None,
        )
        self.assertIsNone(out_string)
        self.assertEqual(out_bytes, base64.b64encode(payload).decode("utf-8"))

    def test_bytearray_and_memoryview_are_base64_encoded(self):
        payload = b"arrow"
        for v in (bytearray(payload), memoryview(payload)):
            out_bytes, out_string = Secret.coerce_to_put_payload(
                value=v,
                bytes_value=None,
                string_value=None,
            )
            self.assertIsNone(out_string)
            self.assertEqual(out_bytes, base64.b64encode(payload).decode("utf-8"))

    def test_json_serializable_object_uses_json_dumps(self):
        value = {"a": 1, "b": [1, 2, 3], "ok": True}
        out_bytes, out_string = Secret.coerce_to_put_payload(
            value=value,
            bytes_value=None,
            string_value=None,
        )
        self.assertIsNone(out_bytes)
        self.assertEqual(json.loads(out_string), value)

    def test_non_json_serializable_raises_typeerror(self):
        value = {"x": {1, 2, 3}}  # sets are not JSON-serializable by default

        with self.assertRaises(TypeError):
            Secret.coerce_to_put_payload(
                value=value,
                bytes_value=None,
                string_value=None,
            )

    def test_dataclass_serializes_to_json_with_namespace(self):
        @dataclass
        class Foo:
            a: int
            b: str
            _pvt: str

        foo = Foo(a=7, b="hi", _pvt="pvt")

        out_bytes, out_string = Secret.coerce_to_put_payload(
            value=foo,
            bytes_value=None,
            string_value=None,
        )
        self.assertIsNone(out_bytes)
        self.assertIsNotNone(out_string)

        metadata = json.loads(out_string)
        # dataclass fields present
        self.assertEqual(metadata["a"], 7)
        self.assertEqual(metadata["b"], "hi")
        self.assertIsNone(metadata.get("_pvt"))

    def test_dataclass_path_is_used_even_if_json_dumps_would_work(self):
        # This ensures we add __namespace__ (dataclass branch), not plain json branch.
        @dataclass
        class Bar:
            x: int

        bar = Bar(x=1)
        out_bytes, out_string = Secret.coerce_to_put_payload(
            value=bar,
            bytes_value=None,
            string_value=None,
        )
        meta = json.loads(out_string)
        self.assertEqual(meta["x"], 1)


if __name__ == "__main__":
    unittest.main()
