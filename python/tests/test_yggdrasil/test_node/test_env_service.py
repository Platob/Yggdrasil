"""Tests for the node env service — system-variable exclusion."""
from __future__ import annotations

import asyncio
import os
import unittest

from yggdrasil.node.config import get_settings
from yggdrasil.node.services.env import EnvService, _is_system_env


class TestEnvSystemFilter(unittest.TestCase):
    def test_is_system_env_names_and_prefixes(self):
        for name in ("PATH", "HOME", "path", "SystemRoot", "PYTHONPATH"):
            self.assertTrue(_is_system_env(name), name)
        for name in ("LC_ALL", "XDG_RUNTIME_DIR", "PROCESSOR_ARCHITECTURE", "PROGRAMFILES(X86)"):
            self.assertTrue(_is_system_env(name), name)
        for name in ("MY_TOKEN", "API_KEY", "REGION", "DATABASE_URL"):
            self.assertFalse(_is_system_env(name), name)

    def test_full_listing_drops_system_vars(self):
        os.environ["YGG_TEST_APP_VAR"] = "keep-me"
        try:
            r = asyncio.run(EnvService(get_settings()).get_env())
            self.assertNotIn("PATH", r.variables)
            self.assertNotIn("HOME", r.variables)
            self.assertIn("YGG_TEST_APP_VAR", r.variables)
        finally:
            os.environ.pop("YGG_TEST_APP_VAR", None)

    def test_include_system_keeps_them(self):
        r = asyncio.run(EnvService(get_settings()).get_env(include_system=True))
        self.assertIn("PATH", r.variables)

    def test_explicit_keys_returned_verbatim(self):
        # Asking for a system var by name returns it even though the full
        # listing would hide it.
        r = asyncio.run(EnvService(get_settings()).get_env(keys=["PATH"]))
        self.assertIn("PATH", r.variables)


if __name__ == "__main__":
    unittest.main()
