"""Static asset ids + @function code inference (signature, dtypes, deps)."""
from __future__ import annotations

import unittest

from yggdrasil.node.api.services.pyfunc import _ann_to_dtype, infer_function
from yggdrasil.node.ids import make_static_id


class TestStaticId(unittest.TestCase):
    def test_deterministic_and_js_safe(self):
        a = make_static_id("main.market.trades")
        b = make_static_id("main.market.trades")
        self.assertEqual(a, b)                       # static across calls
        self.assertNotEqual(a, make_static_id("main.market.other"))
        self.assertLess(a, 2 ** 53)                  # round-trips through JSON


class TestAnnToDtype(unittest.TestCase):
    def test_mapping(self):
        self.assertEqual(_ann_to_dtype("int"), "int64")
        self.assertEqual(_ann_to_dtype("float"), "float64")
        self.assertEqual(_ann_to_dtype("str"), "string")
        self.assertEqual(_ann_to_dtype("list[int]"), "list")
        self.assertEqual(_ann_to_dtype("dict[str, int]"), "struct")
        self.assertEqual(_ann_to_dtype("pd.DataFrame"), "")   # unknown → empty


class TestInferFunction(unittest.TestCase):
    CODE = (
        "import pandas as pd\n"
        "import requests\n"
        "def train(data: list, threshold: float = 0.5, name: str = 'x') -> dict:\n"
        "    '''Fit a model.'''\n"
        "    return {'n': len(data)}\n"
    )

    def test_signature_params_and_return(self):
        r = infer_function(self.CODE, None, pin_versions=False, default_py="3.11")
        self.assertEqual(r.name, "train")
        self.assertEqual(r.signature, "train(data: list, threshold: float = 0.5, name: str = 'x') -> dict")
        self.assertEqual([p.name for p in r.params], ["data", "threshold", "name"])
        self.assertEqual([p.dtype for p in r.params], ["list", "float64", "string"])
        self.assertTrue(r.params[1].has_default)
        self.assertFalse(r.params[0].has_default)
        self.assertEqual(r.return_dtype, "struct")
        self.assertEqual(r.docstring, "Fit a model.")

    def test_dependencies_excludes_stdlib_and_pins(self):
        r = infer_function(self.CODE, None, pin_versions=True, default_py="3.11")
        names = {d.split(">=")[0] for d in r.dependencies}
        self.assertIn("pandas", names)
        self.assertIn("requests", names)
        self.assertNotIn("os", names)                # stdlib excluded
        # installed deps are version-pinned
        self.assertTrue(any(d.startswith("pandas>=") for d in r.dependencies))

    def test_pick_named_function(self):
        code = "def a(x: int): ...\ndef b(y: str) -> bool: ...\n"
        r = infer_function(code, "b", pin_versions=False, default_py="3.12")
        self.assertEqual(r.name, "b")
        self.assertEqual(r.params[0].dtype, "string")
        self.assertEqual(r.return_dtype, "bool")
        self.assertEqual(r.python_version, "3.12")

    def test_bad_code_raises(self):
        from yggdrasil.exceptions.api import BadRequestError
        with self.assertRaises(BadRequestError):
            infer_function("def (:", None, pin_versions=False, default_py="3.11")


if __name__ == "__main__":
    unittest.main()
