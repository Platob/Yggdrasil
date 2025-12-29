import base64
import subprocess
import sys

import dill

from yggdrasil.pyutils.callable_serde import CallableSerde

# ---- helpers for tests ----

GLOBAL_A = 111
GLOBAL_B = 222


def uses_global_a(x: int) -> int:
    return x + GLOBAL_A


def returns_big_bytes(n: int) -> bytes:
    # very compressible
    return b"a" * n


def make_closure(adder: int):
    z = adder

    def inner(x: int) -> int:
        return x + z

    return inner


# ---- tests ----

def test_from_callable_returns_self():
    s = CallableSerde.from_callable(uses_global_a)
    s2 = CallableSerde.from_callable(s)
    assert s2 is s


def test_callable_instance_invokes():
    s = CallableSerde.from_callable(lambda x: x * 3)
    assert s(7) == 21


def test_dump_env_filters_used_globals_only():
    s = CallableSerde.from_callable(uses_global_a)
    d = s.dump(prefer="dill", dump_env="globals", filter_used_globals=True)

    assert "env_b64" in d
    env = dill.loads(base64.b64decode(d["env_b64"]))
    # should include GLOBAL_A but not GLOBAL_B
    assert "globals" in env
    assert "GLOBAL_A" in env["globals"]
    assert "GLOBAL_B" not in env["globals"]


def test_dump_env_closure_capture():
    fn = make_closure(9)
    s = CallableSerde.from_callable(fn)
    d = s.dump(prefer="dill", dump_env="closure", filter_used_globals=True)

    env = dill.loads(base64.b64decode(d["env_b64"]))
    assert "closure" in env
    # closure variable name is "z" in make_closure
    assert env["closure"]["z"] == 9


def test_to_command_roundtrip_result():
    s = CallableSerde.from_callable(lambda a, b: a + b)
    code = s.to_command(args=(2, 5), result_tag="RET", prefer="dill")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    out = (p.stdout or "") + "\n" + (p.stderr or "")
    assert p.returncode == 0, out
    res = CallableSerde.parse_command_result(out, result_tag="RET")
    assert res == 7


def test_to_command_compression_roundtrip():
    s = CallableSerde.from_callable(returns_big_bytes)
    # Force compression by setting a tiny byte_limit
    code = s.to_command(args=(50_000,), result_tag="RET", prefer="dill", byte_limit=1_000)
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    out = (p.stdout or "") + "\n" + (p.stderr or "")
    assert p.returncode == 0, out
    res = CallableSerde.parse_command_result(out, result_tag="RET")
    assert isinstance(res, (bytes, bytearray))
    assert res == b"a" * 50_000
