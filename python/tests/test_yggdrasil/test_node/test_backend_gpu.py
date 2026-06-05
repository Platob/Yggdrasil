"""Tests for GPU resource-metric collection (nvidia-smi parsing).

The node exposes per-GPU utilization, VRAM, temperature, and power. The
parser must read the extra power fields and tolerate ``[N/A]`` (which
some cards report for power) without dropping the whole GPU row.
"""
from __future__ import annotations

import subprocess
import unittest
from types import SimpleNamespace
from unittest import mock

from yggdrasil.node.api.services.backend import BackendService, _gpu_num


class TestGpuNumParsing(unittest.TestCase):
    def test_parses_numbers(self):
        self.assertEqual(_gpu_num("72.5"), 72.5)
        self.assertEqual(_gpu_num("0"), 0.0)

    def test_tolerates_na(self):
        self.assertEqual(_gpu_num("[N/A]"), 0.0)
        self.assertEqual(_gpu_num("[Not Supported]"), 0.0)


class TestCollectGpus(unittest.TestCase):
    def _run_with_smi_output(self, stdout: str):
        fake = SimpleNamespace(returncode=0, stdout=stdout)
        with mock.patch.object(subprocess, "run", return_value=fake):
            return BackendService._collect_gpus()

    def test_parses_power_fields(self):
        # index,name,mem.used,mem.total,util,temp,power.draw,power.limit
        gpus = self._run_with_smi_output(
            "0, NVIDIA A100, 4096, 40960, 65, 58, 210.5, 400\n"
        )
        self.assertEqual(len(gpus), 1)
        g = gpus[0]
        self.assertEqual(g.name, "NVIDIA A100")
        self.assertEqual(g.utilization_percent, 65.0)
        self.assertEqual(g.power_draw_w, 210.5)
        self.assertEqual(g.power_limit_w, 400.0)

    def test_power_na_does_not_drop_gpu(self):
        gpus = self._run_with_smi_output(
            "0, GeForce GTX 1080, 512, 8192, 12, 45, [N/A], [N/A]\n"
        )
        self.assertEqual(len(gpus), 1)
        self.assertEqual(gpus[0].power_draw_w, 0.0)
        self.assertEqual(gpus[0].power_limit_w, 0.0)
        self.assertEqual(gpus[0].memory_total_mb, 8192.0)

    def test_missing_power_columns_default_zero(self):
        # Older 6-field output (no power) still parses.
        gpus = self._run_with_smi_output("0, Tesla T4, 1000, 16000, 30, 50\n")
        self.assertEqual(len(gpus), 1)
        self.assertEqual(gpus[0].power_draw_w, 0.0)

    def test_no_nvidia_smi_returns_empty(self):
        with mock.patch.object(subprocess, "run", side_effect=FileNotFoundError):
            self.assertEqual(BackendService._collect_gpus(), [])


if __name__ == "__main__":
    unittest.main()
