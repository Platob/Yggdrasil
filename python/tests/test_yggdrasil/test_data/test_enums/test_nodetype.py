"""Tests for the :class:`NodeType` enum and :func:`NodeType.from_cpu_and_ram`."""
from __future__ import annotations

import unittest

from yggdrasil.enums import NodeSpec, NodeType


class TestNodeTypeCoercion(unittest.TestCase):
    """``from_`` / ``to_id`` coercion paths."""

    def test_default_member(self):
        self.assertIs(NodeType.DEFAULT, NodeType.FLEET_XLARGE)
        self.assertEqual(NodeType.DEFAULT.value, "rd-fleet.xlarge")

    def test_semantic_aliases_collapse(self):
        self.assertIs(NodeType.SMALL, NodeType.FLEET_XLARGE)
        self.assertIs(NodeType.MEDIUM, NodeType.FLEET_2XLARGE)
        self.assertIs(NodeType.LARGE, NodeType.FLEET_4XLARGE)
        self.assertIs(NodeType.XLARGE, NodeType.FLEET_8XLARGE)

    def test_from_known_strings(self):
        self.assertIs(NodeType.from_("default"), NodeType.FLEET_XLARGE)
        self.assertIs(NodeType.from_("medium"), NodeType.FLEET_2XLARGE)
        self.assertIs(NodeType.from_("rd-fleet.2xlarge"), NodeType.FLEET_2XLARGE)
        self.assertIs(NodeType.from_("M5_LARGE"), NodeType.M5_LARGE)
        # Azure SKU keeps its mixed case.
        self.assertIs(NodeType.from_("Standard_D8ds_v5"), NodeType.AZURE_D8DS_V5)

    def test_from_none_returns_default(self):
        self.assertIs(NodeType.from_(None), NodeType.DEFAULT)

    def test_from_unknown_raises(self):
        with self.assertRaises(ValueError):
            NodeType.from_("ZZZ-not-a-sku")

    def test_to_id_passthrough_unknown(self):
        self.assertEqual(NodeType.to_id("r5d.metal"), "r5d.metal")

    def test_to_id_resolves_aliases(self):
        self.assertEqual(NodeType.to_id("medium"), "rd-fleet.2xlarge")
        self.assertEqual(NodeType.to_id("FLEET_4XLARGE"), "rd-fleet.4xlarge")

    def test_to_id_handles_none(self):
        self.assertEqual(NodeType.to_id(None), NodeType.DEFAULT.value)
        self.assertEqual(NodeType.to_id(None, default=NodeType.LARGE), "rd-fleet.4xlarge")

    def test_to_id_member_round_trip(self):
        self.assertEqual(NodeType.to_id(NodeType.M5_4XLARGE), "m5.4xlarge")

    def test_str_equivalence(self):
        # Members subclass str so they slot into the SDK directly.
        self.assertEqual(NodeType.FLEET_XLARGE, "rd-fleet.xlarge")

    def test_is_known(self):
        self.assertTrue(NodeType.is_known("default"))
        self.assertTrue(NodeType.is_known("Standard_D8ds_v5"))
        self.assertFalse(NodeType.is_known("nope"))


class TestNodeSpec(unittest.TestCase):
    """Per-member hardware specs."""

    def test_fleet_xlarge_specs(self):
        spec = NodeType.FLEET_XLARGE.spec
        self.assertIsInstance(spec, NodeSpec)
        self.assertEqual(spec.cpu_cores, 4)
        self.assertEqual(spec.ram_gib, 16)
        self.assertEqual(spec.gpu_count, 0)
        self.assertEqual(spec.cloud, "fleet")

    def test_r5_xlarge_is_memory_optimized(self):
        # r5.xlarge: 4 vCPU, 32 GiB (vs m5.xlarge's 16 GiB)
        self.assertEqual(NodeType.R5_XLARGE.cpu_cores, 4)
        self.assertEqual(NodeType.R5_XLARGE.ram_gib, 32)
        self.assertEqual(NodeType.R5_XLARGE.cloud, "aws")

    def test_azure_d8_has_local_disk(self):
        self.assertGreater(NodeType.AZURE_D8DS_V5.local_disk_gib, 0)
        self.assertEqual(NodeType.AZURE_D8DS_V5.cloud, "azure")

    def test_gcp_n2_standard_specs(self):
        self.assertEqual(NodeType.GCP_N2_STD_4.cpu_cores, 4)
        self.assertEqual(NodeType.GCP_N2_STD_4.ram_gib, 16)
        self.assertEqual(NodeType.GCP_N2_STD_4.cloud, "gcp")


class TestNodeTypeFromCpuAndRam(unittest.TestCase):
    """Best-fit lookup by hardware requirements."""

    def test_exact_match(self):
        node = NodeType.from_cpu_and_ram(cpu_cores=8, ram_gib=32)
        self.assertIs(node, NodeType.FLEET_2XLARGE)

    def test_picks_smallest_that_fits(self):
        node = NodeType.from_cpu_and_ram(cpu_cores=2, ram_gib=4)
        # M5_LARGE: 2 vCPU, 8 GiB — fits and is the smallest
        # but Fleet has nothing at that size and FLEET_XLARGE is 4 vCPU.
        # The lookup prefers fleet, then sorts by cpu/ram ascending.
        # FLEET_XLARGE (4 cpu, 16 ram) wins because Fleet is preferred over AWS.
        self.assertIs(node, NodeType.FLEET_XLARGE)

    def test_memory_bound_picks_r5(self):
        # 4 vCPU + 32 GiB — Fleet has nothing matching exactly. The
        # cheapest fit across all clouds is R5_XLARGE (4 vCPU / 32 GiB)
        # but Fleet is preferred so FLEET_2XLARGE (8/32) wins.
        node = NodeType.from_cpu_and_ram(cpu_cores=4, ram_gib=32)
        self.assertIs(node, NodeType.FLEET_2XLARGE)

    def test_memory_bound_with_no_cloud_preference(self):
        # Without a cloud preference, the smallest by (cpu, ram) wins.
        node = NodeType.from_cpu_and_ram(cpu_cores=4, ram_gib=32, prefer=None)
        self.assertIs(node, NodeType.R5_XLARGE)

    def test_prefer_azure(self):
        node = NodeType.from_cpu_and_ram(cpu_cores=4, ram_gib=16, prefer="azure")
        self.assertIs(node, NodeType.AZURE_D4DS_V5)

    def test_prefer_gcp(self):
        node = NodeType.from_cpu_and_ram(cpu_cores=8, ram_gib=32, prefer="gcp")
        self.assertIs(node, NodeType.GCP_N2_STD_8)

    def test_prefer_aliases(self):
        # "databricks" → "fleet", "amazon" → "aws", "google" → "gcp"
        node = NodeType.from_cpu_and_ram(cpu_cores=4, prefer="databricks")
        self.assertEqual(node.cloud, "fleet")

        node = NodeType.from_cpu_and_ram(cpu_cores=4, prefer="google")
        self.assertEqual(node.cloud, "gcp")

    def test_local_disk_requirement(self):
        node = NodeType.from_cpu_and_ram(
            cpu_cores=4, ram_gib=16, local_disk_gib=100,
        )
        # Only Azure SKUs carry local disk in our spec table.
        self.assertIs(node, NodeType.AZURE_D4DS_V5)

    def test_no_fit_raises(self):
        with self.assertRaises(ValueError):
            NodeType.from_cpu_and_ram(cpu_cores=128, ram_gib=512)

    def test_gpu_required_raises(self):
        # No GPU members in the spec table yet.
        with self.assertRaises(ValueError):
            NodeType.from_cpu_and_ram(cpu_cores=4, gpu=True)

    def test_bad_cpu_raises(self):
        with self.assertRaises(ValueError):
            NodeType.from_cpu_and_ram(cpu_cores=0)

    def test_unknown_prefer_raises(self):
        with self.assertRaises(ValueError):
            NodeType.from_cpu_and_ram(cpu_cores=4, prefer="oracle")

    def test_candidates_restriction(self):
        # When the candidate list is empty, no fit available.
        with self.assertRaises(ValueError):
            NodeType.from_cpu_and_ram(cpu_cores=4, candidates=[])

    def test_candidates_filters_search(self):
        node = NodeType.from_cpu_and_ram(
            cpu_cores=4,
            candidates=[NodeType.M5_XLARGE, NodeType.M5_2XLARGE],
            prefer=None,
        )
        self.assertIs(node, NodeType.M5_XLARGE)


if __name__ == "__main__":
    unittest.main()
