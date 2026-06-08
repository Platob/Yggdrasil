"""Tests for library deduplication helpers and pool-managed field stripping."""
from __future__ import annotations

from databricks.sdk.service.compute import (
    ClusterDetails,
    ClusterSource,
    Library,
    LibraryFullStatus,
    LibraryInstallStatus,
    PythonPyPiLibrary,
    State,
)

from yggdrasil.databricks.cluster.cluster import (
    Cluster,
    _dedupe_libraries,
    _library_dedup_key,
    _POOL_MANAGED_FIELDS,
)
from yggdrasil.databricks.tests import DatabricksTestCase


# ------------------------------------------------------------------ #
# _library_dedup_key
# ------------------------------------------------------------------ #

class TestLibraryDedupKey(DatabricksTestCase):

    def test_pypi_strips_version(self):
        a = Library(pypi=PythonPyPiLibrary(package="pandas==1.5.0"))
        b = Library(pypi=PythonPyPiLibrary(package="pandas==2.0.0"))
        self.assertEqual(_library_dedup_key(a), _library_dedup_key(b))

    def test_pypi_normalizes_underscores(self):
        a = Library(pypi=PythonPyPiLibrary(package="scikit_learn"))
        b = Library(pypi=PythonPyPiLibrary(package="scikit-learn~=1.0"))
        self.assertEqual(_library_dedup_key(a), _library_dedup_key(b))

    def test_pypi_case_insensitive(self):
        a = Library(pypi=PythonPyPiLibrary(package="PyArrow>=12"))
        b = Library(pypi=PythonPyPiLibrary(package="pyarrow"))
        self.assertEqual(_library_dedup_key(a), _library_dedup_key(b))

    def test_pypi_ignores_repo_difference(self):
        a = Library(pypi=PythonPyPiLibrary(package="mypkg", repo="https://repo1"))
        b = Library(pypi=PythonPyPiLibrary(package="mypkg", repo="https://repo2"))
        self.assertEqual(_library_dedup_key(a), _library_dedup_key(b))

    def test_pypi_with_extras(self):
        a = Library(pypi=PythonPyPiLibrary(package="ygg[databricks]==1.0"))
        b = Library(pypi=PythonPyPiLibrary(package="ygg[http,data]==2.0"))
        # extras are part of the name up to the bracket — both normalize to "ygg"
        self.assertEqual(_library_dedup_key(a), _library_dedup_key(b))

    def test_different_pypi_packages_differ(self):
        a = Library(pypi=PythonPyPiLibrary(package="pandas"))
        b = Library(pypi=PythonPyPiLibrary(package="polars"))
        self.assertNotEqual(_library_dedup_key(a), _library_dedup_key(b))

    def test_jar(self):
        a = Library(jar="dbfs:/jars/my.jar")
        b = Library(jar="dbfs:/jars/my.jar")
        self.assertEqual(_library_dedup_key(a), _library_dedup_key(b))

    def test_different_jars_differ(self):
        a = Library(jar="dbfs:/jars/a.jar")
        b = Library(jar="dbfs:/jars/b.jar")
        self.assertNotEqual(_library_dedup_key(a), _library_dedup_key(b))

    def test_wheel(self):
        a = Library(whl="dbfs:/wheels/pkg.whl")
        self.assertEqual(_library_dedup_key(a), ("whl", "dbfs:/wheels/pkg.whl"))

    def test_requirements(self):
        a = Library(requirements="dbfs:/requirements.txt")
        self.assertEqual(_library_dedup_key(a), ("requirements", "dbfs:/requirements.txt"))

    def test_jar_vs_pypi_differ(self):
        a = Library(jar="pkg.jar")
        b = Library(pypi=PythonPyPiLibrary(package="pkg"))
        self.assertNotEqual(_library_dedup_key(a), _library_dedup_key(b))


# ------------------------------------------------------------------ #
# _dedupe_libraries
# ------------------------------------------------------------------ #

class TestDedupeLibraries(DatabricksTestCase):

    def test_no_duplicates_preserved(self):
        libs = [
            Library(pypi=PythonPyPiLibrary(package="pandas")),
            Library(pypi=PythonPyPiLibrary(package="polars")),
        ]
        result = _dedupe_libraries(libs)
        self.assertEqual(len(result), 2)

    def test_duplicate_pypi_keeps_last(self):
        libs = [
            Library(pypi=PythonPyPiLibrary(package="pandas==1.0")),
            Library(pypi=PythonPyPiLibrary(package="polars")),
            Library(pypi=PythonPyPiLibrary(package="pandas==2.0")),
        ]
        result = _dedupe_libraries(libs)
        self.assertEqual(len(result), 2)
        packages = [lib.pypi.package for lib in result]
        self.assertIn("polars", packages)
        self.assertIn("pandas==2.0", packages)
        self.assertNotIn("pandas==1.0", packages)

    def test_preserves_order(self):
        libs = [
            Library(pypi=PythonPyPiLibrary(package="aaa")),
            Library(pypi=PythonPyPiLibrary(package="ccc")),
            Library(pypi=PythonPyPiLibrary(package="bbb")),
        ]
        result = _dedupe_libraries(libs)
        self.assertEqual([lib.pypi.package for lib in result], ["aaa", "ccc", "bbb"])

    def test_empty_list(self):
        self.assertEqual(_dedupe_libraries([]), [])

    def test_single_element(self):
        libs = [Library(pypi=PythonPyPiLibrary(package="pandas"))]
        result = _dedupe_libraries(libs)
        self.assertEqual(len(result), 1)

    def test_mixed_types_no_cross_dedup(self):
        libs = [
            Library(jar="pkg.jar"),
            Library(pypi=PythonPyPiLibrary(package="pkg")),
        ]
        result = _dedupe_libraries(libs)
        self.assertEqual(len(result), 2)

    def test_version_and_extras_collapse(self):
        libs = [
            Library(pypi=PythonPyPiLibrary(package="ygg[http,data]==1.0")),
            Library(pypi=PythonPyPiLibrary(package="uv")),
            Library(pypi=PythonPyPiLibrary(package="ygg[databricks]==2.0")),
        ]
        result = _dedupe_libraries(libs)
        self.assertEqual(len(result), 2)
        packages = [lib.pypi.package for lib in result]
        self.assertEqual(packages, ["uv", "ygg[databricks]==2.0"])


# ------------------------------------------------------------------ #
# _dedupe_uninstalled_libraries
# ------------------------------------------------------------------ #

class TestDedupeUninstalledLibraries(DatabricksTestCase):

    def _make_cluster_with_installed(self, installed_packages: list[str]) -> Cluster:
        cluster = Cluster(service=self.clusters, cluster_id="c-dedup-1")
        statuses = [
            LibraryFullStatus(
                library=Library(pypi=PythonPyPiLibrary(package=pkg)),
                status=LibraryInstallStatus.INSTALLED,
            )
            for pkg in installed_packages
        ]
        self.libraries_api.cluster_status.return_value = statuses
        return cluster

    def test_filters_already_installed(self):
        cluster = self._make_cluster_with_installed(["pandas==1.5.0"])
        to_install = [Library(pypi=PythonPyPiLibrary(package="pandas==1.5.0"))]
        result = cluster._dedupe_uninstalled_libraries(to_install)
        self.assertEqual(result, [])

    def test_keeps_not_installed(self):
        cluster = self._make_cluster_with_installed(["pandas==1.5.0"])
        to_install = [Library(pypi=PythonPyPiLibrary(package="polars"))]
        result = cluster._dedupe_uninstalled_libraries(to_install)
        self.assertEqual(len(result), 1)

    def test_filters_by_name_not_version(self):
        """Installed pandas==1.5 should suppress requested pandas==2.0."""
        cluster = self._make_cluster_with_installed(["pandas==1.5.0"])
        to_install = [Library(pypi=PythonPyPiLibrary(package="pandas==2.0.0"))]
        result = cluster._dedupe_uninstalled_libraries(to_install)
        self.assertEqual(result, [])

    def test_filters_case_insensitive(self):
        cluster = self._make_cluster_with_installed(["PyArrow>=12"])
        to_install = [Library(pypi=PythonPyPiLibrary(package="pyarrow==14.0"))]
        result = cluster._dedupe_uninstalled_libraries(to_install)
        self.assertEqual(result, [])


# ------------------------------------------------------------------ #
# _editable_details_from — pool-managed field stripping
# ------------------------------------------------------------------ #

class TestEditableDetailsPoolFields(DatabricksTestCase):

    def test_no_pool_keeps_node_type(self):
        details = ClusterDetails(
            cluster_id="c-1",
            cluster_name="test",
            spark_version="15.4.x-scala2.12",
            node_type_id="m5.xlarge",
        )
        result = Cluster._editable_details_from(details)
        self.assertEqual(result.get("node_type_id"), "m5.xlarge")

    def test_pool_strips_node_type_id(self):
        details = ClusterDetails(
            cluster_id="c-1",
            cluster_name="test",
            spark_version="15.4.x-scala2.12",
            instance_pool_id="pool-1",
            node_type_id="m5.xlarge",
        )
        result = Cluster._editable_details_from(details)
        self.assertNotIn("node_type_id", result)

    def test_pool_strips_driver_node_type_id(self):
        details = ClusterDetails(
            cluster_id="c-1",
            cluster_name="test",
            spark_version="15.4.x-scala2.12",
            instance_pool_id="pool-1",
            driver_node_type_id="m5.xlarge",
        )
        result = Cluster._editable_details_from(details)
        self.assertNotIn("driver_node_type_id", result)

    def test_pool_strips_all_managed_fields(self):
        details = ClusterDetails(
            cluster_id="c-1",
            cluster_name="test",
            spark_version="15.4.x-scala2.12",
            instance_pool_id="pool-1",
            node_type_id="m5.xlarge",
            driver_node_type_id="m5.xlarge",
            enable_elastic_disk=True,
        )
        result = Cluster._editable_details_from(details)
        for field in _POOL_MANAGED_FIELDS:
            self.assertNotIn(field, result, f"{field} should be stripped when pool is set")

    def test_pool_preserves_non_managed_fields(self):
        details = ClusterDetails(
            cluster_id="c-1",
            cluster_name="test",
            spark_version="15.4.x-scala2.12",
            instance_pool_id="pool-1",
            node_type_id="m5.xlarge",
            num_workers=4,
        )
        result = Cluster._editable_details_from(details)
        self.assertIn("instance_pool_id", result)
        self.assertEqual(result["instance_pool_id"], "pool-1")

    def test_none_details_returns_empty(self):
        result = Cluster._editable_details_from(None)
        self.assertEqual(result, {})

    def test_update_skips_when_only_pool_fields_differ(self):
        """Simulates the real update flow: current details have pool-filled
        node_type_id, desired details have it nullified by check_details.
        The diff should be empty after stripping."""
        current = ClusterDetails(
            cluster_id="c-1",
            cluster_name="test",
            spark_version="15.4.x-scala2.12",
            instance_pool_id="pool-1",
            node_type_id="m5.xlarge",
            driver_node_type_id="m5.xlarge",
            enable_elastic_disk=True,
        )
        desired = ClusterDetails(
            cluster_id="c-1",
            cluster_name="test",
            spark_version="15.4.x-scala2.12",
            instance_pool_id="pool-1",
            node_type_id=None,
            driver_node_type_id=None,
            enable_elastic_disk=None,
        )
        from yggdrasil.pyutils.equality import dicts_equal
        from yggdrasil.databricks.cluster.cluster import _EDIT_ARG_NAMES

        current_edit = Cluster._editable_details_from(current)
        desired_edit = Cluster._editable_details_from(desired)
        self.assertTrue(
            dicts_equal(current_edit, desired_edit, keys=_EDIT_ARG_NAMES),
            "Update should be skipped when only pool-managed fields differ",
        )


# ------------------------------------------------------------------ #
# Clusters.list / find_cluster — cluster-source filtering
# ------------------------------------------------------------------ #

class TestClusterSourceFilter(DatabricksTestCase):
    """``Clusters.list`` / ``find_cluster`` always apply a source filter."""

    def setUp(self) -> None:
        super().setUp()
        self.clusters_api.list.return_value = []

    def test_list_applies_source_filter_without_name(self):
        # Previously the filter was only built when a name was given, so an
        # unnamed list silently ignored sources and returned every source.
        list(self.clusters.list())
        _, kwargs = self.clusters_api.list.call_args
        filter_by = kwargs["filter_by"]
        self.assertIsNotNone(filter_by)
        self.assertEqual(
            filter_by.cluster_sources, [ClusterSource.API, ClusterSource.UI]
        )

    def test_list_default_sources_include_ui(self):
        list(self.clusters.list(name="etl"))
        _, kwargs = self.clusters_api.list.call_args
        self.assertEqual(
            kwargs["filter_by"].cluster_sources,
            [ClusterSource.API, ClusterSource.UI],
        )

    def test_list_honors_explicit_sources(self):
        list(self.clusters.list(sources=[ClusterSource.JOB]))
        _, kwargs = self.clusters_api.list.call_args
        self.assertEqual(kwargs["filter_by"].cluster_sources, [ClusterSource.JOB])

    def test_find_cluster_list_scan_defaults_to_api_and_ui(self):
        # No cached id + lookup by name → falls through to the list scan,
        # which must default sources to [API, UI] so UI clusters are found.
        details = ClusterDetails(
            cluster_id="0303-000000-ccc33333",
            cluster_name="ui-made-cluster",
            state=State.RUNNING,
        )
        self.clusters_api.list.return_value = [details]
        # find_cluster compares against ``cluster.details.cluster_name``,
        # which lazily re-fetches via clusters.get — return the same record.
        self.clusters_api.get.return_value = details

        found = self.clusters.find_cluster(cluster_name="ui-made-cluster")

        self.assertIsNotNone(found)
        self.assertEqual(found.cluster_id, "0303-000000-ccc33333")
        _, kwargs = self.clusters_api.list.call_args
        self.assertEqual(
            kwargs["filter_by"].cluster_sources,
            [ClusterSource.API, ClusterSource.UI],
        )
