"""Unit tests for the Databricks Vector Search service and resources.

Exercises endpoint / index lifecycle, the query path, the
``data_array`` → Arrow conversion, and the defaults dataclass on top of
mocked SDK calls.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import replace
from unittest.mock import MagicMock

from databricks.sdk.errors import AlreadyExists, NotFound
from databricks.sdk.service.vectorsearch import (
    ColumnInfo,
    DeltaSyncVectorIndexSpecRequest,
    DirectAccessVectorIndexSpec,
    EmbeddingSourceColumn,
    EmbeddingVectorColumn,
    EndpointInfo,
    EndpointStatus,
    EndpointStatusState,
    EndpointType,
    IndexSubtype,
    MiniVectorIndex,
    PipelineType,
    QueryVectorIndexResponse,
    ResultData,
    ResultManifest,
    VectorIndex,
    VectorIndexStatus,
    VectorIndexType,
)

from yggdrasil.databricks.ai import (
    DEFAULT_VS_WAIT,
    DatabricksAI,
    VectorSearch,
    VectorSearchDefaults,
    VectorSearchEndpoint,
    VectorSearchIndex,
    VectorSearchQueryResult,
)
from yggdrasil.databricks.tests import DatabricksTestCase
from yggdrasil.dataclasses import WaitingConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_endpoint_info(
    *,
    name: str = "rag-endpoint",
    state: EndpointStatusState = EndpointStatusState.ONLINE,
    endpoint_type: EndpointType = EndpointType.STANDARD,
    num_indexes: int = 0,
) -> EndpointInfo:
    return EndpointInfo(
        name=name,
        endpoint_type=endpoint_type,
        endpoint_status=EndpointStatus(state=state, message="ok"),
        num_indexes=num_indexes,
    )


def _build_vector_index(
    *,
    name: str = "main.rag.docs",
    endpoint_name: str = "rag-endpoint",
    index_type: VectorIndexType = VectorIndexType.DELTA_SYNC,
    primary_key: str = "id",
    ready: bool = True,
    indexed_row_count: int | None = 42,
) -> VectorIndex:
    return VectorIndex(
        name=name,
        endpoint_name=endpoint_name,
        index_type=index_type,
        index_subtype=IndexSubtype.VECTOR,
        primary_key=primary_key,
        status=VectorIndexStatus(
            ready=ready,
            indexed_row_count=indexed_row_count,
            message="ok",
        ),
    )


def _build_query_response(
    *,
    column_specs: list[tuple[str, str]],
    rows: list[list[str | None]],
    next_page_token: str | None = None,
) -> QueryVectorIndexResponse:
    manifest = ResultManifest(
        column_count=len(column_specs),
        columns=[ColumnInfo(name=n, type_text=t) for n, t in column_specs],
    )
    return QueryVectorIndexResponse(
        manifest=manifest,
        result=ResultData(data_array=rows, row_count=len(rows)),
        next_page_token=next_page_token,
    )


# ---------------------------------------------------------------------------
# Test base
# ---------------------------------------------------------------------------


class VectorSearchTestCase(DatabricksTestCase):
    """Helper base that exposes vector-search API mocks."""

    def setUp(self) -> None:
        super().setUp()
        self.endpoints_api = self.workspace_client.vector_search_endpoints
        self.indexes_api = self.workspace_client.vector_search_indexes

    @property
    def vs(self) -> VectorSearch:
        return self.client.ai.vector_search


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


class TestWiring(VectorSearchTestCase):
    def test_client_ai_returns_singleton(self):
        ai = self.client.ai
        self.assertIsInstance(ai, DatabricksAI)
        self.assertIs(self.client.ai, ai)

    def test_vector_search_cached_on_ai(self):
        vs = self.client.ai.vector_search
        self.assertIsInstance(vs, VectorSearch)
        self.assertIs(self.client.ai.vector_search, vs)

    def test_service_shortcut_via_inherited_property(self):
        # Every DatabricksService inherits ``.ai`` from the base class.
        self.assertIs(self.client.sql.ai, self.client.ai)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


class TestDefaults(VectorSearchTestCase):
    def test_defaults_attached_to_service(self):
        self.assertIsInstance(self.vs.defaults, VectorSearchDefaults)
        self.assertIs(self.vs.defaults.wait, DEFAULT_VS_WAIT)
        self.assertEqual(self.vs.defaults.endpoint_type, "STANDARD")
        self.assertEqual(self.vs.defaults.index_type, "DELTA_SYNC")
        self.assertEqual(self.vs.defaults.pipeline_type, "TRIGGERED")
        self.assertIsNone(self.vs.defaults.endpoint_name)
        self.assertIsNone(self.vs.defaults.embedding_model_endpoint_name)

    def test_defaults_replace_in_place(self):
        self.vs.defaults = replace(
            self.vs.defaults,
            endpoint_name="rag-endpoint",
            embedding_model_endpoint_name="databricks-bge-large-en",
            wait=WaitingConfig.from_(60),
        )
        self.assertEqual(self.vs.defaults.endpoint_name, "rag-endpoint")
        self.assertEqual(
            self.vs.defaults.embedding_model_endpoint_name,
            "databricks-bge-large-en",
        )
        self.assertEqual(
            self.vs.defaults.wait.timeout_timedelta,
            dt.timedelta(seconds=60),
        )


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


class TestVectorSearchEndpoint(VectorSearchTestCase):
    def test_endpoint_handle_without_default_raises(self):
        with self.assertRaises(ValueError):
            self.vs.endpoint()

    def test_endpoint_handle_uses_default_endpoint_name(self):
        self.vs.defaults = replace(self.vs.defaults, endpoint_name="rag-endpoint")
        ep = self.vs.endpoint()
        self.assertIsInstance(ep, VectorSearchEndpoint)
        self.assertEqual(ep.endpoint_name, "rag-endpoint")

    def test_endpoint_handle_explicit_name(self):
        ep = self.vs.endpoint("rag-explicit")
        self.assertEqual(ep.endpoint_name, "rag-explicit")
        self.assertIn("rag-explicit", repr(ep))

    def test_infos_caches_after_first_call(self):
        self.endpoints_api.get_endpoint.return_value = _build_endpoint_info()
        ep = self.vs.endpoint("rag-endpoint")
        info1 = ep.infos
        info2 = ep.infos
        self.assertIs(info1, info2)
        self.endpoints_api.get_endpoint.assert_called_once_with(
            endpoint_name="rag-endpoint",
        )

    def test_exists_false_on_not_found(self):
        self.endpoints_api.get_endpoint.side_effect = NotFound("missing")
        ep = self.vs.endpoint("rag-endpoint")
        self.assertFalse(ep.exists)

    def test_is_online_reads_status(self):
        self.endpoints_api.get_endpoint.return_value = _build_endpoint_info(
            state=EndpointStatusState.PROVISIONING,
        )
        ep = self.vs.endpoint("rag-endpoint")
        self.assertFalse(ep.is_online)
        self.assertEqual(ep.state, "PROVISIONING")

    def test_create_passes_endpoint_type_and_caches_infos(self):
        info = _build_endpoint_info(state=EndpointStatusState.PROVISIONING)
        wait_handle = MagicMock()
        wait_handle.response = info
        self.endpoints_api.create_endpoint.return_value = wait_handle

        ep = self.vs.endpoint("rag-endpoint").create()

        self.endpoints_api.create_endpoint.assert_called_once_with(
            name="rag-endpoint",
            endpoint_type=EndpointType.STANDARD,
            budget_policy_id=None,
            target_qps=None,
        )
        self.assertIs(ep.infos, info)
        # ``infos`` should NOT have triggered an extra get_endpoint call.
        self.endpoints_api.get_endpoint.assert_not_called()

    def test_create_with_storage_optimized_string(self):
        info = _build_endpoint_info(endpoint_type=EndpointType.STORAGE_OPTIMIZED)
        wait_handle = MagicMock()
        wait_handle.response = info
        self.endpoints_api.create_endpoint.return_value = wait_handle

        self.vs.endpoint("rag-endpoint").create(endpoint_type="STORAGE_OPTIMIZED")
        call = self.endpoints_api.create_endpoint.call_args
        self.assertEqual(call.kwargs["endpoint_type"], EndpointType.STORAGE_OPTIMIZED)

    def test_create_if_not_exists_swallows_already_exists(self):
        self.endpoints_api.create_endpoint.side_effect = AlreadyExists("dup")
        existing = _build_endpoint_info()
        self.endpoints_api.get_endpoint.return_value = existing

        ep = self.vs.endpoint("rag-endpoint").create(if_not_exists=True)
        self.assertIs(ep.infos, existing)
        self.endpoints_api.get_endpoint.assert_called_once()

    def test_create_if_not_exists_false_propagates(self):
        self.endpoints_api.create_endpoint.side_effect = AlreadyExists("dup")
        with self.assertRaises(AlreadyExists):
            self.vs.endpoint("rag-endpoint").create(if_not_exists=False)

    def test_ensure_created_skips_when_exists(self):
        self.endpoints_api.get_endpoint.return_value = _build_endpoint_info()
        ep = self.vs.endpoint("rag-endpoint").ensure_created()
        self.endpoints_api.create_endpoint.assert_not_called()
        self.assertTrue(ep.is_online)

    def test_ensure_created_invokes_create_when_missing(self):
        self.endpoints_api.get_endpoint.side_effect = NotFound("missing")
        wait_handle = MagicMock()
        wait_handle.response = _build_endpoint_info()
        self.endpoints_api.create_endpoint.return_value = wait_handle

        self.vs.endpoint("rag-endpoint").ensure_created()
        self.endpoints_api.create_endpoint.assert_called_once()

    def test_delete_calls_sdk(self):
        ep = self.vs.endpoint("rag-endpoint")
        ep._details = _build_endpoint_info()
        ep.delete()
        self.endpoints_api.delete_endpoint.assert_called_once_with(
            endpoint_name="rag-endpoint",
        )
        self.assertIsNone(ep._details)

    def test_delete_missing_ok_swallows_not_found(self):
        self.endpoints_api.delete_endpoint.side_effect = NotFound("gone")
        self.vs.endpoint("rag-endpoint").delete(missing_ok=True)

    def test_delete_missing_not_ok_raises(self):
        self.endpoints_api.delete_endpoint.side_effect = NotFound("gone")
        with self.assertRaises(NotFound):
            self.vs.endpoint("rag-endpoint").delete()

    def test_wait_online_routes_through_sdk_helper(self):
        info = _build_endpoint_info()
        self.endpoints_api.wait_get_endpoint_vector_search_endpoint_online.return_value = info
        ep = self.vs.endpoint("rag-endpoint").wait_online(wait=30)
        self.endpoints_api.wait_get_endpoint_vector_search_endpoint_online.assert_called_once()
        call = self.endpoints_api.wait_get_endpoint_vector_search_endpoint_online.call_args
        self.assertEqual(call.kwargs["endpoint_name"], "rag-endpoint")
        self.assertEqual(call.kwargs["timeout"], dt.timedelta(seconds=30))
        self.assertIs(ep.infos, info)

    def test_explore_url_format(self):
        ep = self.vs.endpoint("rag-endpoint")
        self.assertEqual(
            ep.explore_url.to_string(),
            "https://test.databricks.net/compute/vector-search/rag-endpoint",
        )

    def test_endpoint_indexes_iterates_via_service(self):
        mini = MiniVectorIndex(name="main.rag.docs", endpoint_name="rag-endpoint")
        self.indexes_api.list_indexes.return_value = iter([mini])
        ep = self.vs.endpoint("rag-endpoint")
        indexes = list(ep.indexes())
        self.assertEqual(len(indexes), 1)
        self.assertEqual(indexes[0].index_name, "main.rag.docs")
        self.indexes_api.list_indexes.assert_called_once_with(
            endpoint_name="rag-endpoint",
        )


# ---------------------------------------------------------------------------
# Listing / discovery
# ---------------------------------------------------------------------------


class TestVectorSearchListing(VectorSearchTestCase):
    def test_list_endpoints_wraps_each_info(self):
        infos = [
            _build_endpoint_info(name="rag-a"),
            _build_endpoint_info(name="rag-b", state=EndpointStatusState.PROVISIONING),
        ]
        self.endpoints_api.list_endpoints.return_value = iter(infos)
        eps = list(self.vs.list_endpoints())
        self.assertEqual([ep.endpoint_name for ep in eps], ["rag-a", "rag-b"])
        self.assertTrue(eps[0].is_online)
        self.assertFalse(eps[1].is_online)

    def test_find_endpoint_match(self):
        infos = [
            _build_endpoint_info(name="rag-a"),
            _build_endpoint_info(name="rag-b"),
        ]
        self.endpoints_api.list_endpoints.return_value = iter(infos)
        match = self.vs.find_endpoint(name="rag-b")
        self.assertIsNotNone(match)
        self.assertEqual(match.endpoint_name, "rag-b")

    def test_find_endpoint_returns_none_when_default_unset(self):
        self.assertIsNone(self.vs.find_endpoint())

    def test_list_indexes_requires_endpoint(self):
        with self.assertRaises(ValueError):
            list(self.vs.list_indexes())

    def test_list_indexes_uses_default_endpoint(self):
        self.vs.defaults = replace(self.vs.defaults, endpoint_name="rag-endpoint")
        mini = MiniVectorIndex(name="main.rag.docs", endpoint_name="rag-endpoint")
        self.indexes_api.list_indexes.return_value = iter([mini])
        indexes = list(self.vs.list_indexes())
        self.assertEqual(len(indexes), 1)
        self.assertEqual(indexes[0].index_name, "main.rag.docs")
        self.assertEqual(indexes[0].endpoint_name, "rag-endpoint")


# ---------------------------------------------------------------------------
# Index lifecycle
# ---------------------------------------------------------------------------


class TestVectorSearchIndexCreate(VectorSearchTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.vs.defaults = replace(
            self.vs.defaults,
            endpoint_name="rag-endpoint",
            embedding_model_endpoint_name="databricks-bge-large-en",
        )

    def test_create_delta_sync_managed_embedding(self):
        info = _build_vector_index()
        self.indexes_api.create_index.return_value = info

        idx = self.vs.index("main.rag.docs").create_delta_sync(
            source_table="main.rag.source",
            primary_key="id",
            embedding_source_columns=["text"],
        )
        self.assertIs(idx.infos, info)
        call = self.indexes_api.create_index.call_args
        self.assertEqual(call.kwargs["name"], "main.rag.docs")
        self.assertEqual(call.kwargs["endpoint_name"], "rag-endpoint")
        self.assertEqual(call.kwargs["primary_key"], "id")
        self.assertEqual(call.kwargs["index_type"], VectorIndexType.DELTA_SYNC)
        spec: DeltaSyncVectorIndexSpecRequest = call.kwargs["delta_sync_index_spec"]
        self.assertEqual(spec.source_table, "main.rag.source")
        self.assertEqual(spec.pipeline_type, PipelineType.TRIGGERED)
        self.assertEqual(len(spec.embedding_source_columns), 1)
        sc = spec.embedding_source_columns[0]
        self.assertIsInstance(sc, EmbeddingSourceColumn)
        self.assertEqual(sc.name, "text")
        self.assertEqual(
            sc.embedding_model_endpoint_name,
            "databricks-bge-large-en",
        )
        self.assertIsNone(spec.embedding_vector_columns)

    def test_create_delta_sync_self_managed_embedding(self):
        info = _build_vector_index()
        self.indexes_api.create_index.return_value = info

        self.vs.index("main.rag.docs").create_delta_sync(
            source_table="main.rag.source",
            primary_key="id",
            embedding_vector_columns=[{"name": "vec", "embedding_dimension": 768}],
            pipeline_type="CONTINUOUS",
        )
        call = self.indexes_api.create_index.call_args
        spec: DeltaSyncVectorIndexSpecRequest = call.kwargs["delta_sync_index_spec"]
        self.assertEqual(spec.pipeline_type, PipelineType.CONTINUOUS)
        self.assertIsNone(spec.embedding_source_columns)
        self.assertEqual(len(spec.embedding_vector_columns), 1)
        vc = spec.embedding_vector_columns[0]
        self.assertIsInstance(vc, EmbeddingVectorColumn)
        self.assertEqual(vc.name, "vec")
        self.assertEqual(vc.embedding_dimension, 768)

    def test_create_delta_sync_rejects_both_embedding_shapes(self):
        with self.assertRaises(ValueError):
            self.vs.index("main.rag.docs").create_delta_sync(
                source_table="main.rag.source",
                primary_key="id",
                embedding_source_columns=["text"],
                embedding_vector_columns=[{"name": "vec", "embedding_dimension": 768}],
            )

    def test_create_delta_sync_rejects_no_embedding_shape(self):
        with self.assertRaises(ValueError):
            self.vs.index("main.rag.docs").create_delta_sync(
                source_table="main.rag.source",
                primary_key="id",
            )

    def test_create_delta_sync_requires_model_endpoint_when_managed(self):
        self.vs.defaults = replace(self.vs.defaults, embedding_model_endpoint_name=None)
        with self.assertRaises(ValueError) as ctx:
            self.vs.index("main.rag.docs").create_delta_sync(
                source_table="main.rag.source",
                primary_key="id",
                embedding_source_columns=["text"],
            )
        self.assertIn("embedding_model_endpoint_name", str(ctx.exception))

    def test_create_delta_sync_requires_endpoint_when_default_unset(self):
        self.vs.defaults = replace(self.vs.defaults, endpoint_name=None)
        with self.assertRaises(ValueError):
            self.vs.index("main.rag.docs").create_delta_sync(
                source_table="main.rag.source",
                primary_key="id",
                embedding_source_columns=["text"],
            )

    def test_create_delta_sync_if_not_exists_swallows_already_exists(self):
        self.indexes_api.create_index.side_effect = AlreadyExists("dup")
        existing = _build_vector_index()
        self.indexes_api.get_index.return_value = existing
        idx = self.vs.index("main.rag.docs").create_delta_sync(
            source_table="main.rag.source",
            primary_key="id",
            embedding_source_columns=["text"],
        )
        self.assertIs(idx.infos, existing)

    def test_create_direct_access(self):
        info = _build_vector_index(index_type=VectorIndexType.DIRECT_ACCESS)
        self.indexes_api.create_index.return_value = info

        self.vs.index("main.rag.docs").create_direct_access(
            primary_key="id",
            schema_json='{"id": "string", "text": "string", "vec": "array<float>"}',
            embedding_vector_columns=[
                EmbeddingVectorColumn(name="vec", embedding_dimension=768),
            ],
        )
        call = self.indexes_api.create_index.call_args
        self.assertEqual(call.kwargs["index_type"], VectorIndexType.DIRECT_ACCESS)
        spec: DirectAccessVectorIndexSpec = call.kwargs["direct_access_index_spec"]
        self.assertEqual(spec.schema_json, '{"id": "string", "text": "string", "vec": "array<float>"}')
        self.assertEqual(len(spec.embedding_vector_columns), 1)

    def test_create_direct_access_requires_some_embedding(self):
        with self.assertRaises(ValueError):
            self.vs.index("main.rag.docs").create_direct_access(
                primary_key="id",
                schema_json="{}",
            )


# ---------------------------------------------------------------------------
# Index runtime
# ---------------------------------------------------------------------------


class TestVectorSearchIndexRuntime(VectorSearchTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.vs.defaults = replace(self.vs.defaults, endpoint_name="rag-endpoint")
        self.index_info = _build_vector_index()
        self.indexes_api.get_index.return_value = self.index_info

    def _idx(self) -> VectorSearchIndex:
        return self.vs.index("main.rag.docs")

    def test_infos_caches_first_call(self):
        idx = self._idx()
        _ = idx.infos
        _ = idx.infos
        self.indexes_api.get_index.assert_called_once_with(index_name="main.rag.docs")

    def test_is_ready_and_indexed_row_count(self):
        idx = self._idx()
        self.assertTrue(idx.is_ready)
        self.assertEqual(idx.indexed_row_count, 42)
        self.assertEqual(idx.primary_key, "id")
        self.assertEqual(idx.index_type, "DELTA_SYNC")

    def test_sync_calls_sdk(self):
        self._idx().sync()
        self.indexes_api.sync_index.assert_called_once_with(index_name="main.rag.docs")

    def test_delete_calls_sdk(self):
        idx = self._idx()
        idx._details = self.index_info
        idx.delete()
        self.indexes_api.delete_index.assert_called_once_with(index_name="main.rag.docs")
        self.assertIsNone(idx._details)

    def test_wait_ready_returns_immediately_when_ready(self):
        # status already ready=True from setUp
        idx = self._idx()
        idx.wait_ready(wait=5)
        # One refresh call from wait_ready, plus no time-passing.
        self.assertTrue(idx.is_ready)

    def test_wait_ready_times_out_when_not_ready(self):
        self.indexes_api.get_index.return_value = _build_vector_index(ready=False)
        idx = self.vs.index("main.rag.docs")
        with self.assertRaises(TimeoutError):
            # Use a fractional second timeout + tiny interval to keep the
            # test snappy. The polling loop sleeps min(interval, remaining)
            # and the SDK mock returns ready=False every time.
            idx.wait_ready(wait=WaitingConfig(timeout=0.05, interval=0.01))

    def test_upsert_serialises_rows_through_ygg_json(self):
        from yggdrasil.pickle import json as ygg_json

        rows = [{"id": "a", "text": "alpha"}, {"id": "b", "text": "beta"}]
        self._idx().upsert(rows)
        call = self.indexes_api.upsert_data_vector_index.call_args
        self.assertEqual(call.kwargs["index_name"], "main.rag.docs")
        parsed = ygg_json.loads(call.kwargs["inputs_json"])
        self.assertEqual(parsed, rows)

    def test_upsert_skips_when_empty(self):
        self.assertIsNone(self._idx().upsert([]))
        self.indexes_api.upsert_data_vector_index.assert_not_called()

    def test_delete_rows_passes_primary_keys(self):
        self._idx().delete_rows(["a", "b", "c"])
        self.indexes_api.delete_data_vector_index.assert_called_once_with(
            index_name="main.rag.docs",
            primary_keys=["a", "b", "c"],
        )

    def test_delete_rows_skips_when_empty(self):
        self.assertIsNone(self._idx().delete_rows([]))
        self.indexes_api.delete_data_vector_index.assert_not_called()

    def test_scan_passes_kwargs(self):
        self._idx().scan(num_results=10, last_primary_key="abc")
        self.indexes_api.scan_index.assert_called_once_with(
            index_name="main.rag.docs",
            num_results=10,
            last_primary_key="abc",
        )

    def test_explore_url_format(self):
        idx = self.vs.index("main.rag.docs")
        self.assertEqual(
            idx.explore_url.to_string(),
            "https://test.databricks.net/explore/data/main/rag/docs",
        )


# ---------------------------------------------------------------------------
# Querying
# ---------------------------------------------------------------------------


class TestVectorSearchQuery(VectorSearchTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.vs.defaults = replace(self.vs.defaults, endpoint_name="rag-endpoint")
        self.indexes_api.get_index.return_value = _build_vector_index()

    def _set_query_response(self, response: QueryVectorIndexResponse) -> None:
        self.indexes_api.query_index.return_value = response

    def test_query_requires_text_or_vector(self):
        with self.assertRaises(ValueError):
            self.vs.index("main.rag.docs").query(columns=["id"])

    def test_query_rejects_both_text_and_vector(self):
        with self.assertRaises(ValueError):
            self.vs.index("main.rag.docs").query(
                columns=["id"],
                query_text="foo",
                query_vector=[0.1, 0.2],
            )

    def test_query_text_path_passes_kwargs(self):
        self._set_query_response(_build_query_response(
            column_specs=[("id", "string"), ("score", "double")],
            rows=[["a", "0.9"], ["b", "0.8"]],
        ))
        result = self.vs.index("main.rag.docs").query(
            columns=["id"],
            query_text="how do I bake bread?",
            num_results=5,
        )
        call = self.indexes_api.query_index.call_args
        self.assertEqual(call.kwargs["index_name"], "main.rag.docs")
        self.assertEqual(call.kwargs["columns"], ["id"])
        self.assertEqual(call.kwargs["query_text"], "how do I bake bread?")
        self.assertIsNone(call.kwargs["query_vector"])
        self.assertEqual(call.kwargs["num_results"], 5)
        self.assertEqual(result.row_count, 2)

    def test_query_vector_path(self):
        self._set_query_response(_build_query_response(
            column_specs=[("id", "string")],
            rows=[["a"]],
        ))
        self.vs.index("main.rag.docs").query(
            columns=["id"],
            query_vector=[0.1, 0.2, 0.3],
        )
        call = self.indexes_api.query_index.call_args
        self.assertEqual(call.kwargs["query_vector"], [0.1, 0.2, 0.3])
        self.assertIsNone(call.kwargs["query_text"])

    def test_query_filters_mapping_serialised(self):
        from yggdrasil.pickle import json as ygg_json

        self._set_query_response(_build_query_response(
            column_specs=[("id", "string")],
            rows=[],
        ))
        self.vs.index("main.rag.docs").query(
            columns=["id"],
            query_text="x",
            filters={"category": "news", "year": 2026},
        )
        call = self.indexes_api.query_index.call_args
        parsed = ygg_json.loads(call.kwargs["filters_json"])
        self.assertEqual(parsed, {"category": "news", "year": 2026})

    def test_query_filters_string_passed_through(self):
        self._set_query_response(_build_query_response(
            column_specs=[("id", "string")],
            rows=[],
        ))
        self.vs.index("main.rag.docs").query(
            columns=["id"],
            query_text="x",
            filters='{"category": "news"}',
        )
        call = self.indexes_api.query_index.call_args
        self.assertEqual(call.kwargs["filters_json"], '{"category": "news"}')


# ---------------------------------------------------------------------------
# Query result materialisation
# ---------------------------------------------------------------------------


class TestVectorSearchQueryResult(VectorSearchTestCase):
    def _make_result(
        self,
        *,
        column_specs: list[tuple[str, str]],
        rows: list[list[str | None]],
        next_page_token: str | None = None,
    ) -> VectorSearchQueryResult:
        self.vs.defaults = replace(self.vs.defaults, endpoint_name="rag-endpoint")
        return VectorSearchQueryResult(
            index=self.vs.index("main.rag.docs"),
            response=_build_query_response(
                column_specs=column_specs,
                rows=rows,
                next_page_token=next_page_token,
            ),
        )

    def test_to_dicts_keys_by_column_name(self):
        result = self._make_result(
            column_specs=[("id", "string"), ("score", "double")],
            rows=[["a", "0.9"], ["b", "0.8"]],
        )
        self.assertEqual(result.to_dicts(), [
            {"id": "a", "score": "0.9"},
            {"id": "b", "score": "0.8"},
        ])

    def test_to_arrow_table_casts_per_column(self):
        import pyarrow as pa

        result = self._make_result(
            column_specs=[
                ("id", "string"),
                ("price", "double"),
                ("count", "bigint"),
                ("active", "boolean"),
                ("unknown", "weird-type"),
            ],
            rows=[
                ["a", "1.5", "10", "true", "left-as-string"],
                ["b", "2.5", "20", "false", "still-string"],
            ],
        )
        table = result.to_arrow_table()
        self.assertEqual(table.num_rows, 2)
        self.assertEqual(table.column_names, ["id", "price", "count", "active", "unknown"])
        self.assertEqual(table.schema.field("id").type, pa.string())
        self.assertEqual(table.schema.field("price").type, pa.float64())
        self.assertEqual(table.schema.field("count").type, pa.int64())
        self.assertEqual(table.schema.field("active").type, pa.bool_())
        # Unknown type_text falls back to string so the byte payload survives.
        self.assertEqual(table.schema.field("unknown").type, pa.string())
        self.assertEqual(table.column("price").to_pylist(), [1.5, 2.5])
        self.assertEqual(table.column("count").to_pylist(), [10, 20])
        self.assertEqual(table.column("active").to_pylist(), [True, False])

    def test_to_arrow_table_handles_empty_rows(self):
        result = self._make_result(
            column_specs=[("id", "string"), ("score", "double")],
            rows=[],
        )
        table = result.to_arrow_table()
        self.assertEqual(table.num_rows, 0)
        self.assertEqual(table.column_names, ["id", "score"])

    def test_to_polars(self):
        result = self._make_result(
            column_specs=[("id", "string"), ("score", "double")],
            rows=[["a", "0.9"]],
        )
        try:
            df = result.to_polars()
        except Exception as exc:  # pragma: no cover - exercised under [data] extras
            self.skipTest(f"polars not installed: {exc}")
        self.assertEqual(df.shape, (1, 2))
        self.assertEqual(df["id"].to_list(), ["a"])

    def test_next_page_iteration(self):
        # First page carries a token; second page is empty + no token.
        page1_response = _build_query_response(
            column_specs=[("id", "string")],
            rows=[["a"]],
            next_page_token="next-1",
        )
        page2_response = _build_query_response(
            column_specs=[("id", "string")],
            rows=[["b"]],
            next_page_token=None,
        )
        self.indexes_api.query_next_page.return_value = page2_response

        self.vs.defaults = replace(self.vs.defaults, endpoint_name="rag-endpoint")
        result = VectorSearchQueryResult(
            index=self.vs.index("main.rag.docs"),
            response=page1_response,
        )
        pages = list(result.iter_pages())
        self.assertEqual(len(pages), 2)
        self.assertEqual(pages[0].column_names, ("id",))
        self.assertEqual([row[0] for row in pages[1].data_array], ["b"])
        self.indexes_api.query_next_page.assert_called_once_with(
            index_name="main.rag.docs",
            endpoint_name="rag-endpoint",
            page_token="next-1",
        )
