# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for ElasticsearchSearchEngine.

These tests require a running Elasticsearch cluster.  Set the following
environment variables before running::

    ES_ENDPOINT   — cluster URL, e.g. http://localhost:9200
    ES_API_KEY    — API key (or ES_USERNAME + ES_PASSWORD for basic auth)

Run with::

    source .venv/bin/activate
    pytest tests/core/tools/engines/search/test_elasticsearch.py --integration -v

No env-var guards are applied — missing credentials will surface as connection
errors, which is the intended signal that the environment is not configured.
"""

# Third Party
from dotenv import load_dotenv
import pytest

# Local
from fms_dgt.core.tools.data_objects import Tool, ToolCall
from fms_dgt.core.tools.engines.search.elasticsearch import ElasticsearchSearchEngine
from fms_dgt.core.tools.registry import ToolRegistry

# ---------------------------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_INDEX = "dgt_integration_test"
_NS = "ns"
_DOCS = [
    {"id": "1", "text": "Photosynthesis converts sunlight into chemical energy stored as glucose."},
    {
        "id": "2",
        "text": "The mitochondria generate most of a cell's ATP through cellular respiration.",
    },
    {
        "id": "3",
        "text": "Continental drift describes the movement of tectonic plates over geological time.",
    },
    {
        "id": "4",
        "text": "Newton's laws of motion describe the relationship between force and acceleration.",
    },
    {
        "id": "5",
        "text": "The water cycle moves water between oceans, atmosphere, and land via evaporation and precipitation.",
    },
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_registry() -> ToolRegistry:
    return ToolRegistry(tools=[Tool(name="search", namespace=_NS, description="Search")])


def _tc(query: str, call_id: str = "c1", **extra) -> ToolCall:
    return ToolCall(
        name=f"{_NS}::search", arguments={"query": query, "index": _INDEX, **extra}, call_id=call_id
    )


# ---------------------------------------------------------------------------
# Module-scoped fixture: index documents once, clean up after all tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def es_engine():
    engine = ElasticsearchSearchEngine(_make_registry(), index=_INDEX, limit=5)
    engine.setup("integration")
    yield engine
    engine.teardown("integration")


@pytest.fixture(scope="module", autouse=True)
def populated_index(es_engine):
    # Third Party

    client = es_engine._client
    if client.indices.exists(index=_INDEX):
        client.indices.delete(index=_INDEX)
    client.indices.create(
        index=_INDEX, body={"mappings": {"properties": {"text": {"type": "text"}}}}
    )
    for doc in _DOCS:
        client.index(index=_INDEX, id=doc["id"], document={"text": doc["text"]})
    client.indices.refresh(index=_INDEX)
    yield
    client.indices.delete(index=_INDEX, ignore_unavailable=True)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestElasticsearchIntegration:
    def test_search_returns_results(self, es_engine):
        [result] = es_engine.execute("integration", [_tc("photosynthesis sunlight")])
        assert result.error is None
        assert len(result.result) >= 1

    def test_result_shape(self, es_engine):
        [result] = es_engine.execute("integration", [_tc("mitochondria ATP")])
        assert result.error is None
        for doc in result.result:
            assert "id" in doc
            assert "text" in doc

    def test_relevant_document_returned(self, es_engine):
        [result] = es_engine.execute("integration", [_tc("tectonic plates continental drift")])
        assert result.error is None
        texts = [d["text"] for d in result.result]
        assert any("continental" in t.lower() or "tectonic" in t.lower() for t in texts)

    def test_limit_respected(self, es_engine):
        engine = ElasticsearchSearchEngine(_make_registry(), index=_INDEX, limit=2)
        engine.setup("integration-limit")
        try:
            [result] = engine.execute("integration-limit", [_tc("water cycle energy motion")])
            assert result.error is None
            assert len(result.result) <= 2
        finally:
            engine.teardown("integration-limit")

    def test_empty_query_returns_empty_list(self, es_engine):
        tc = ToolCall(
            name=f"{_NS}::search", arguments={"query": "", "index": _INDEX}, call_id="empty"
        )
        [result] = es_engine.execute("integration", [tc])
        assert result.result == []

    def test_call_id_and_name_propagated(self, es_engine):
        [result] = es_engine.execute("integration", [_tc("newton force", call_id="my-id")])
        assert result.call_id == "my-id"
        assert result.name == f"{_NS}::search"

    def test_missing_index_surfaces_error(self, es_engine):
        tc = ToolCall(
            name=f"{_NS}::search",
            arguments={"query": "test", "index": "nonexistent_index_xyz"},
            call_id="bad",
        )
        [result] = es_engine.execute("integration", [tc])
        assert result.error is not None

    def test_corpus_returns_all_indexed_documents(self, es_engine):
        docs = es_engine.corpus()
        assert len(docs) == len(_DOCS)
        returned_ids = {d.id for d in docs}
        expected_ids = {d["id"] for d in _DOCS}
        assert returned_ids == expected_ids

    def test_corpus_documents_have_text(self, es_engine):
        docs = es_engine.corpus()
        for doc in docs:
            assert doc.text
