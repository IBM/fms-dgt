# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for FileSearchEngine and RandomDocumentSampler.

Corpus files live in test_data/ alongside this module:
  corpus_canonical.jsonl  — standard {id, text} fields, 20 documents
  corpus_projected.jsonl  — non-standard {doc_id, body, source_url, category} fields, 10 documents
"""

# Standard
from pathlib import Path

# Third Party
import pytest

# Local
from fms_dgt.core.tools.data_objects import Tool, ToolCall
from fms_dgt.core.tools.engines.base import ErrorCategory
from fms_dgt.core.tools.engines.search.base import Document
from fms_dgt.core.tools.engines.search.file import FileSearchEngine
from fms_dgt.core.tools.engines.search.samplers.random import RandomDocumentSampler
from fms_dgt.core.tools.registry import ToolRegistry

# ---------------------------------------------------------------------------
# Paths to static test corpora
# ---------------------------------------------------------------------------

_TEST_DATA = Path(__file__).parent / "test_data"
_CORPUS_CANONICAL = str(_TEST_DATA / "corpus_canonical.jsonl")
_CORPUS_PROJECTED = str(_TEST_DATA / "corpus_projected.jsonl")

_CANONICAL_SIZE = 20
_PROJECTED_SIZE = 10

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_registry() -> ToolRegistry:
    return ToolRegistry(tools=[Tool(name="search", namespace="ns", description="Search documents")])


def _make_engine(path: str, **kwargs) -> FileSearchEngine:
    return FileSearchEngine(_make_registry(), path=path, **kwargs)


def _tc(call_id: str = "c1", size: int | None = None) -> ToolCall:
    args = {} if size is None else {"size": size}
    return ToolCall(name="ns::search", arguments=args, call_id=call_id)


# ---------------------------------------------------------------------------
# FileSearchEngine — setup / teardown
# ---------------------------------------------------------------------------


class TestFileSearchEngineLifecycle:
    def test_corpus_empty_before_setup(self):
        engine = _make_engine(_CORPUS_CANONICAL)
        assert engine._corpus == []

    def test_setup_loads_full_corpus(self):
        engine = _make_engine(_CORPUS_CANONICAL)
        engine.setup("s1")
        try:
            assert len(engine._corpus) == _CANONICAL_SIZE
        finally:
            engine.teardown("s1")

    def test_teardown_releases_corpus_when_no_sessions_remain(self):
        engine = _make_engine(_CORPUS_CANONICAL)
        engine.setup("s1")
        engine.teardown("s1")
        assert engine._corpus == []

    def test_corpus_retained_while_any_session_active(self):
        engine = _make_engine(_CORPUS_CANONICAL)
        engine.setup("s1")
        engine.setup("s2")
        engine.teardown("s1")
        assert len(engine._corpus) == _CANONICAL_SIZE
        engine.teardown("s2")
        assert engine._corpus == []

    def test_duplicate_session_raises(self):
        engine = _make_engine(_CORPUS_CANONICAL)
        engine.setup("s1")
        try:
            with pytest.raises(ValueError, match="already active"):
                engine.setup("s1")
        finally:
            engine.teardown("s1")


# ---------------------------------------------------------------------------
# FileSearchEngine — result shape
# ---------------------------------------------------------------------------


class TestFileSearchEngineResults:
    def test_returns_one_result_per_call(self):
        engine = _make_engine(_CORPUS_CANONICAL, limit=5)
        engine.setup("s1")
        try:
            results = engine.execute("s1", [_tc(size=5)])
            assert len(results) == 1
            assert results[0].error is None
            assert isinstance(results[0].result, list)
            assert len(results[0].result) == 5
        finally:
            engine.teardown("s1")

    def test_each_document_has_id_and_text(self):
        engine = _make_engine(_CORPUS_CANONICAL)
        engine.setup("s1")
        try:
            [result] = engine.execute("s1", [_tc(size=5)])
            for doc in result.result:
                assert "id" in doc
                assert "text" in doc
        finally:
            engine.teardown("s1")

    def test_canonical_ids_match_corpus(self):
        engine = _make_engine(_CORPUS_CANONICAL)
        engine.setup("s1")
        try:
            [result] = engine.execute("s1", [_tc(size=_CANONICAL_SIZE)])
            returned_ids = {doc["id"] for doc in result.result}
            expected_ids = {f"doc_{i}" for i in range(_CANONICAL_SIZE)}
            assert returned_ids == expected_ids
        finally:
            engine.teardown("s1")

    def test_size_argument_controls_count(self):
        engine = _make_engine(_CORPUS_CANONICAL)
        engine.setup("s1")
        try:
            for size in (1, 5, 10):
                [result] = engine.execute("s1", [_tc(size=size)])
                assert len(result.result) == size
        finally:
            engine.teardown("s1")

    def test_size_capped_at_corpus_size(self):
        engine = _make_engine(_CORPUS_CANONICAL)
        engine.setup("s1")
        try:
            [result] = engine.execute("s1", [_tc(size=999)])
            assert len(result.result) == _CANONICAL_SIZE
        finally:
            engine.teardown("s1")

    def test_default_limit_used_when_size_absent(self):
        engine = _make_engine(_CORPUS_CANONICAL, limit=3)
        engine.setup("s1")
        try:
            [result] = engine.execute("s1", [_tc()])
            assert len(result.result) == 3
        finally:
            engine.teardown("s1")

    def test_call_id_and_name_propagated(self):
        engine = _make_engine(_CORPUS_CANONICAL)
        engine.setup("s1")
        try:
            tc = ToolCall(name="ns::search", arguments={}, call_id="my-call-id")
            [result] = engine.execute("s1", [tc])
            assert result.call_id == "my-call-id"
            assert result.name == "ns::search"
        finally:
            engine.teardown("s1")

    def test_simulate_returns_same_count_as_execute(self):
        engine = _make_engine(_CORPUS_CANONICAL, limit=4)
        engine.setup("s1")
        try:
            tc = _tc(size=4)
            [r_sim] = engine.simulate("s1", [tc])
            [r_exe] = engine.execute("s1", [tc])
            assert len(r_sim.result) == len(r_exe.result)
        finally:
            engine.teardown("s1")

    def test_score_is_none_for_file_engine(self):
        engine = _make_engine(_CORPUS_CANONICAL)
        engine.setup("s1")
        try:
            [result] = engine.execute("s1", [_tc(size=3)])
            for doc in result.result:
                assert "score" not in doc
        finally:
            engine.teardown("s1")


# ---------------------------------------------------------------------------
# FileSearchEngine — projection
# ---------------------------------------------------------------------------


class TestFileSearchEngineProjection:
    def test_projection_remaps_body_to_text_and_doc_id_to_id(self):
        engine = _make_engine(
            _CORPUS_PROJECTED,
            projection={"body": "text", "doc_id": "id"},
        )
        engine.setup("s1")
        try:
            [result] = engine.execute("s1", [_tc(size=_PROJECTED_SIZE)])
            for doc in result.result:
                assert "text" in doc
                assert "id" in doc
        finally:
            engine.teardown("s1")

    def test_projection_ids_match_doc_id_field(self):
        engine = _make_engine(
            _CORPUS_PROJECTED,
            projection={"body": "text", "doc_id": "id"},
        )
        engine.setup("s1")
        try:
            [result] = engine.execute("s1", [_tc(size=_PROJECTED_SIZE)])
            returned_ids = {doc["id"] for doc in result.result}
            expected_ids = {f"p_{i}" for i in range(_PROJECTED_SIZE)}
            assert returned_ids == expected_ids
        finally:
            engine.teardown("s1")

    def test_projection_text_comes_from_body_field(self):
        engine = _make_engine(
            _CORPUS_PROJECTED,
            projection={"body": "text", "doc_id": "id"},
        )
        engine.setup("s1")
        try:
            [result] = engine.execute("s1", [_tc(size=_PROJECTED_SIZE)])
            for doc in result.result:
                assert len(doc["text"]) > 10
        finally:
            engine.teardown("s1")

    def test_unprojected_fields_land_in_metadata(self):
        engine = _make_engine(
            _CORPUS_PROJECTED,
            projection={"body": "text", "doc_id": "id"},
        )
        engine.setup("s1")
        try:
            [result] = engine.execute("s1", [_tc(size=_PROJECTED_SIZE)])
            for doc in result.result:
                assert "metadata" in doc
                meta = doc["metadata"]
                assert "source_url" in meta
                assert "category" in meta
                assert meta["source_url"].startswith("https://example.com/")
        finally:
            engine.teardown("s1")

    def test_no_projection_uses_id_and_text_verbatim(self):
        engine = _make_engine(_CORPUS_CANONICAL)
        engine.setup("s1")
        try:
            [result] = engine.execute("s1", [_tc(size=3)])
            for doc in result.result:
                assert doc["id"].startswith("doc_")
                assert "water cycle" in doc["text"] or len(doc["text"]) > 10
        finally:
            engine.teardown("s1")


# ---------------------------------------------------------------------------
# FileSearchEngine — error injection
# ---------------------------------------------------------------------------


class TestFileSearchEngineErrorInjection:
    def test_network_error_populates_error_field(self):
        engine = _make_engine(
            _CORPUS_CANONICAL,
            error_categories=[
                ErrorCategory(type="network_error", probability=1.0, message="timeout")
            ],
        )
        engine.setup("s1")
        try:
            [result] = engine.execute("s1", [_tc(call_id="ce")])
            assert result.error == "timeout"
            assert result.result is None
            assert result.call_id == "ce"
            assert result.name == "ns::search"
        finally:
            engine.teardown("s1")

    def test_empty_result_error_returns_empty_list(self):
        engine = _make_engine(
            _CORPUS_CANONICAL,
            error_categories=[ErrorCategory(type="empty_result", probability=1.0)],
        )
        engine.setup("s1")
        try:
            [result] = engine.execute("s1", [_tc()])
            assert result.error is None
            assert result.result == []
        finally:
            engine.teardown("s1")

    def test_unparseable_result_error_returns_garbled(self):
        engine = _make_engine(
            _CORPUS_CANONICAL,
            error_categories=[ErrorCategory(type="unparseable_result", probability=1.0)],
        )
        engine.setup("s1")
        try:
            [result] = engine.execute("s1", [_tc()])
            assert result.result == "<garbled>"
        finally:
            engine.teardown("s1")

    def test_zero_probability_never_fires(self):
        engine = _make_engine(
            _CORPUS_CANONICAL,
            error_categories=[ErrorCategory(type="network_error", probability=0.0)],
            limit=2,
        )
        engine.setup("s1")
        try:
            for _ in range(20):
                [result] = engine.execute("s1", [_tc()])
                assert result.error is None
        finally:
            engine.teardown("s1")


# ---------------------------------------------------------------------------
# FileSearchEngine — relevance threshold
# ---------------------------------------------------------------------------


class TestFileSearchEngineRelevanceThreshold:
    def test_threshold_does_not_filter_scoreless_docs(self):
        # FileSearchEngine returns score=None — per design, None scores pass
        # the threshold check regardless of the configured threshold value.
        engine = _make_engine(_CORPUS_CANONICAL, relevance_threshold=0.9, limit=5)
        engine.setup("s1")
        try:
            [result] = engine.execute("s1", [_tc()])
            assert len(result.result) == 5
        finally:
            engine.teardown("s1")


# ---------------------------------------------------------------------------
# RandomDocumentSampler
# ---------------------------------------------------------------------------


class TestRandomDocumentSampler:
    def test_sample_returns_k_documents(self):
        engine = _make_engine(_CORPUS_CANONICAL)
        sampler = RandomDocumentSampler(engine)
        engine.setup("s1")
        try:
            docs = sampler.sample(session_id="s1", k=4)
            assert len(docs) == 4
            assert all(isinstance(d, Document) for d in docs)
        finally:
            engine.teardown("s1")

    def test_sample_docs_have_non_empty_id_and_text(self):
        engine = _make_engine(_CORPUS_CANONICAL)
        sampler = RandomDocumentSampler(engine)
        engine.setup("s1")
        try:
            for doc in sampler.sample(session_id="s1", k=5):
                assert doc.id
                assert doc.text
        finally:
            engine.teardown("s1")

    def test_sample_ids_are_from_corpus(self):
        engine = _make_engine(_CORPUS_CANONICAL)
        sampler = RandomDocumentSampler(engine)
        engine.setup("s1")
        try:
            docs = sampler.sample(session_id="s1", k=_CANONICAL_SIZE)
            returned_ids = {d.id for d in docs}
            expected_ids = {f"doc_{i}" for i in range(_CANONICAL_SIZE)}
            assert returned_ids == expected_ids
        finally:
            engine.teardown("s1")

    def test_sample_ignores_query_arg(self):
        engine = _make_engine(_CORPUS_CANONICAL)
        sampler = RandomDocumentSampler(engine)
        engine.setup("s1")
        try:
            docs = sampler.sample(session_id="s1", k=3, query="this query should be ignored")
            assert len(docs) == 3
        finally:
            engine.teardown("s1")

    def test_sample_k_larger_than_corpus_returns_full_corpus(self):
        engine = _make_engine(_CORPUS_CANONICAL)
        sampler = RandomDocumentSampler(engine)
        engine.setup("s1")
        try:
            docs = sampler.sample(session_id="s1", k=999)
            assert len(docs) == _CANONICAL_SIZE
        finally:
            engine.teardown("s1")

    def test_sample_with_projection_returns_remapped_fields(self):
        engine = _make_engine(
            _CORPUS_PROJECTED,
            projection={"body": "text", "doc_id": "id"},
        )
        sampler = RandomDocumentSampler(engine)
        engine.setup("s1")
        try:
            docs = sampler.sample(session_id="s1", k=_PROJECTED_SIZE)
            for doc in docs:
                assert doc.id.startswith("p_")
                assert len(doc.text) > 10
        finally:
            engine.teardown("s1")

    def test_sample_produces_distinct_draws(self):
        # With 20 docs sampling 5, P(two identical draws) ≈ 0.3% — safe enough.
        engine = _make_engine(_CORPUS_CANONICAL)
        sampler = RandomDocumentSampler(engine)
        engine.setup("s1")
        try:
            ids_1 = frozenset(d.id for d in sampler.sample(session_id="s1", k=5))
            ids_2 = frozenset(d.id for d in sampler.sample(session_id="s1", k=5))
            assert ids_1 != ids_2
        finally:
            engine.teardown("s1")
