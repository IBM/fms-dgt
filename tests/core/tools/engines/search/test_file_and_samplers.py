# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for FileSearchEngine and RandomDocumentSampler.

Corpus files live in test_data/ alongside this module:
  corpus_canonical.jsonl  — standard {id, text} fields, 20 documents
  corpus_projected.jsonl  — non-standard {doc_id, body, source_url, category} fields, 10 documents
"""

# Standard
from pathlib import Path
from unittest.mock import MagicMock
import random as _random

# Third Party
import pytest

# Local
from fms_dgt.core.tools.data_objects import Tool, ToolCall
from fms_dgt.core.tools.engines.base import ErrorCategory
from fms_dgt.core.tools.engines.search.base import Document, SearchToolEngine
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


# ---------------------------------------------------------------------------
# FileSearchEngine.corpus()
# ---------------------------------------------------------------------------


class TestFileSearchEngineCorpus:
    def test_corpus_returns_all_documents(self):
        engine = _make_engine(_CORPUS_CANONICAL)
        docs = engine.corpus()
        assert len(docs) == _CANONICAL_SIZE
        assert all(isinstance(d, Document) for d in docs)

    def test_corpus_ids_cover_full_set(self):
        engine = _make_engine(_CORPUS_CANONICAL)
        ids = {d.id for d in engine.corpus()}
        assert ids == {f"doc_{i}" for i in range(_CANONICAL_SIZE)}

    def test_corpus_loads_without_prior_setup(self):
        # corpus() must trigger lazy load even if setup() was never called.
        engine = _make_engine(_CORPUS_CANONICAL)
        docs = engine.corpus()
        assert len(docs) == _CANONICAL_SIZE

    def test_corpus_with_projection_remaps_fields(self):
        engine = _make_engine(_CORPUS_PROJECTED, projection={"body": "text", "doc_id": "id"})
        docs = engine.corpus()
        assert len(docs) == _PROJECTED_SIZE
        for doc in docs:
            assert doc.id.startswith("p_")
            assert len(doc.text) > 10

    def test_corpus_metadata_preserved(self):
        engine = _make_engine(_CORPUS_PROJECTED, projection={"body": "text", "doc_id": "id"})
        docs = engine.corpus()
        for doc in docs:
            assert "category" in doc.metadata
            assert "source_url" in doc.metadata

    def test_corpus_returns_independent_list(self):
        # Mutating the returned list must not affect subsequent calls.
        engine = _make_engine(_CORPUS_CANONICAL)
        docs1 = engine.corpus()
        docs1.clear()
        docs2 = engine.corpus()
        assert len(docs2) == _CANONICAL_SIZE


# ---------------------------------------------------------------------------
# RandomDocumentSampler — grouped sampling
# ---------------------------------------------------------------------------

_PROJECTED_CATEGORIES = {"biology", "earth-science", "physics", "chemistry"}
_BIOLOGY_IDS = {"p_1", "p_2", "p_6", "p_9"}
_EARTH_SCIENCE_IDS = {"p_0", "p_3", "p_8"}
_PHYSICS_IDS = {"p_4", "p_5"}
_CHEMISTRY_IDS = {"p_7"}


def _make_grouped_sampler(strategy="uniform") -> RandomDocumentSampler:
    engine = _make_engine(_CORPUS_PROJECTED, projection={"body": "text", "doc_id": "id"})
    return RandomDocumentSampler(engine, group_by="category", strategy=strategy)


class TestRandomDocumentSamplerGrouped:
    def test_group_pools_built_at_construction(self):
        sampler = _make_grouped_sampler()
        assert sampler._group_pools is not None
        assert set(sampler._group_pools.keys()) == _PROJECTED_CATEGORIES

    def test_group_pool_sizes_match_corpus(self):
        sampler = _make_grouped_sampler()
        assert len(sampler._group_pools["biology"]) == 4
        assert len(sampler._group_pools["earth-science"]) == 3
        assert len(sampler._group_pools["physics"]) == 2
        assert len(sampler._group_pools["chemistry"]) == 1

    def test_uniform_strategy_equal_probabilities(self):
        sampler = _make_grouped_sampler(strategy="uniform")
        probs = sampler._group_probs
        assert set(probs.keys()) == _PROJECTED_CATEGORIES
        for p in probs.values():
            assert abs(p - 0.25) < 1e-9

    def test_proportional_strategy_weights_by_size(self):
        sampler = _make_grouped_sampler(strategy="proportional")
        probs = sampler._group_probs
        # biology=4, earth-science=3, physics=2, chemistry=1 → total=10
        assert abs(probs["biology"] - 0.4) < 1e-9
        assert abs(probs["earth-science"] - 0.3) < 1e-9
        assert abs(probs["physics"] - 0.2) < 1e-9
        assert abs(probs["chemistry"] - 0.1) < 1e-9

    def test_sample_returns_single_group_documents(self):
        sampler = _make_grouped_sampler()
        for _ in range(30):
            docs = sampler.sample(session_id="s1", k=2)
            categories = {d.metadata["category"] for d in docs}
            assert len(categories) == 1, f"Mixed categories in sample: {categories}"

    def test_sample_k_capped_at_group_size(self):
        # chemistry has only 1 doc; requesting k=5 should return 1.
        sampler = _make_grouped_sampler()
        # Seed random to always pick the chemistry group (size=1).
        _random.seed(0)
        for _ in range(50):
            docs = sampler.sample(session_id="s1", k=5)
            category = docs[0].metadata["category"]
            pool_size = len(sampler._group_pools[category])
            assert len(docs) <= pool_size

    def test_sample_ids_belong_to_selected_group(self):
        sampler = _make_grouped_sampler()
        group_id_sets = {
            "biology": _BIOLOGY_IDS,
            "earth-science": _EARTH_SCIENCE_IDS,
            "physics": _PHYSICS_IDS,
            "chemistry": _CHEMISTRY_IDS,
        }
        for _ in range(20):
            docs = sampler.sample(session_id="s1", k=2)
            category = docs[0].metadata["category"]
            for doc in docs:
                assert doc.id in group_id_sets[category]

    def test_no_group_by_falls_back_to_engine_path(self):
        engine = _make_engine(_CORPUS_CANONICAL)
        sampler = RandomDocumentSampler(engine)
        assert sampler._group_pools is None
        assert sampler._group_probs is None
        engine.setup("s1")
        try:
            docs = sampler.sample(session_id="s1", k=3)
            assert len(docs) == 3
        finally:
            engine.teardown("s1")

    def test_group_by_unknown_field_buckets_to_empty_string(self):
        # When the group_by field is absent from all docs the sampler must not
        # crash — all documents land in a single "" bucket.
        engine = _make_engine(_CORPUS_CANONICAL)  # canonical corpus has no "category" field
        sampler = RandomDocumentSampler(engine, group_by="category")
        assert list(sampler._group_pools.keys()) == [""]
        docs = sampler.sample(session_id="s1", k=3)
        assert len(docs) == 3

    def test_corpus_called_on_non_enumerable_engine_raises(self):
        engine = MagicMock(spec=SearchToolEngine)
        engine.corpus = MagicMock(side_effect=NotImplementedError("no corpus"))
        with pytest.raises(NotImplementedError):
            RandomDocumentSampler(engine, group_by="domain")


# ---------------------------------------------------------------------------
# RandomDocumentSampler — exclude_groups / include_groups
# ---------------------------------------------------------------------------


class TestRandomDocumentSamplerGroupFilter:
    def _sampler(self, strategy="uniform") -> RandomDocumentSampler:
        engine = _make_engine(_CORPUS_PROJECTED, projection={"body": "text", "doc_id": "id"})
        return RandomDocumentSampler(engine, group_by="category", strategy=strategy)

    def test_exclude_groups_never_selected(self):
        sampler = self._sampler()
        for _ in range(50):
            docs = sampler.sample(session_id="s1", k=2, exclude_groups=["biology"])
            categories = {d.metadata["category"] for d in docs}
            assert "biology" not in categories

    def test_exclude_multiple_groups(self):
        sampler = self._sampler()
        excluded = ["biology", "earth-science"]
        for _ in range(50):
            docs = sampler.sample(session_id="s1", k=2, exclude_groups=excluded)
            categories = {d.metadata["category"] for d in docs}
            assert not categories.intersection(excluded)

    def test_include_groups_only_selected(self):
        sampler = self._sampler()
        for _ in range(30):
            docs = sampler.sample(session_id="s1", k=2, include_groups=["physics"])
            categories = {d.metadata["category"] for d in docs}
            assert categories == {"physics"}

    def test_include_multiple_groups_restricted_to_those(self):
        sampler = self._sampler()
        allowed = {"biology", "chemistry"}
        for _ in range(30):
            docs = sampler.sample(session_id="s1", k=2, include_groups=list(allowed))
            categories = {d.metadata["category"] for d in docs}
            assert categories.issubset(allowed)

    def test_include_and_exclude_together_raises(self):
        sampler = self._sampler()
        with pytest.raises(ValueError, match="mutually exclusive"):
            sampler.sample(
                session_id="s1",
                k=2,
                include_groups=["biology"],
                exclude_groups=["physics"],
            )

    def test_exclude_all_groups_raises(self):
        sampler = self._sampler()
        with pytest.raises(ValueError, match="no groups remain"):
            sampler.sample(
                session_id="s1",
                k=2,
                exclude_groups=list(_PROJECTED_CATEGORIES),
            )

    def test_include_unknown_group_raises(self):
        sampler = self._sampler()
        with pytest.raises(ValueError, match="unknown group"):
            sampler.sample(session_id="s1", k=2, include_groups=["nonexistent"])

    def test_exclude_does_not_mutate_instance_state(self):
        sampler = self._sampler()
        original_pools = dict(sampler._group_pools)
        original_probs = dict(sampler._group_probs)
        sampler.sample(session_id="s1", k=2, exclude_groups=["biology"])
        assert sampler._group_pools == original_pools
        assert sampler._group_probs == original_probs

    def test_include_does_not_mutate_instance_state(self):
        sampler = self._sampler()
        original_pools = dict(sampler._group_pools)
        original_probs = dict(sampler._group_probs)
        sampler.sample(session_id="s1", k=2, include_groups=["physics"])
        assert sampler._group_pools == original_pools
        assert sampler._group_probs == original_probs

    def test_no_filter_still_works_after_filtered_call(self):
        sampler = self._sampler()
        sampler.sample(session_id="s1", k=2, exclude_groups=["biology"])
        # Next call without filter should still have access to all groups.
        all_seen = set()
        for _ in range(100):
            docs = sampler.sample(session_id="s1", k=2)
            all_seen.update(d.metadata["category"] for d in docs)
        assert "biology" in all_seen

    def test_exclude_on_non_grouped_sampler_has_no_effect(self):
        engine = _make_engine(_CORPUS_CANONICAL)
        sampler = RandomDocumentSampler(engine)
        engine.setup("s1")
        try:
            docs = sampler.sample(session_id="s1", k=3, exclude_groups=["anything"])
            assert len(docs) == 3
        finally:
            engine.teardown("s1")
