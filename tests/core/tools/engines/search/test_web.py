# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for web search engines and supporting utilities.

Unit tests run without any network access — ``fetch_url_raw`` is mocked
wherever HTTP calls would otherwise be made.

Integration tests are marked ``@pytest.mark.integration`` and skipped by
default.  Run them explicitly::

    source .venv/bin/activate
    pytest tests/core/tools/engines/search/test_web.py -v --integration
"""

# Standard
from unittest.mock import MagicMock, patch

# Third Party
import pytest

# Local
from fms_dgt.core.tools.data_objects import Tool, ToolCall
from fms_dgt.core.tools.engines.search.web.duckduckgo import DuckDuckGoSearchEngine
from fms_dgt.core.tools.engines.search.web.utils import (
    _build_document,
    _validate_content,
    extract_text,
)
from fms_dgt.core.tools.registry import ToolRegistry

# ---------------------------------------------------------------------------
# Minimal HTML pages used across tests — no network needed
# ---------------------------------------------------------------------------

_SIMPLE_HTML = """<!DOCTYPE html>
<html>
<head><title>Mount Rushmore</title></head>
<body>
  <nav>Site navigation</nav>
  <article>
    <h1>Mount Rushmore National Memorial</h1>
    <p>Mount Rushmore features the carved faces of four United States presidents:
    George Washington, Thomas Jefferson, Theodore Roosevelt, and Abraham Lincoln.
    The sculpture was carved between 1927 and 1941 by sculptor Gutzon Borglum
    and his son Lincoln Borglum.</p>
    <p>The memorial is located in the Black Hills of South Dakota and attracts
    approximately three million visitors each year.</p>
  </article>
  <footer>Copyright example.com</footer>
</body>
</html>"""

_BOILERPLATE_ONLY_HTML = """<!DOCTYPE html>
<html>
<head><title>Empty</title></head>
<body>
  <nav>Home | About | Contact</nav>
  <footer>© 2024 Example Corp. All rights reserved. Privacy Policy Terms of Use.</footer>
</body>
</html>"""

# Snippet returned by DDG for the same topic
_SNIPPET = "Mount Rushmore features the faces of four US presidents."

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_registry() -> ToolRegistry:
    return ToolRegistry(tools=[Tool(name="search", namespace="ns", description="Search")])


def _make_ddg_engine(**kwargs) -> DuckDuckGoSearchEngine:
    return DuckDuckGoSearchEngine(_make_registry(), **kwargs)


def _tc(query: str = "mount rushmore faces", call_id: str = "c1", **extra) -> ToolCall:
    args = {"query": query, **extra}
    return ToolCall(name="ns::search", arguments=args, call_id=call_id)


# ---------------------------------------------------------------------------
# _validate_content
# ---------------------------------------------------------------------------


class TestValidateContent:
    def test_valid_single(self):
        assert _validate_content(["snippet"]) == ["snippet"]

    def test_valid_multiple(self):
        assert _validate_content(["snippet", "text", "raw"]) == ["snippet", "text", "raw"]

    def test_deduplicates_preserving_order(self):
        assert _validate_content(["text", "snippet", "text"]) == ["text", "snippet"]

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown content type"):
            _validate_content(["snippet", "html"])

    def test_empty_list_is_valid(self):
        assert _validate_content([]) == []


# ---------------------------------------------------------------------------
# extract_text
# ---------------------------------------------------------------------------


class TestExtractText:
    def test_extracts_article_content(self):
        text = extract_text(_SIMPLE_HTML)
        assert text is not None
        assert "Mount Rushmore" in text
        assert "four" in text.lower()

    def test_excludes_nav_and_footer_boilerplate(self):
        text = extract_text(_SIMPLE_HTML)
        assert text is not None
        assert "Site navigation" not in text
        assert "Copyright example.com" not in text

    def test_returns_none_for_boilerplate_only_page(self):
        text = extract_text(_BOILERPLATE_ONLY_HTML)
        # trafilatura may return None or a very short string for nav/footer-only pages
        assert text is None or len(text) < 50

    def test_returns_none_for_empty_string(self):
        assert extract_text("") is None


# ---------------------------------------------------------------------------
# _build_document — snippet only (no network)
# ---------------------------------------------------------------------------


class TestBuildDocumentSnippetOnly:
    def test_snippet_only_default(self):
        doc = _build_document(
            idx=0,
            snippet=_SNIPPET,
            url="https://example.com/rushmore",
            title="Mount Rushmore",
            content=["snippet"],
        )
        assert doc.text == _SNIPPET
        assert doc.metadata["snippet"] == _SNIPPET
        assert doc.metadata["source"] == "https://example.com/rushmore"
        assert doc.metadata["title"] == "Mount Rushmore"
        assert "raw" not in doc.metadata
        assert "text" not in doc.metadata

    def test_snippet_populates_id_from_idx(self):
        doc = _build_document(
            idx=3, snippet=_SNIPPET, url="https://x.com", title="X", content=["snippet"]
        )
        assert doc.id == "3"

    def test_empty_snippet_falls_back_gracefully(self):
        doc = _build_document(
            idx=0, snippet="", url="https://x.com", title="X", content=["snippet"]
        )
        assert doc.text == ""


# ---------------------------------------------------------------------------
# _build_document — with raw and text (fetch_url_raw mocked)
# ---------------------------------------------------------------------------


class TestBuildDocumentWithFetch:
    def test_raw_content_stored_in_metadata(self):
        with patch(
            "fms_dgt.core.tools.engines.search.web.utils.fetch_url_raw",
            return_value=_SIMPLE_HTML,
        ):
            doc = _build_document(
                idx=0,
                snippet=_SNIPPET,
                url="https://example.com",
                title="T",
                content=["snippet", "raw"],
            )
        assert "raw" in doc.metadata
        assert doc.metadata["raw"] == _SIMPLE_HTML

    def test_text_extraction_stored_in_metadata(self):
        with patch(
            "fms_dgt.core.tools.engines.search.web.utils.fetch_url_raw",
            return_value=_SIMPLE_HTML,
        ):
            doc = _build_document(
                idx=0,
                snippet=_SNIPPET,
                url="https://example.com",
                title="T",
                content=["snippet", "text"],
            )
        assert "text" in doc.metadata
        assert "Mount Rushmore" in doc.metadata["text"]

    def test_text_wins_over_snippet_in_document_text(self):
        with patch(
            "fms_dgt.core.tools.engines.search.web.utils.fetch_url_raw",
            return_value=_SIMPLE_HTML,
        ):
            doc = _build_document(
                idx=0,
                snippet=_SNIPPET,
                url="https://example.com",
                title="T",
                content=["snippet", "text"],
            )
        # text > snippet in priority order
        assert doc.text == doc.metadata["text"]
        assert doc.text != _SNIPPET

    def test_snippet_wins_when_text_extraction_returns_none(self):
        with (
            patch(
                "fms_dgt.core.tools.engines.search.web.utils.fetch_url_raw",
                return_value=_SIMPLE_HTML,
            ),
            patch(
                "fms_dgt.core.tools.engines.search.web.utils.extract_text",
                return_value=None,
            ),
        ):
            doc = _build_document(
                idx=0,
                snippet=_SNIPPET,
                url="https://example.com",
                title="T",
                content=["snippet", "text"],
            )
        # extract_text returned None — snippet is the fallback
        assert doc.text == _SNIPPET
        assert "text" not in doc.metadata

    def test_snippet_fallback_when_fetch_fails(self):
        with patch(
            "fms_dgt.core.tools.engines.search.web.utils.fetch_url_raw",
            return_value=None,
        ):
            doc = _build_document(
                idx=0,
                snippet=_SNIPPET,
                url="https://example.com",
                title="T",
                content=["snippet", "text", "raw"],
            )
        assert doc.text == _SNIPPET
        assert "raw" not in doc.metadata
        assert "text" not in doc.metadata

    def test_all_three_content_types(self):
        with patch(
            "fms_dgt.core.tools.engines.search.web.utils.fetch_url_raw",
            return_value=_SIMPLE_HTML,
        ):
            doc = _build_document(
                idx=0,
                snippet=_SNIPPET,
                url="https://example.com",
                title="T",
                content=["snippet", "text", "raw"],
            )
        assert "snippet" in doc.metadata
        assert "text" in doc.metadata
        assert "raw" in doc.metadata
        # text is highest fidelity
        assert doc.text == doc.metadata["text"]


# ---------------------------------------------------------------------------
# DuckDuckGoSearchEngine — constructor and content validation
# ---------------------------------------------------------------------------


class TestDuckDuckGoEngineConstructor:
    def test_default_content_is_snippet(self):
        engine = _make_ddg_engine()
        assert engine._content == ["snippet"]

    def test_custom_content_stored(self):
        engine = _make_ddg_engine(content=["snippet", "text"])
        assert engine._content == ["snippet", "text"]

    def test_invalid_content_raises_at_construction(self):
        with pytest.raises(ValueError, match="Unknown content type"):
            _make_ddg_engine(content=["snippet", "html"])


# ---------------------------------------------------------------------------
# DuckDuckGoSearchEngine — execute with mocked DDGS
# ---------------------------------------------------------------------------

_DDG_RAW_RESULTS = [
    {
        "title": "Mount Rushmore - Wikipedia",
        "href": "https://en.wikipedia.org/wiki/Mount_Rushmore",
        "body": _SNIPPET,
    },
    {
        "title": "Mount Rushmore History",
        "href": "https://example.com/history",
        "body": "The sculpture was carved between 1927 and 1941.",
    },
]


class TestDuckDuckGoEngineExecute:
    def _run(self, engine, tc):
        engine.setup("s1")
        try:
            return engine.execute("s1", [tc])
        finally:
            engine.teardown("s1")

    def _mock_ddgs(self, results=None):
        mock = MagicMock()
        mock.text.return_value = results if results is not None else _DDG_RAW_RESULTS
        return mock

    def test_returns_one_result_per_call(self):
        engine = _make_ddg_engine(limit=2)
        engine._ddgs = self._mock_ddgs()
        [result] = self._run(engine, _tc())
        assert result.error is None
        assert isinstance(result.result, list)
        assert len(result.result) == 2

    def test_each_document_has_id_text_and_source(self):
        engine = _make_ddg_engine()
        engine._ddgs = self._mock_ddgs()
        [result] = self._run(engine, _tc())
        for doc in result.result:
            assert "id" in doc
            assert "text" in doc
            assert "metadata" in doc
            assert "source" in doc["metadata"]
            assert "title" in doc["metadata"]

    def test_snippet_content_populates_text_and_metadata(self):
        engine = _make_ddg_engine(content=["snippet"])
        engine._ddgs = self._mock_ddgs()
        [result] = self._run(engine, _tc())
        first = result.result[0]
        assert first["text"] == _SNIPPET
        assert first["metadata"]["snippet"] == _SNIPPET

    def test_per_call_content_override(self):
        engine = _make_ddg_engine(content=["snippet"])
        engine._ddgs = self._mock_ddgs()
        with patch(
            "fms_dgt.core.tools.engines.search.web.utils.fetch_url_raw",
            return_value=_SIMPLE_HTML,
        ):
            [result] = self._run(engine, _tc(content=["snippet", "text"]))
        first = result.result[0]
        assert "text" in first["metadata"]
        assert "Mount Rushmore" in first["metadata"]["text"]

    def test_empty_query_returns_empty_list(self):
        engine = _make_ddg_engine()
        engine._ddgs = self._mock_ddgs()
        [result] = self._run(
            engine, ToolCall(name="ns::search", arguments={"query": ""}, call_id="c0")
        )
        assert result.result == []

    def test_call_id_and_name_propagated(self):
        engine = _make_ddg_engine()
        engine._ddgs = self._mock_ddgs()
        [result] = self._run(engine, _tc(call_id="my-id"))
        assert result.call_id == "my-id"
        assert result.name == "ns::search"

    def test_ddg_exception_raises_runtime_error(self):
        # Third Party
        from ddgs.exceptions import DDGSException

        engine = _make_ddg_engine()
        mock = MagicMock()
        mock.text.side_effect = DDGSException("rate limited")
        engine._ddgs = mock
        engine.setup("s1")
        try:
            with pytest.raises(RuntimeError, match="DuckDuckGo search failed"):
                engine._search({"query": "test"}, limit=3)
        finally:
            engine.teardown("s1")


# ---------------------------------------------------------------------------
# DuckDuckGoSearchEngine — live network call
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestDuckDuckGoIntegration:
    def test_real_search_returns_results(self):
        engine = _make_ddg_engine(limit=3, content=["snippet"])
        engine.setup("s1")
        try:
            tc = _tc("how many faces on mount rushmore")
            [result] = engine.execute("s1", [tc])
            assert result.error is None
            # DDG may return 0 results under rate limiting — tolerate but verify shape
            for doc in result.result:
                assert doc["text"]
                assert doc["metadata"]["source"].startswith("http")
        finally:
            engine.teardown("s1")

    def test_real_search_with_full_text(self):
        engine = _make_ddg_engine(limit=2, content=["snippet", "text"])
        engine.setup("s1")
        try:
            tc = _tc("mount rushmore south dakota")
            [result] = engine.execute("s1", [tc])
            assert result.error is None
            # At least one result should have extracted text longer than snippet
            texts = [
                d["metadata"].get("text") for d in result.result if "text" in d.get("metadata", {})
            ]
            if texts:
                assert any(len(t) > 200 for t in texts)
        finally:
            engine.teardown("s1")
