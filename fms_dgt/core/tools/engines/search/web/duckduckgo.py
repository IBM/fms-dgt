# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any, Dict, List, Optional
import logging

# Local
from fms_dgt.core.tools.engines.base import register_tool_engine
from fms_dgt.core.tools.engines.search.base import Document, SearchToolEngine
from fms_dgt.core.tools.engines.search.web.utils import (
    _build_document,
    _validate_content,
)

logger = logging.getLogger(__name__)


@register_tool_engine("search/web/duckduckgo")
class DuckDuckGoSearchEngine(SearchToolEngine):
    """Web search via DuckDuckGo (no API key required).

    Rate-limited by DuckDuckGo's informal policy.  Appropriate for open-domain
    web search recipes during development.  ``setup``/``teardown`` are no-ops
    since this engine is stateless.

    Requires ``duckduckgo-search`` (``pip install fms_dgt[search]``).

    **Content types** — the ``content`` parameter controls what each result
    document contains.  Pass a list of one or more of:

    - ``"snippet"`` — short excerpt from the search engine (default; no extra fetch)
    - ``"text"`` — full page text extracted via trafilatura (requires
      ``pip install fms_dgt[search-full-text]``)
    - ``"raw"`` — raw HTML of the linked page

    ``Document.text`` is populated with the highest-fidelity type successfully
    retrieved: ``text`` > ``snippet`` > ``raw``.  Each requested type is also
    stored individually in ``Document.metadata`` under its type name.

    The per-call ``arguments`` dict may include a ``"content"`` key to override
    the engine default for that call.

    Args:
        registry: Shared ``ToolRegistry``.
        limit: Default number of results per call.
        content: Default content types to fetch.  Defaults to ``["snippet"]``.
        region: DuckDuckGo region code.  Defaults to ``"us-en"``.
        safesearch: ``"off"``, ``"moderate"``, or ``"on"``.
        timelimit: Time filter — ``"d"``, ``"w"``, ``"m"``, ``"y"``, or ``None``.
    """

    def __init__(
        self,
        registry,
        limit: int = 5,
        content: Optional[List[str]] = None,
        region: str = "us-en",
        safesearch: str = "off",
        timelimit: str = "y",
        **kwargs: Any,
    ) -> None:
        super().__init__(registry, **kwargs)
        self._limit = limit
        self._content = _validate_content(content or ["snippet"])
        self._region = region
        self._safesearch = safesearch
        self._timelimit = timelimit
        self._ddgs = None

    # ------------------------------------------------------------------
    # SearchToolEngine contract
    # ------------------------------------------------------------------

    def _search(self, arguments: Dict[str, Any], limit: int, **kwargs: Any) -> List[Document]:
        try:
            # Third Party
            from ddgs import DDGS
            from ddgs.exceptions import DDGSException
        except ImportError as exc:
            raise ImportError(
                "DuckDuckGoSearchEngine requires 'ddgs'. "
                "Install with: pip install fms_dgt[search]"
            ) from exc

        if self._ddgs is None:
            self._ddgs = DDGS()

        query = arguments.get("query", "")
        if not query:
            return []

        content = _validate_content(arguments.get("content") or self._content)

        try:
            raw_results = self._ddgs.text(
                query,
                region=self._region,
                safesearch=self._safesearch,
                timelimit=self._timelimit,
                max_results=limit,
            )
        except DDGSException as exc:
            raise RuntimeError(f"DuckDuckGo search failed: {exc}") from exc

        return [
            _build_document(
                idx=i,
                snippet=result.get("body", ""),
                url=result.get("href", ""),
                title=result.get("title", ""),
                content=content,
            )
            for i, result in enumerate(raw_results or [])
        ]

    def _default_limit(self) -> int:
        return self._limit
