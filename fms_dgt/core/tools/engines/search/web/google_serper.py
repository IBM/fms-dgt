# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any, Dict, List, Optional
import logging
import os

# Third Party
import requests

# Local
from fms_dgt.core.tools.engines.base import register_tool_engine
from fms_dgt.core.tools.engines.search.base import Document, SearchToolEngine
from fms_dgt.core.tools.engines.search.web.utils import (
    _build_document,
    _validate_content,
)

logger = logging.getLogger(__name__)

_SERPER_URL = "https://google.serper.dev/search"


@register_tool_engine("search/web/google_serper")
class GoogleSerperSearchEngine(SearchToolEngine):
    """Google Search via the Serper.dev API.

    Requires a ``SERPER_API_KEY`` environment variable or ``serper_api_key``
    constructor argument.  Sign up at https://serper.dev for a free-tier key.

    ``setup``/``teardown`` are no-ops — this engine is stateless.

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
        serper_api_key: API key.  Falls back to ``SERPER_API_KEY`` env var.
        limit: Default number of organic results to return per call.
        content: Default content types to fetch.  Defaults to ``["snippet"]``.
        gl: Google country code (e.g. ``"us"``).
        hl: Google language code (e.g. ``"en"``).
    """

    def __init__(
        self,
        registry,
        serper_api_key: Optional[str] = None,
        limit: int = 5,
        content: Optional[List[str]] = None,
        gl: str = "us",
        hl: str = "en",
        **kwargs: Any,
    ) -> None:
        super().__init__(registry, **kwargs)
        key = serper_api_key or os.environ.get("SERPER_API_KEY")
        if not key:
            raise ValueError(
                "GoogleSerperSearchEngine requires a Serper API key. "
                "Set the SERPER_API_KEY environment variable or pass serper_api_key."
            )
        self._api_key = key
        self._limit = limit
        self._content = _validate_content(content or ["snippet"])
        self._gl = gl
        self._hl = hl

    # ------------------------------------------------------------------
    # SearchToolEngine contract
    # ------------------------------------------------------------------

    def _search(self, arguments: Dict[str, Any], limit: int, **kwargs: Any) -> List[Document]:
        query = arguments.get("query", "")
        if not query:
            return []

        content = _validate_content(arguments.get("content") or self._content)

        headers = {
            "X-API-KEY": self._api_key,
            "Content-Type": "application/json",
        }
        params = {"q": query, "gl": self._gl, "hl": self._hl, "num": limit}

        try:
            resp = requests.get(_SERPER_URL, headers=headers, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            raise RuntimeError(f"Google Serper request failed: {exc}") from exc

        organic = sorted(data.get("organic", []), key=lambda x: x.get("position", 999))
        return [
            _build_document(
                idx=i,
                snippet=item.get("snippet", ""),
                url=item.get("link", ""),
                title=item.get("title", ""),
                content=content,
            )
            for i, item in enumerate(organic[:limit])
        ]

    def _default_limit(self) -> int:
        return self._limit
