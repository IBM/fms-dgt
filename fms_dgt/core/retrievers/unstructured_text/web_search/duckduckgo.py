# Standard
from typing import Any, Dict, List

# Third Party
from ddgs import DDGS
from ddgs.exceptions import DDGSException
import aiohttp

# Local
from fms_dgt.core.retrievers.registry import register_retriever
from fms_dgt.core.retrievers.unstructured_text.web_search.base import (
    SearchAPIException,
    SearchEngineRetriever,
    SearchResult,
)


@register_retriever("core/web/duckduckgo")
class DuckDuckGoRetriever(SearchEngineRetriever):
    """Search the web with duck duck go (free)"""

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.ddgs = DDGS()

    async def _search_query(
        self, session: aiohttp.ClientSession, query: str
    ) -> List[Dict[str, Any]]:
        try:
            return self.ddgs.text(
                query,
                region="us-en",
                safesearch="off",
                timelimit="y",
                max_results=self.try_limit,
            )
        except DDGSException as e:
            raise SearchAPIException(f"Error while searching DuckDuckGo: {e}")

    def _parse_result(self, search_result: List[Dict[str, Any]]) -> List[SearchResult]:
        return [
            SearchResult(
                id=f"{i}",
                text=result["body"],
                title=result["title"],
                metadata={"source": result["href"]},
            )
            for i, result in enumerate(search_result)
        ]

    def _result_to_str(self, search_results: List[SearchResult]) -> str:
        formatted = ""

        for result in search_results:
            formatted += f"### Search Result: {result.title}\n\n"
            formatted += f"{result.text}\n\n"

        return formatted
