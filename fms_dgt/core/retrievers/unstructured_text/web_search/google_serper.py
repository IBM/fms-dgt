# Standard
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence
import os

# Third Party
import aiohttp

# Local
from fms_dgt.core.retrievers.registry import register_retriever
from fms_dgt.core.retrievers.unstructured_text.web_search.base import (
    SearchAPIException,
    SearchEngineRetriever,
    SearchResult,
    SearchResultMetadata,
)
from fms_dgt.utils import dgt_logger


@dataclass(kw_only=True)
class GoogleSerperMetadata(SearchResultMetadata):
    attributes: Dict[str, Any]


@dataclass(kw_only=True)
class GoogleSearchResult(SearchResult):
    metadata: GoogleSerperMetadata = field(default_factory=lambda: {"source": "", "attributes": {}})


@dataclass(kw_only=True)
class GoogleSummarySearchResult(GoogleSearchResult):
    """Google Serper block data."""

    answer_box: Optional[str] = None
    knowledge_panel: Optional[str] = None
    people_also_ask: Optional[str] = None
    related_searches: Optional[List[str]] = None


@register_retriever("core/web/google")
class GoogleSearchBlock(SearchEngineRetriever):
    """Wrapper around the Serper.dev Google Search API.
    You can create a free API key at https://serper.dev.
    To use, you should have the environment variable ``SERPER_API_KEY``
    set with your API key, or pass `serper_api_key` as a named parameter
    to the constructor."""

    def __init__(
        self,
        *,
        serper_api_key: Optional[str] = None,
        gl: str = "us",
        hl: str = "en",
        **kwargs,
    ):
        super().__init__(**kwargs)

        if serper_api_key is None:
            serper_api_key = os.getenv("SERPER_API_KEY")
            if serper_api_key is None:
                raise ValueError(
                    "serper_api_key must be set or SERPER_API_KEY environment variable must be set"
                )

        self._serper_api_key = serper_api_key
        self._gl = gl
        self._hl = hl

    async def _search_query(self, session: aiohttp.ClientSession, query: str) -> Dict[str, Any]:
        headers = {
            "X-API-KEY": self._serper_api_key or "",
            "Content-Type": "application/json",
        }
        params = {"q": query, "gl": self._gl, "hl": self._hl}
        try:
            async with session.post(
                "https://google.serper.dev/search",
                headers=headers,
                params=params,
                raise_for_status=True,
            ) as response:
                return await response.json()
        except aiohttp.ClientResponseError as e:
            raise SearchAPIException(
                f"Error while searching Google Serper: {e.status} - {e.message}"
            ) from e

    def _parse_result(self, search_result: Dict[str, Any]):
        if type(search_result) is aiohttp.ClientResponseError:
            dgt_logger.warning(
                "ClientResponseError: No good Google Search Result was found"
            )
            return {}

        summary_search_result = GoogleSummarySearchResult(id="summary", title="summary", text="")

        if search_result.get("answerBox"):
            # Answer box might appear as a snippet or a standalone answer
            answer_box = res.get("answerBox", {})
            answer = answer_box.get("answer")
            snippet = answer_box.get("snippet")
            highlights = answer_box.get("snippetHighlighted", [])

            if snippet:
                info = snippet
                for highlight in highlights:
                    info = info.replace(highlight, f"*{highlight}*")
                if answer:
                    info = f"**{answer}**\n\n{info}"
            elif answer:
                info = f"**{answer}**"
            else:
                info = None

            summary_search_result.answer_box = info

        if search_result.get("knowledgeGraph"):
            # The summary on the right side of the search results
            kg = search_result.get("knowledgeGraph", {})
            title = kg.get("title")
            entity_type = kg.get("type")
            description = kg.get("description")
            attributes = kg.get("attributes")

            info = f"**{title}: {entity_type}**" if entity_type else f"**{title}**"
            if description:
                info += f"\n\n{description}"
            if attributes:
                info += "\n" + "\n".join([f"{k}: {v}" for k, v in attributes.items()])
            summary_search_result.knowledge_panel = info

        sorted_organic = sorted(search_result["organic"], key=lambda x: x["position"])
        skip_no_snippet = (
            len([o for o in sorted_organic if "snippet" in o]) >= self.try_limit
        )
        organic_search_results = []
        for i, result in enumerate(sorted_organic):
            if len(organic_search_results) >= self.try_limit:
                break

            title = result.get("title")
            snippet = result.get("snippet")
            attributes = result.get("attributes", {})
            link = result.get("link")

            if not snippet and skip_no_snippet:
                continue

            organic_search_results.append(
                GoogleSearchResult(
                    id=f"{i}",
                    title=title,
                    text=snippet,
                    metadata={"source": link, "attributes": attributes},
                )
            )

        if search_result.get("peopleAlsoAsk"):
            people_also_ask = search_result.get("peopleAlsoAsk", [])
            for res in people_also_ask:
                title = res.get("title")
                question = res.get("question")
                snippet = res.get("snippet")

                info = f"**{question}**\n\n{title}:\n\n{snippet}"
            summary_search_result.people_also_ask = info


        summary_search_result.related_searches = search_result.get(
            "relatedSearches", []
        )

        return [summary_search_result] + organic_search_results

    def _result_to_str(self, search_results: list[GoogleSearchResult]) -> str:
        formatted = ""

        if isinstance(search_results[0], GoogleSummarySearchResult):
            if search_results[0].answer_box:
                formatted += f"### Featured Snippet\n\n{search_results[0].answer_box}\n\n"
            if search_results[0].knowledge_panel:
                formatted += f"### Knowledge Panel\n\n{search_results[0].knowledge_panel}\n\n"
            # if search_data.people_also_ask:
            #     formatted += f"### People Also Ask\n\n{search_data.people_also_ask}"
            # TODO: suggested improvement - use `related_searches` to further enhance the context
        else:
            raise TypeError("Expected first search result to be a GoogleSummarySearchResult")

        formatted += "### Search Results\n\n"
        for search_result in search_results[1:]:
            if isinstance(search_result, GoogleSearchResult):
                formatted += f"**Search Result: {search_result.title}**\n\n{search_result.text}\n\n"
                if search_result.metadata["attributes"]:
                    formatted += "\n".join(
                        [f"{k}: {v}" for k, v in search_result.metadata["attributes"].items()]
                    )
            else:
                raise TypeError("Expected search result to be a GoogleSearchResult")

        return formatted.strip()
