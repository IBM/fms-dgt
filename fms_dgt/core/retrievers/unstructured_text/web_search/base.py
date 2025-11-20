# Standard
from abc import abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Dict, Iterable, List, Optional, Tuple, TypedDict
import asyncio
import json
import logging

# Third Party
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, PipelineOptions
from docling.datamodel.settings import settings
from docling.document_converter import (
    DocumentConverter,
    HTMLFormatOption,
    PdfFormatOption,
)
from firecrawl import FirecrawlApp
from markitdown import MarkItDown
from tqdm import tqdm
import aiohttp
import torch

# Local
from fms_dgt.core.retrievers.registry import get_unstructured_text_retriever
from fms_dgt.core.retrievers.unstructured_text.base import (
    UnstructuredTextDocument,
    UnstructuredTextRetriever,
)
from fms_dgt.utils import dgt_logger


class SearchAPIException(Exception):
    """Exception raised when there is an error with the search API."""


class SearchResultMetadata(TypedDict):
    source: str


@dataclass(kw_only=True)
class SearchResult(UnstructuredTextDocument):
    title: str
    metadata: SearchResultMetadata = field(default_factory=lambda: {"source": ""})


class WebpageProcessor(StrEnum):
    DOCLING = "docling"
    FIRECRAWL = "firecrawl"


logging.getLogger("docling").setLevel(logging.WARNING)


class SearchEngineRetriever(UnstructuredTextRetriever):
    """Block that performs a web search"""

    def __init__(
        self,
        process_webpages: bool = True,
        deduplicate_sources: bool = True,
        reorder_organic: bool = True,
        try_limit: int = 8,
        webpage_processor: WebpageProcessor = WebpageProcessor.DOCLING,
        fallback_retriever: Optional[str] = None,
        cache_file: Optional[str] = None,
        limit: int = 2,
        **kwargs,
    ):
        """
        Initialize the base search generator.
        Args:
            process_webpages (bool, optional): Whether to process webpages HTML during
                the search. Defaults to True.
            deduplicate_sources (bool, optional): Whether to deduplicate sources
                in the search results. Useful when all the queries in a single session
                are about the same thing. Defaults to True.
            reorder_organic (bool, optional): Whether to reorder the organic results
                such that the first result is the result that appears the most
                across other searches (relevant only if `deduplicate_sources` is True).
            try_limit (int, optional): The upper bound on the number of results parsed by docling.
                For cases where docling fails to parse a webpage, used only if `process_webpages` is True.
                Defaults to 8.
            webpage_processor (WebpageProcessor, optional): The processor to use for processing webpages.
                Defaults to WebpageProcessor.DOCLING. For using Firecrawl, you need to run it locally -
                more information can be found at: `https://github.com/mendableai/firecrawl/blob/main/CONTRIBUTING.md`.
            fallback_retriever (Optional[str], optional): The name of the fallback retriever to use
                when the retriever fails to search the web. Defaults to None.
            **kwargs: Additional keyword arguments to be passed to base block.
        """

        super().__init__(limit=limit, **kwargs)

        self.process_webpages = process_webpages
        self.deduplicate_sources = deduplicate_sources
        self.reorder_organic = reorder_organic
        self.try_limit = try_limit
        self.webpage_processor = webpage_processor
        if webpage_processor == "firecrawl":
            self.firecrawl_app = FirecrawlApp(api_url="http://localhost:3002")
        elif webpage_processor == "docling":
            accelerator_options = AcceleratorOptions(
                device=(
                    "cuda"
                    if torch.cuda.is_available()
                    else ("mps" if torch.mps.is_available() else "auto")
                ),
                cuda_use_flash_attention2=torch.cuda.is_available(),
            )
            self.docling_processor = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=PdfPipelineOptions(
                            document_timeout=5,
                            accelerator_options=accelerator_options,
                        ),
                    ),
                    InputFormat.HTML: HTMLFormatOption(
                        pipeline_options=PipelineOptions(
                            document_timeout=5,
                            accelerator_options=accelerator_options,
                        )
                    ),
                }
            )
        elif webpage_processor == "markitdown":
            self.md = MarkItDown(enable_plugins=False)

        if fallback_retriever:
            self.fallback_retriever: Optional[SearchEngineRetriever] = (
                get_unstructured_text_retriever(
                    fallback_retriever,
                    process_webpages=process_webpages,
                    deduplicate_sources=deduplicate_sources,
                    reorder_organic=reorder_organic,
                    try_limit=try_limit,
                    webpage_processor=webpage_processor,
                    fallback_retriever=None,  # Avoid circular fallback
                )
            )
        else:
            self.fallback_retriever = None
        if cache_file:
            with open(cache_file, "r") as f:
                self._search_results_cache = json.load(f)
        else:
            self._search_results_cache = {}

        # settings.perf.doc_batch_concurrency = 2
        settings.perf.doc_batch_size = 4

    async def __parallel_searches(
        self, search_queries: Iterable[str], disable_tqdm: bool = False
    ) -> Tuple[List[Dict[str, Any]], Dict[int, str]]:
        """
        Perform parallel searches for a list of search queries using asynchronous requests.
        Args:
            search_queries (Iterable[str]): An iterable of search query strings to be processed.
            disable_tqdm (bool, optional): Whether to disable the progress bar. Defaults to False.
        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the search results.
            If an exception occurs during a search, an empty dictionary is returned for that query.
        """
        progress_bar = tqdm(
            total=len(list(search_queries)),
            desc="Searching The Web",
            unit="query",
            disable=disable_tqdm,
        )

        fallback_queries = {}
        async with aiohttp.ClientSession() as session:

            async def search(i: int, query: str, remaining_tries: int = 10):
                try:
                    result = self._search_results_cache.get(query)
                    if not result:
                        result = await self._search_query(session=session, query=query)
                        self._search_results_cache[query] = result
                except SearchAPIException as e:
                    if self.fallback_retriever:
                        dgt_logger.error(
                            f"Search API exception occurred while searching {query!r}: {e}."
                        )
                        fallback_queries[i] = query
                        return None
                    else:
                        dgt_logger.error(
                            f"Error occurred while searching: {e}. Retrying in 5 seconds..."
                        )
                        await asyncio.sleep(5)
                        if remaining_tries > 0:
                            return await search(i, query, remaining_tries - 1)
                        else:
                            dgt_logger.warning(
                                f"Max retries (10) reached for query {query!r}. Returning empty search results..."
                            )
                            return None
                progress_bar.update(1)
                return result

            tasks = [search(i, query) for i, query in enumerate(search_queries)]
            search_results = await asyncio.gather(*tasks, return_exceptions=True)

            def transform_exception(x):
                return [] if isinstance(x, Exception) else x

            return [transform_exception(result) for result in search_results], fallback_queries

    async def _process_webpages_docling(
        self, search_results: List[SearchResult], fallback=True
    ) -> None:
        processed_htmls_iter = self.docling_processor.convert_all(
            source=[
                webpage.metadata["source"]
                for webpage in search_results
                if webpage.metadata["source"]
            ],
            raises_on_error=False,
        )

        # class TimeoutException(Exception):
        #     pass

        # # 1) Install a SIGALRM handler
        # def _timeout_handler(signum, frame):
        #     raise TimeoutException(f"Operation timed out after {timeout_secs} seconds")

        i = 0
        success_indices = []
        while True:
            if i >= len(search_results):
                break
            if len(success_indices) >= self.limit:
                break
            if not search_results[i].metadata["source"]:
                i += 1
                continue

            try:
                # signal.signal(signal.SIGALRM, _timeout_handler)
                # timeout_secs = 5
                # signal.alarm(timeout_secs)
                # Some sources can throw forbidden errors or other errors
                processed_html = next(processed_htmls_iter)
                page_content = processed_html.document.export_to_markdown()
                # some webpages are not parsable when initially accessed and might return an empty string
                if len(page_content) > len(search_results[i].text):
                    search_results[i].text = page_content
                    success_indices.append(i)
            except StopIteration:
                break
            # except TimeoutException:
            #     dgt_logger.warning(
            #         f"Timeout occurred while processing source: {search_results[i].metadata['source']}"
            #     )
            #     continue
            except Exception as e:
                src = search_results[i].metadata["source"]
                dgt_logger.warning(f'Failed to access source "{src}": {e}')
                continue
            finally:
                # signal.alarm(0)
                # signal.signal(signal.SIGALRM, signal.SIG_DFL)
                i += 1
                torch.cuda.empty_cache()

        # change the list such that the indices in success_indices are first
        search_results[:] = [search_results[i] for i in success_indices] + [
            search_results[i] for i in range(len(search_results)) if i not in success_indices
        ]

    async def _process_webpages_firecrawl(
        self, search_results: List[SearchResult], fallback=True
    ) -> None:
        urls = [
            search_result.metadata["source"]
            for search_result in search_results
            if search_result.metadata["source"]
        ]

        def fetch_all_mds() -> Tuple[List[str], List[int]]:
            success_indices = []
            mds = []
            i = 0
            while len(mds) < self.limit and i < len(urls):
                try:
                    resp = self.firecrawl_app.batch_scrape_urls([urls[i]])
                    if not resp.completed:
                        raise ValueError("Failed to scrape URL for an unknown reason")
                except Exception as e:
                    dgt_logger.error(
                        f"Error occurred while scraping URLs {urls[i:i+self.limit]}: {e}"
                    )
                else:
                    candidate = resp.data[0].markdown
                    if candidate is not None and len(candidate) > 0:
                        mds.append(candidate)
                        success_indices.append(i)
                i += 1
            if len(mds) < self.limit:
                dgt_logger.error(
                    f"Only {len(mds)} out of {self.try_limit} sources in total were successfully processed (need {self.limit})."
                )
            return mds, success_indices

        mds, success_indices = fetch_all_mds()
        for i in success_indices:
            search_results[i].text = mds.pop(0)

        # Reorder search results such that those with text are first
        search_results[:] = [
            search_result for i, search_result in enumerate(search_results) if i in success_indices
        ] + [
            search_result
            for i, search_result in enumerate(search_results)
            if i not in success_indices
        ]

    async def _process_webpages_markitdown(
        self, search_results: List[SearchResult], fallback=True
    ) -> None:
        success_indices = []
        for i, webpage in enumerate(search_results):
            if not webpage.metadata["source"]:
                continue
            if len(success_indices) >= self.limit:
                break
            try:
                page_content = self.md.convert(webpage.metadata["source"]).text_content
            except Exception as e:
                dgt_logger.warning(f'Failed to access source "{webpage.metadata["source"]}": {e}')
                continue
            # some webpages are not parsable when initially accessed and might return an empty string
            if len(page_content) > len(webpage.text):
                webpage.text = page_content
                success_indices.append(i)

        # change the list such that the indices in success_indices are first
        search_results[:] = [search_results[i] for i in success_indices] + [
            search_results[i] for i in range(len(search_results)) if i not in success_indices
        ]

    async def _process_webpages(self, search_results: List[SearchResult]) -> None:
        """
        Processes the organic search results from a SearchResult object by replacing
        their `text` field with the source HTML content in Markdown format.
        Args:
            search_docs (List[SearchResult]): The list of search documents to process.
        Notes:
            - If an error occurs while accessing a source, the corresponding organic result
              will not have its content updated.
        """
        if not search_results:
            return

        process_fn = getattr(self, f"_process_webpages_{self.webpage_processor}", None)
        if process_fn is None:
            raise ValueError(f"Webpage processor {self.webpage_processor} is not supported.")
        await process_fn(search_results)

    async def run(self, queries: List[str], disable_tqdm: bool = False) -> List[List[SearchResult]]:
        """
        Executes the search operation for the given search queries and processes the results.
        Args:
            queries (List[str]): An iterable of the queries to be executed.
        Returns:
            List[List[SearchResult]]: A list of lists containing the search results.
            Each list corresponds to a search query and contains `SearchResult` objects
            representing the search results.
        Note:
            If `process_webpages` is enabled, processes the webpages for each search query.
        """

        results, fallback_queries = await self.__parallel_searches(
            queries, disable_tqdm=disable_tqdm
        )
        searches_results = []

        fallback_results_dict = {}
        if self.fallback_retriever and len(fallback_queries) > 0:
            dgt_logger.error(
                f"Error occurred while searching {len(fallback_queries)} queries. Falling back to {self.fallback_retriever.__class__.__name__}..."
            )
            fallback_results, _ = await self.fallback_retriever.__parallel_searches(
                fallback_queries.values(), disable_tqdm=disable_tqdm
            )
            fallback_results_dict = {
                i: result for i, result in zip(fallback_queries.keys(), fallback_results)
            }

        for i, res in enumerate(results):
            if self.fallback_retriever and i in fallback_results_dict:
                # If the query is in fallback_queries, use the fallback result
                searches_results.append(
                    self.fallback_retriever._parse_result(fallback_results_dict[i])
                )
            elif res is None:
                dgt_logger.warning(
                    f"Search API exception occurred for query {queries[i]!r}. No results returned."
                )
                searches_results.append([])
            else:
                # Otherwise, parse the original result
                searches_results.append(self._parse_result(res))

        for search_results, query in zip(searches_results, queries):
            for search_result in search_results:
                search_result.id = f"{query}_{search_result.id}"

        if self.deduplicate_sources:
            sources_freq = {}
            num_duplicates = 0
            for search_results in searches_results:
                if not search_results:
                    continue

                # update sources_freq
                for search_result in search_results:
                    source = search_result.metadata["source"]
                    if source not in sources_freq:
                        sources_freq[source] = 1
                    else:
                        num_duplicates += 1
                        sources_freq[source] += 1

                # remove duplicates
                search_results = [
                    organic
                    for organic in search_results
                    if sources_freq[organic.metadata["source"]] == 1
                ]

            dgt_logger.info(
                f"Removed {num_duplicates} duplicate sources out of {sum(sources_freq.values())}."
            )
            if self.reorder_organic:
                for search_results in searches_results:
                    if not search_results:
                        continue

                    # sort organic results by frequency
                    search_results.sort(
                        key=lambda x: sources_freq[x.metadata["source"]], reverse=True
                    )

        if self.process_webpages:
            for result in tqdm(searches_results, desc="Processing webpages", disable=disable_tqdm):
                await self._process_webpages(result)

        searches_results = [results[: self.limit] for results in searches_results]

        return searches_results

    def __call__(self, requests: List[str], disable_tqdm: bool = False) -> List[List[SearchResult]]:
        return asyncio.run(self.run(requests, disable_tqdm=disable_tqdm))

    def result_to_str(self, query: str, search_results: List[SearchResult]) -> str:
        """
        Converts search results into a formatted string representation.
        Args:
            query (str): The search query string.
            search_results (List[SearchResult]): The data object containing search results.
        Returns:
            str: A formatted string representation of the search results.
        """

        title = f'## Search Results For: "{query}"\n\n'
        formatted = self._result_to_str(search_results=search_results).strip()

        return title + formatted if formatted else ""

    @abstractmethod
    async def _search_query(self, session: aiohttp.ClientSession, query: str) -> Any:
        """
        Perform a search query using the provided aiohttp client session.
        This is an abstract method that must be implemented by subclasses to define
        the specific behavior for executing a search query.
        Args:
            session (aiohttp.ClientSession): The aiohttp client session to use for making the request.
            query (str): The search query string.
        Returns:
            Any: The result of the search query returned by the specific API used by the subclass.
        """

        raise NotImplementedError

    @abstractmethod
    def _parse_result(self, search_result: Any) -> List[SearchResult]:
        """
        Parses the search result returned from `self._search_query()`.
        Args:
            search_result (Any): The raw result returned by the search engine.
        Returns:
            List[SearchResult]: A list of `SearchResult` objects containing the parsed search results.
        """

        raise NotImplementedError

    @abstractmethod
    def _result_to_str(self, search_results: List[SearchResult]) -> str:
        """
        Converts the search results of a query into a string representation.
        Args:
            search_results (List[SearchResult]): The search results to be represented.
        Returns:
            str: A string representation of the search results.
        Note:
            There is no need to relate to the search query, this method is wrapped by `self.result_to_str()`
            it is recommended to use markdown formatting for the string representation.
        """

        raise NotImplementedError
