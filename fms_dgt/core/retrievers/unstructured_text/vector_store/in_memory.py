# Standard
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union
import asyncio
import os

# Third Party
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_docling.loader import DoclingLoader, ExportType
from tqdm import tqdm

# Local
from fms_dgt.core.retrievers.registry import register_retriever
from fms_dgt.core.retrievers.unstructured_text.base import (
    UnstructuredTextDocument,
    UnstructuredTextRetriever,
)
from fms_dgt.utils import dgt_logger


class SplitUnit(Enum):
    """
    Enum for split by options.
    """

    CHAR = "char"
    WORD = "word"
    TOKEN = "token"
    MARKDOWN = "markdown"


class EmbeddingModelProvider(Enum):
    """
    Enum for embedding model providers.
    """

    RITS = "rits"
    OPENAI = "openai"
    WATSONX = "watsonx"


WATSONX_EMB_MAX_TOKENS = {
    "ibm/slate-125m-english-rtrvr": 512,
    "ibm/slate-30m-english-rtrvr": 512,
    "sentence-transformers/all-minilm-l6-v2": 256,
    "intfloat/multilingual-e5-large": 512,
}


@register_retriever("core/vector/in_memory")
class InMemoryRetriever(UnstructuredTextRetriever):
    """
    Block for document retrieval.
    """

    def __init__(
        self,
        docs_source: Union[List[str], Path],
        limit: int,
        embedding_model_provider: EmbeddingModelProvider = EmbeddingModelProvider.WATSONX,
        embedding_model_id: str = "ibm/slate-125m-english-rtrvr",
        split_unit: SplitUnit = SplitUnit.TOKEN,
        split_chunk_size: Optional[int] = None,
        split_chunk_overlap: Optional[int] = None,
        **kwargs,
    ):
        """
        Initializes the document retrieval generator.
        Args:
            docs_source (Optional[Union[List[str], Path]]): The source of documents to be used.
                Can be a list of web links or a Path to a directory with all the documents.
            limit (int): The maximum number of documents to retrieve.
            embedding_model_provider (EmbeddingModelProvider): The provider of the embedding
                model. Defaults to `EmbeddingModelProvider.WATSONX`.
            embedding_model_id (str): The identifier for the embedding model in the provider service.
                Defaults to "ibm/slate-125m-english-rtrvr".
            split_unit (Optional[SplitUnit]): The unit in splitting documents.
            split_chunk_size (Optional[int]): The size (n.o. units) of each chunk when splitting documents.
                Must be provided if `split_unit` is not TOKEN.
            split_chunk_overlap (Optional[int]): The overlap size (n.o. units) between chunks when splitting documents.
                Must be provided if `split_unit` is not TOKEN.
            **kwargs: Additional keyword arguments to be passed to the parent class.
        Raises:
            ValueError: If neither `docs_source` nor `milvus_uri` is provided, or if both are provided.
        """
        super().__init__(limit=limit, **kwargs)

        self.embedding_model_provider = embedding_model_provider
        self.embedding_model_id = embedding_model_id

        self.split_unit = split_unit
        self.split_chunk_size = split_chunk_size
        self.split_chunk_overlap = split_chunk_overlap

        self.embedding_model = self._get_embedding_model()
        self.vector_store = InMemoryVectorStore(self.embedding_model)

        docs = self._load_and_split_docs(docs_source)
        self._index_docs(docs)

    def _get_embedding_model(self) -> Embeddings:
        """
        Retrieves the appropriate embedding model based on the specified provider and model ID.
        """

        if self.embedding_model_provider in [
            EmbeddingModelProvider.RITS,
            EmbeddingModelProvider.OPENAI,
        ]:
            # Third Party
            from langchain_openai import OpenAIEmbeddings

            model_to_url_path = {
                "ibm/slate-125m-english-rtrvr-v2": "slate-125m-english-rtrvr-v2",
                "meta-llama/llama-3-3-70b-instruct-embeddings": "llama-3-3-70b-instruct-e",
            }
            if self.embedding_model_id not in model_to_url_path:
                model_to_url_path[self.embedding_model_id] = self.embedding_model_id

            return OpenAIEmbeddings(
                base_url="/".join(
                    [
                        os.environ["RITS_API_BASE_URL"],
                        model_to_url_path[self.embedding_model_id],
                        "v1",
                    ]
                ),
                model=self.embedding_model_id,
                api_key=os.environ["RITS_API_KEY"],  # type: ignore
                default_headers={"RITS_API_KEY": os.environ["RITS_API_KEY"]},
            )
        elif self.embedding_model_provider == EmbeddingModelProvider.WATSONX:
            # Third Party
            from langchain_ibm import WatsonxEmbeddings

            return WatsonxEmbeddings(
                model_id=self.embedding_model_id,
                url="https://us-south.ml.cloud.ibm.com",  # type: ignore
                apikey=os.environ["WATSONX_API_KEY"],  # type: ignore
                project_id=os.environ["WATSONX_PROJECT_ID"],
            )
        else:
            raise ValueError(
                f"Unsupported embedding model provider: {self.embedding_model_provider}"
            )

    def _load_and_split_docs(self, docs_source: Union[List[str], Path]) -> List[Document]:
        """
        Loads and splits documents from the specified source.
        Args:
            docs_source (Union[List[str], Path]): The source of documents to be loaded.
                Either list of web links or a Path to a directory with all the documents.
        Raises:
            ValueError: If the split type is not specified or if chunk size and overlap are not provided for char/word split types.
        Returns:
            List[Document]: A list of loaded and split documents.
        """
        if isinstance(docs_source, str):
            docs_source = Path(docs_source)

        docs_sources = []
        if isinstance(docs_source, Path):
            for doc in docs_source.iterdir():
                if doc.is_file() and doc.suffix in [".pdf", ".md", ".docx"]:
                    docs_sources.append(doc)
                else:
                    dgt_logger.warning(
                        "File %s is not a supported document type. Supported types are: .pdf, .md, .docx",
                        doc,
                    )

        elif isinstance(docs_source, list):
            docs_sources = docs_source

        dgt_logger.info("Loading and splitting %s documents...", len(docs_sources))

        if self.split_unit == SplitUnit.MARKDOWN:
            doc_loader = DoclingLoader(docs_sources)
            return doc_loader.load()

        doc_loader = DoclingLoader(docs_sources, export_type=ExportType.MARKDOWN)
        docs = doc_loader.load()

        if self.split_unit == SplitUnit.CHAR:
            if not self.split_chunk_size:
                raise ValueError("split_chunk_size must be provided for char split type.")
            if not self.split_chunk_overlap:
                raise ValueError("split_chunk_overlap must be provided for char split type.")

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.split_chunk_size,
                chunk_overlap=self.split_chunk_overlap,
            )
        elif self.split_unit == SplitUnit.WORD:
            if not self.split_chunk_size:
                raise ValueError("split_chunk_size must be provided for word split type.")
            if not self.split_chunk_overlap:
                raise ValueError("split_chunk_overlap must be provided for word split type.")

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.split_chunk_size,
                chunk_overlap=self.split_chunk_overlap,
                length_function=lambda s: len(s.split()),
            )
        else:
            if self.split_chunk_size:
                chunk_size = self.split_chunk_size
            else:
                if self.embedding_model_provider in [
                    EmbeddingModelProvider.RITS,
                    EmbeddingModelProvider.OPENAI,
                ]:
                    try:
                        max_allowed_tokens = self.embedding_model.embedding_ctx_length
                    except Exception:
                        max_allowed_tokens = 512
                        dgt_logger.warning(
                            "Could not find max allowed tokens for %s, using %s",
                            self.embedding_model_id,
                            max_allowed_tokens,
                        )
                if (
                    self.embedding_model_provider == EmbeddingModelProvider.WATSONX
                    and self.embedding_model_id in WATSONX_EMB_MAX_TOKENS
                ):
                    max_allowed_tokens = WATSONX_EMB_MAX_TOKENS[self.embedding_model_id]
                else:
                    max_allowed_tokens = 512
                    dgt_logger.warning(
                        "Could not find max allowed tokens for %s, using %s",
                        self.embedding_model_id,
                        max_allowed_tokens,
                    )

                chunk_size = 0.9 * max_allowed_tokens
                dgt_logger.info(
                    "Automatically set chunk size of %s which is 90%% of the maximum context length for %s",
                    chunk_size,
                    self.embedding_model_id,
                )

            if self.split_chunk_overlap:
                chunk_overlap = self.split_chunk_overlap
            else:
                chunk_overlap = 0.3 * chunk_size
                dgt_logger.info(
                    "Automatically set chunk overlap of %s which is 30%% of the chunk size %s",
                    chunk_overlap,
                    chunk_size,
                )

            splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                # model_name="gpt-4o",
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

        return splitter.split_documents(docs)

    def _index_docs(self, docs: List[Document]) -> None:
        """
        Indexes the documents into the vector store.
        Args:
            docs (List[Document]): The documents to be indexed.
        """
        dgt_logger.info("Indexing %s documents...", len(docs))
        _ = self.vector_store.add_documents(docs)

    async def retrieve_docs(self, query: str) -> List[UnstructuredTextDocument]:
        """
        Asynchronously retrieves documents from the vector store based on the provided query.
        Args:
            query (str): The query string to search for in the vector store.
        Returns:
            RetrievalBlockData: The updated data with retrieved documents.
        """
        docs = self.vector_store.similarity_search_with_score(query, k=self.limit)
        return [
            UnstructuredTextDocument(id=f"{query} ({i})", text=doc.page_content, score=score)
            for i, (doc, score) in enumerate(docs)
        ]

    def __call__(
        self,
        requests: List[Union[str, dict]],
    ) -> List[List[UnstructuredTextDocument]]:
        """
        Processes a list of requests to retrieve documents.
        Args:
            requests (List[Union[str, dict, None]]): A list of requests, where each request can be a query string,
                or a dictionary with a "query" key.
        Returns:
            List[List[UnstructuredTextDocument]]: A list of lists, where each inner list contains the retrieved documents
            for the corresponding request.
        """
        lock = asyncio.Lock()
        progress_bar = tqdm(total=len(requests), desc="Retrieving documents")

        async def retrieve_and_update(query: str) -> List[UnstructuredTextDocument]:
            docs = await self.retrieve_docs(query)
            async with lock:
                progress_bar.update(1)
            return docs

        async def run() -> List[List[UnstructuredTextDocument]]:
            tasks = []
            for req in requests:
                if isinstance(req, dict) and "query" in req:
                    query = req["query"]
                elif isinstance(req, str):
                    query = req
                else:
                    raise ValueError("Each request must be a string or a dict with a 'query' key.")
                tasks.append(retrieve_and_update(query))
            return await asyncio.gather(*tasks)

        return asyncio.run(run())
