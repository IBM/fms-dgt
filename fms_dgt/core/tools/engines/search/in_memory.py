# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any, Dict, List, Optional
import logging

# Third Party
from sentence_transformers import SentenceTransformer

# Local
from fms_dgt.core.tools.engines.base import register_tool_engine
from fms_dgt.core.tools.engines.search.base import Document, SearchToolEngine
from fms_dgt.core.tools.registry import ToolRegistry
from fms_dgt.utils import read_json, read_jsonl

logger = logging.getLogger(__name__)


@register_tool_engine("search/in_memory")
class InMemoryVectorSearchEngine(SearchToolEngine):
    """In-memory vector index backed by a local JSONL or JSON corpus file.

    Builds a cosine-similarity index from ``sentence-transformers`` embeddings
    at engine construction time.  Suitable for corpora up to ~100K documents;
    for larger corpora use ``ElasticsearchSearchEngine``.

    Requires ``sentence-transformers`` and ``numpy``.  Import errors are raised
    at construction time so misconfigured recipes fail early.

    Args:
        registry: Shared ``ToolRegistry``.
        path: Path to the corpus file.
        format: ``"jsonl"`` (default) or ``"json"``.
        projection: Field name mapping (same convention as ``FileSearchEngine``).
        embedding_model: ``sentence-transformers`` model name or path.
            Defaults to ``"sentence-transformers/all-minilm-l6-v2"``.
        limit: Default number of documents to return per call.
        relevance_threshold: Forwarded to ``SearchToolEngine``.
        error_categories: Forwarded to ``SearchToolEngine``.
        namespaces: Forwarded to ``SearchToolEngine``.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        path: str,
        format: str = "jsonl",
        projection: Optional[Dict[str, str]] = None,
        embedding_model: str = "sentence-transformers/all-minilm-l6-v2",
        limit: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(registry, **kwargs)
        self._path = path
        self._format = format
        self._projection = projection or {}
        self._embedding_model_name = embedding_model
        self._limit = limit

        # Built eagerly at construction time.
        self._documents: List[Document] = []
        self._embeddings = None  # numpy array, shape (N, D)
        self._model = None
        self._corpus_size: int = 0

        self._build_index()

    # ------------------------------------------------------------------
    # SearchToolEngine contract
    # ------------------------------------------------------------------

    def corpus(self) -> List[Document]:
        """Return all documents in the in-memory index."""
        return list(self._documents)

    def corpus_size(self) -> int:
        """Return the number of documents in the index."""
        return self._corpus_size

    def _search(self, arguments: Dict[str, Any], limit: int, **kwargs: Any) -> List[Document]:
        # Third Party

        query = arguments.get("query", "")
        if not query or self._embeddings is None:
            return []

        q_emb = self._model.encode([query], normalize_embeddings=True)
        scores = (self._embeddings @ q_emb.T).flatten()
        top_indices = scores.argsort()[::-1][:limit]

        results = []
        for idx in top_indices:
            doc = self._documents[idx]
            results.append(
                Document(
                    id=doc.id,
                    text=doc.text,
                    score=float(scores[idx]),
                    metadata=doc.metadata,
                )
            )
        return results

    def _default_limit(self) -> int:
        return self._limit

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        corpus = self._load_corpus()
        self._documents = [self._to_document(i, raw) for i, raw in enumerate(corpus)]
        self._corpus_size = len(self._documents)
        texts = [d.text for d in self._documents]

        logger.debug(
            "InMemoryVectorSearchEngine: encoding %d documents with %s",
            len(texts),
            self._embedding_model_name,
        )
        self._model = SentenceTransformer(self._embedding_model_name)
        self._embeddings = self._model.encode(texts, normalize_embeddings=True)

    def _load_corpus(self) -> List[Dict[str, Any]]:
        if self._format == "json":
            return read_json(self._path)
        return read_jsonl(self._path)

    def _to_document(self, idx: int, raw: Dict[str, Any]) -> Document:
        proj = self._projection
        canonical_to_corpus: Dict[str, str] = {v: k for k, v in proj.items()}

        text_corpus_key = canonical_to_corpus.get("text", "text")
        id_corpus_key = canonical_to_corpus.get("id", "id")

        text = raw.get(text_corpus_key, raw.get("text", ""))
        doc_id = raw.get(id_corpus_key, raw.get("id", str(idx)))

        known_corpus_keys = {text_corpus_key, id_corpus_key}
        metadata = {k: v for k, v in raw.items() if k not in known_corpus_keys}

        return Document(id=str(doc_id), text=str(text), score=None, metadata=metadata)
