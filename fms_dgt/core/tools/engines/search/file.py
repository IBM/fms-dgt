# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any, Dict, List, Optional
import json
import logging
import random

# Local
from fms_dgt.core.tools.engines.base import register_tool_engine
from fms_dgt.core.tools.engines.search.base import Document, SearchToolEngine
from fms_dgt.core.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


@register_tool_engine("search/file")
class FileSearchEngine(SearchToolEngine):
    """Corpus-backed retriever that loads documents from a local JSONL or JSON file.

    Documents are selected uniformly at random — no query processing is
    performed.  This engine is appropriate for simple corpus-driven recipes
    where diversity matters more than topical relevance.

    The ``projection`` map translates corpus field names to the canonical
    ``Document`` fields (``text``, ``id``).  Fields not listed in
    ``projection`` are stored verbatim in ``Document.metadata``.

    ``setup()`` loads the corpus into memory once per session.  ``teardown()``
    releases it.  For small corpora (< ~100K docs) this is fast enough to call
    per conversation; for very large corpora prefer ``ElasticsearchSearchEngine``.

    Args:
        registry: Shared ``ToolRegistry``.
        path: Path to the corpus file (JSONL or JSON array).
        format: ``"jsonl"`` (default) or ``"json"``.
        projection: Field name mapping, e.g. ``{"body": "text", "doc_id": "id"}``.
            Keys are corpus field names; values are canonical names.
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
        limit: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(registry, **kwargs)
        self._path = path
        self._format = format
        self._projection = projection or {}
        self._limit = limit
        # Corpus is populated lazily in setup(); _corpus holds raw dicts.
        self._corpus: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def setup(self, session_id: str, *args: Any, **kwargs: Any) -> None:
        """Load corpus into memory and register the session."""
        super().setup(session_id, *args, **kwargs)
        if not self._corpus:
            self._corpus = self._load_corpus()
            logger.debug(
                "FileSearchEngine loaded %d documents from %s", len(self._corpus), self._path
            )

    def teardown(self, session_id: str) -> None:
        """Unregister the session; release corpus when no sessions remain."""
        super().teardown(session_id)
        with self._sessions_lock:
            if not self._sessions:
                self._corpus = []

    # ------------------------------------------------------------------
    # SearchToolEngine contract
    # ------------------------------------------------------------------

    def _search(self, arguments: Dict[str, Any], limit: int, **kwargs: Any) -> List[Document]:
        corpus = self._corpus
        if not corpus:
            corpus = self._load_corpus()

        sample = random.sample(corpus, min(limit, len(corpus)))
        return [self._to_document(i, raw) for i, raw in enumerate(sample)]

    def _default_limit(self) -> int:
        return self._limit

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_corpus(self) -> List[Dict[str, Any]]:
        with open(self._path, "r", encoding="utf-8") as fh:
            if self._format == "json":
                return json.load(fh)
            return [json.loads(line) for line in fh if line.strip()]

    def _to_document(self, idx: int, raw: Dict[str, Any]) -> Document:
        proj = self._projection

        # Reverse-map: keys in projection map FROM corpus names TO canonical names.
        # Build a lookup: canonical_name -> corpus_field_name.
        canonical_to_corpus: Dict[str, str] = {}
        for corpus_key, canonical in proj.items():
            canonical_to_corpus[canonical] = corpus_key

        text_corpus_key = canonical_to_corpus.get("text", "text")
        id_corpus_key = canonical_to_corpus.get("id", "id")

        text = raw.get(text_corpus_key, raw.get("text", ""))
        doc_id = raw.get(id_corpus_key, raw.get("id", str(idx)))

        known_corpus_keys = {text_corpus_key, id_corpus_key}
        metadata = {k: v for k, v in raw.items() if k not in known_corpus_keys}

        return Document(id=str(doc_id), text=str(text), score=None, metadata=metadata)
