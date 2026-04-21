# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any, Dict, List, Optional
import logging
import os
import random

# Local
from fms_dgt.core.tools.data_objects import ToolCall, ToolResult
from fms_dgt.core.tools.engines.base import ErrorCategory, register_tool_engine
from fms_dgt.core.tools.engines.search.base import Document, SearchToolEngine
from fms_dgt.core.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


@register_tool_engine("search/elasticsearch")
class ElasticsearchSearchEngine(SearchToolEngine):
    """Elasticsearch / OpenSearch retriever using natural-language (BM25) queries.

    Requires the ``elasticsearch`` Python client (``pip install elasticsearch``).
    The client is created lazily on the first ``setup()`` call.

    ``setup()`` opens a connection pool; ``teardown()`` closes it when the last
    session is removed.

    Supports an ``index_not_found`` error category (Elasticsearch-specific) in
    addition to the base error categories.

    **Authentication and TLS** are configured via dedicated constructor args.
    Use ``api_key`` for token-based auth (preferred for Elastic Cloud and
    modern self-managed clusters) or ``username``/``password`` for basic auth.
    For clusters with self-signed or private CA certificates, pass the CA bundle
    path via ``ca_certs``; set ``ssl_verify=False`` only in development
    environments where cert verification is intentionally disabled.

    Credentials and the endpoint should be provided via environment variables
    rather than hardcoded in task YAML files.  The constructor args take
    priority; env vars are the fallback.

    +-----------------+--------------------+
    | Constructor arg | Env var            |
    +=================+====================+
    | ``host``        | ``ES_ENDPOINT``    |
    +-----------------+--------------------+
    | ``api_key``     | ``ES_API_KEY``     |
    +-----------------+--------------------+
    | ``username``    | ``ES_USERNAME``    |
    +-----------------+--------------------+
    | ``password``    | ``ES_PASSWORD``    |
    +-----------------+--------------------+
    | ``ssl_verify``  | ``ES_SSL_VERIFY``  |
    +-----------------+--------------------+

    Typical YAML configurations::

        # Local development — host defaults to http://localhost:9200
        elastic_retriever:
          type: search/elasticsearch
          index: knowledge_base

        # Elastic Cloud / production — credentials from env vars
        elastic_retriever:
          type: search/elasticsearch
          index: knowledge_base
          # ES_ENDPOINT and ES_API_KEY set in environment

        # Self-managed with custom CA
        elastic_retriever:
          type: search/elasticsearch
          index: knowledge_base
          ca_certs: /etc/ssl/certs/elastic-ca.pem
          # ES_ENDPOINT, ES_USERNAME, ES_PASSWORD set in environment

    Args:
        registry: Shared ``ToolRegistry``.
        host: Elasticsearch host URL.  Defaults to ``ES_ENDPOINT`` env var,
            then ``"http://localhost:9200"``.
        index: Default index name.  Can be overridden per call via arguments.
        limit: Default number of documents to return per call.
        api_key: API key for token-based auth.  Falls back to ``ES_API_KEY``.
        username: Username for basic auth.  Falls back to ``ES_USERNAME``.
        password: Password for basic auth.  Falls back to ``ES_PASSWORD``.
        ca_certs: Path to a CA bundle file for verifying the server certificate.
        ssl_verify: Set to ``False`` to disable TLS certificate verification.
            Falls back to ``ES_SSL_VERIFY`` env var (``"false"`` disables).
            Only for development; never use in production.
        relevance_threshold: Forwarded to ``SearchToolEngine``.
        error_categories: Forwarded to ``SearchToolEngine`` (plus
            ``index_not_found`` support).
        namespaces: Forwarded to ``SearchToolEngine``.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        host: Optional[str] = None,
        index: str = "_all",
        limit: int = 5,
        api_key: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        ca_certs: Optional[str] = None,
        ssl_verify: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(registry, **kwargs)
        self._host = host or os.environ.get("ES_ENDPOINT", "http://localhost:9200")
        self._default_index = index
        self._limit = limit
        self._api_key = api_key or os.environ.get("ES_API_KEY")
        self._username = username or os.environ.get("ES_USERNAME")
        self._password = password or os.environ.get("ES_PASSWORD")
        self._ca_certs = ca_certs
        if not ssl_verify:
            self._ssl_verify = False
        else:
            self._ssl_verify = os.environ.get("ES_SSL_VERIFY", "true").lower() != "false"
        self._client = None

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def setup(self, session_id: str, *args: Any, **kwargs: Any) -> None:
        super().setup(session_id, *args, **kwargs)
        if self._client is None:
            self._client = self._make_client()
            self._check_version_compatibility()

    def teardown(self, session_id: str) -> None:
        super().teardown(session_id)
        with self._sessions_lock:
            if not self._sessions and self._client is not None:
                try:
                    self._client.close()
                except Exception:
                    pass
                self._client = None

    # ------------------------------------------------------------------
    # Error injection (adds index_not_found)
    # ------------------------------------------------------------------

    def _inject_error(
        self,
        tc: ToolCall,
        category: Optional[ErrorCategory] = None,
    ) -> Optional[ToolResult]:
        if category is None:
            fired = [ec for ec in self._error_categories if ec.should_fire()]
            if not fired:
                return None
            category = random.choice(fired)

        if category.type == "index_not_found":
            index = (tc.arguments or {}).get("index", self._default_index)
            return ToolResult(
                call_id=tc.call_id,
                name=tc.name,
                result=None,
                error=f"Index '{index}' does not exist",
            )

        return super()._inject_error(tc, category=category)

    # ------------------------------------------------------------------
    # SearchToolEngine contract
    # ------------------------------------------------------------------

    def _search(self, arguments: Dict[str, Any], limit: int, **kwargs: Any) -> List[Document]:
        if self._client is None:
            self._client = self._make_client()

        query = arguments.get("query", "")
        index = arguments.get("index", self._default_index)

        body = {
            "query": {"match": {"_all" if index == "_all" else "text": query}},
            "size": limit,
        }

        try:
            response = self._client.search(index=index, body=body)
        except Exception as exc:
            raise RuntimeError(f"Elasticsearch query failed: {exc}") from exc

        hits = response.get("hits", {}).get("hits", [])
        return [self._hit_to_document(hit) for hit in hits]

    def _default_limit(self) -> int:
        return self._limit

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_client(self):
        try:
            # Third Party
            from elasticsearch import Elasticsearch
        except ImportError as exc:
            raise ImportError(
                "ElasticsearchSearchEngine requires 'elasticsearch'. "
                "Install with: pip install fms_dgt[search]"
            ) from exc

        client_kwargs: Dict[str, Any] = {}

        if self._api_key:
            client_kwargs["api_key"] = self._api_key
        elif self._username and self._password:
            client_kwargs["basic_auth"] = (self._username, self._password)

        if self._ca_certs:
            client_kwargs["ca_certs"] = self._ca_certs

        if not self._ssl_verify:
            client_kwargs["verify_certs"] = False
            client_kwargs["ssl_show_warn"] = False

        return Elasticsearch(self._host, **client_kwargs)

    def _check_version_compatibility(self) -> None:
        try:
            # Third Party
            import elasticsearch as es_pkg
        except ImportError:
            return

        raw = es_pkg.__version__
        client_version_str = ".".join(str(x) for x in raw) if isinstance(raw, tuple) else str(raw)
        client_major = int(client_version_str.split(".")[0])

        try:
            info = self._client.info()
            cluster_version = info.get("version", {}).get("number", "")
            if not cluster_version:
                return
            cluster_major = int(cluster_version.split(".")[0])
            if client_major != cluster_major:
                raise RuntimeError(
                    f"Elasticsearch client v{client_version_str} is not compatible with "
                    f"cluster version {cluster_version}. "
                    f"Install the matching client: pip install 'elasticsearch>={cluster_major}.0,<{cluster_major + 1}'"
                )
        except RuntimeError:
            raise
        except Exception:
            # info() failed — likely a version mismatch causing a protocol error.
            # Surface a helpful message rather than a raw API exception.
            raise RuntimeError(
                f"Elasticsearch client v{client_version_str} could not communicate with the cluster at {self._host}. "
                f"This is often caused by a client/cluster version mismatch. "
                f"Check your cluster version and install the matching client, "
                f"e.g. pip install 'elasticsearch>=8.0,<9' for an ES 8.x cluster."
            )

    def _hit_to_document(self, hit: Dict[str, Any]) -> Document:
        source = hit.get("_source", {})
        doc_id = hit.get("_id", "")
        score = hit.get("_score")
        text = source.get("text", source.get("body", ""))
        metadata = {k: v for k, v in source.items() if k not in ("text", "body")}
        return Document(id=str(doc_id), text=str(text), score=score, metadata=metadata)
