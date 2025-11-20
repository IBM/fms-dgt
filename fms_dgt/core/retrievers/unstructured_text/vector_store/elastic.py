# Standard
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4
import json
import logging
import os

# Third Party
from elasticsearch import BadRequestError, Elasticsearch

# Local
from fms_dgt.core.retrievers.registry import (
    register_retriever,
)
from fms_dgt.core.retrievers.unstructured_text.base import (
    PROJECTION_FIELD_ID,
    PROJECTION_FIELD_TEXT,
    UnstructuredTextDocument,
    UnstructuredTextRetriever,
)
from fms_dgt.utils import dgt_logger

# Disable third party logging
logging.getLogger("elastic_transport.transport").setLevel(logging.WARNING)

# ===========================================================================
#                       CONSTANTS
# ===========================================================================
CONNECTION_FIELD_ENDPOINT = "endpoint"
CONNECTION_FIELD_API_KEY = "api_key"
CONNECTION_FIELD_USERNAME = "username"
CONNECTION_FIELD_PASSWORD = "password"
CONNECTION_FIELD_SSL_FINGERPRINT = "ssl_fingerprint"


@register_retriever("core/vector/elastic")
class ElasticRetriever(UnstructuredTextRetriever):
    r"""Class for ElasticSearch Retriever

    NOTE
    - `ES_ENDPOINT` environment variable must be set to establish connection with ElasticSearch.
    - `ES_API_KEY` or `ES_USERNAME` and `ES_USERNAME` environment variables must be specified to authenticate connection.

    Args:
        index_name (str): index name
        projection (Dict[str, str]): mappings between returned document's fields and response object's 'text' and 'id' fields. Default is set to {'text': 'text', 'document_id': 'id'}
        limit (Optional[int]): number of hits to return. Default is set to 10.

    .. code-block:: python

        # Initialize retriever
        retriever = ElasticRetriever(index_name="mt-rag-documents", projection={"text": "text", "document_id": "id"})


        # Invoke retriever
        retriever(query="")


    """

    def __init__(
        self,
        index_name: str,
        query_template: str,
        connection: Dict[str, str] = None,
        projection: Dict[str, str] = {"text": "text", "document_id": "id"},
        limit: Optional[int] = 10,
        _id: Optional[str] = str(uuid4()),
        **kwargs: Any,
    ) -> None:
        super().__init__(projection=projection, limit=limit, _id=_id, **kwargs)

        # Step 1: Initialize variables
        self._index_name = index_name

        # Step 2: If connection details are provided, use them
        if connection:
            # Step 2.a: Verify connection field
            if CONNECTION_FIELD_ENDPOINT not in connection:
                raise ValueError("Missing mandaroty 'endpoint' field in the connection field.")

            if CONNECTION_FIELD_API_KEY not in connection or (
                CONNECTION_FIELD_USERNAME not in connection
                and CONNECTION_FIELD_PASSWORD not in connection
            ):
                raise ValueError(
                    "Either 'api_key' or 'username' and 'password' fields must be specified in the connection field."
                )

            # Step 2.b: Establish connection
            es_client_parameters = {}

            # Step 2.b.i: Check if SSL fingerprint is provided
            if (
                connection[CONNECTION_FIELD_SSL_FINGERPRINT]
                and connection[CONNECTION_FIELD_SSL_FINGERPRINT]
            ):
                es_client_parameters["ssl_assert_fingerprint"] = connection[
                    CONNECTION_FIELD_SSL_FINGERPRINT
                ]
            else:
                es_client_parameters["verify_certs"] = False

            # Step 2.b.ii: Determine authentication strategy
            if CONNECTION_FIELD_API_KEY in connection and connection[CONNECTION_FIELD_API_KEY]:
                os.environ["ES_API_KEY"] = connection[CONNECTION_FIELD_API_KEY]
            else:
                try:
                    es_client_parameters["basic_auth"] = (
                        connection[CONNECTION_FIELD_USERNAME],
                        connection[CONNECTION_FIELD_PASSWORD],
                    )
                except KeyError as err:
                    raise ValueError(
                        "Missing mandatory 'username' and 'password' fields in the connection field when 'api_key' field is not specified."
                    ) from err

            # Step 2.b.iii: Instatiate elastic client
            self._client = Elasticsearch(
                connection[CONNECTION_FIELD_ENDPOINT], **es_client_parameters
            )
        else:
            es_client_parameters = {}
            # Step 2.a: Check if SSL fingerprint is provided
            ssl_fingerprint = os.getenv("ES_SSL_FINGERPRINT")
            if ssl_fingerprint:
                es_client_parameters["ssl_assert_fingerprint"] = ssl_fingerprint
            else:
                es_client_parameters["verify_certs"] = False

            # Step 2.b: Determine authentication strategy
            api_key = os.getenv("ES_API_KEY")
            if api_key is None:
                username = os.getenv("ES_USERNAME")
                password = os.getenv("ES_PASSWORD")
                if username and password:
                    es_client_parameters["basic_auth"] = (username, password)
                else:
                    raise ValueError(
                        "Missing mandatory 'ES_USERNAME' and 'ES_PASSWORD' environment variables."
                    )

            # Step 2.c: Instatiate elastic client
            self._client = Elasticsearch(os.getenv("ES_ENDPOINT"), **es_client_parameters)

        # Step 3: Verify query template has necessary variable and is valid JSON
        # Step 3.b: Check mandatory variable presence
        if "${QUERY}" not in query_template:
            raise ValueError('Missing mandatory "${QUERY} variable in the query template.')
        # Step 3.b: Check JSON validity
        try:
            json.loads(query_template)
        except ValueError as err:
            raise ValueError(
                'Provided "query template" must be a valid JSON as per ElasticSearch guidelines.'
            ) from err

        self._query_template = query_template

    # ===========================================================================
    #                       HELPER FUNCTIONS
    # ===========================================================================
    def form_query(self, query_text: str):
        return json.loads(
            self._query_template.replace("${QUERY}", json.dumps(query_text).strip('"'))
        )

    # ===========================================================================
    #                       MAIN PROCESS
    # ===========================================================================
    def __call__(
        self, *args, requests: List[Union[str, dict, None]], limit: int = None, **kwargs
    ) -> List[UnstructuredTextDocument]:
        """
        Top-level process method to retrieving unstructured text documents

        Args:
            requests (List[Union[str, dict]]): requests to be run
            limit (Optional[int]): number of documents to fetch per query.

        Returns:
            List[Document]: retrieved documents
        """
        # Step 1: Set the limt of hits to return
        limit = limit if limit else self._limit

        # Step 2: Execute requests
        hits = []
        for request in requests:
            # Step 2.a: Fetch results based on query in the request
            if request is None:
                # Step 2.a.i: Create random document fetch query, if necessary
                query = {"query": {"function_score": {"random_score": {}}}}
            else:
                # Step 2.a.i: Copy requested query
                query = request

            # Step 2.a.ii: Execute query
            try:
                response = self._client.search(
                    index=self._index_name,
                    **query,
                    size=limit,
                )
            except BadRequestError:
                dgt_logger.warning("Incorrect request: %s", json.dumps(query))

            # Step 2.b: Process response
            processed_results = []
            if (
                "hits" in response.body
                and response.body["hits"]
                and "hits" in response.body["hits"]
                and response.body["hits"]["hits"]
            ):
                for result in response.body["hits"]["hits"]:
                    processed_result = UnstructuredTextDocument(
                        id=result["_source"][self._mappings[PROJECTION_FIELD_ID]],
                        text=result["_source"][self._mappings[PROJECTION_FIELD_TEXT]]
                        .strip()
                        .strip("\n")
                        .strip(),
                    )
                    metadata = {}
                    for dest, source in self._mappings.items():
                        if (
                            dest not in [PROJECTION_FIELD_ID, PROJECTION_FIELD_TEXT]
                            and source in result["_source"]
                            and result["_source"][source]
                        ):
                            metadata[dest] = result["_source"][source]

                    if metadata:
                        processed_results.metadata = metadata

                    # Add created document
                    processed_results.append(processed_result)

                # Step 2.b.ii: Add processed results
                hits.append(processed_results)

        # Step 3: Return
        return hits
