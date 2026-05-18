# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any, Dict, List, Optional
import logging

# Third Party
import requests

# Local
from fms_dgt.core.tools.engines.search.base import Document

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 10
_USER_AGENT = "fms-dgt/0.1"


def fetch_url_raw(url: str, timeout: int = _DEFAULT_TIMEOUT) -> Optional[str]:
    """Fetch a URL and return the raw HTML response body.

    Returns ``None`` on any HTTP or network error.

    Args:
        url: Page URL to fetch.
        timeout: Request timeout in seconds.
    """
    try:
        resp = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": _USER_AGENT},
        )
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as exc:
        logger.debug("fetch_url_raw: failed for %s: %s", url, exc)
        return None


def extract_text(html: str) -> Optional[str]:
    """Extract clean main-content text from an HTML string using trafilatura.

    Returns ``None`` if trafilatura produces no output (e.g. the page is
    mostly boilerplate with no extractable article content).

    Requires ``trafilatura>=1.8.0`` (Apache 2.0).  Raises ``ImportError`` with
    an actionable install message if the package is absent.

    Args:
        html: Raw HTML string to process.
    """
    try:
        # Third Party
        import trafilatura
    except ImportError as exc:
        raise ImportError(
            "Full-text extraction requires 'trafilatura>=1.8.0'. "
            "Install with: pip install fms_dgt[search]"
        ) from exc

    return trafilatura.extract(html, include_comments=False, include_tables=True) or None


# ---------------------------------------------------------------------------
# Shared helpers for web engine result assembly
# ---------------------------------------------------------------------------


def _validate_content(content: List[str]) -> List[str]:
    """Validate and deduplicate a content type list.

    Raises:
        ValueError: If any entry is not one of ``"snippet"``, ``"text"``, ``"raw"``.
    """
    unknown = set(content) - {"snippet", "text", "raw"}
    if unknown:
        raise ValueError(
            f"Unknown content type(s): {unknown}. " f'Valid values are: "snippet", "text", "raw"'
        )
    # Preserve order but deduplicate.
    seen = set()
    return [c for c in content if not (c in seen or seen.add(c))]


def _build_document(
    idx: int,
    snippet: str,
    url: str,
    title: str,
    content: List[str],
) -> Document:
    """Assemble a ``Document`` from a web search hit.

    Fetches and extracts additional content types as requested.  ``Document.text``
    is set to the highest-fidelity type successfully retrieved, in priority
    order: ``text`` > ``snippet`` > ``raw``.  Each requested type is stored
    in ``Document.metadata`` under its type name.

    Args:
        idx: Result position (used as document ID).
        snippet: Short excerpt returned by the search engine.
        url: Source URL of the result.
        title: Page title.
        content: List of requested content types.
    """
    collected: Dict[str, str] = {}

    if "snippet" in content and snippet:
        collected["snippet"] = snippet

    if "raw" in content or "text" in content:
        html = fetch_url_raw(url)
        if html is not None:
            if "raw" in content:
                collected["raw"] = html
            if "text" in content:
                extracted = extract_text(html)
                if extracted:
                    collected["text"] = extracted

    # Populate Document.text with highest-fidelity available type.
    primary_text = ""
    for preferred in ["text", "snippet", "raw"]:
        if preferred in collected:
            primary_text = collected[preferred]
            break
    # Final fallback: use snippet even if not explicitly requested.
    if not primary_text and snippet:
        primary_text = snippet

    metadata: Dict[str, Any] = {"title": title, "source": url}
    metadata.update(collected)

    return Document(id=str(idx), text=primary_text, score=None, metadata=metadata)
