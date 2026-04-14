# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared cache helpers for ToolEnrichment result persistence.

Cache files live under::

    {DGT_CACHE_DIR}/enrichments/{enrichment_type}/{fingerprint}.json

The ``fingerprint`` is a content-addressed SHA-256 hash over inputs specific to
each enrichment type (see individual enrichment modules for what goes into the
hash).  Two tasks that share the same tool set and enrichment config will
automatically reuse the same cache file with no explicit coordination needed.

**Delta-merge** — the cache is a flat dict keyed by ``qualified_tool_name``.
When loading, only entries for tool names that are *not yet in the registry*
are returned as "hits".  When saving, new computed entries are merged into the
existing cache dict and the whole file is rewritten.

Public API
----------
- ``enrichment_cache_path(enrichment_type, fingerprint) -> Path``
- ``load_cache(path) -> dict``
- ``save_cache(path, entries) -> None``
- ``compute_fingerprint(*parts) -> str``
"""

# Standard
from pathlib import Path
from typing import Any, Dict
import hashlib
import json
import logging
import os

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _cache_root() -> Path:
    """Return the DGT cache root directory.

    Reads ``DGT_CACHE_DIR`` from the environment, falling back to ``.cache``
    relative to the current working directory (the project root when the CLI
    is invoked from there).
    """
    return Path(os.environ.get("DGT_CACHE_DIR", ".cache"))


def enrichment_cache_path(enrichment_type: str, fingerprint: str) -> Path:
    """Return the canonical cache file path for a given enrichment run.

    Args:
        enrichment_type: Registry key of the enrichment (e.g. ``"embeddings"``).
        fingerprint: Content-addressed hex digest produced by
            ``compute_fingerprint()``.

    Returns:
        ``Path`` object.  Parent directories are not created here; call
        ``path.parent.mkdir(parents=True, exist_ok=True)`` before writing.
    """
    return _cache_root() / "enrichments" / enrichment_type / f"{fingerprint}.json"


# ---------------------------------------------------------------------------
# Fingerprint helper
# ---------------------------------------------------------------------------


def compute_fingerprint(*parts: Any) -> str:
    """Produce a stable SHA-256 hex digest from an ordered sequence of parts.

    Each part is serialized via ``json.dumps(sort_keys=True)`` before hashing
    so that dict ordering has no effect.

    Args:
        *parts: Any JSON-serializable values (str, dict, list, int, …).

    Returns:
        64-character hex string.
    """
    h = hashlib.sha256()
    for part in parts:
        h.update(json.dumps(part, sort_keys=True, separators=(",", ":")).encode())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Load / save
# ---------------------------------------------------------------------------


def load_cache(path: Path) -> Dict[str, Any]:
    """Load a cache file, returning its contents as a dict.

    Args:
        path: Path returned by ``enrichment_cache_path()``.

    Returns:
        Dict mapping ``qualified_tool_name`` to cached payload.  Returns
        ``{}`` if the file does not exist or cannot be parsed.
    """
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            logger.warning("Enrichment cache at %s is not a dict; ignoring.", path)
            return {}
        return data
    except Exception as exc:
        logger.warning("Could not read enrichment cache at %s: %s", path, exc)
        return {}


def save_cache(path: Path, entries: Dict[str, Any]) -> None:
    """Write (or delta-merge) entries into a cache file.

    If the file already exists, its current contents are read and merged with
    ``entries`` (``entries`` wins on key conflicts) before writing back.

    Args:
        path: Cache file path (parent directories are created if needed).
        entries: New entries to persist; keyed by ``qualified_tool_name``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_cache(path)
    merged = {**existing, **entries}
    try:
        with path.open("w", encoding="utf-8") as fh:
            json.dump(merged, fh, indent=2)
        logger.debug("Enrichment cache saved: %s (%d entries)", path, len(merged))
    except Exception as exc:
        logger.warning("Could not write enrichment cache at %s: %s", path, exc)
