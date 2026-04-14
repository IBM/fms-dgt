# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any, Dict, List

# Third Party
from sentence_transformers import SentenceTransformer
import numpy as np
import torch

# Local
from fms_dgt.core.tools.constants import (
    TOOL_DESCRIPTION,
    TOOL_PROPERTIES,
    TOOL_TYPE,
)
from fms_dgt.core.tools.data_objects import Tool
from fms_dgt.core.tools.enrichments.base import ToolEnrichment, register_tool_enrichment
from fms_dgt.core.tools.enrichments.cache import (
    compute_fingerprint,
    enrichment_cache_path,
    load_cache,
    save_cache,
)
from fms_dgt.core.tools.registry import schema_fingerprint

# ===========================================================================
#                       TEXT ENCODING HELPER
# ===========================================================================


def _tool_to_text(tool: Tool) -> str:
    """Convert a tool to a compact text representation for embedding.

    Mirrors the encoding used by the internal ``cluster_tools.py`` script:
    tool name + description + input parameter names/types/descriptions +
    output parameter names/types/descriptions (when present).
    """
    lines = [f'This is a "{tool.name}" tool.']
    if tool.description:
        lines.append(tool.description)

    input_props = (tool.parameters or {}).get(TOOL_PROPERTIES) or {}
    if input_props:
        lines.append("Args:")
        for param_name, param_info in input_props.items():
            if not isinstance(param_info, dict):
                lines.append(f"  {param_name}")
                continue
            type_str = param_info.get(TOOL_TYPE, "")
            descr = param_info.get(TOOL_DESCRIPTION, "")
            type_part = f" ({type_str})" if type_str else ""
            descr_part = f": {descr}" if descr else ""
            lines.append(f"  {param_name}{type_part}{descr_part}")

    output_props = (tool.output_parameters or {}).get(TOOL_PROPERTIES) or {}
    if output_props:
        lines.append("Returns:")
        for param_name, param_info in output_props.items():
            if not isinstance(param_info, dict):
                lines.append(f"  {param_name}")
                continue
            type_str = param_info.get(TOOL_TYPE, "")
            descr = param_info.get(TOOL_DESCRIPTION, "")
            type_part = f" ({type_str})" if type_str else ""
            descr_part = f": {descr}" if descr else ""
            lines.append(f"  {param_name}{type_part}{descr_part}")

    return "\n".join(lines)


# ===========================================================================
#                       ENRICHMENT
# ===========================================================================


@register_tool_enrichment("embeddings")
class EmbeddingsEnrichment(ToolEnrichment):
    """Embed each tool and store vectors in ``registry.artifacts["embeddings"]``.

    Uses a ``SentenceTransformer`` model.  Does not modify any ``Tool`` object.

    The artifact written to ``registry.artifacts["embeddings"]`` is a
    ``dict[qualified_tool_name, dict[schema_fp, np.ndarray]]`` where
    ``schema_fp`` is the ``schema_fingerprint()`` of the tool's input
    parameters.  This two-level structure uniquely identifies each overload:
    tools with the same name but different input schemas get separate entries
    under the same ``qualified_name`` key.

    Results are cached under ``{DGT_CACHE_DIR}/enrichments/embeddings/
    {fingerprint}.json`` and delta-merged on subsequent runs so that only
    tools missing from the cache are re-embedded.

    Args:
        model: Sentence-transformer model identifier or local path.
            Defaults to ``"sentence-transformers/all-mpnet-base-v2"``.
        force: If ``True``, bypass the cache and re-embed all tools.
    """

    artifact_key: str = "embeddings"

    def __init__(
        self,
        model: str = "sentence-transformers/all-mpnet-base-v2",
        force: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._model_name = model
        self._force = force
        self._model = None  # lazy-loaded on first enrich() call

    def _get_model(self):
        if self._model is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model = SentenceTransformer(self._model_name).to(
                device=device, dtype=torch.float32
            )
        return self._model

    def _cache_fingerprint(self, tools: List[Tool]) -> str:
        """Fingerprint: sorted (qualified_name, schema_fp, tool_text) triples + model name."""
        tool_ids = sorted(
            (t.qualified_name, schema_fingerprint(t.parameters), _tool_to_text(t)) for t in tools
        )
        return compute_fingerprint(tool_ids, self._model_name)

    @staticmethod
    def _tool_cache_key(tool: Tool) -> str:
        """Stable cache key for a single tool: ``qualified_name::schema_fp``."""
        return f"{tool.qualified_name}::{schema_fingerprint(tool.parameters)}"

    def enrich(self, registry: Any) -> None:
        """Embed all tools and store in ``registry.artifacts["embeddings"]``.

        Args:
            registry: ``ToolRegistry`` instance.
        """
        tools = registry.all_tools()
        if not tools:
            registry.artifacts[EmbeddingsEnrichment.artifact_key] = {}
            return

        # --- Cache lookup ---------------------------------------------------
        fingerprint = self._cache_fingerprint(tools)
        cache_path = enrichment_cache_path("embeddings", fingerprint)

        # Cache is keyed by "qualified_name::schema_fp" → list (serialized vector).
        cached_raw: Dict[str, Any] = {}
        if not getattr(self, "_force", False):
            cached_raw = load_cache(cache_path)
            if cached_raw:
                self.logger.debug(
                    "EmbeddingsEnrichment: loaded cache from %s (%d entries)",
                    cache_path,
                    len(cached_raw),
                )

        # Convert cached lists back to numpy arrays keyed by cache key.
        cached_flat: Dict[str, np.ndarray] = {
            k: np.array(vec, dtype=np.float32) for k, vec in cached_raw.items()
        }

        tools_to_embed = [t for t in tools if self._tool_cache_key(t) not in cached_flat]

        if not tools_to_embed:
            self.logger.info(
                "EmbeddingsEnrichment: all %d tool(s) satisfied from cache", len(tools)
            )
            registry.artifacts[EmbeddingsEnrichment.artifact_key] = self._build_artifact(
                tools, cached_flat
            )
            return

        self.logger.info(
            "EmbeddingsEnrichment: embedding %d tool(s) (%d cache hits)",
            len(tools_to_embed),
            len(tools) - len(tools_to_embed),
        )

        texts = [_tool_to_text(t) for t in tools_to_embed]
        model = self._get_model()

        with torch.no_grad():
            vectors = model.encode(
                texts,
                convert_to_tensor=True,
                show_progress_bar=False,
            )
            vectors_np = vectors.cpu().numpy()

        new_flat: Dict[str, np.ndarray] = {
            self._tool_cache_key(t): vectors_np[i] for i, t in enumerate(tools_to_embed)
        }

        # Persist new entries (serialized as lists).
        save_cache(
            cache_path,
            {k: vec.tolist() for k, vec in new_flat.items()},
        )

        merged_flat = {**cached_flat, **new_flat}
        registry.artifacts[EmbeddingsEnrichment.artifact_key] = self._build_artifact(
            tools, merged_flat
        )

        self.logger.info(
            "EmbeddingsEnrichment: stored embeddings for %d tool(s) (dim=%d)",
            len(tools),
            vectors_np.shape[1] if vectors_np.ndim == 2 else vectors_np.shape[0],
        )

    @staticmethod
    def _build_artifact(
        tools: List[Tool], flat: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Build the two-level artifact ``{qualified_name: {schema_fp: vector}}``."""
        artifact: Dict[str, Dict[str, np.ndarray]] = {}
        for tool in tools:
            fp = schema_fingerprint(tool.parameters)
            vec = flat.get(f"{tool.qualified_name}::{fp}")
            if vec is not None:
                artifact.setdefault(tool.qualified_name, {})[fp] = vec
        return artifact
