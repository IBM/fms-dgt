# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import hashlib

# Third Party
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np
import torch

# Local
from fms_dgt.base.block import Block, BlockData
from fms_dgt.base.registry import register_block


@dataclass(kw_only=True)
class MagpieDistanceBlockData(BlockData):
    """
    Data type for Magpie tagging block
    """

    # ===========================================================================
    #                       INPUT FIELDS
    # ===========================================================================
    magpie_input: Optional[str] = None

    # For multi-turn setting, the input must be array of dictionaries
    # with each item in dictionary containing "text"/"content" field
    magpie_mt_input: Optional[List[Dict[str, Any]]] = None

    # UUID for each entry
    id: Optional[str] = None

    # ===========================================================================
    #                       OUTPUT FIELDS
    # ===========================================================================
    magpie_tags: Optional[Dict] = None


@register_block("magpie_distance")
class MagpieDistance(Block):
    r"""Class for Magpie Distance based duplicate identification.

    Embeds every input, finds near-duplicates via cosine similarity, and tags
    each row with the *canonical* member of its duplicate cluster
    (``min_similar_uuid``). A downstream ``magpie_filter`` keeps a row only when
    its ``min_similar_uuid`` points to itself, so each duplicate cluster
    collapses to a single survivor.

    The canonical member is the one with the lexicographically smallest ``id``
    in the cluster. Because ``id`` is a stable property of the row (and is
    derived from a content hash when not supplied), the surviving set does not
    depend on the order in which inputs are presented.

    Two search backends are available:

    * **matmul** (default for the exact path) computes cosine similarity with a
      batched matrix multiply on the device that ``SentenceTransformer`` picks
      (GPU when available, else CPU). This is exact and needs no GPU build of
      FAISS.
    * **faiss** (HNSW / IVF) is an approximate escape hatch for very large ``N``
      on CPU-only machines, where the exact O(N^2) path is impractical.

    Args:
        sentence_model (str): sentence model to use for encoding. Defaults to
            ``sentence-transformers/all-mpnet-base-v2``.
        distance_threshold (float): squared-L2 distance cutoff below which two
            inputs are considered duplicates (smaller distance == more similar).
            Defaults to ``0.05``. Internally, embeddings are L2-normalized and
            this distance is converted to an equivalent cosine-similarity cutoff
            via ``cosine = 1 - distance / 2`` (so ``0.05`` -> cosine ``0.975``);
            the public knob keeps its original distance units and meaning.
        index_type (str): one of ``auto`` | ``flat`` | ``hnsw`` | ``ivf``.
            ``auto`` (default) picks exact matmul when it fits the device and an
            approximate FAISS index otherwise.
        max_rows_cpu (int): in ``auto`` mode on a CPU-only device, the largest
            number of input rows (``N``) handled by the exact all-pairs search.
            Above this, ``auto`` falls back to an approximate FAISS index because
            the exact path is O(N^2) and impractical on CPU at scale. Defaults to
            ``50_000``.
        max_rows_gpu (int): in ``auto`` mode, the largest number of input rows
            (``N``) handled by the exact all-pairs search even when a GPU is
            available. Above this, ``auto`` falls back to an approximate FAISS
            index. Defaults to ``500_000``.
        search_space_size (int): retained for backward compatibility; unused by
            the threshold-based duplicate detection. Defaults to ``500``.
        search_batch_size (int): number of query rows per search batch. Defaults
            to ``1024``.
        encoding_batch_size (int): number of entries to encode in a batch.
            Defaults to ``65536``.

    .. code-block:: python

        # Initialize dedup calculator
        dedup_calculator = MagpieDistance()

        # Sample data
        data = [
                {
                    "question": "what is capital of the United States of America?",
                    "answer": "Washington D.C"
                },
                {
                    "question": "What is biggest star in our solar system?",
                    "answer": "Sun is the biggest star in our solar system."
                }
            ]

        # Invoke dedup calculator
        dedup_calculator(data)
    """

    DATA_TYPE = MagpieDistanceBlockData

    def __init__(
        self,
        sentence_model: str = "sentence-transformers/all-mpnet-base-v2",
        distance_threshold: float = 0.05,
        index_type: str = "auto",
        max_rows_cpu: int = 50000,
        max_rows_gpu: int = 500000,
        search_space_size: int = 500,
        search_batch_size: int = 1024,
        encoding_batch_size: int = 65536,
        input_map: Optional[Union[List, Dict]] = None,
        output_map: Optional[Union[List, Dict]] = None,
        **kwargs: Any,
    ) -> None:
        # Set default values for "input_map" & "output_map", if necessary
        if input_map is None:
            input_map = {
                "id": "id",
                "input": "magpie_input",
            }

        if output_map is None:
            output_map = {
                "id": "id",
                "magpie_tags": "magpie_tags",
            }

        # Initialize parent
        super().__init__(input_map=input_map, output_map=output_map, **kwargs)

        # The public knob is a squared-L2 distance (smaller == more similar).
        # Embeddings are L2-normalized, so squared-L2 and cosine are equivalent
        # (``squared_l2 = 2 - 2 * cosine``); convert once to a cosine cutoff that
        # the search paths use directly.
        self._distance_threshold = distance_threshold
        self._similarity_threshold = 1.0 - (distance_threshold / 2.0)

        index_type = index_type.lower()
        if index_type not in ("auto", "flat", "hnsw", "ivf"):
            raise ValueError(
                f"index_type must be one of 'auto', 'flat', 'hnsw', 'ivf'; got {index_type!r}"
            )
        self._index_type = index_type
        self._max_rows_cpu = max_rows_cpu
        self._max_rows_gpu = max_rows_gpu

        # Retained for backward compatibility; not used by the duplicate logic.
        self._search_space_size = search_space_size
        self._search_batch_size = search_batch_size
        self._encoding_batch_size = encoding_batch_size

        self._model = SentenceTransformer(sentence_model)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(device=device, dtype=torch.float32)

        self.logger.info("The model is loaded on device: %s", self._model.device)

    # ------------------------------------------------------------------
    #  Embedding
    # ------------------------------------------------------------------
    def embed(self, texts: List[str]) -> torch.Tensor:
        """Encode ``texts`` into L2-normalized embeddings on the model device.

        Returns a contiguous ``float32`` tensor of shape ``(N, d)`` left on the
        model's device (GPU when available) so the exact matmul path can run
        there without an extra host transfer.
        """
        chunks: List[torch.Tensor] = []
        for i in range(0, len(texts), self._encoding_batch_size):
            batch = texts[i : i + self._encoding_batch_size]
            emb = self._model.encode(
                batch,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=True,
            )
            chunks.append(emb.to(dtype=torch.float32))
        return torch.cat(chunks, dim=0).contiguous()

    # ------------------------------------------------------------------
    #  Backend selection
    # ------------------------------------------------------------------
    def _select_backend(self, n: int) -> str:
        """Resolve ``index_type='auto'`` to a concrete backend for ``n`` rows.

        Returns ``"matmul"`` for the exact path or ``"hnsw"`` for the
        approximate path. Explicit ``index_type`` values are honored as-is
        (``flat`` maps to ``matmul``).
        """
        if self._index_type == "flat":
            return "matmul"
        if self._index_type in ("hnsw", "ivf"):
            return self._index_type

        # auto
        on_gpu = self._model.device.type == "cuda"
        max_rows = self._max_rows_gpu if on_gpu else self._max_rows_cpu
        if n <= max_rows and n <= self._max_rows_gpu:
            return "matmul"

        backend = "hnsw"
        self.logger.warning(
            "magpie_distance: N=%d on device=%s exceeds the exact-search limit "
            "(%d rows); routing to approximate '%s' index. Set index_type "
            "explicitly to override.",
            n,
            self._model.device.type,
            max_rows,
            backend,
        )
        return backend

    # ------------------------------------------------------------------
    #  Exact search (matmul)
    # ------------------------------------------------------------------
    def _search_matmul(self, embeddings: torch.Tensor) -> List[Tuple[List[int], List[float]]]:
        """Exact cosine duplicate search via batched matrix multiply.

        For each row, returns ``(neighbor_indices, neighbor_similarities)`` for
        all other rows whose cosine similarity is at or above the threshold.
        """
        n = embeddings.shape[0]
        results: List[Tuple[List[int], List[float]]] = [([], []) for _ in range(n)]
        for start in tqdm(range(0, n, self._search_batch_size)):
            end = min(start + self._search_batch_size, n)
            # (B, N) cosine similarity slab on the active device.
            sims = embeddings[start:end] @ embeddings.T
            mask = sims >= self._similarity_threshold
            rows, cols = torch.nonzero(mask, as_tuple=True)
            scores = sims[rows, cols].cpu().numpy()
            rows = rows.cpu().numpy()
            cols = cols.cpu().numpy()
            for r, c, s in zip(rows, cols, scores):
                q = start + int(r)
                c = int(c)
                if c != q:
                    results[q][0].append(c)
                    results[q][1].append(float(s))
        return results

    # ------------------------------------------------------------------
    #  Approximate search (FAISS HNSW / IVF) -- escape hatch
    # ------------------------------------------------------------------
    def _search_faiss(
        self, embeddings: torch.Tensor, backend: str
    ) -> List[Tuple[List[int], List[float]]]:
        """Approximate cosine duplicate search via a FAISS HNSW/IVF index.

        Used only for very large ``N`` on CPU-only machines, where the exact
        O(N^2) matmul is impractical. ``faiss`` is imported lazily so it never
        loads on the default (exact) path.
        """
        # Third Party
        import faiss  # noqa: PLC0415 -- lazy: keep faiss off the default code path

        vectors = embeddings.cpu().numpy().astype(np.float32)
        n, d = vectors.shape
        # Inner product on unit vectors == cosine similarity.
        if backend == "ivf":
            nlist = max(1, min(int(np.sqrt(n)), n))
            quantizer = faiss.IndexFlatIP(d)
            index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
            index.train(vectors)
            index.nprobe = min(nlist, 16)
        else:  # hnsw
            index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efSearch = 64
        index.add(vectors)
        self.logger.info(
            "magpie_distance: built approximate FAISS '%s' index over %d vectors.",
            backend,
            n,
        )

        # range_search returns all neighbors with inner product >= threshold.
        results: List[Tuple[List[int], List[float]]] = [([], []) for _ in range(n)]
        for start in tqdm(range(0, n, self._search_batch_size)):
            end = min(start + self._search_batch_size, n)
            lims, dists, idxs = index.range_search(
                vectors[start:end], float(self._similarity_threshold)
            )
            for local_q in range(end - start):
                q = start + local_q
                sl = slice(lims[local_q], lims[local_q + 1])
                for c, s in zip(idxs[sl], dists[sl]):
                    if int(c) != q:
                        results[q][0].append(int(c))
                        results[q][1].append(float(s))
        return results

    # ------------------------------------------------------------------
    #  Tagging
    # ------------------------------------------------------------------
    def _tag_duplicates(
        self,
        neighbors: List[Tuple[List[int], List[float]]],
        ids: List[str],
        instances: List[MagpieDistanceBlockData],
    ) -> None:
        """Populate ``magpie_tags`` with the order-independent dedup result.

        The canonical survivor of a cluster is the row with the smallest ``id``
        among the cluster. A row keeps itself iff it *is* that canonical row.
        ``min_neighbor_distance`` is the squared-L2 distance to the nearest
        duplicate (converted from cosine similarity via ``2 - 2 * cosine``).
        """
        for i, (dup_indices, dup_sims) in enumerate(neighbors):
            if dup_indices:
                cluster_ids = [ids[i]] + [ids[j] for j in dup_indices]
                min_similar_uuid = min(cluster_ids)
                repeat_count = len(dup_indices)
                # Convert max cosine similarity back to squared-L2 distance to
                # match the original field's units (smaller == more similar).
                min_neighbor_distance = 2.0 - 2.0 * max(dup_sims)
            else:
                min_similar_uuid = None
                repeat_count = 0
                min_neighbor_distance = None

            if not instances[i].magpie_tags:
                instances[i].magpie_tags = {}

            instances[i].magpie_tags["min_neighbor_distance"] = min_neighbor_distance
            instances[i].magpie_tags["repeat_count"] = repeat_count
            instances[i].magpie_tags["min_similar_uuid"] = min_similar_uuid

    # ------------------------------------------------------------------
    #  Entry point
    # ------------------------------------------------------------------
    def execute(self, inputs: List[MagpieDistanceBlockData]) -> List[MagpieDistanceBlockData]:
        # Cast to list, if necessary
        inputs = [entry for entry in inputs] if isinstance(inputs, map) else inputs

        texts, ids = self._prepare(inputs)
        if not texts:
            return inputs

        embeddings = self.embed(texts)
        backend = self._select_backend(len(texts))
        if backend == "matmul":
            neighbors = self._search_matmul(embeddings)
        else:
            neighbors = self._search_faiss(embeddings, backend)

        self._tag_duplicates(neighbors, ids, inputs)
        return inputs

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------
    def _prepare(self, inputs: List[MagpieDistanceBlockData]) -> Tuple[List[str], List[str]]:
        """Flatten each entry to its text and resolve a stable id.

        Missing ids are derived from an md5 of the text so they are stable
        across runs (matching the annotate pipeline's id convention), rather
        than a random UUID.
        """
        texts: List[str] = []
        ids: List[str] = []
        for entry in inputs:
            text = self._extract_text(entry)
            if entry.id is None:
                entry.id = hashlib.md5(text.encode("utf-8")).hexdigest()
            texts.append(text)
            ids.append(entry.id)
        return texts, ids

    @staticmethod
    def _extract_text(entry: MagpieDistanceBlockData) -> str:
        """Return the user-side text for an entry (single- or multi-turn)."""
        if entry.magpie_mt_input:
            user_utterance_texts = []
            for utterance in entry.magpie_mt_input:
                # Identify role field
                if "from" in utterance:
                    role_field = "from"
                elif "role" in utterance:
                    role_field = "role"
                elif "speaker" in utterance:
                    role_field = "speaker"
                else:
                    raise ValueError(
                        "\"magpie_mt_input\" should have a 'from' field or a 'role' field or a 'speaker' field to signify whether it was a user or assistant utterance"
                    )

                # Identify text/content field
                if "value" in utterance:
                    txt_field = "value"
                elif "content" in utterance:
                    txt_field = "content"
                elif "text" in utterance:
                    txt_field = "text"
                else:
                    # Skipping messages without content
                    continue
                if utterance[role_field] == "user":
                    user_utterance_texts.append(utterance[txt_field])

            return ("\n\n".join(user_utterance_texts)).strip()
        return entry.magpie_input.strip()
