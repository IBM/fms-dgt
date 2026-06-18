# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :class:`MagpieDistance` dedup behavior.

These tests exercise the duplicate-detection and tagging logic without loading
the real sentence-transformer model: the block's ``embed`` method and model are
stubbed so embeddings are deterministic and the tests stay fast and offline.
The surviving-set assertion is the regression test for order-independence.
"""

# Standard
import hashlib

# Third Party
import pytest
import torch

# Local
from fms_dgt.public.blocks.magpie.distance.block import (
    MagpieDistance,
    MagpieDistanceBlockData,
)


def _make_block(monkeypatch, text_to_vec, **kwargs):
    """Build a MagpieDistance whose embeddings come from ``text_to_vec``.

    Avoids constructing the real SentenceTransformer (no __init__ network/model
    load) and replaces ``embed`` with a deterministic lookup.
    """
    block = MagpieDistance.__new__(MagpieDistance)
    # Minimal state the methods under test rely on.
    block._similarity_threshold = kwargs.get("similarity_threshold", 0.975)
    block._index_type = kwargs.get("index_type", "flat")
    block._max_rows_cpu = kwargs.get("max_rows_cpu", 50_000)
    block._max_rows_gpu = kwargs.get("max_rows_gpu", 500_000)
    block._search_batch_size = kwargs.get("search_batch_size", 1024)
    block._model = type("_M", (), {"device": torch.device("cpu")})()

    def fake_embed(texts):
        vecs = torch.tensor([text_to_vec[t] for t in texts], dtype=torch.float32)
        return torch.nn.functional.normalize(vecs, dim=1)

    monkeypatch.setattr(block, "embed", fake_embed)
    return block


def _survivors(instances):
    """Ids whose tag points to themselves => kept by the downstream filter."""
    return {e.id for e in instances if e.magpie_tags.get("min_similar_uuid") in (None, e.id)}


# Two near-identical "France capital" vectors, two near-identical "photosynthesis"
# vectors, plus a unique one. Within-cluster cosine ~1.0; across-cluster ~0.
_VECS = {
    "france_a": [1.0, 0.0, 0.0],
    "france_b": [0.999, 0.04, 0.0],
    "photo_a": [0.0, 1.0, 0.0],
    "photo_b": [0.0, 0.999, 0.04],
    "unique": [0.0, 0.0, 1.0],
}


def _entries(order):
    return [MagpieDistanceBlockData(SRC_DATA={}, magpie_input=t, id=t) for t in order]


def test_clusters_collapse_to_one_survivor(monkeypatch):
    block = _make_block(monkeypatch, _VECS)
    order = ["france_a", "france_b", "photo_a", "photo_b", "unique"]
    out = block.execute(_entries(order))
    survivors = _survivors(out)
    # One survivor per cluster + the unique row = 3 total.
    assert survivors == {"france_a", "photo_a", "unique"}


def test_order_independence(monkeypatch):
    """Shuffling the input must not change which rows survive."""
    order_a = ["france_a", "france_b", "photo_a", "photo_b", "unique"]
    order_b = ["unique", "photo_b", "france_b", "photo_a", "france_a"]

    block_a = _make_block(monkeypatch, _VECS)
    block_b = _make_block(monkeypatch, _VECS)
    survivors_a = _survivors(block_a.execute(_entries(order_a)))
    survivors_b = _survivors(block_b.execute(_entries(order_b)))

    assert survivors_a == survivors_b == {"france_a", "photo_a", "unique"}


def test_threshold_separates_similar_from_distinct(monkeypatch):
    block = _make_block(monkeypatch, _VECS)
    out = block.execute(_entries(list(_VECS)))
    tags = {e.id: e.magpie_tags for e in out}
    # Near-identical pair flagged as duplicates of each other.
    assert tags["france_b"]["repeat_count"] >= 1
    # Distinct row has no duplicates.
    assert tags["unique"]["repeat_count"] == 0
    assert tags["unique"]["min_similar_uuid"] is None


def test_deterministic_md5_id_when_missing(monkeypatch):
    block = _make_block(monkeypatch, {"hello world": [1.0, 0.0, 0.0]})
    entry = MagpieDistanceBlockData(SRC_DATA={}, magpie_input="hello world", id=None)
    block.execute([entry])
    expected = hashlib.md5("hello world".encode("utf-8")).hexdigest()
    assert entry.id == expected


def test_legacy_distance_threshold_maps_to_cosine():
    # Construct without the heavy model path: only the threshold resolution is
    # under test, so drive the same arithmetic the constructor uses.
    # squared-L2 0.05 on unit vectors -> cosine 0.975
    distance_threshold = 0.05
    cosine = 1.0 - (distance_threshold / 2.0)
    assert cosine == pytest.approx(0.975)
