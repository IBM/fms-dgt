# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for GenerationTask.sample_examples().

Uses a minimal stub that bypasses GenerationTask.__init__ and wires up only
the attributes sample_examples() touches:
    - _seed_batch_size
    - _machine_batch_size
    - machine_data
    - get_seed_examples  (patched directly; sample_examples delegates to it)
    - _dataloader        (sentinel to verify it is never advanced)

seed_fraction=None behaviour: fraction = seed_batch_size / (seed_batch_size +
machine_batch_size) is applied to k, so k is always the total cap.
"""

# Standard
from dataclasses import dataclass
from typing import List
from unittest.mock import MagicMock
import random

# Third Party
import pytest

# Local
from fms_dgt.base.data_objects import DataPoint
from fms_dgt.base.task import GenerationTask


# ===========================================================================
#                       STUBS
# ===========================================================================
@dataclass(kw_only=True)
class SeedDataPoint(DataPoint):
    text: str = ""
    is_seed: bool = True


@dataclass(kw_only=True)
class SynthDataPoint(DataPoint):
    text: str = ""


def _make_seeds(n: int) -> List[SeedDataPoint]:
    return [SeedDataPoint(task_name="test", text=f"seed-{i}") for i in range(n)]


def _make_synth(n: int) -> List[SynthDataPoint]:
    return [SynthDataPoint(task_name="test", text=f"synth-{i}") for i in range(n)]


def _make_stub_task(
    seed_pool: List[SeedDataPoint],
    machine_items: List[SynthDataPoint],
    seed_batch_size: int = 2,
    machine_batch_size: int = 2,
) -> GenerationTask:
    """Create a GenerationTask stub wired to sample_examples() internals only."""
    task = object.__new__(GenerationTask)
    task._seed_batch_size = seed_batch_size
    task._machine_batch_size = machine_batch_size
    task.machine_data = list(machine_items)
    task.get_seed_examples = MagicMock(return_value=list(seed_pool))
    task._dataloader = MagicMock(name="shared_dataloader")
    return task


# ===========================================================================
#                       DEFAULTS (seed_fraction=None)
# ===========================================================================
class TestSampleExamplesDefaults:
    def test_returns_expected_total_when_pools_sufficient(self):
        # fraction = 2/(2+2) = 0.5, k=4 → n_seed=2, n_synth=2
        task = _make_stub_task(
            _make_seeds(10), _make_synth(10), seed_batch_size=2, machine_batch_size=2
        )
        results = task.sample_examples(k=4)
        assert len(results) == 4

    def test_seed_synthetic_split_matches_ratio(self):
        # fraction = 3/(3+1) = 0.75, k=8 → n_seed=6, n_synth=2
        task = _make_stub_task(
            _make_seeds(10), _make_synth(10), seed_batch_size=3, machine_batch_size=1
        )
        results = task.sample_examples(k=8)
        seeds = [r for r in results if r.is_seed]
        synths = [r for r in results if not r.is_seed]
        assert len(seeds) == 6
        assert len(synths) == 2

    def test_seeds_only_when_machine_batch_size_zero(self):
        # fraction = 2/(2+0) = 1.0 → n_seed=k, n_synth=0
        task = _make_stub_task(_make_seeds(10), [], seed_batch_size=2, machine_batch_size=0)
        results = task.sample_examples(k=4)
        assert len(results) == 4
        assert all(r.is_seed for r in results)

    def test_synthetic_only_when_seed_batch_size_zero(self):
        # fraction = 0/(0+3) = 0.0 → n_seed=0, n_synth=k
        task = _make_stub_task([], _make_synth(10), seed_batch_size=0, machine_batch_size=3)
        results = task.sample_examples(k=4)
        assert len(results) == 4
        assert all(not r.is_seed for r in results)

    def test_shortfall_on_seeds_no_backfill(self):
        # fraction=0.5, k=4 → n_seed=2, n_synth=2; seed pool has only 1
        task = _make_stub_task(
            _make_seeds(1), _make_synth(10), seed_batch_size=2, machine_batch_size=2
        )
        results = task.sample_examples(k=4)
        seeds = [r for r in results if r.is_seed]
        synths = [r for r in results if not r.is_seed]
        assert len(seeds) == 1  # only 1 available, no backfill
        assert len(synths) == 2  # synthetic quota unaffected

    def test_shortfall_on_synthetic_no_backfill(self):
        # fraction=0.5, k=4 → n_seed=2, n_synth=2; machine_data has only 1
        task = _make_stub_task(
            _make_seeds(10), _make_synth(1), seed_batch_size=2, machine_batch_size=2
        )
        results = task.sample_examples(k=4)
        seeds = [r for r in results if r.is_seed]
        synths = [r for r in results if not r.is_seed]
        assert len(seeds) == 2  # seed quota unaffected
        assert len(synths) == 1  # only 1 available, no backfill

    def test_does_not_advance_shared_dataloader(self):
        task = _make_stub_task(_make_seeds(4), _make_synth(4))
        task.sample_examples(k=4)
        task._dataloader.__next__.assert_not_called()

    def test_empty_both_pools_returns_empty(self):
        task = _make_stub_task([], [], seed_batch_size=2, machine_batch_size=2)
        assert task.sample_examples(k=4) == []


# ===========================================================================
#                       EXPLICIT SEED_FRACTION
# ===========================================================================
class TestSampleExamplesSeedFraction:
    def test_seeds_only(self):
        task = _make_stub_task(_make_seeds(10), _make_synth(10))
        results = task.sample_examples(k=4, seed_fraction=1.0)
        assert len(results) == 4
        assert all(r.is_seed for r in results)

    def test_synthetic_only(self):
        task = _make_stub_task(_make_seeds(10), _make_synth(10))
        results = task.sample_examples(k=4, seed_fraction=0.0)
        assert len(results) == 4
        assert all(not r.is_seed for r in results)

    def test_mixed_fraction(self):
        task = _make_stub_task(_make_seeds(10), _make_synth(10))
        results = task.sample_examples(k=4, seed_fraction=0.5)
        # round(4 * 0.5) = 2 seeds, 2 synthetic
        assert sum(1 for r in results if r.is_seed) == 2
        assert sum(1 for r in results if not r.is_seed) == 2

    def test_invalid_seed_fraction_raises(self):
        task = _make_stub_task([], [])
        with pytest.raises(ValueError):
            task.sample_examples(k=4, seed_fraction=1.5)

    def test_negative_seed_fraction_raises(self):
        task = _make_stub_task([], [])
        with pytest.raises(ValueError):
            task.sample_examples(k=4, seed_fraction=-0.1)

    def test_k_governs_total_with_explicit_fraction(self):
        task = _make_stub_task(_make_seeds(10), _make_synth(10))
        results = task.sample_examples(k=6, seed_fraction=0.5)
        # round(6 * 0.5) = 3 seeds, 3 synthetic
        assert len(results) == 6


# ===========================================================================
#                       RANDOMNESS AND UNIQUENESS
# ===========================================================================
class TestRandomSampling:
    def test_synthetic_subset_is_distinct(self):
        random.seed(0)
        task = _make_stub_task([], _make_synth(100), seed_batch_size=0, machine_batch_size=10)
        results = task.sample_examples(k=10, seed_fraction=0.0)
        assert len(results) == len(set(id(r) for r in results))

    def test_seed_subset_is_distinct(self):
        random.seed(0)
        task = _make_stub_task(_make_seeds(100), [], seed_batch_size=10, machine_batch_size=0)
        results = task.sample_examples(k=10, seed_fraction=1.0)
        assert len(results) == len(set(id(r) for r in results))

    def test_pool_smaller_than_quota_returns_all_without_error(self):
        task = _make_stub_task([], _make_synth(3), seed_batch_size=0, machine_batch_size=10)
        results = task.sample_examples(k=100, seed_fraction=0.0)
        assert len(results) == 3

    def test_get_seed_examples_called_once_per_invocation(self):
        task = _make_stub_task(_make_seeds(10), _make_synth(10))
        task.sample_examples(k=4)
        task.get_seed_examples.assert_called_once()
