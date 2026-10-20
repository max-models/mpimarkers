import cunumpy as xp
import numpy as np
import pytest

from mpiparticles import IdAllocator
from mpiparticles.tests.conftest import FakeComm


def test_unique_across_simulated_ranks():
    size = 5
    allocators = [IdAllocator(rank, size) for rank in range(size)]
    drawn = []
    for round_ in range(4):
        for allocator in allocators:
            drawn.extend(
                np.asarray(xp.to_numpy(allocator.allocate(round_ + 1))).tolist()
            )
    assert len(drawn) == len(set(drawn)) == 5 * (1 + 2 + 3 + 4)
    for rank, allocator in enumerate(allocators):
        assert (allocator.next_id - rank) % size == 0


def test_allocate_is_int64_and_strided():
    ids = IdAllocator(2, 3).allocate(4)
    assert ids.dtype == np.int64
    assert np.asarray(xp.to_numpy(ids)).tolist() == [2, 5, 8, 11]
    assert IdAllocator().allocate(0).shape == (0,)


def test_from_comm_is_independent_of_any_decomposition():
    allocator = IdAllocator.from_comm(FakeComm(3, 8))
    assert (allocator.rank, allocator.size) == (3, 8)


def test_advance_past_stays_in_the_rank_sequence():
    allocator = IdAllocator(1, 4)
    allocator.advance_past(xp.asarray([2**60 + 6]))
    assert allocator.next_id > 2**60 + 6 and (allocator.next_id - 1) % 4 == 0
    before = allocator.next_id
    allocator.advance_past(xp.asarray([3]))  # never goes back
    allocator.advance_past(xp.zeros(0, dtype=np.int64))
    assert allocator.next_id == before


def test_next_id_must_belong_to_the_sequence():
    allocator = IdAllocator(1, 4)
    allocator.next_id = 9
    with pytest.raises(ValueError):
        allocator.next_id = 10
    with pytest.raises(ValueError):
        IdAllocator(4, 4)
    with pytest.raises(ValueError):
        IdAllocator(0, 4, next_id=3)
